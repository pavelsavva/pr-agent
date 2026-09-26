from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pr_agent.algo.review_finding_state import (
    append_review_state,
    build_review_state_comment,
    build_round_summary,
    extract_reviewed_files,
    fit_review_state,
    insert_round_summary,
    parse_review_state,
    reconcile_review_findings,
    serialize_review_state,
)
from pr_agent.algo.utils import _ALL_COMMENT_IDENTITIES, PRReviewStateIdentity
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.git_provider import GitProvider


def _finding(body="body", path="app.py", start=10, end=10):
    finding = {"body": body, "path": path}
    if start:
        finding["line_start"] = start
        finding["line_end"] = end or start
    return finding


def _state_comment(state):
    return build_review_state_comment(state)


def _reviewer(provider, monkeypatch=None):
    from pr_agent.tools.pr_reviewer import PRReviewer

    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.pr_url = "https://example.test/pull/1"
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.remaining_files_list = []
    reviewer.prediction = "prediction"
    reviewer.patches_diff = ""
    reviewer._review_calls = None
    reviewer.review_failed_chunk_count = 0
    reviewer._review_inline_published = 0
    reviewer._review_state_result = None
    reviewer._review_state_blocked = False
    reviewer._review_state_block_reason = None
    provider.supports_review_finding_state.return_value = True
    provider.is_comment_authored_by_pr_agent.return_value = True
    provider.get_issue_comments_newest_first.side_effect = (
        lambda: list(reversed(provider.get_issue_comments()))
    )
    if monkeypatch is not None:
        settings = get_settings()
        monkeypatch.setattr(settings.config, "publish_output", True)
        monkeypatch.setattr(settings.config, "is_auto_command", False, raising=False)
        monkeypatch.setattr(settings.pr_reviewer, "persistent_comment", True)
        monkeypatch.setattr(settings.pr_reviewer, "persistent_finding_state", True, raising=False)
        monkeypatch.setattr(settings.pr_reviewer, "inline_key_issues", False)
    return reviewer


def test_state_identity_is_distinct_from_review_identities():
    assert PRReviewStateIdentity.STATE.value == "<!-- pr-agent:review:state -->"
    assert PRReviewStateIdentity.STATE.value in _ALL_COMMENT_IDENTITIES
    from pr_agent.algo.utils import comment_carries_other_identity, comment_matches_identity

    state_body = (
        "<!-- pr-agent:review:state -->\n\n<sub>state</sub>\n\n"
        + serialize_review_state(
            reconcile_review_findings(None, [_finding()], allow_resolution=True,
                                      timestamp="2026-01-01T00:00:00Z").state
        )
    )
    assert comment_matches_identity(state_body, PRReviewStateIdentity.STATE.value) is True
    assert comment_carries_other_identity(state_body, "<!-- pr-agent:review:full -->") is True
    review_body = "<!-- pr-agent:review:full -->\n\n## PR Reviewer Guide"
    assert comment_matches_identity(review_body, PRReviewStateIdentity.STATE.value) is False
    assert comment_carries_other_identity(review_body, PRReviewStateIdentity.STATE.value) is True


def test_oversized_state_fits_and_review_stays_verbatim():
    active = []
    for index in range(400):
        active.append({
            "finding_id": f"active-{index:04d}",
            "state": "ACTIVE",
            "body": f"finding {index} " + ("x" * 1000),
            "path": f"file-{index}.py",
            "line_start": 10,
            "line_end": 10,
            "first_seen": f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
            "last_seen": f"2026-02-01T00:{index // 60:02d}:{index % 60:02d}Z",
        })
    resolved = []
    for index in range(50):
        resolved.append({
            "finding_id": f"resolved-{index:04d}",
            "state": "RESOLVED",
            "body": "old " + ("y" * 500),
            "path": f"old-{index}.py",
            "resolved_at": f"2025-12-{index % 28 + 1:02d}T00:00:00Z",
        })
    state = {"schema_version": 1, "findings": active + resolved, "last_run": {"head_sha": "h1"}}
    fitted = fit_review_state(state, 65000)
    body = build_review_state_comment(fitted)
    assert len(body) <= 65000
    assert body.splitlines()[0].strip() == PRReviewStateIdentity.STATE.value
    parsed = parse_review_state(body)
    assert parsed.present is True
    assert parsed.valid is True
    assert parsed.state == fitted
    kept_active = [finding for finding in fitted["findings"] if finding["state"] == "ACTIVE"]
    assert kept_active
    assert all(len(finding["body"]) <= 200 for finding in fitted["findings"])
    kept_ids = {finding["finding_id"] for finding in kept_active}
    assert "active-0399" in kept_ids
    assert "active-0000" not in kept_ids

    review_markdown = "## PR Reviewer Guide \U0001f50d\n\nHuman review " + ("z" * 1000)
    published = append_review_state(review_markdown, fitted, max_chars=65000)
    assert review_markdown in published
    assert "<!-- pr-agent-review-state:" not in published


def test_visible_review_never_truncated_only_resolved_shrinks():
    review_markdown = "## PR Reviewer Guide \U0001f50d\n\n" + ("y" * 60000)
    findings = []
    for index in range(20):
        findings.append({
            "finding_id": f"r-{index:02d}",
            "state": "RESOLVED",
            "body": f"resolved body {index} " + ("w" * 100),
            "path": f"f{index}.py",
            "line_start": index + 1,
            "line_end": index + 1,
            "resolved_at": f"2026-01-01T00:00:{index:02d}Z",
        })
    state = {"schema_version": 1, "findings": findings, "last_run": {}}
    budget = len(review_markdown) + 600
    published = append_review_state(review_markdown, state, max_chars=budget)
    assert review_markdown in published
    assert "<!-- pr-agent-review-state:" not in published
    assert "\u2026 and " in published
    assert "more resolved findings not shown" in published
    assert len(published) <= budget
    tiny = append_review_state(review_markdown, state, max_chars=len(review_markdown) + 10)
    assert tiny.startswith(review_markdown)
    assert "<details>" not in tiny


def test_partial_run_reconciliation_matrix(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "num_max_findings", 3)

    def previous_with(*paths):
        findings = [_finding(f"body-{path}", path, start=5) for path in paths]
        return reconcile_review_findings(
            None, findings, allow_resolution=True, head_sha="head-1",
            timestamp="2026-01-01T00:00:00Z",
        ).state

    previous = previous_with("a.py", "b.py", "c.py", "d.py", "e.py", "f.py")
    by_path = {finding["path"]: finding["finding_id"] for finding in previous["findings"]}

    def reconcile_empty(resolvable):
        return reconcile_review_findings(
            previous, [], allow_resolution=True, head_sha="head-2",
            timestamp="2026-01-01T00:01:00Z", resolvable_paths=resolvable,
            complete=False,
        )

    resolved = reconcile_empty({"a.py", "f.py"})
    states = {finding["path"]: finding["state"] for finding in resolved.state["findings"]}
    assert states["a.py"] == "RESOLVED"
    assert states["f.py"] == "RESOLVED"
    assert states["b.py"] == "ACTIVE"
    assert states["c.py"] == "ACTIVE"
    assert states["d.py"] == "ACTIVE"
    assert states["e.py"] == "ACTIVE"
    assert set(resolved.resolved_ids) == {by_path["a.py"], by_path["f.py"]}

    same_head = reconcile_review_findings(
        previous, [], allow_resolution=True, head_sha="head-1",
        timestamp="2026-01-01T00:01:00Z", resolvable_paths={"a.py", "b.py", "f.py"},
        complete=True,
    )
    assert all(finding["state"] == "ACTIVE" for finding in same_head.state["findings"])
    assert same_head.resolved_ids == ()

    assert resolved.state["last_run"]["complete"] is False
    assert resolved.state["last_run"]["kind"] == "partial"


def test_chunked_complete_with_total_over_cap_still_resolves(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "num_max_findings", 3)
    previous = reconcile_review_findings(
        None,
        [_finding("old-a", "a.py", start=5), _finding("old-b", "e.py", start=5)],
        allow_resolution=True, head_sha="head-1", timestamp="2026-01-01T00:00:00Z",
    ).state
    provider = MagicMock()
    provider.get_diff_files.return_value = [
        SimpleNamespace(filename="a.py"), SimpleNamespace(filename="b.py"),
        SimpleNamespace(filename="c.py"), SimpleNamespace(filename="d.py"),
        SimpleNamespace(filename="e.py"),
    ]
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.remaining_files_list = []
    reviewer.review_failed_chunk_count = 0
    reviewer._review_calls = [
        (["a.py", "b.py"], set(), 2),
        (["c.py", "d.py"], set(), 2),
    ]
    resolvable, complete = reviewer._compute_resolvable_paths(previous)
    assert complete is True
    assert {"a.py", "b.py", "c.py", "d.py"} <= set(resolvable)
    result = reconcile_review_findings(
        previous,
        [_finding("new-b", "b.py", start=6)],
        allow_resolution=True, head_sha="head-2", timestamp="2026-01-01T00:01:00Z",
        resolvable_paths=resolvable, complete=complete,
    )
    states = {finding["path"]: finding["state"] for finding in result.state["findings"]}
    assert states["a.py"] == "RESOLVED"
    assert states["e.py"] == "ACTIVE"
    assert result.state["last_run"]["complete"] is True
    assert result.state["last_run"]["kind"] == "full"


def test_location_carry_over_reworded_finding(monkeypatch):
    previous = reconcile_review_findings(
        None, [_finding("the lock is never released here", "app.py", start=20, end=23)],
        allow_resolution=True, head_sha="head-1", timestamp="2026-01-01T00:00:00Z",
    ).state
    previous_id = previous["findings"][0]["finding_id"]
    result = reconcile_review_findings(
        previous,
        [_finding("completely different wording about cleanup", "app.py", start=21, end=24)],
        allow_resolution=True, head_sha="head-2", timestamp="2026-01-01T00:01:00Z",
        resolvable_paths={"app.py"}, complete=True,
    )
    assert len(result.state["findings"]) == 1
    kept = result.state["findings"][0]
    assert kept["finding_id"] == previous_id
    assert kept["state"] == "ACTIVE"
    assert "completely different wording" in kept["body"]
    assert result.open_ids == (previous_id,)
    assert result.new_ids == ()
    assert result.resolved_ids == ()

    previous2 = reconcile_review_findings(
        None, [_finding("old issue", "app.py", start=20, end=23)],
        allow_resolution=True, head_sha="head-1", timestamp="2026-01-01T00:00:00Z",
    ).state
    old_id = previous2["findings"][0]["finding_id"]
    moved = reconcile_review_findings(
        previous2,
        [_finding("brand new wording elsewhere", "app.py", start=100, end=105)],
        allow_resolution=True, head_sha="head-2", timestamp="2026-01-01T00:01:00Z",
        resolvable_paths={"app.py"}, complete=True,
    )
    states = {finding["finding_id"]: finding["state"] for finding in moved.state["findings"]}
    assert states[old_id] == "RESOLVED"
    assert len(moved.new_ids) == 1
    assert moved.new_ids[0] != old_id
    assert moved.open_ids == ()
    assert moved.resolved_ids == (old_id,)


def test_legacy_marker_in_review_is_ignored_and_migrates(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.config, "publish_output", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_comment", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_finding_state", True, raising=False)
    legacy_findings = [
        {"finding_id": f"legacy-{index}", "state": "ACTIVE", "body": f"legacy body {index}",
         "path": f"legacy-{index}.py", "first_seen": "2026-01-01T00:00:00Z",
         "last_seen": "2026-01-01T00:00:00Z"}
        for index in range(64)
    ]
    legacy_state = {"schema_version": 1, "findings": legacy_findings, "last_run": {"head_sha": "old"}}
    human_part = "H" * 560
    legacy_body = f"## PR Reviewer Guide \U0001f50d\n\n{human_part}\n\n{serialize_review_state(legacy_state)}"
    legacy_body = legacy_body.ljust(65000, "x")
    review_comment = SimpleNamespace(body=legacy_body, user=SimpleNamespace(login="agent"))
    provider = MagicMock()
    provider.get_issue_comments.return_value = [review_comment]
    provider.get_diff_files.return_value = []
    provider.is_supported.side_effect = lambda capability: capability == "get_issue_comments"
    provider.last_commit_id = "head-new"
    provider.max_comment_chars = 65000
    provider.get_latest_commit_url.return_value = "commit-url"
    provider.get_comment_url.return_value = "comment-url"

    def edit_comment(comment_obj, body):
        comment_obj.body = body
        return True

    created = []

    def publish_comment(body, is_temporary=False, **kwargs):
        comment = SimpleNamespace(body=body, user=SimpleNamespace(login="agent"))
        created.append(comment)
        return comment

    provider.edit_comment.side_effect = edit_comment
    provider.publish_comment.side_effect = publish_comment
    provider.publish_persistent_comment_full = (
        lambda *args, **kwargs: GitProvider.publish_persistent_comment_full(provider, *args, **kwargs)
    )
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.git_provider = provider

    parsed = reviewer._load_review_finding_state()
    assert parsed.present is False
    assert parsed.valid is True
    assert parsed.state is None

    current = [_finding("fresh one", "new.py", start=3), _finding("fresh two", "other.py", start=4)]
    reviewer.prediction = "prediction"
    reviewer._review_calls = [(["new.py", "other.py"], set(), 2)]
    provider.get_diff_files.return_value = [SimpleNamespace(filename="new.py"),
                                            SimpleNamespace(filename="other.py")]
    reviewer._prepare_review_finding_state({"review": {"key_issues_to_review": [
        {"relevant_file": "new.py", "issue_content": "fresh one", "start_line": 3, "end_line": 3},
        {"relevant_file": "other.py", "issue_content": "fresh two", "start_line": 4, "end_line": 4},
    ]}})
    assert reviewer._review_state_result is not None
    assert len(reviewer._review_state_result.new_ids) == 2
    assert reviewer._review_state_result.resolved_ids == ()

    new_review = "## PR Reviewer Guide \U0001f50d\n\nFull new review " + ("n" * 1000)
    visible = append_review_state(new_review, reviewer._review_state_result.state,
                                  max_chars=reviewer._review_comment_max_chars())
    assert new_review in visible
    assert "<!-- pr-agent-review-state:" not in visible
    result = GitProvider.publish_persistent_comment_full(
        provider, visible, initial_header=new_review.split("\n", 1)[0], update_header=True,
        final_update_message=False, identity_marker="<!-- pr-agent:review:full -->",
        legacy_initial_header="## PR Reviewer Guide", require_agent_authorship=True,
        fallback_on_error=False,
    )
    assert result is review_comment
    assert "Full new review" in review_comment.body
    assert "<!-- pr-agent-review-state:" not in review_comment.body

    reviewer._publish_review_finding_state()
    assert len(created) == 1
    state_body = created[0].body
    assert PRReviewStateIdentity.STATE.value in state_body.splitlines()[:5]
    parsed_state = parse_review_state(state_body)
    assert parsed_state.valid is True
    assert len(parsed_state.state["findings"]) == 2
    assert all(not finding["finding_id"].startswith("legacy-") for finding in parsed_state.state["findings"])

    provider.get_issue_comments.return_value = [review_comment, created[0]]
    second = _reviewer(provider, monkeypatch)
    second.git_provider = provider
    reparsed = second._load_review_finding_state()
    assert reparsed.valid is True
    assert len(reparsed.state["findings"]) == 2


def test_state_comment_security_and_write_failure(monkeypatch):
    previous = reconcile_review_findings(
        None, [_finding("real", "app.py", start=2)],
        allow_resolution=True, timestamp="2026-01-01T00:00:00Z",
    ).state
    spoofed = reconcile_review_findings(
        None, [_finding("spoofed", "evil.py", start=1)],
        allow_resolution=True, timestamp="2026-01-01T00:01:00Z",
    ).state
    header_state = PRReviewStateIdentity.STATE.value
    human = "<sub>PR-Agent review state \u2014 machine-readable, updated in place by the review bot; do not edit.</sub>"
    spoof_body = f"{header_state}\n\n{human}\n\n{serialize_review_state(spoofed)}"
    real_body = f"{header_state}\n\n{human}\n\n{serialize_review_state(previous)}"
    spoofed_comment = SimpleNamespace(body=spoof_body, user=SimpleNamespace(login="human"))
    real_comment = SimpleNamespace(body=real_body, user=SimpleNamespace(login="agent"))
    provider = MagicMock()
    provider.get_issue_comments.return_value = [real_comment, spoofed_comment]
    reviewer = _reviewer(provider, monkeypatch)
    provider.is_comment_authored_by_pr_agent.side_effect = lambda comment: comment.user.login == "agent"
    parsed = reviewer._load_review_finding_state()
    assert parsed.valid is True
    assert parsed.state == previous

    provider2 = MagicMock()
    provider2.get_issue_comments.return_value = [spoofed_comment]
    reviewer2 = _reviewer(provider2, monkeypatch)
    provider2.is_comment_authored_by_pr_agent.return_value = False
    parsed2 = reviewer2._load_review_finding_state()
    assert parsed2.present is False
    assert parsed2.state is None

    malformed = SimpleNamespace(body=f"{header_state}\n\n{human}\n\n<!-- pr-agent-review-state:v1\nbad\n-->",
                                user=SimpleNamespace(login="agent"))
    provider3 = MagicMock()
    provider3.get_issue_comments.return_value = [real_comment, malformed]
    reviewer3 = _reviewer(provider3, monkeypatch)
    provider3.get_issue_comments_newest_first.side_effect = lambda: [malformed, real_comment]
    parsed3 = reviewer3._load_review_finding_state()
    assert parsed3.valid is True
    assert parsed3.state == previous

    failing = MagicMock()
    failing.publish_persistent_comment_full.side_effect = RuntimeError("write failed")
    failing.max_comment_chars = 65000
    reviewer4 = _reviewer(failing, monkeypatch)
    reviewer4._review_state_result = reconcile_review_findings(
        None, [_finding("x", "a.py", start=1)], allow_resolution=False,
        timestamp="2026-01-01T00:00:00Z",
    )
    # A failed state write must not raise and must warn that the next round
    # will treat these findings as new — for an exception and for a None return.
    with patch("pr_agent.tools.pr_reviewer.get_logger") as mock_logger:
        reviewer4._publish_review_finding_state()
    failing.publish_persistent_comment_full.assert_called_once()
    failing.edit_comment.assert_not_called()
    warning = mock_logger.return_value.warning.call_args.args[0]
    assert "treat these findings as new" in warning

    silent = MagicMock()
    silent.publish_persistent_comment_full.return_value = None
    silent.max_comment_chars = 65000
    reviewer5 = _reviewer(silent, monkeypatch)
    reviewer5._review_state_result = reviewer4._review_state_result
    with patch("pr_agent.tools.pr_reviewer.get_logger") as mock_logger:
        reviewer5._publish_review_finding_state()
    silent.publish_persistent_comment_full.assert_called_once()
    warning = mock_logger.return_value.warning.call_args.args[0]
    assert "treat these findings as new" in warning


def test_round_summary_counts_and_exact_format():
    previous = reconcile_review_findings(
        None,
        [_finding("issue one", "file1.py", start=10, end=10),
         _finding("issue two", "file2.py", start=20, end=20),
         _finding("old wording", "file3.py", start=30, end=32)],
        allow_resolution=True, head_sha="head-1", timestamp="2026-01-01T00:00:00Z",
    ).state
    result = reconcile_review_findings(
        previous,
        [_finding("issue one", "file1.py", start=10, end=10),
         _finding("reworded third finding text", "file3.py", start=31, end=33),
         _finding("brand new fourth", "file4.py", start=5, end=5)],
        allow_resolution=True, head_sha="head-2", timestamp="2026-01-01T00:01:00Z",
        resolvable_paths={"file1.py", "file2.py", "file3.py", "file4.py"}, complete=False,
        excluded_files=["big.py", "huge.py", "massive.py"],
    )
    assert len(result.new_ids) == 1
    assert len(result.open_ids) == 2
    assert len(result.resolved_ids) == 1
    summary = build_round_summary("abc123def", len(result.new_ids), len(result.open_ids),
                                  len(result.resolved_ids), 2, False, 3)
    hidden, visible = summary.split("\n")
    assert hidden == "<!-- pr-agent-review-round:v1 head=abc123def new=1 open=2 resolved=1 inline=2 complete=false -->"
    assert visible == ("**Review round:** 1 new \u00b7 2 still open from earlier rounds "
                       "\u00b7 1 resolved since the last review \u00b7 2 new inline threads "
                       "\u00b7 coverage: partial \u2014 3 file(s) not reviewed")
    markdown = "## PR Reviewer Guide \U0001f50d\n\nBody text"
    inserted = insert_round_summary(markdown, summary)
    assert inserted.startswith("## PR Reviewer Guide \U0001f50d\n\n" + hidden)
    assert visible in inserted
    complete_summary = build_round_summary("abc123def", 0, 1, 0, 0, True, 0)
    assert complete_summary.split("\n")[0].endswith("complete=true -->")
    assert build_round_summary("", 0, 0, 0, 0, True, 0).split("\n")[0] == (
        "<!-- pr-agent-review-round:v1 head=unknown new=0 open=0 resolved=0 inline=0 complete=true -->"
    )
    assert complete_summary.split("\n")[1].endswith("coverage: complete")


def test_dockerfile_covers_every_funnel_fork_file():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    dockerfile = root / "Dockerfile.github_action_dockerhub"
    text = dockerfile.read_text()
    copied = set(re.findall(r"^COPY\s+(\S+)", text, re.MULTILINE))
    missing = []
    for path in sorted((root / "pr_agent").rglob("*.py")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "Funnel fork" in content:
            rel = path.relative_to(root).as_posix()
            if rel not in copied:
                missing.append(rel)
    assert missing == [], f"Funnel fork files missing from Dockerfile COPY list: {missing}"


def _previous_with_paths(*paths, head_sha="head-1"):
    return reconcile_review_findings(
        None,
        [_finding(f"body of {path}", path, start=5) for path in paths],
        allow_resolution=True,
        head_sha=head_sha,
        timestamp="2026-01-01T00:00:00Z",
    ).state


def test_compute_resolvable_paths_classifies_each_file(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "num_max_findings", 3)
    previous = _previous_with_paths("a.py", "b.py", "c.py", "d.py", "f.py", "g.py", "e.py")
    provider = MagicMock()
    provider.get_diff_files.return_value = [
        SimpleNamespace(filename=name) for name in ["a.py", "b.py", "c.py", "d.py", "f.py", "g.py"]
    ]
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.remaining_files_list = ["b.py"]
    reviewer.review_failed_chunk_count = 1
    reviewer._review_calls = [
        (["a.py"], set(), 1),
        (["c.py"], set(), 3),
        (["f.py", "g.py"], {"f.py"}, 1),
    ]

    resolvable, complete = reviewer._compute_resolvable_paths(previous)

    assert resolvable == {"a.py", "g.py", "e.py"}
    assert complete is False

    result = reconcile_review_findings(
        previous,
        [],
        allow_resolution=True,
        excluded_files=["b.py"],
        head_sha="head-2",
        timestamp="2026-01-01T00:01:00Z",
        resolvable_paths=resolvable,
        complete=complete,
    )
    states = {finding["path"]: finding["state"] for finding in result.state["findings"]}
    assert states == {
        "a.py": "RESOLVED",
        "b.py": "ACTIVE",
        "c.py": "ACTIVE",
        "d.py": "ACTIVE",
        "f.py": "ACTIVE",
        "g.py": "RESOLVED",
        "e.py": "RESOLVED",
    }
    assert result.state["last_run"]["kind"] == "partial"
    assert result.state["last_run"]["complete"] is False


def test_extract_reviewed_files_marks_truncated_section():
    diff = (
        "## File: 'alpha.py'\n\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n\n"
        "## File: 'beta.py'\n\n"
        "@@ -10,3 +10,3 @@\n"
        "-old line\n"
        "+new line\n"
        "...(truncated)"
    )
    files, clipped = extract_reviewed_files(diff)
    assert files == ["alpha.py", "beta.py"]
    assert clipped == {"beta.py"}


async def test_chunked_prediction_records_per_call_files_and_counts(monkeypatch):
    _reviewer_settings = get_settings()
    monkeypatch.setattr(_reviewer_settings.pr_reviewer, "max_number_of_calls", 3, raising=False)
    provider = MagicMock()
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.token_handler = MagicMock()
    chunk_diffs = [
        "## File: 'one.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
        "## File: 'two.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
        "## File: 'three.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
    ]
    chunk_one = (
        "review:\n"
        "  key_issues_to_review:\n"
        "    - relevant_file: one.py\n"
        "      issue_content: first finding body here\n"
        "      start_line: 1\n"
        "      end_line: 1\n"
    )
    chunk_three = (
        "review:\n"
        "  key_issues_to_review:\n"
        "    - relevant_file: three.py\n"
        "      issue_content: third finding one\n"
        "      start_line: 1\n"
        "      end_line: 1\n"
        "    - relevant_file: three.py\n"
        "      issue_content: third finding two\n"
        "      start_line: 2\n"
        "      end_line: 2\n"
        "    - relevant_file: three.py\n"
        "      issue_content: third finding three\n"
        "      start_line: 3\n"
        "      end_line: 3\n"
    )
    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_multi_diffs",
        return_value=(chunk_diffs, ["left.py"]),
    ):
        reviewer._get_prediction = AsyncMock(
            side_effect=[chunk_one, RuntimeError("chunk 2 exploded"), chunk_three]
        )
        assert await reviewer._prepare_chunked_prediction("model") is True

    assert reviewer._review_calls == [(["one.py"], set(), 1), (["three.py"], set(), 3)]
    assert reviewer.review_failed_chunk_count == 1


async def test_single_call_prediction_records_call_with_deferred_count():
    provider = MagicMock()
    reviewer = _reviewer(provider)
    reviewer.token_handler = MagicMock()
    reviewer.vars = {}
    diff = "## File: 'solo.py'\n\n@@ -1 +1 @@\n-a\n+b\n"
    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_diff",
        return_value=(diff, []),
    ):
        reviewer._get_prediction = AsyncMock(return_value="review:\n  key_issues_to_review: []\n")
        await reviewer._prepare_prediction("model")

    assert reviewer._review_calls == [(["solo.py"], set(), None)]
    assert reviewer.remaining_files_list == []


def _issue_yaml_row(filename, content, start):
    return (
        f"    - relevant_file: {filename}\n"
        f"      issue_content: {content}\n"
        f"      start_line: {start}\n"
        f"      end_line: {start}\n"
    )


def _review_yaml(rows):
    return "review:\n  key_issues_to_review:\n" + "".join(rows)


class _RunFakeProvider:
    """Minimal provider driving the real persistent-comment publisher."""

    def __init__(self, comments, diff_files):
        self.comments = comments
        self.diff_files = diff_files
        self.last_commit_id = "head-1"
        self.max_comment_chars = 65000
        self.published = []
        self.edits = []
        self.removed = []

    def get_files(self):
        return [item.filename for item in self.diff_files]

    def get_diff_files(self):
        return list(self.diff_files)

    def should_publish_review_as_thread(self):
        return False

    def supports_review_finding_state(self):
        return True

    def supports_review_comment_identity(self):
        return False

    def is_supported(self, capability):
        return capability == "get_issue_comments"

    def is_comment_authored_by_pr_agent(self, comment):
        return comment.user.login == "agent"

    def get_issue_comments(self):
        return list(self.comments)

    def get_issue_comments_newest_first(self):
        return list(reversed(self.comments))

    def get_latest_commit_url(self):
        return "commit-url"

    def get_comment_url(self, comment):
        return f"comment-url-{id(comment)}"

    def edit_comment(self, comment, body):
        comment.body = body
        self.edits.append(comment)
        return True

    def publish_comment(self, body, is_temporary=False, **kwargs):
        comment = SimpleNamespace(body=body, user=SimpleNamespace(login="agent"))
        self.comments.append(comment)
        self.published.append((body, bool(is_temporary)))
        return comment

    def remove_comment(self, comment):
        self.removed.append(comment)

    publish_persistent_comment_full = GitProvider.publish_persistent_comment_full


def _bot_comment(body):
    return SimpleNamespace(body=body, user=SimpleNamespace(login="agent"))


def test_review_markdown_with_quoted_marker_survives_byte_for_byte():
    quoted_marker = "<!-- pr-agent-review-state:v1\nfake payload\n-->"
    review = (
        "## PR Reviewer Guide \U0001f50d\n\n"
        "First finding stands.\n\n"
        f"Second finding quotes {quoted_marker} inline.\n"
    )
    empty_state = {"schema_version": 1, "findings": [], "last_run": {}}
    assert append_review_state(review, empty_state) == review

    resolved_state = reconcile_review_findings(
        None, [_finding("x", "a.py", start=1)], allow_resolution=False,
        timestamp="2026-01-01T00:00:00Z",
    ).state
    resolved_state["findings"][0]["state"] = "RESOLVED"
    resolved_state["findings"][0]["resolved_at"] = "2026-01-01T00:01:00Z"
    with_resolved = append_review_state(review, resolved_state)
    assert with_resolved.startswith(review)
    assert quoted_marker in with_resolved
    assert "\u2705 Resolved findings" in with_resolved


def test_file_header_regex_ignores_quoted_header_in_diff():
    diff = (
        "## File: 'real.py'\n\n"
        "@@ -1 +1 @@\n"
        "+    \"## File: 'fake.py'\"\n"
    )
    files, clipped = extract_reviewed_files(diff)
    assert files == ["real.py"]
    assert clipped == set()


def test_serialize_escapes_angle_brackets_and_round_trips():
    body = "quotes --> and <!-- pr-agent-review-state inside one body"
    state = reconcile_review_findings(
        None, [_finding(body, "a.py", start=1)], allow_resolution=False,
        timestamp="2026-01-01T00:00:00Z",
    ).state
    marker = serialize_review_state(state)
    assert "<!-- pr-agent-review-state:v1" in marker
    assert marker.count("<!-- pr-agent-review-state") == 1
    assert "-->" not in marker[: -len("-->")]
    parsed = parse_review_state(marker)
    assert parsed.valid is True
    assert parsed.state["findings"][0]["body"] == body


def test_fit_caps_excluded_files_before_dropping_findings():
    findings = [
        {
            "finding_id": f"keep-{index:02d}",
            "state": "ACTIVE",
            "body": f"finding body {index}",
            "path": f"keep-{index}.py",
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-01-01T00:00:00Z",
        }
        for index in range(10)
    ]
    state = {
        "schema_version": 1,
        "findings": findings,
        "last_run": {
            "head_sha": "h",
            "excluded_files": [f"excluded-{index:04d}-{'p' * 40}.py" for index in range(1500)],
        },
    }
    fitted = fit_review_state(state, 65000)
    assert len(build_review_state_comment(fitted)) <= 65000
    assert [finding["finding_id"] for finding in fitted["findings"]] == [
        f"keep-{index:02d}" for index in range(10)
    ]
    assert len(fitted["last_run"]["excluded_files"]) == 50
    assert fitted["last_run"]["excluded_files_truncated"] == 1500
    assert parse_review_state(build_review_state_comment(fitted)).valid is True


def test_unknown_issue_counts_resolve_nothing(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "num_max_findings", 3)
    previous = _previous_with_paths("a.py", "s.py")
    provider = MagicMock()
    provider.get_diff_files.return_value = [SimpleNamespace(filename="a.py"), SimpleNamespace(filename="s.py")]
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.remaining_files_list = []
    reviewer.review_failed_chunk_count = 0
    reviewer._review_calls = [(["a.py"], set(), None)]

    resolvable, _ = reviewer._compute_resolvable_paths(previous)
    assert resolvable == set()

    reviewer._review_calls = [(["s.py"], set(), None)]
    resolvable_single, _ = reviewer._compute_resolvable_paths(previous)
    assert resolvable_single == set()

    result = reconcile_review_findings(
        previous, [], allow_resolution=True, head_sha="head-2",
        timestamp="2026-01-01T00:01:00Z", resolvable_paths=resolvable, complete=True,
    )
    assert {finding["path"]: finding["state"] for finding in result.state["findings"]} == {
        "a.py": "ACTIVE",
        "s.py": "ACTIVE",
    }


def test_first_round_with_no_findings_renders_round_line_and_baseline_state(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.config, "publish_output", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_comment", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_finding_state", True, raising=False)
    monkeypatch.setattr(settings.pr_reviewer, "inline_key_issues", False)
    provider = MagicMock()
    provider.last_commit_id = "head-1"
    provider.get_issue_comments.return_value = []
    provider.get_diff_files.return_value = []
    provider.is_supported.side_effect = lambda capability: capability == "get_issue_comments"
    reviewer = _reviewer(provider, monkeypatch)
    reviewer._review_calls = [([], set(), 0)]
    reviewer.review_failed_chunk_count = 0

    with (
        patch("pr_agent.tools.pr_reviewer.load_yaml", return_value={"review": {"key_issues_to_review": []}}),
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch(
            "pr_agent.tools.pr_reviewer.convert_to_markdown_v2",
            return_value="## PR Reviewer Guide \U0001f50d\n\nAll clear",
        ),
        patch("pr_agent.tools.pr_reviewer.push_outputs"),
    ):
        review = reviewer._prepare_pr_review()

    assert "<!-- pr-agent-review-round:v1 head=head-1 new=0 open=0 resolved=0 inline=0 complete=true -->" in review
    assert reviewer._review_state_result is not None
    assert reviewer._review_state_result.state["findings"] == []
    baseline = build_review_state_comment(reviewer._review_state_result.state)
    assert parse_review_state(baseline).valid is True


def test_inline_counts_threads_not_issues():
    from pr_agent.algo.types import FilePatchInfo
    from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
    from pr_agent.tools.pr_reviewer import PRReviewer

    provider = AzureDevopsProvider.__new__(AzureDevopsProvider)
    provider.azure_devops_client = MagicMock()
    provider.azure_devops_client.get_threads.return_value = []
    provider.repo_slug = "repo"
    provider.workspace_slug = "project"
    provider.pr_num = 1
    provider.get_diff_files = MagicMock(
        return_value=[FilePatchInfo(base_file="", head_file="one\ntwo\nthree\nfour\n", patch="", filename="app.py")]
    )
    provider.publish_code_suggestions = MagicMock(return_value=True)
    provider.max_comment_chars = None
    provider._inline_comment_store = None
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer._review_inline_published = 0
    reviewer._published_inline_key_issue_fingerprints = MagicMock(
        side_effect=lambda _store, fingerprints: fingerprints
    )
    shared = {
        "relevant_file": "app.py",
        "issue_header": "Possible Issue",
        "issue_content": "The new branch never releases the lock.",
        "start_line": 2,
        "end_line": 3,
    }
    other = dict(shared, issue_content="A different finding on another line.", start_line=1, end_line=1)
    data = {"review": {"key_issues_to_review": [dict(shared), dict(shared), other]}}

    reviewer._publish_key_issues_as_inline_comments(data)

    assert provider.publish_code_suggestions.call_count == 1
    assert reviewer._review_inline_published == 2


def test_location_pairing_is_one_to_one():
    previous = reconcile_review_findings(
        None, [_finding("original wording", "app.py", start=20, end=23)],
        allow_resolution=True, head_sha="head-1", timestamp="2026-01-01T00:00:00Z",
    ).state
    previous_id = previous["findings"][0]["finding_id"]
    result = reconcile_review_findings(
        previous,
        [
            _finding("reworded once more", "app.py", start=21, end=24),
            _finding("another rewording", "app.py", start=22, end=25),
        ],
        allow_resolution=True, head_sha="head-2", timestamp="2026-01-01T00:01:00Z",
        resolvable_paths={"app.py"}, complete=True,
    )
    assert len(result.open_ids) == 1
    assert len(result.new_ids) == 1
    assert result.open_ids[0] == previous_id
    assert result.new_ids[0] != previous_id
    assert result.resolved_ids == ()
    assert len(result.state["findings"]) == 2
    assert all(finding["state"] == "ACTIVE" for finding in result.state["findings"])


async def test_unknown_chunk_files_are_uncovered_and_run_partial(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "max_number_of_calls", 3, raising=False)
    monkeypatch.setattr(settings.config, "publish_output", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_comment", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_finding_state", True, raising=False)
    monkeypatch.setattr(settings.pr_reviewer, "inline_key_issues", False)
    provider = MagicMock()
    provider.last_commit_id = "head-1"
    provider.get_issue_comments.return_value = []
    provider.get_diff_files.return_value = [
        SimpleNamespace(filename="one.py"), SimpleNamespace(filename="mystery.py"),
    ]
    provider.is_supported.side_effect = lambda capability: capability == "get_issue_comments"
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.token_handler = MagicMock()
    chunk_diffs = [
        "## File: 'one.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
        "## File: 'mystery.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
    ]
    known_yaml = (
        "review:\n"
        "  key_issues_to_review:\n"
        "    - relevant_file: one.py\n"
        "      issue_content: known finding\n"
        "      start_line: 1\n"
        "      end_line: 1\n"
    )
    unknown_yaml = "review:\n  summary: nothing structured here\n"
    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_multi_diffs",
        return_value=(chunk_diffs, []),
    ):
        reviewer._get_prediction = AsyncMock(side_effect=[known_yaml, unknown_yaml])
        assert await reviewer._prepare_chunked_prediction("model") is True

    assert reviewer._review_calls == [(["one.py"], set(), 1), (["mystery.py"], set(), None)]
    assert reviewer._review_uncovered_files == {"mystery.py"}

    with (
        patch("pr_agent.tools.pr_reviewer.github_action_output"),
        patch(
            "pr_agent.tools.pr_reviewer.convert_to_markdown_v2",
            return_value="## PR Reviewer Guide \U0001f50d\n\nPartial review",
        ),
        patch("pr_agent.tools.pr_reviewer.push_outputs"),
    ):
        review = reviewer._prepare_pr_review()

    assert "<!-- pr-agent-review-round:v1 head=head-1 new=1 open=0 resolved=0 inline=0 complete=false -->" in review
    assert reviewer._round_uncovered_count(False) == 1


async def test_uncovered_files_count_six(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.pr_reviewer, "max_number_of_calls", 3, raising=False)
    provider = MagicMock()
    reviewer = _reviewer(provider, monkeypatch)
    reviewer.token_handler = MagicMock()
    chunk_diffs = [
        "## File: 'ok.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
        "## File: 'fail1.py'\n\n@@ -1 +1 @@\n-a\n+b\n## File: 'fail2.py'\n\n@@ -1 +1 @@\n-a\n+b\n"
        "## File: 'fail3.py'\n\n@@ -1 +1 @@\n-a\n+b\n",
        "## File: 'clip.py'\n\n@@ -1 +1 @@\n-a\n+b\n...(truncated)",
    ]
    chunk_yaml = (
        "review:\n"
        "  key_issues_to_review:\n"
        "    - relevant_file: ok.py\n"
        "      issue_content: ok finding\n"
        "      start_line: 1\n"
        "      end_line: 1\n"
    )
    with patch(
        "pr_agent.tools.pr_reviewer.get_pr_multi_diffs",
        return_value=(chunk_diffs, ["big1.py", "big2.py"]),
    ):
        reviewer._get_prediction = AsyncMock(
            side_effect=[chunk_yaml, RuntimeError("chunk 2 exploded"), chunk_yaml]
        )
        assert await reviewer._prepare_chunked_prediction("model") is True

    assert reviewer._review_uncovered_files == {
        "big1.py", "big2.py", "fail1.py", "fail2.py", "fail3.py", "clip.py",
    }
    assert reviewer._round_uncovered_count(False) == 6


async def test_run_migrates_legacy_review_and_reconciles_second_run(monkeypatch):
    from pr_agent.tools.pr_reviewer import PRReviewer

    settings = get_settings()
    monkeypatch.setattr(settings.config, "publish_output", True)
    monkeypatch.setattr(settings.config, "is_auto_command", False)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_comment", True)
    monkeypatch.setattr(settings.pr_reviewer, "persistent_finding_state", True, raising=False)
    monkeypatch.setattr(settings.pr_reviewer, "inline_key_issues", False)
    monkeypatch.setattr(settings.pr_reviewer, "num_max_findings", 3)

    legacy_findings = [
        {
            "finding_id": f"legacy-{index:02d}",
            "state": "ACTIVE",
            "body": f"legacy finding body {index} " + ("L" * 930),
            "path": f"legacy-{index}.py",
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-01-01T00:00:00Z",
        }
        for index in range(64)
    ]
    legacy_state = {"schema_version": 1, "findings": legacy_findings, "last_run": {"head_sha": "old"}}
    legacy_body = (
        "## PR Reviewer Guide \U0001f50d\n\n"
        + ("H" * 560)
        + "\n\n"
        + serialize_review_state(legacy_state)
        + "\n\n"
        + ("x" * 60000)
    )
    assert len(legacy_body) >= 60000
    legacy_comment = _bot_comment(legacy_body)
    diff_files = [SimpleNamespace(filename=name) for name in ["n1.py", "n2.py", "n3.py"]]
    provider = _RunFakeProvider([legacy_comment], diff_files)

    full_yaml = _review_yaml([
        _issue_yaml_row("n1.py", "first finding locks the widget", 3),
        _issue_yaml_row("n2.py", "second finding leaks the handle", 4),
        _issue_yaml_row("n3.py", "third finding races on close", 5),
    ])
    partial_yaml = _review_yaml([
        _issue_yaml_row("n1.py", "first finding locks the widget", 3),
        _issue_yaml_row("n2.py", "second finding leaks the handle", 4),
    ])
    holder = {"yaml": full_yaml}
    markdown = "## PR Reviewer Guide \U0001f50d\n\nFull new review " + ("R" * 1500)

    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = provider
    reviewer.pr_url = "https://example.test/pull/93"
    reviewer.args = []
    reviewer.vars = {}
    reviewer.incremental = SimpleNamespace(is_incremental=False)
    reviewer.prediction = None
    reviewer.prediction_data = None
    reviewer.patches_diff = ""
    reviewer.remaining_files_list = []
    reviewer.review_chunk_count = 1
    reviewer.review_failed_chunk_count = 0
    reviewer._review_calls = None
    reviewer._review_inline_published = 0
    reviewer._review_state_result = None
    reviewer._review_state_blocked = False
    reviewer._review_state_block_reason = None

    async def _fake_prepare(model):
        reviewer.prediction = holder["yaml"]
        reviewer.patches_diff = (
            "## File: 'n1.py'\n+x\n\n## File: 'n2.py'\n+x\n\n## File: 'n3.py'\n+x\n"
        )
        reviewer._review_calls = [(["n1.py", "n2.py", "n3.py"], set(), None)]
        reviewer.remaining_files_list = []
        reviewer.review_failed_chunk_count = 0

    async def _through_retry(prepare_fn, *args, **kwargs):
        await prepare_fn("model")

    reviewer._prepare_prediction = AsyncMock(side_effect=_fake_prepare)
    with (
        patch("pr_agent.tools.pr_reviewer.retry_with_fallback_models", new=_through_retry),
        patch("pr_agent.tools.pr_reviewer.convert_to_markdown_v2", new=lambda *args, **kwargs: markdown),
        patch("pr_agent.tools.pr_reviewer.push_outputs", new=MagicMock()),
        patch("pr_agent.tools.pr_reviewer.github_action_output", new=MagicMock()),
    ):
        await reviewer.run()

        assert "Full new review" in legacy_comment.body
        assert ("R" * 1500) in legacy_comment.body
        assert "<!-- pr-agent-review-round:v1 head=head-1 new=3 open=0 resolved=0 inline=0 complete=true -->" in (
            legacy_comment.body
        )
        assert "pr-agent-review-state" not in legacy_comment.body
        assert all("Failed to review PR" not in comment.body for comment in provider.comments)

        state_comments = [
            comment
            for comment in provider.comments
            if any(
                line.strip() == PRReviewStateIdentity.STATE.value
                for line in comment.body.splitlines()[:5]
            )
        ]
        assert len(state_comments) == 1
        assert state_comments[0] is not legacy_comment
        parsed = parse_review_state(state_comments[0].body)
        assert parsed.valid is True
        assert len(parsed.state["findings"]) == 3
        assert {finding["path"] for finding in parsed.state["findings"]} == {"n1.py", "n2.py", "n3.py"}
        assert all(finding["state"] == "ACTIVE" for finding in parsed.state["findings"])
        assert all(
            not finding["finding_id"].startswith("legacy-") for finding in parsed.state["findings"]
        )
        review_comments = [
            comment for comment in provider.comments if "<!-- pr-agent:review:full -->" in comment.body
        ]
        assert review_comments == [legacy_comment]

        bodies_before = {id(comment): comment.body for comment in provider.comments}
        holder["yaml"] = partial_yaml
        provider.last_commit_id = "head-2"
        await reviewer.run()

        review_after = [c for c in provider.comments if "<!-- pr-agent:review:full -->" in c.body]
        state_after = [
            c
            for c in provider.comments
            if any(
                line.strip() == PRReviewStateIdentity.STATE.value for line in c.body.splitlines()[:5]
            )
        ]
        assert review_after == [legacy_comment]
        assert len(state_after) == 1
        assert state_after[0] is state_comments[0]
        created = [comment for comment in provider.comments if id(comment) not in bodies_before]
        assert all(
            "<!-- pr-agent:review:full -->" not in comment.body
            and PRReviewStateIdentity.STATE.value not in comment.body.splitlines()[:5]
            for comment in created
        )
        assert "<!-- pr-agent-review-round:v1 head=head-2 new=0 open=2 resolved=1 inline=0 complete=true -->" in (
            legacy_comment.body
        )
        assert "\u2705 Resolved findings" in legacy_comment.body
        assert "n3.py" in legacy_comment.body.split("\u2705 Resolved findings", 1)[1]
        assert all("Failed to review PR" not in comment.body for comment in provider.comments)
