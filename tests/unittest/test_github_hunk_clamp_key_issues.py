from types import SimpleNamespace

from pr_agent.git_providers.github_provider import GithubProvider

PATCH = "@@ -184,12 +184,9 @@ function renderPanelField(\n context\n"


def _make_provider(patch=PATCH):
    """Build a GithubProvider without running __init__ and stub the diff so
    only hunk-clamping behaviour is under test (new-file hunk: 184-192)."""
    provider = GithubProvider.__new__(GithubProvider)
    provider.get_diff_files = lambda: [SimpleNamespace(filename="brighterRegistry.tsx", patch=patch)]
    return provider


def _suggestion(start, end, **extra):
    suggestion = {
        "body": "**[P1] Unconditional hideLabel hides generic labels**",
        "relevant_file": "brighterRegistry.tsx",
        "relevant_lines_start": start,
        "relevant_lines_end": end,
    }
    suggestion.update(extra)
    return suggestion


def test_key_issue_range_clamped_into_hunk():
    """PR #50: the model cited 183-190 but the hunk starts at 184, so the
    unclamped range 422'd at the API and the finding was lost to the summary.
    Key-issue comments (no original_suggestion) must clamp, not skip."""
    provider = _make_provider()

    (clamped,) = provider.validate_comments_inside_hunks([_suggestion(183, 190)])

    assert clamped["relevant_lines_start"] == 184
    assert clamped["relevant_lines_end"] == 190
    assert clamped["body"].startswith("**[P1]")


def test_range_missing_hunk_collapses_to_nearest_edge():
    """A range fully above the hunk (within the 10-line window) inverts under
    clamping; collapse to a single line at the nearest hunk edge instead."""
    provider = _make_provider()

    (clamped,) = provider.validate_comments_inside_hunks([_suggestion(175, 178)])

    assert clamped["relevant_lines_start"] == 184
    assert clamped["relevant_lines_end"] == 184


def test_far_range_left_alone():
    """A range nowhere near any hunk keeps its lines (the API fallback, not
    the clamp, owns that case)."""
    provider = _make_provider()

    (untouched,) = provider.validate_comments_inside_hunks([_suggestion(100, 105)])

    assert untouched["relevant_lines_start"] == 100
    assert untouched["relevant_lines_end"] == 105


def test_code_suggestion_body_rewrite_unchanged():
    """Code suggestions (original_suggestion present) keep the upstream
    behaviour: clamp plus diff-code body rewrite."""
    provider = _make_provider()
    suggestion = _suggestion(
        183,
        190,
        body="fix it\n```suggestion\nnew\n```",
        original_suggestion={"existing_code": "old\n", "improved_code": "new\n"},
    )

    (clamped,) = provider.validate_comments_inside_hunks([suggestion])

    assert clamped["relevant_lines_start"] == 184
    assert clamped["relevant_lines_end"] == 190
    assert "```suggestion" not in clamped["body"]
    assert "New proposed code" in clamped["body"]
