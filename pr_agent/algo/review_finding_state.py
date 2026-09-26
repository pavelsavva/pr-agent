"""Persist review finding state across runs."""

# Funnel fork: finding state lives in its own hidden comment, compacts to fit,
# carries over reworded findings by location, and resolves per reviewed file.

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from pr_agent.algo.inline_comment_dedup import key_issue_fingerprint
from pr_agent.log import get_logger

STATE_SCHEMA_VERSION = 1
DEFAULT_MAX_RESOLVED_FINDINGS = 20
REVIEW_STATE_COMMENT_IDENTITY = "<!-- pr-agent:review:state -->"
REVIEW_STATE_VISIBLE_LINE = (
    "<sub>PR-Agent review state \u2014 machine-readable, updated in place "
    "by the review bot; do not edit.</sub>"
)
REVIEW_STATE_BODY_TRUNCATE_LEN = 200
_ROUND_MARKER_VERSION = 1
_STATE_MARKER_RE = re.compile(
    r"<!-- pr-agent-review-state:v(?P<version>\d+)\n(?P<payload>.*?)\n-->",
    re.DOTALL,
)
_STATE_MARKER_NAMESPACE = "<!-- pr-agent-review-state"
_WHITESPACE_RE = re.compile(r"\s+")
# Funnel fork: anchored to line starts so quoted headers inside diff content never match.
_REVIEW_FILE_HEADER_RE = re.compile(r"^## File:\s*'(?P<path>[^']+)'", re.MULTILINE)
_VALID_STATES = {"ACTIVE", "RESOLVED"}


@dataclass(frozen=True)
class ParsedReviewState:
    state: dict[str, Any] | None
    present: bool
    valid: bool


@dataclass(frozen=True)
class ReconciliationResult:
    state: dict[str, Any]
    changed: bool
    resolved_ids: tuple[str, ...]
    reopened_ids: tuple[str, ...]
    new_ids: tuple[str, ...] = ()
    open_ids: tuple[str, ...] = ()


def _timestamp(value: str | None) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_line(value: Any) -> int | None:
    try:
        line = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return line if line > 0 else None


def _normalize_path(path: Any) -> str:
    return str(path or "").strip().strip(chr(96)).lstrip("/")


def normalize_finding(finding: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the stable, display-oriented fields needed for reconciliation."""
    if not isinstance(finding, Mapping):
        return None

    path = _normalize_path(finding.get("path") or finding.get("relevant_file"))
    body = str(
        finding.get("body")
        or finding.get("issue_content")
        or finding.get("description")
        or ""
    ).strip()
    if not path or not body:
        return None

    body = _WHITESPACE_RE.sub(" ", body)
    finding_id = key_issue_fingerprint(path, body.lower())
    start = _as_line(
        finding.get("line_start")
        or finding.get("relevant_lines_start")
        or finding.get("start_line")
    )
    end = _as_line(
        finding.get("line_end")
        or finding.get("relevant_lines_end")
        or finding.get("end_line")
    )
    if start is not None and end is None:
        end = start
    if start is not None and end is not None and end < start:
        end = start

    normalized = {
        "finding_id": finding_id,
        "state": "ACTIVE",
        "body": body,
        "path": path,
    }
    if start is not None:
        normalized["line_start"] = start
    if end is not None:
        normalized["line_end"] = end
    return normalized


def normalize_findings(findings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and de-duplicate current structured findings deterministically."""
    by_id: dict[str, dict[str, Any]] = {}
    for finding in findings or []:
        normalized = normalize_finding(finding)
        if normalized is not None:
            by_id.setdefault(normalized["finding_id"], normalized)
    return [by_id[finding_id] for finding_id in sorted(by_id)]


def _is_valid_state(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        return False
    if not isinstance(state.get("findings"), list) or not isinstance(state.get("last_run"), dict):
        return False
    finding_ids = set()
    for finding in state["findings"]:
        if not isinstance(finding, dict):
            return False
        if finding.get("state") not in _VALID_STATES:
            return False
        finding_id = finding.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id or finding_id in finding_ids:
            return False
        finding_ids.add(finding_id)
        reopened_count = finding.get("reopened_count", 0)
        if type(reopened_count) is not int or reopened_count < 0:
            return False
        if not finding.get("path") or not finding.get("body"):
            return False
    return True


def parse_review_state(comment_body: str) -> ParsedReviewState:
    """Parse the versioned state marker, treating malformed state as unsafe."""
    body = comment_body or ""
    namespace_count = body.count(_STATE_MARKER_NAMESPACE)
    if namespace_count == 0:
        return ParsedReviewState(None, present=False, valid=True)
    if namespace_count != 1:
        return ParsedReviewState(None, present=True, valid=False)
    matches = list(_STATE_MARKER_RE.finditer(body))
    if len(matches) != 1:
        return ParsedReviewState(None, present=True, valid=False)
    match = matches[0]
    try:
        version = int(match.group("version"))
        state = json.loads(match.group("payload"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ParsedReviewState(None, present=True, valid=False)
    if version != STATE_SCHEMA_VERSION or not _is_valid_state(state):
        return ParsedReviewState(None, present=True, valid=False)
    return ParsedReviewState(state, present=True, valid=True)


def serialize_review_state(state: Mapping[str, Any]) -> str:
    """Serialize state deterministically so repeated updates are diffable."""
    if not _is_valid_state(state):
        raise ValueError("Invalid review finding state")
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Escape angle brackets so a finding body quoting `-->` or the marker namespace
    # can never break or duplicate the marker; json.loads still round-trips them.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<!-- pr-agent-review-state:v{STATE_SCHEMA_VERSION}\n{payload}\n-->"


def build_review_state_comment(state: Mapping[str, Any]) -> str:
    """Render the standalone hidden state comment body for a valid state."""
    marker = serialize_review_state(state)
    return f"{REVIEW_STATE_COMMENT_IDENTITY}\n\n{REVIEW_STATE_VISIBLE_LINE}\n\n{marker}\n"


def extract_reviewed_files(diff_text: str | None) -> tuple[list[str], set[str]]:
    """Split a model-call diff into reviewed files and clipped files.

    Files come from the ``## File: '<path>'`` headers in the diff text actually
    sent to the model. A file is clipped when its section ends with
    ``...(truncated)``.
    """
    if not diff_text:
        return ([], set())
    matches = list(_REVIEW_FILE_HEADER_RE.finditer(diff_text))
    files: list[str] = []
    clipped: set[str] = set()
    for index, match in enumerate(matches):
        path = _normalize_path(match.group("path"))
        if not path:
            continue
        files.append(path)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(diff_text)
        section = diff_text[start:end]
        if section.rstrip().endswith("...(truncated)"):
            clipped.add(path)
    deduped = list(dict.fromkeys(files))
    return (deduped, clipped)


def fit_review_state(state: Mapping[str, Any], max_chars: int | None) -> dict[str, Any]:
    """Compact a valid state until its standalone comment fits ``max_chars``.

    Compaction is deterministic: keep the state as-is when it fits, otherwise
    truncate every finding body to 200 chars ending in ``…``, then shrink the
    informational ``last_run.excluded_files`` list, then drop RESOLVED findings
    oldest first, then ACTIVE findings oldest first.
    """
    _MAX_EXCLUDED_FILES = 50
    fitted = copy.deepcopy(dict(state))
    fitted["findings"] = [copy.deepcopy(finding) for finding in state.get("findings", [])]
    if max_chars is None:
        return fitted
    try:
        budget = int(max_chars)
    except (TypeError, ValueError):
        return fitted
    if len(build_review_state_comment(fitted)) <= budget:
        return fitted

    truncated = 0
    for finding in fitted["findings"]:
        body = str(finding.get("body") or "")
        if len(body) > REVIEW_STATE_BODY_TRUNCATE_LEN:
            finding["body"] = body[: REVIEW_STATE_BODY_TRUNCATE_LEN - 1] + "\u2026"
            truncated += 1
    if len(build_review_state_comment(fitted)) <= budget:
        get_logger().info(
            f"Compacted review state to fit {budget} chars: truncated {truncated} bodies, dropped 0 findings"
        )
        return fitted

    last_run = fitted.get("last_run")
    if isinstance(last_run, dict) and isinstance(last_run.get("excluded_files"), list):
        excluded = [str(path) for path in last_run["excluded_files"]]
        if len(excluded) > _MAX_EXCLUDED_FILES:
            last_run["excluded_files"] = sorted(excluded)[:_MAX_EXCLUDED_FILES]
            last_run["excluded_files_truncated"] = len(excluded)
            get_logger().info(
                f"Compacted review state to fit {budget} chars: capped excluded_files "
                f"at {_MAX_EXCLUDED_FILES} of {len(excluded)} paths"
            )
            if len(build_review_state_comment(fitted)) <= budget:
                return fitted

    by_id = {finding["finding_id"]: finding for finding in fitted["findings"]}
    resolved_sorted = sorted(
        (finding for finding in fitted["findings"] if finding.get("state") == "RESOLVED"),
        key=lambda finding: (str(finding.get("resolved_at") or ""), str(finding["finding_id"])),
    )
    dropped_resolved = 0
    for finding in resolved_sorted:
        if len(build_review_state_comment({"schema_version": fitted["schema_version"],
                                           "findings": list(by_id.values()),
                                           "last_run": fitted["last_run"]})) <= budget:
            break
        by_id.pop(finding["finding_id"], None)
        dropped_resolved += 1
    if dropped_resolved:
        fitted["findings"] = sorted(by_id.values(), key=lambda finding: finding["finding_id"])
        if len(build_review_state_comment(fitted)) <= budget:
            get_logger().info(
                f"Compacted review state to fit {budget} chars: truncated {truncated} bodies, "
                f"dropped {dropped_resolved} resolved findings"
            )
            return fitted

    active_sorted = sorted(
        (finding for finding in by_id.values() if finding.get("state") == "ACTIVE"),
        key=lambda finding: (
            str(finding.get("last_seen") or ""),
            str(finding.get("first_seen") or ""),
            str(finding["finding_id"]),
        ),
    )
    dropped_active = 0
    for finding in active_sorted:
        candidate = {"schema_version": fitted["schema_version"],
                     "findings": list(by_id.values()), "last_run": fitted["last_run"]}
        if len(build_review_state_comment(candidate)) <= budget:
            break
        by_id.pop(finding["finding_id"], None)
        dropped_active += 1
    fitted["findings"] = sorted(by_id.values(), key=lambda finding: finding["finding_id"])
    get_logger().info(
        f"Compacted review state to fit {budget} chars: truncated {truncated} bodies, "
        f"dropped {dropped_resolved} resolved and {dropped_active} active findings"
    )
    return fitted


def _retained_findings(
    findings: Iterable[dict[str, Any]],
    max_resolved_findings: int,
) -> list[dict[str, Any]]:
    active = [finding for finding in findings if finding["state"] == "ACTIVE"]
    resolved = [finding for finding in findings if finding["state"] == "RESOLVED"]
    resolved.sort(
        key=lambda finding: (
            str(finding.get("resolved_at") or ""),
            str(finding["finding_id"]),
        ),
        reverse=True,
    )
    return sorted(
        active + resolved[:max(0, max_resolved_findings)],
        key=lambda finding: finding["finding_id"],
    )


def _lines_overlap(current: Mapping[str, Any], previous: Mapping[str, Any]) -> bool:
    current_start = current.get("line_start")
    current_end = current.get("line_end", current_start)
    previous_start = previous.get("line_start")
    previous_end = previous.get("line_end", previous_start)
    if not isinstance(current_start, int) or not isinstance(previous_start, int):
        return False
    if not isinstance(current_end, int) or not isinstance(previous_end, int):
        return False
    return not (current_end < previous_start or current_start > previous_end)


def reconcile_review_findings(
    previous_state: Mapping[str, Any] | None,
    current_findings: Iterable[Mapping[str, Any]],
    *,
    allow_resolution: bool,
    excluded_files: Iterable[str] | None = None,
    head_sha: str = "",
    run_id: str = "",
    timestamp: str | None = None,
    max_resolved_findings: int = DEFAULT_MAX_RESOLVED_FINDINGS,
    resolvable_paths: Iterable[str] | None = None,
    complete: bool | None = None,
) -> ReconciliationResult:
    """Reconcile current structured findings against the previous state.

    Resolution is deliberately conservative. The caller must only pass
    allow_resolution=True for a successful, non-incremental review, and the
    previous and current reviewed HEADs must both be known and different.

    When ``resolvable_paths`` is None every absent path counts as resolvable
    (backward compatible). Otherwise only paths in the set resolve. ``complete``
    defaults to ``bool(allow_resolution)`` for backward compatibility.
    """
    now = _timestamp(timestamp)
    current = normalize_findings(current_findings)
    previous_findings = list((previous_state or {}).get("findings", []))
    previous_last_run = (previous_state or {}).get("last_run", {})
    previous_head_sha = (
        previous_last_run.get("head_sha", "")
        if isinstance(previous_last_run, Mapping)
        else ""
    )
    resolution_allowed = (
        allow_resolution
        and isinstance(previous_head_sha, str)
        and bool(previous_head_sha.strip())
        and isinstance(head_sha, str)
        and bool(head_sha.strip())
        and previous_head_sha != head_sha
    )
    resolvable_set: set[str] | None = None
    if resolvable_paths is not None:
        resolvable_set = {_normalize_path(path) for path in resolvable_paths if str(path or "").strip()}
    previous_by_id = {finding["finding_id"]: finding for finding in previous_findings}
    current_by_id = {finding["finding_id"]: finding for finding in current}
    reconciled: dict[str, dict[str, Any]] = {}
    resolved_ids: list[str] = []
    reopened_ids: list[str] = []
    new_ids: list[str] = []
    open_ids: list[str] = []
    changed = previous_state is None and bool(current)
    matched_previous: set[str] = set()

    for finding_id, current_finding in current_by_id.items():
        previous = previous_by_id.get(finding_id)
        if previous is None:
            continue
        record = copy.deepcopy(previous)
        old_state = record.get("state")
        record.update(current_finding)
        record["state"] = "ACTIVE"
        record["last_seen"] = now
        if old_state == "RESOLVED":
            record["reopened_at"] = now
            record["reopened_count"] = int(record.get("reopened_count", 0)) + 1
            reopened_ids.append(finding_id)
            new_ids.append(finding_id)
        else:
            open_ids.append(finding_id)
        if record != previous:
            changed = True
        if head_sha:
            record["last_seen_head_sha"] = head_sha
        reconciled[finding_id] = record
        matched_previous.add(finding_id)

    unmatched_current = [finding for finding_id, finding in current_by_id.items() if finding_id not in reconciled]
    unmatched_current.sort(key=lambda finding: finding["finding_id"])
    unmatched_previous_active = [
        finding for finding_id, finding in previous_by_id.items()
        if finding_id not in matched_previous and finding.get("state") == "ACTIVE"
    ]
    remaining_previous = {finding["finding_id"]: finding for finding in unmatched_previous_active}

    for current_finding in unmatched_current:
        best_id: str | None = None
        best_delta: int | None = None
        current_start = current_finding.get("line_start")
        for candidate in unmatched_previous_active:
            if candidate["finding_id"] not in remaining_previous:
                continue
            if candidate.get("path") != current_finding.get("path"):
                continue
            if "line_start" not in current_finding or "line_start" not in candidate:
                continue
            if not _lines_overlap(current_finding, candidate):
                continue
            try:
                delta = abs(int(current_start) - int(candidate.get("line_start")))
            except (TypeError, ValueError):
                continue
            candidate_id = str(candidate["finding_id"])
            if best_id is None or delta < best_delta or (delta == best_delta and candidate_id < best_id):
                best_id = candidate_id
                best_delta = delta
        if best_id is None:
            record = dict(current_finding)
            record.update(first_seen=now, last_seen=now)
            changed = True
            if head_sha:
                record["last_seen_head_sha"] = head_sha
            reconciled[current_finding["finding_id"]] = record
            new_ids.append(current_finding["finding_id"])
        else:
            previous = remaining_previous.pop(best_id)
            record = copy.deepcopy(previous)
            record.update(current_finding)
            record["finding_id"] = best_id
            record["state"] = "ACTIVE"
            record["last_seen"] = now
            if head_sha:
                record["last_seen_head_sha"] = head_sha
            if record != previous:
                changed = True
            reconciled[best_id] = record
            open_ids.append(best_id)
            matched_previous.add(best_id)

    for finding_id, previous in previous_by_id.items():
        if finding_id in reconciled or finding_id in matched_previous:
            continue
        record = copy.deepcopy(previous)
        if record.get("state") == "ACTIVE":
            if resolution_allowed:
                path = _normalize_path(record.get("path"))
                resolvable = True if resolvable_set is None else path in resolvable_set
                if resolvable:
                    record["state"] = "RESOLVED"
                    record["resolved_at"] = now
                    if head_sha:
                        record["resolved_head_sha"] = head_sha
                    if run_id:
                        record["resolution_run_id"] = run_id
                    resolved_ids.append(finding_id)
                    changed = True
            if record.get("state") == "ACTIVE":
                # Still open after this run (unreviewed file, capped call,
                # same-head rerun, or blocked resolution): count it so the
                # round line never claims "0 still open" while state holds it.
                open_ids.append(finding_id)
        reconciled[finding_id] = record

    run_complete = bool(allow_resolution) if complete is None else bool(complete)
    excluded = sorted({str(path) for path in (excluded_files or []) if path})
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "findings": _retained_findings(reconciled.values(), max_resolved_findings),
        "last_run": {
            "complete": run_complete,
            "excluded_files": excluded,
            "head_sha": head_sha,
            "kind": "full" if run_complete else "partial",
            "run_id": run_id,
        },
    }
    if previous_state is not None and state["findings"] != previous_state.get("findings", []):
        changed = True
    return ReconciliationResult(
        state=state,
        changed=changed,
        resolved_ids=tuple(sorted(resolved_ids)),
        reopened_ids=tuple(sorted(reopened_ids)),
        new_ids=tuple(sorted(new_ids)),
        open_ids=tuple(sorted(open_ids)),
    )


def render_round_marker(
    head_sha: str | None,
    new_count: int,
    open_count: int,
    resolved_count: int,
    inline_count: int,
    complete: bool,
) -> str:
    complete_token = "true" if complete else "false"
    head_token = head_sha if head_sha else "unknown"
    return (
        f"<!-- pr-agent-review-round:v{_ROUND_MARKER_VERSION} head={head_token} "
        f"new={int(new_count)} open={int(open_count)} resolved={int(resolved_count)} "
        f"inline={int(inline_count)} complete={complete_token} -->"
    )


def render_round_visible(
    new_count: int,
    open_count: int,
    resolved_count: int,
    inline_count: int,
    complete: bool,
    uncovered_count: int = 0,
) -> str:
    if complete:
        coverage = "coverage: complete"
    else:
        coverage = f"coverage: partial \u2014 {int(uncovered_count)} file(s) not reviewed"
    return (
        f"**Review round:** {int(new_count)} new \u00b7 {int(open_count)} still open from earlier rounds "
        f"\u00b7 {int(resolved_count)} resolved since the last review "
        f"\u00b7 {int(inline_count)} new inline threads \u00b7 {coverage}"
    )


def build_round_summary(
    head_sha: str | None,
    new_count: int,
    open_count: int,
    resolved_count: int,
    inline_count: int,
    complete: bool,
    uncovered_count: int = 0,
) -> str:
    return (
        render_round_marker(head_sha, new_count, open_count, resolved_count, inline_count, complete)
        + "\n"
        + render_round_visible(new_count, open_count, resolved_count, inline_count, complete, uncovered_count)
    )


def insert_round_summary(review_markdown: str, summary_block: str) -> str:
    """Insert the round summary after the heading block when clean, else append."""
    body = str(review_markdown or "")
    summary = str(summary_block or "")
    if not summary:
        return body
    heading, separator, remainder = body.partition("\n\n")
    if separator and heading.strip() and remainder.strip():
        return f"{heading.rstrip()}\n\n{summary}\n\n{remainder.lstrip()}"
    stripped = body.rstrip()
    if not stripped:
        return summary
    return f"{stripped}\n\n{summary}"


def _render_resolved_entries(state: Mapping[str, Any]) -> tuple[list[str], int]:
    resolved = [finding for finding in state.get("findings", []) if finding.get("state") == "RESOLVED"]
    if not resolved:
        return ([], 0)
    resolved.sort(
        key=lambda finding: (
            str(finding.get("resolved_at") or ""),
            str(finding.get("finding_id") or ""),
        ),
        reverse=True,
    )
    entries = []
    for finding in resolved:
        location = finding["path"]
        if finding.get("line_start"):
            location += f":{finding['line_start']}"
            if finding.get("line_end") and finding["line_end"] != finding["line_start"]:
                location += f"-{finding['line_end']}"
        entries.append(f"### {location}\n\n{finding['body']}")
    return (entries, len(entries))


def append_review_state(
    review_body: str,
    state: Mapping[str, Any],
    max_chars: int | None = None,
) -> str:
    """Return human-visible review markdown with the resolved section appended.

    The review markdown is passed through byte-for-byte: it is freshly rendered
    and state is never read from the review comment, so there is nothing to
    strip. Only the appended resolved section shrinks to fit ``max_chars``:
    newest entries are kept and a ``\u2026 and N more`` line covers the rest.
    """
    body = str(review_body or "")
    entries, total = _render_resolved_entries(state or {})
    if total == 0:
        return body
    header = "<details>\n<summary>\u2705 Resolved findings</summary>\n\n"
    footer = "\n</details>"
    full_section = header + "\n\n".join(entries) + footer
    if max_chars is None:
        combined = body + ("\n\n" + full_section if body else full_section)
        return combined.rstrip() + "\n"
    try:
        budget = int(max_chars)
    except (TypeError, ValueError):
        combined = body + ("\n\n" + full_section if body else full_section)
        return combined.rstrip() + "\n"
    separator_len = len("\n\n") if body else 0
    if len(body) + separator_len + len(full_section) <= budget:
        combined = body + ("\n\n" + full_section if body else full_section)
        return combined.rstrip() + "\n"
    for keep in range(total, 0, -1):
        shown = entries[:keep]
        hidden = total - keep
        more = f"\u2026 and {hidden} more resolved findings not shown" if hidden else ""
        section = header + "\n\n".join(shown)
        if more:
            section += "\n\n" + more
        section += footer
        if len(body) + separator_len + len(section) <= budget:
            combined = body + ("\n\n" + section if body else section)
            return combined.rstrip() + "\n"
    return body
