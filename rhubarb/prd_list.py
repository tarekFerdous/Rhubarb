"""Compute the sorted, blockage-annotated PRD list for the "To be
implemented" card.

Pure data transform -- no `gh`/subprocess calls here. The caller fetches two
already-open-issue-scoped lists (see `rhubarb.web.app`'s endpoint) and this
module turns them into the sorted `{number, title, blocked}` shape the
frontend renders as `PRD: N` buttons.
"""

import re

_PARENT_RE = re.compile(r"##\s*Parent\s*\n+\s*#(\d+)", re.IGNORECASE)
_BLOCKED_SECTION_RE = re.compile(r"##\s*Blocked by\s*\n+(.*?)(?=\n##|\Z)", re.IGNORECASE | re.DOTALL)
_BLOCKER_NUM_RE = re.compile(r"#(\d+)")


def _parent_number(body: str) -> int | None:
    match = _PARENT_RE.search(body or "")
    return int(match.group(1)) if match else None


def _blocker_numbers(body: str) -> list[int]:
    match = _BLOCKED_SECTION_RE.search(body or "")
    if not match:
        return []
    section = match.group(1)
    return [int(n) for n in _BLOCKER_NUM_RE.findall(section)]


def _child_is_unblocked(child_body: str, open_issue_numbers: set[int]) -> bool:
    blockers = _blocker_numbers(child_body)
    if not blockers:
        return True
    # A blocker still counts as open (blocking) only if it appears in the
    # open-issues list the caller fetched -- a number no longer present there
    # is closed/resolved.
    return not any(b in open_issue_numbers for b in blockers)


def compute_prd_list(prds: list[dict], all_open_issues: list[dict]) -> list[dict]:
    """`prds`: open issues labeled `prd` (each `{number, title, body, ...}`) --
    the caller's fetch filters by that label, so every entry here is already
    guaranteed to be a standalone, top-level PRD, not a child slice.
    `all_open_issues`: every other open issue (each `{number, title, body}`),
    used to find each PRD's children (via `## Parent` -> `#<prd_number>`) and
    to resolve whether a child's blockers are still open.

    Returns `[{"number": int, "title": str, "blocked": bool}, ...]`, unblocked
    entries first, then blocked, ties broken by ascending issue number.
    """
    open_issue_numbers = {issue["number"] for issue in all_open_issues}

    children_by_parent: dict[int, list[dict]] = {}
    for issue in all_open_issues:
        parent = _parent_number(issue.get("body", ""))
        if parent is not None:
            children_by_parent.setdefault(parent, []).append(issue)

    results = []
    for prd in prds:
        children = children_by_parent.get(prd["number"], [])
        blocked = bool(children) and not any(
            _child_is_unblocked(child.get("body", ""), open_issue_numbers) for child in children
        )
        results.append({"number": prd["number"], "title": prd["title"], "blocked": blocked})

    results.sort(key=lambda p: (p["blocked"], p["number"]))
    return results
