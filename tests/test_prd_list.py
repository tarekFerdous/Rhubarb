from rhubarb.prd_list import compute_prd_list


def _prd(number, title):
    return {"number": number, "title": title, "body": "## Problem Statement\n...", "labels": []}


def _child(number, parent, blocked_by=None):
    body = f"## Parent\n\n#{parent}\n\n## What to build\n\nStuff.\n\n## Blocked by\n\n"
    if blocked_by:
        body += "\n".join(f"- #{b}" for b in blocked_by) + "\n"
    else:
        body += "None - can start immediately.\n"
    return {"number": number, "title": f"child {number}", "body": body}


def test_prd_with_unblocked_child_sorts_unblocked():
    prds = [_prd(34, "Feature")]
    all_open = [_child(35, 34)]
    result = compute_prd_list(prds, all_open)
    assert result == [{"number": 34, "title": "Feature", "blocked": False}]


def test_prd_whose_only_child_is_blocked_by_open_issue_sorts_blocked():
    prds = [_prd(34, "Feature")]
    # 40's blocker (38) is itself present in the open-issues list, so it's
    # still open -- 40 is blocked, and it's the PRD's only child.
    all_open = [_child(40, 34, blocked_by=[38]), _child(38, 99)]
    result = compute_prd_list(prds, all_open)
    assert result == [{"number": 34, "title": "Feature", "blocked": True}]


def test_blocker_not_in_open_issues_counts_as_closed_so_child_is_unblocked():
    prds = [_prd(34, "Feature")]
    # Child 41's only blocker (#40) is not present in all_open -- it's closed.
    all_open = [_child(41, 34, blocked_by=[40])]
    result = compute_prd_list(prds, all_open)
    assert result == [{"number": 34, "title": "Feature", "blocked": False}]


def test_prd_with_zero_children_counts_as_unblocked():
    prds = [_prd(34, "Feature")]
    result = compute_prd_list(prds, [])
    assert result == [{"number": 34, "title": "Feature", "blocked": False}]


def test_unblocked_prds_sort_before_blocked_ties_broken_by_ascending_number():
    prds = [_prd(50, "E"), _prd(20, "B"), _prd(30, "C"), _prd(10, "A")]
    all_open = [
        _child(51, 50, blocked_by=[99]),  # 50's only child is blocked (99 still open)
        {"number": 99, "title": "unrelated open issue", "body": "no parent here"},
        _child(21, 20),  # 20 unblocked
        _child(31, 30),  # 30 unblocked
        _child(11, 10, blocked_by=[99]),  # 10's only child is blocked
    ]
    result = compute_prd_list(prds, all_open)
    assert [p["number"] for p in result] == [20, 30, 10, 50]
    assert [p["blocked"] for p in result] == [False, False, True, True]
