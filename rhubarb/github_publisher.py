"""Publishes a PRD draft to GitHub via `gh issue create`, run as a subprocess.

Deliberately independent of the Claude CLI: GitHub publishing is a scripted
operation that needs no AI reasoning, so it must not consume subscription
turns. See CLAUDE.md and issue #55.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

_ISSUE_URL_RE = re.compile(r"/issues/(\d+)\s*$")


class GithubPublishError(RuntimeError):
    pass


def create_issue(title: str, body: str, labels: list[str], *, cwd: str) -> tuple[int, str]:
    """Create one GitHub issue via `gh issue create` and return its
    `(number, title)`. Public (issue #154) so a caller like
    `session_runner._run_publish_step` can drive the per-issue loop itself
    -- or just pass an `on_progress` callback into `publish_draft` below,
    which is built on top of this same function."""
    args = ["gh", "issue", "create", "--title", title, "--body", body]
    for label in labels:
        args += ["--label", label]

    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd, check=False)

    if result.returncode != 0:
        raise GithubPublishError(f"gh issue create failed for {title!r}: {result.stderr.strip()}")

    match = _ISSUE_URL_RE.search(result.stdout.strip())
    if match is None:
        raise GithubPublishError(f"could not parse issue number from gh output for {title!r}: {result.stdout.strip()}")

    return int(match.group(1)), title


# Backwards-compat alias -- `create_issue` used to be private (`_create_issue`).
_create_issue = create_issue


def publish_draft(
    draft_path: Path,
    cwd: str,
    *,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> str:
    """Read `draft_path` and create the PRD issue, then each child issue in
    `draft["issues"]`, via `create_issue()` above.

    `on_progress` (issue #154), if given, is called synchronously right
    after each issue is created -- once for the PRD
    (`{"kind": "prd", "number": N, "title": ...}`) and once per child issue
    (`{"kind": "issue", "number": N, "title": ...}`) -- so a caller can
    observe progress mid-call instead of only seeing the final joined-string
    summary this function still returns once everything is done. Omit it
    (the default) to get exactly the old behavior.

    Preserves every other existing behavior: the draft file is deleted via
    `finally` regardless of outcome, and a failure on issue N is reported
    with the already-created PRD's number folded into the message.
    """
    draft_path = Path(draft_path)

    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GithubPublishError(f"could not read draft file {draft_path}: {e}") from e

    prd_number = None
    try:
        prd = draft["prd"]
        prd_number, prd_title = create_issue(prd["title"], prd["body"], prd.get("labels", []), cwd=cwd)
        lines = [f"PRD #{prd_number}: {prd_title}"]
        if on_progress is not None:
            on_progress({"kind": "prd", "number": prd_number, "title": prd_title})

        for issue in draft.get("issues", []):
            issue_number, issue_title = create_issue(issue["title"], issue["body"], issue.get("labels", []), cwd=cwd)
            lines.append(f"Issue #{issue_number}: {issue_title}")
            if on_progress is not None:
                on_progress({"kind": "issue", "number": issue_number, "title": issue_title})

        return "\n".join(lines)
    except GithubPublishError as e:
        if prd_number is not None:
            raise GithubPublishError(f"PRD #{prd_number} was created, but a later step failed: {e}") from e
        raise
    finally:
        draft_path.unlink(missing_ok=True)
