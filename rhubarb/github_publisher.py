"""Publishes a PRD draft to GitHub via `gh issue create`, run as a subprocess.

Deliberately independent of the Claude CLI: GitHub publishing is a scripted
operation that needs no AI reasoning, so it must not consume subscription
turns. See CLAUDE.md and issue #55.
"""

import json
import re
import subprocess
from pathlib import Path

_ISSUE_URL_RE = re.compile(r"/issues/(\d+)\s*$")


class GithubPublishError(RuntimeError):
    pass


def _create_issue(title: str, body: str, labels: list[str], *, cwd: str) -> tuple[int, str]:
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


def publish_draft(draft_path: Path, cwd: str) -> str:
    draft_path = Path(draft_path)

    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise GithubPublishError(f"could not read draft file {draft_path}: {e}") from e

    prd_number = None
    try:
        prd = draft["prd"]
        prd_number, prd_title = _create_issue(prd["title"], prd["body"], prd.get("labels", []), cwd=cwd)
        lines = [f"PRD #{prd_number}: {prd_title}"]

        for issue in draft.get("issues", []):
            issue_number, issue_title = _create_issue(issue["title"], issue["body"], issue.get("labels", []), cwd=cwd)
            lines.append(f"Issue #{issue_number}: {issue_title}")

        return "\n".join(lines)
    except GithubPublishError as e:
        if prd_number is not None:
            raise GithubPublishError(f"PRD #{prd_number} was created, but a later step failed: {e}") from e
        raise
    finally:
        draft_path.unlink(missing_ok=True)
