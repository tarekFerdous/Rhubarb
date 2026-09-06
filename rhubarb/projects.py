"""Discover git repos with a GitHub remote under a root directory.

Pure local filesystem/git inspection -- no `gh` call needed just to
enumerate candidates, since `git remote -v` already gives us what we need.
"""

import subprocess
from pathlib import Path

_GITHUB_MARKERS = ("github.com",)


def has_github_remote(repo_path: Path) -> bool:
    if not (repo_path / ".git").exists():
        return False

    result = subprocess.run(
        ["git", "remote", "-v"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return False

    return any(marker in result.stdout for marker in _GITHUB_MARKERS)


def get_current_branch(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def scan_projects(root_dir: str) -> list[dict]:
    """Scan `root_dir` one level deep for git repos with a GitHub remote."""
    root = Path(root_dir)
    if not root.is_dir():
        return []

    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not has_github_remote(entry):
            continue
        found.append(
            {
                "path": str(entry.resolve()),
                "name": entry.name,
                "branch": get_current_branch(entry),
            }
        )
    return found
