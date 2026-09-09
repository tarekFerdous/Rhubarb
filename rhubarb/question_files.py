"""Read/delete helpers for a session's pending "question file" under
`.claude/` in the project directory (issue #124, PRD #123).

A skill (`/rhubarb:grilling`, `/rhubarb:qa-grilling`, `/rhubarb:implement`)
writes its structured question/blocked-marker output to one of these files
via the Write tool -- a deterministic tool call, unaffected by terminal
rendering or PTY capture timing -- in addition to printing the same content
to chat as it already does. `session_runner.py` prefers reading this file
over scraping the turn's rendered terminal text, falling back to the
terminal-text parse (and Ollama rescue) only when the file is absent.

Resolved against a session's `cwd` -- the same working directory the
`claude` CLI subprocess runs in, so a relative Write-tool path from inside
that process lands exactly where these functions look."""

from pathlib import Path


def _resolve(cwd: str | None, filename: str) -> Path | None:
    if not cwd:
        return None
    return Path(cwd) / ".claude" / filename


def read_question_file(cwd: str | None, filename: str) -> str | None:
    """Return `.claude/<filename>`'s exact text content under `cwd`, or
    `None` if `cwd` is unset or the file doesn't exist. Never deletes --
    the file is left in place until the round it describes is actually
    answered (see `delete_question_file`)."""
    path = _resolve(cwd, filename)
    if path is None or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def delete_question_file(cwd: str | None, filename: str) -> None:
    """Delete `.claude/<filename>` under `cwd` if present. A no-op, not an
    error, when `cwd` is unset or the file doesn't exist."""
    path = _resolve(cwd, filename)
    if path is None:
        return
    path.unlink(missing_ok=True)
