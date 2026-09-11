"""App-wide rotating error log (issue #153).

Every error that already flows through `session_runner._turn_event` (the
one shared point where every error `turn` event is built before being
published on a session's live stream -- see that function and its docstring)
is additionally written here as one JSON line. This is a standing
diagnostics trail on disk, independent of and in addition to the live-stream
error event -- it does not replace or change that reporting path.

Follows the same lazy-`mkdir` app-data-directory convention already used by
`rhubarb.db` (`DEFAULT_DB_PATH` / `get_connection()`, which creates
`~/.rhubarb/` and its `rhubarb.db` file on first use rather than requiring
either to pre-exist): nothing under `~/.rhubarb/logs/` is created until the
first error is actually logged.
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

# Mirrors `db.DEFAULT_DB_PATH`'s convention -- a module-level Path constant
# so tests can monkeypatch it to an isolated tmp path instead of writing to
# the real `~/.rhubarb/` during a test run.
DEFAULT_LOG_PATH = Path.home() / ".rhubarb" / "logs" / "errors.log"

# Not user-configurable -- just sane constants keeping the file bounded.
# 5 MB per file, 3 backups (errors.log.1 .. errors.log.3) => ~20 MB worst
# case on disk for this log.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_LOGGER_NAME = "rhubarb.errors"


def _build_logger(log_path: Path) -> logging.Logger:
    """A fresh `Logger` + `RotatingFileHandler` pointed at `log_path`,
    constructed on every call rather than cached at import time (or behind
    `logging.getLogger`, which *would* cache/reuse a single instance) -- so
    a test's monkeypatch of `DEFAULT_LOG_PATH`, or a caller's explicit
    `log_path`, always takes effect for that call, and the parent directory
    (mkdir'd here, same as `db.get_connection()` does for `DEFAULT_DB_PATH`'s
    parent) is only ever created the first time an error is actually logged.

    Not caching the logger also sidesteps handler accumulation: since each
    call gets its own unregistered `Logger` instance (not the shared one
    `logging.getLogger(name)` would hand back), there's no risk of the same
    process re-adding a handler to a long-lived cached logger on every error.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.Logger(_LOGGER_NAME)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_error(
    *,
    project_id: int | None,
    card_id: int | None,
    phase: str | None,
    message: str,
    log_path: Path | None = None,
) -> None:
    """Append one JSON line describing an error to the rotating error log.

    This is the single hook `session_runner._turn_event` calls whenever it
    builds an error `turn` event (i.e. whenever `error is not None`) --
    every existing and future call site that reports an error through
    `_turn_event` gets a log line here automatically, with no per-call-site
    instrumentation. `log_path` is for tests that want to point this at an
    isolated tmp file explicitly rather than relying on a `DEFAULT_LOG_PATH`
    monkeypatch.
    """
    path = log_path if log_path is not None else DEFAULT_LOG_PATH
    logger = _build_logger(path)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "card_id": card_id,
        "phase": phase,
        "message": message,
    }
    try:
        logger.error(json.dumps(record))
    finally:
        for handler in logger.handlers:
            handler.close()


# --- Reader (issue #155) ---------------------------------------------------


def _rotated_paths(log_path: Path) -> list[Path]:
    """The rotated backup files for `log_path`, oldest first, followed by
    the live file itself last -- `RotatingFileHandler` numbers backups
    `.1` (most recently rotated) through `.BACKUP_COUNT` (oldest), so this
    reverses that order to read chronologically."""
    paths = []
    for i in range(BACKUP_COUNT, 0, -1):
        backup = log_path.with_name(f"{log_path.name}.{i}")
        if backup.exists():
            paths.append(backup)
    if log_path.exists():
        paths.append(log_path)
    return paths


def _iter_records(log_path: Path):
    for path in _rotated_paths(log_path):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A line clipped mid-write (e.g. process killed mid-append)
                # shouldn't take down the whole read.
                continue


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, treating a naive value as UTC so it can
    be compared against the always-UTC-aware timestamps `log_error` writes."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def query_errors(
    project_id: int,
    *,
    phase: str | None = None,
    since: str | None = None,
    until: str | None = None,
    card_id: int | None = None,
    q: str | None = None,
    log_path: Path | None = None,
) -> list[dict]:
    """Return logged error entries for `project_id`, oldest first, narrowed
    by whichever optional filters are given (AND semantics across all of
    them): exact `phase` match, a `since`/`until` ISO-timestamp range
    (inclusive), exact `card_id` match, and a case-insensitive substring
    match for `q` against `message`. Reads the live log file plus any
    rotated backups (see `_rotated_paths`); a missing log file (nothing
    logged yet) yields an empty list rather than raising.
    """
    path = log_path if log_path is not None else DEFAULT_LOG_PATH
    since_dt = _parse_iso(since) if since else None
    until_dt = _parse_iso(until) if until else None
    needle = q.lower() if q else None

    matches = []
    for record in _iter_records(path):
        if record.get("project_id") != project_id:
            continue
        if phase is not None and record.get("phase") != phase:
            continue
        if card_id is not None and record.get("card_id") != card_id:
            continue
        if needle is not None and needle not in (record.get("message") or "").lower():
            continue
        if since_dt is not None or until_dt is not None:
            timestamp = record.get("timestamp")
            if not timestamp:
                continue
            try:
                record_dt = _parse_iso(timestamp)
            except ValueError:
                continue
            if since_dt is not None and record_dt < since_dt:
                continue
            if until_dt is not None and record_dt > until_dt:
                continue
        matches.append(record)
    return matches
