import asyncio
import json

import pytest

from rhubarb import error_log, session_runner


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_log_path(tmp_path, monkeypatch):
    """Redirect the error log to an isolated tmp path instead of the real
    `~/.rhubarb/logs/errors.log`, mirroring how `tests/conftest.py` points
    `db.DEFAULT_DB_PATH` at a tmp db for every test."""
    log_path = tmp_path / "rhubarb-home" / ".rhubarb" / "logs" / "errors.log"
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", log_path)
    return log_path


def test_log_error_creates_directory_lazily(tmp_path):
    """The `logs/` directory (and its `.rhubarb` parent) must not need to
    pre-exist -- same lazy-mkdir convention as `db.get_connection()`."""
    log_path = tmp_path / "fresh-home" / ".rhubarb" / "logs" / "errors.log"
    assert not log_path.parent.exists()

    error_log.log_error(project_id=1, card_id=2, phase="grilling", message="boom", log_path=log_path)

    assert log_path.exists()


def test_log_error_writes_expected_json_shape(_isolated_log_path):
    error_log.log_error(project_id=7, card_id=42, phase="implementing", message="something broke")

    lines = _isolated_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert set(record.keys()) == {"timestamp", "project_id", "card_id", "phase", "message"}
    assert record["project_id"] == 7
    assert record["card_id"] == 42
    assert record["phase"] == "implementing"
    assert record["message"] == "something broke"

    # ISO 8601 -- `datetime.fromisoformat` round-trips it without error.
    from datetime import datetime

    datetime.fromisoformat(record["timestamp"])


def test_log_error_appends_one_line_per_call(_isolated_log_path):
    error_log.log_error(project_id=1, card_id=1, phase="grilling", message="first")
    error_log.log_error(project_id=1, card_id=1, phase="grilling", message="second")

    lines = _isolated_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "first"
    assert json.loads(lines[1])["message"] == "second"


def test_log_error_rotates_when_size_limit_exceeded(tmp_path, monkeypatch):
    """Configure a tiny `MAX_BYTES` so a handful of log lines force at least
    one rollover, and confirm the rotated backup file actually shows up."""
    log_path = tmp_path / "rotating-home" / ".rhubarb" / "logs" / "errors.log"
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", log_path)
    monkeypatch.setattr(error_log, "MAX_BYTES", 200)
    monkeypatch.setattr(error_log, "BACKUP_COUNT", 2)

    for i in range(20):
        error_log.log_error(
            project_id=1, card_id=1, phase="grilling", message=f"error number {i} padded out a bit"
        )

    assert log_path.exists()
    backup_path = log_path.with_name(log_path.name + ".1")
    assert backup_path.exists()

    # Never more backups than configured (errors.log.1 and .2, no .3).
    assert not log_path.with_name(log_path.name + ".3").exists()

    # Every line still on disk (across the live file and its backups) is
    # still well-formed JSON with the expected shape.
    for path in [log_path, backup_path]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            assert set(record.keys()) == {"timestamp", "project_id", "card_id", "phase", "message"}


def test_log_error_handles_none_project_and_card_id(_isolated_log_path):
    """`_turn_event`'s lock-busy path can end up with no session row (a
    stale/unknown card_id) -- `log_error` must not blow up on `None`s."""
    error_log.log_error(project_id=None, card_id=None, phase="grilling", message="no ids available")

    lines = _isolated_log_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["project_id"] is None
    assert record["card_id"] is None


# --- Integration: the shared `_turn_event` hook actually logs -------------


def test_turn_event_with_error_logs_via_shared_hook(_isolated_log_path):
    """Issue #153's whole point: every caller that reports an error through
    `session_runner._turn_event` gets a log line for free, without that
    call site instrumenting logging itself."""
    event = session_runner._turn_event(
        phase="implementing",
        error="the lock was busy",
        needs_github_login=False,
        card_id=99,
        project_id=3,
    )

    # The published event shape is untouched by the new logging hook (aside
    # from issue #154's additive `status_message` field and issue #169's
    # additive `stalled`/`stalled_context` fields, unused here).
    assert event == {
        "type": "turn",
        "phase": "implementing",
        "interview": None,
        "details": None,
        "error": "the lock was busy",
        "needs_github_login": False,
        "blocked": None,
        "status_message": None,
        "stalled": False,
        "stalled_context": None,
    }

    lines = _isolated_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["project_id"] == 3
    assert record["card_id"] == 99
    assert record["phase"] == "implementing"
    assert record["message"] == "the lock was busy"


def test_turn_event_without_error_does_not_log(_isolated_log_path):
    session_runner._turn_event(phase="grilling", interview={"questions": []})

    assert not _isolated_log_path.exists()


def test_run_turn_lock_busy_logs_exactly_once(monkeypatch, tmp_path, _isolated_log_path):
    """Reproduces the issue #149 lock-busy case end to end: a genuine
    in-flight turn for a card_id causes a second concurrent call to publish
    an error `turn` event -- which must also produce exactly one log line,
    with the project_id looked up from the session row."""
    from rhubarb import db

    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_path / "rhubarb.db")
    monkeypatch.setattr(db, "OLD_DB_PATH", tmp_path / "not-a-real-legacy-db" / "baton.db")
    # `_turn_locks` is process-global and keyed by card_id (like the other
    # module dicts `tests/conftest.py` already resets per test) -- card_ids
    # restart at 1 in every test's fresh tmp db, so start with a clean slate
    # rather than risk a leftover Lock object from another test.
    monkeypatch.setattr(session_runner, "_turn_locks", {})

    conn = db.get_connection()
    db.upsert_project(conn, str(tmp_path), "proj", "main")
    project_row = db.list_projects(conn)[0]
    card_id = db.create_session(conn, project_row["id"], session_type="do", phase="grilling")

    lock = session_runner._get_turn_lock(card_id)

    async def _scenario():
        async with lock:
            return await session_runner._run_turn(
                card_id, "hello", session_id=None, cwd=None, phase="grilling"
            )

    result = run(_scenario())
    assert result is None

    lines = _isolated_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["card_id"] == card_id
    assert record["project_id"] == project_row["id"]
    assert record["phase"] == "grilling"
    assert "already in progress" in record["message"]
