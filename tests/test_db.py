import json
import sqlite3

from rhubarb import db


def test_root_dir_round_trip(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    assert db.get_root_dir(conn) is None

    db.set_root_dir(conn, "/some/root")
    assert db.get_root_dir(conn) == "/some/root"

    db.set_root_dir(conn, "/other/root")
    assert db.get_root_dir(conn) == "/other/root"


def test_afk_hours_round_trip(tmp_path):
    db_path = tmp_path / "rhubarb.db"
    conn = db.get_connection(db_path)
    assert db.get_afk_hours(conn) == 6

    db.set_afk_hours(conn, 10)
    assert db.get_afk_hours(conn) == 10

    reopened = db.get_connection(db_path)
    assert db.get_afk_hours(reopened) == 10


def test_parallel_implementation_round_trip(tmp_path):
    db_path = tmp_path / "rhubarb.db"
    conn = db.get_connection(db_path)
    assert db.get_parallel_implementation(conn) is False

    db.set_parallel_implementation(conn, True)
    assert db.get_parallel_implementation(conn) is True

    reopened = db.get_connection(db_path)
    assert db.get_parallel_implementation(reopened) is True

    db.set_parallel_implementation(reopened, False)
    assert db.get_parallel_implementation(reopened) is False


def test_parallel_implementation_column_migrates_existing_db_defaulting_off(tmp_path):
    """A DB created before `settings.parallel_implementation` existed (simulated
    here by building the pre-migration schema by hand) must gain the column
    defaulting to off (sequential), not the old on-by-default behavior, and
    keep its other settings intact when `get_connection` runs its guarded
    ALTER TABLE."""
    db_path = tmp_path / "rhubarb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root_dir TEXT,
            afk_hours INTEGER NOT NULL DEFAULT 6
        )
        """
    )
    conn.execute("INSERT INTO settings (id, root_dir, afk_hours) VALUES (1, '/some/root', 9)")
    conn.commit()
    conn.close()

    migrated = db.get_connection(db_path)
    assert db.get_parallel_implementation(migrated) is False
    # Pre-existing data survived the migration untouched.
    assert db.get_root_dir(migrated) == "/some/root"
    assert db.get_afk_hours(migrated) == 9


def test_terminal_view_hidden_round_trip(tmp_path):
    db_path = tmp_path / "rhubarb.db"
    conn = db.get_connection(db_path)
    assert db.get_terminal_view_hidden(conn) is False

    db.set_terminal_view_hidden(conn, True)
    assert db.get_terminal_view_hidden(conn) is True

    reopened = db.get_connection(db_path)
    assert db.get_terminal_view_hidden(reopened) is True

    db.set_terminal_view_hidden(reopened, False)
    assert db.get_terminal_view_hidden(reopened) is False


def test_terminal_view_hidden_column_migrates_existing_db_defaulting_visible(tmp_path):
    """A DB created before `settings.terminal_view_hidden` existed (simulated
    here by building the pre-migration schema by hand) must gain the column
    defaulting to visible/expanded (issue #89's default), and keep its other
    settings intact when `get_connection` runs its guarded ALTER TABLE."""
    db_path = tmp_path / "rhubarb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root_dir TEXT,
            afk_hours INTEGER NOT NULL DEFAULT 6
        )
        """
    )
    conn.execute("INSERT INTO settings (id, root_dir, afk_hours) VALUES (1, '/some/root', 9)")
    conn.commit()
    conn.close()

    migrated = db.get_connection(db_path)
    assert db.get_terminal_view_hidden(migrated) is False
    # Pre-existing data survived the migration untouched.
    assert db.get_root_dir(migrated) == "/some/root"
    assert db.get_afk_hours(migrated) == 9


def test_model_round_trip(tmp_path):
    db_path = tmp_path / "rhubarb.db"
    conn = db.get_connection(db_path)
    assert db.get_model(conn) == "claude-sonnet-4-6"

    db.set_model(conn, "claude-opus-4-8")
    assert db.get_model(conn) == "claude-opus-4-8"

    reopened = db.get_connection(db_path)
    assert db.get_model(reopened) == "claude-opus-4-8"


def test_model_column_migrates_existing_db_without_data_loss(tmp_path):
    """A DB created before `settings.model` existed (simulated here by
    building the pre-migration schema by hand) must gain the column,
    default to claude-sonnet-4-6, and keep its other settings intact when
    `get_connection` runs its guarded ALTER TABLE."""
    db_path = tmp_path / "rhubarb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root_dir TEXT,
            afk_hours INTEGER NOT NULL DEFAULT 6,
            parallel_implementation INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute("INSERT INTO settings (id, root_dir, afk_hours) VALUES (1, '/some/root', 9)")
    conn.commit()
    conn.close()

    migrated = db.get_connection(db_path)
    assert db.get_model(migrated) == "claude-sonnet-4-6"
    # Pre-existing data survived the migration untouched.
    assert db.get_root_dir(migrated) == "/some/root"
    assert db.get_afk_hours(migrated) == 9


def test_effort_round_trip(tmp_path):
    db_path = tmp_path / "rhubarb.db"
    conn = db.get_connection(db_path)
    assert db.get_effort(conn) == "auto"

    db.set_effort(conn, "high")
    assert db.get_effort(conn) == "high"

    reopened = db.get_connection(db_path)
    assert db.get_effort(reopened) == "high"


def test_effort_column_migrates_existing_db_without_data_loss(tmp_path):
    """A DB created before `settings.effort` existed (simulated here by
    building the pre-migration schema by hand) must gain the column,
    default to "auto", and keep its other settings intact when
    `get_connection` runs its guarded ALTER TABLE."""
    db_path = tmp_path / "rhubarb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root_dir TEXT,
            afk_hours INTEGER NOT NULL DEFAULT 6,
            parallel_implementation INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL DEFAULT 'claude-sonnet-4-6'
        )
        """
    )
    conn.execute("INSERT INTO settings (id, root_dir, afk_hours) VALUES (1, '/some/root', 9)")
    conn.commit()
    conn.close()

    migrated = db.get_connection(db_path)
    assert db.get_effort(migrated) == "auto"
    # Pre-existing data survived the migration untouched.
    assert db.get_root_dir(migrated) == "/some/root"
    assert db.get_afk_hours(migrated) == 9


def test_create_session_seeds_effort_column(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.upsert_project(conn, "/repos/foo", "foo", "main")
    [project] = db.list_projects(conn)

    row_id = db.create_session(conn, project["id"], effort="high")
    row = db.get_session(conn, row_id)
    assert row["effort"] == "high"

    default_row_id = db.create_session(conn, project["id"])
    default_row = db.get_session(conn, default_row_id)
    assert default_row["effort"] == "auto"


def test_get_connection_migrates_legacy_baton_db_when_new_path_absent(tmp_path, monkeypatch):
    """Issue #93: a user upgrading across the #91 package rename has data at
    the old `~/.baton/baton.db` path but nothing yet at the new
    `~/.rhubarb/rhubarb.db` path. `get_connection()` (called with no
    explicit path, as the real app does) must copy the old file over before
    opening the new path, and must leave the old file untouched afterward."""
    old_path = tmp_path / "old_home" / ".baton" / "baton.db"
    new_path = tmp_path / "new_home" / ".rhubarb" / "rhubarb.db"
    old_path.parent.mkdir(parents=True)

    legacy_conn = db.get_connection(old_path)
    db.set_root_dir(legacy_conn, "/legacy/root")
    legacy_conn.close()
    old_mtime = old_path.stat().st_mtime

    monkeypatch.setattr(db, "OLD_DB_PATH", old_path)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", new_path)

    migrated_conn = db.get_connection()
    assert new_path.exists()
    assert db.get_root_dir(migrated_conn) == "/legacy/root"

    # Old file is a fallback/backup -- left in place, unmodified.
    assert old_path.exists()
    assert old_path.stat().st_mtime == old_mtime


def test_get_connection_does_not_clobber_existing_new_db_with_legacy_data(tmp_path, monkeypatch):
    """If `~/.rhubarb/rhubarb.db` already exists, migration must never run --
    even if an old `~/.baton/baton.db` is also present, the new path's own
    data wins and is left exactly as it was."""
    old_path = tmp_path / "old_home" / ".baton" / "baton.db"
    new_path = tmp_path / "new_home" / ".rhubarb" / "rhubarb.db"
    old_path.parent.mkdir(parents=True)
    new_path.parent.mkdir(parents=True)

    legacy_conn = db.get_connection(old_path)
    db.set_root_dir(legacy_conn, "/legacy/root")
    legacy_conn.close()

    existing_conn = db.get_connection(new_path)
    db.set_root_dir(existing_conn, "/current/root")
    existing_conn.close()

    monkeypatch.setattr(db, "OLD_DB_PATH", old_path)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", new_path)

    reopened = db.get_connection()
    assert db.get_root_dir(reopened) == "/current/root"


def test_get_connection_creates_fresh_db_when_neither_path_exists(tmp_path, monkeypatch):
    """No `.baton` and no `.rhubarb` data anywhere -- behaves exactly as
    today, creating a fresh DB at the new path with no migration attempted."""
    old_path = tmp_path / "old_home" / ".baton" / "baton.db"
    new_path = tmp_path / "new_home" / ".rhubarb" / "rhubarb.db"

    monkeypatch.setattr(db, "OLD_DB_PATH", old_path)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", new_path)

    assert not old_path.exists()
    assert not new_path.exists()

    conn = db.get_connection()
    assert new_path.exists()
    assert not old_path.exists()
    assert db.get_root_dir(conn) is None


def test_project_open_close_reopen_preserves_state(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.upsert_project(conn, "/repos/foo", "foo", "main")

    [row] = db.list_projects(conn)
    assert db.load_session_state(row) == {}

    db.mark_opened(conn, row["id"])
    db.save_session_state(conn, row["id"], {"session_id": "abc123", "console_text": "hello"})

    reopened = db.get_project(conn, row["id"])
    assert db.load_session_state(reopened) == {"session_id": "abc123", "console_text": "hello"}
    assert reopened["last_opened"] is not None


def test_clear_projects_removes_all_rows(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.upsert_project(conn, "/repos/foo", "foo", "main")
    db.upsert_project(conn, "/repos/bar", "bar", "dev")
    assert len(db.list_projects(conn)) == 2

    db.clear_projects(conn)
    assert db.list_projects(conn) == []


def test_create_session_starts_in_grilling_phase(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(conn, project_id=1)

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    assert row["available_for_reuse"] == 0


def test_session_phase_transitions_through_state_machine(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(conn, project_id=1, claude_session_id="s1")

    for phase in ("creating_prd", "creating_issues", "details"):
        db.update_session(conn, row_id, phase=phase)
        assert db.get_session(conn, row_id)["phase"] == phase


def test_claim_available_session_is_scoped_to_its_project(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    session_a = db.create_session(conn, project_id=1, claude_session_id="a1")
    db.mark_session_available(conn, session_a, "a1")

    # Project B has no available session of its own.
    assert db.claim_available_session(conn, project_id=2) is None

    claimed = db.claim_available_session(conn, project_id=1)
    assert claimed["claude_session_id"] == "a1"
    assert claimed["available_for_reuse"] == 0

    # Already claimed -- not handed out twice.
    assert db.claim_available_session(conn, project_id=1) is None


def test_create_session_seeds_session_type_phase_and_details(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(
        conn,
        project_id=1,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 3, "title": "T"}},
    )

    row = db.get_session(conn, row_id)
    assert row["session_type"] == "implement"
    assert row["phase"] == "implementing"
    assert json.loads(row["details_json"]) == {"prd": {"number": 3, "title": "T"}}


def test_create_session_defaults_model_to_claude_sonnet(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(conn, project_id=1)

    row = db.get_session(conn, row_id)
    assert row["model"] == "claude-sonnet-4-6"


def test_create_session_accepts_explicit_model(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(conn, project_id=1, model="claude-opus-4-8")

    row = db.get_session(conn, row_id)
    assert row["model"] == "claude-opus-4-8"


def test_create_session_defaults_session_type_to_do(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    row_id = db.create_session(conn, project_id=1)

    row = db.get_session(conn, row_id)
    assert row["session_type"] == "do"


def test_has_active_implement_session_detects_live_session_for_prd_number(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 9, "title": "X"}}
    )

    assert db.has_active_implement_session(conn, project_id=1, prd_number=9) is True
    assert db.has_active_implement_session(conn, project_id=1, prd_number=10) is False
    assert db.has_active_implement_session(conn, project_id=2, prd_number=9) is False


def test_has_active_implement_session_ignores_terminal_or_errored_sessions(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.create_session(
        conn, project_id=1, session_type="implement", phase="implemented", details={"prd": {"number": 4, "title": "Y"}}
    )
    errored_id = db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 4, "title": "Y"}}
    )
    db.update_session(conn, errored_id, error_text="boom")

    assert db.has_active_implement_session(conn, project_id=1, prd_number=4) is False


def test_has_any_active_implement_session_ignores_prd_number(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    assert db.has_any_active_implement_session(conn, project_id=1) is False

    db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 9, "title": "X"}}
    )

    # True regardless of which PRD number the live session is for.
    assert db.has_any_active_implement_session(conn, project_id=1) is True
    assert db.has_any_active_implement_session(conn, project_id=2) is False


def test_has_any_active_implement_session_ignores_terminal_or_errored_sessions(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    db.create_session(
        conn, project_id=1, session_type="implement", phase="implemented", details={"prd": {"number": 4, "title": "Y"}}
    )
    errored_id = db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 4, "title": "Y"}}
    )
    db.update_session(conn, errored_id, error_text="boom")

    assert db.has_any_active_implement_session(conn, project_id=1) is False


def test_cleanup_sessions_on_shutdown_keeps_only_most_recently_used(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    old = db.create_session(conn, project_id=1, claude_session_id="old")
    conn.execute("UPDATE sessions SET last_activity = '2020-01-01T00:00:00' WHERE id = ?", (old,))

    newer = db.create_session(conn, project_id=2, claude_session_id="newer")
    conn.execute("UPDATE sessions SET last_activity = '2020-01-02T00:00:00' WHERE id = ?", (newer,))

    newest = db.create_session(conn, project_id=1, claude_session_id="newest")
    conn.execute("UPDATE sessions SET last_activity = '2020-01-03T00:00:00' WHERE id = ?", (newest,))
    conn.commit()

    db.cleanup_sessions_on_shutdown(conn)

    remaining = conn.execute("SELECT id FROM sessions").fetchall()
    assert [r["id"] for r in remaining] == [newest]


def test_recover_interrupted_implement_sessions_marks_stuck_row(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    stuck_id = db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 1}}
    )

    db.recover_interrupted_implement_sessions(conn)

    row = db.get_session(conn, stuck_id)
    assert row["error_text"]
    assert "interrupted" in row["error_text"].lower()


def test_recover_interrupted_implement_sessions_leaves_existing_error_untouched(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    errored_id = db.create_session(
        conn, project_id=1, session_type="implement", phase="implementing", details={"prd": {"number": 1}}
    )
    db.update_session(conn, errored_id, error_text="original failure")

    db.recover_interrupted_implement_sessions(conn)

    row = db.get_session(conn, errored_id)
    assert row["error_text"] == "original failure"


def test_recover_interrupted_implement_sessions_leaves_implemented_untouched(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    implemented_id = db.create_session(
        conn, project_id=1, session_type="implement", phase="implemented", details={"prd": {"number": 1}}
    )

    db.recover_interrupted_implement_sessions(conn)

    row = db.get_session(conn, implemented_id)
    assert row["error_text"] is None


def test_recover_interrupted_implement_sessions_leaves_do_session_untouched(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    do_id = db.create_session(conn, project_id=1, session_type="do", phase="implementing")

    db.recover_interrupted_implement_sessions(conn)

    row = db.get_session(conn, do_id)
    assert row["error_text"] is None


def test_recover_interrupted_implement_sessions_leaves_grilling_do_session_untouched(tmp_path):
    conn = db.get_connection(tmp_path / "rhubarb.db")
    do_id = db.create_session(conn, project_id=1, session_type="do", phase="grilling")

    db.recover_interrupted_implement_sessions(conn)

    row = db.get_session(conn, do_id)
    assert row["error_text"] is None
