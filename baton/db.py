"""SQLite persistence for root directory settings and discovered projects.

Uses stdlib `sqlite3` -- no new dependency, per CLAUDE.md's minimal-footprint
approach. `db_path` is accepted explicitly everywhere so tests can point at
an isolated temp database instead of the real one under the user's home dir.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".baton" / "baton.db"

# The model Baton passes to every `claude` CLI invocation (via `--model`)
# unless a session already has one recorded -- see `get_model`/`set_model`
# (the global setting) and the `sessions.model` column (the value a given
# session was created with) below.
DEFAULT_MODEL = "claude-sonnet-4-6"

# The reasoning-effort level Baton passes to every `claude` CLI invocation
# (via `--effort`, omitted entirely for "auto") unless a session already has
# one recorded -- see `get_effort`/`set_effort` (the global setting) and the
# `sessions.effort` column (the value a given session was created with, or
# updated to if changed before its first turn) below.
DEFAULT_EFFORT = "auto"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Concurrent sessions each open their own connection (sqlite3 connections
    # aren't shareable across threads) and can write around the same time --
    # WAL lets readers and a writer proceed together, and busy_timeout makes
    # a writer wait out a brief conflict instead of raising "database is locked".
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            root_dir TEXT
        )
        """
    )
    # CREATE TABLE IF NOT EXISTS above won't add new columns to a settings
    # table that already exists on disk from before this column existed --
    # add it here, no-op'ing if it's already present.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(settings)")}
    if "afk_hours" not in existing_columns:
        conn.execute("ALTER TABLE settings ADD COLUMN afk_hours INTEGER NOT NULL DEFAULT 6")
    if "parallel_implementation" not in existing_columns:
        conn.execute(
            "ALTER TABLE settings ADD COLUMN parallel_implementation INTEGER NOT NULL DEFAULT 0"
        )
    if "terminal_view_hidden" not in existing_columns:
        conn.execute(
            "ALTER TABLE settings ADD COLUMN terminal_view_hidden INTEGER NOT NULL DEFAULT 0"
        )
    if "model" not in existing_columns:
        conn.execute(
            f"ALTER TABLE settings ADD COLUMN model TEXT NOT NULL DEFAULT '{DEFAULT_MODEL}'"
        )
    if "effort" not in existing_columns:
        conn.execute(
            f"ALTER TABLE settings ADD COLUMN effort TEXT NOT NULL DEFAULT '{DEFAULT_EFFORT}'"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            last_opened TEXT,
            branch TEXT,
            session_state TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            claude_session_id TEXT,
            phase TEXT NOT NULL DEFAULT 'grilling',
            console_text TEXT NOT NULL DEFAULT '',
            interview_json TEXT,
            details_json TEXT,
            error_text TEXT,
            needs_github_login INTEGER NOT NULL DEFAULT 0,
            last_activity TEXT NOT NULL,
            available_for_reuse INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    # Same guarded-migration approach as afk_hours above: a sessions table
    # that already exists on disk from before session_type existed won't get
    # the new column from CREATE TABLE IF NOT EXISTS.
    session_columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "session_type" not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT NOT NULL DEFAULT 'do'")
    if "model" not in session_columns:
        conn.execute(
            f"ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT '{DEFAULT_MODEL}'"
        )
    if "effort" not in session_columns:
        conn.execute(
            f"ALTER TABLE sessions ADD COLUMN effort TEXT NOT NULL DEFAULT '{DEFAULT_EFFORT}'"
        )
    if "context_pct" not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN context_pct REAL")
    if "blocked_json" not in session_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN blocked_json TEXT")
    conn.commit()
    return conn


def get_root_dir(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT root_dir FROM settings WHERE id = 1").fetchone()
    return row["root_dir"] if row else None


def set_root_dir(conn: sqlite3.Connection, root_dir: str) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, root_dir) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET root_dir = excluded.root_dir
        """,
        (root_dir,),
    )
    conn.commit()


def get_afk_hours(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT afk_hours FROM settings WHERE id = 1").fetchone()
    return row["afk_hours"] if row else 6


def set_afk_hours(conn: sqlite3.Connection, hours: int) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, afk_hours) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET afk_hours = excluded.afk_hours
        """,
        (hours,),
    )
    conn.commit()


def get_parallel_implementation(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT parallel_implementation FROM settings WHERE id = 1").fetchone()
    return bool(row["parallel_implementation"]) if row else False


def set_parallel_implementation(conn: sqlite3.Connection, value: bool) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, parallel_implementation) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET parallel_implementation = excluded.parallel_implementation
        """,
        (1 if value else 0,),
    )
    conn.commit()


def get_terminal_view_hidden(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT terminal_view_hidden FROM settings WHERE id = 1").fetchone()
    return bool(row["terminal_view_hidden"]) if row else False


def set_terminal_view_hidden(conn: sqlite3.Connection, value: bool) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, terminal_view_hidden) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET terminal_view_hidden = excluded.terminal_view_hidden
        """,
        (1 if value else 0,),
    )
    conn.commit()


def get_model(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT model FROM settings WHERE id = 1").fetchone()
    return row["model"] if row and row["model"] is not None else DEFAULT_MODEL


def set_model(conn: sqlite3.Connection, model: str) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, model) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET model = excluded.model
        """,
        (model,),
    )
    conn.commit()


def get_effort(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT effort FROM settings WHERE id = 1").fetchone()
    return row["effort"] if row and row["effort"] is not None else DEFAULT_EFFORT


def set_effort(conn: sqlite3.Connection, effort: str) -> None:
    conn.execute(
        """
        INSERT INTO settings (id, effort) VALUES (1, ?)
        ON CONFLICT(id) DO UPDATE SET effort = excluded.effort
        """,
        (effort,),
    )
    conn.commit()


def clear_projects(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM projects")
    conn.commit()


def list_projects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM projects ORDER BY last_opened IS NULL, last_opened DESC, name ASC"
    ).fetchall()


def get_project(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()


def upsert_project(conn: sqlite3.Connection, path: str, name: str, branch: str) -> None:
    conn.execute(
        """
        INSERT INTO projects (path, name, branch) VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET name = excluded.name, branch = excluded.branch
        """,
        (path, name, branch),
    )
    conn.commit()


def mark_opened(conn: sqlite3.Connection, project_id: int) -> None:
    conn.execute(
        "UPDATE projects SET last_opened = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), project_id),
    )
    conn.commit()


def save_session_state(conn: sqlite3.Connection, project_id: int, state: dict) -> None:
    conn.execute(
        "UPDATE projects SET session_state = ? WHERE id = ?",
        (json.dumps(state), project_id),
    )
    conn.commit()


def load_session_state(row: sqlite3.Row) -> dict:
    raw = row["session_state"]
    return json.loads(raw) if raw else {}


# --- Concurrent session orchestration (PRD 2) -----------------------------

PHASES = ("grilling", "creating_prd", "creating_issues", "details", "implementing", "implemented")


def create_session(
    conn: sqlite3.Connection,
    project_id: int,
    claude_session_id: str | None = None,
    *,
    session_type: str = "do",
    phase: str = "grilling",
    details: dict | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    details_json = json.dumps(details) if details is not None else None
    cur = conn.execute(
        """
        INSERT INTO sessions (project_id, claude_session_id, session_type, phase, details_json, model, effort, last_activity, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            claude_session_id,
            session_type,
            phase,
            details_json,
            model or DEFAULT_MODEL,
            effort or DEFAULT_EFFORT,
            now,
            now,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_session(conn: sqlite3.Connection, session_row_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_row_id,)).fetchone()


def list_sessions_for_project(conn: sqlite3.Connection, project_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sessions WHERE project_id = ? ORDER BY created_at ASC",
        (project_id,),
    ).fetchall()


def update_session(conn: sqlite3.Connection, session_row_id: int, **fields) -> None:
    if not fields:
        return
    fields = {**fields, "last_activity": datetime.now(timezone.utc).isoformat()}
    columns = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(
        f"UPDATE sessions SET {columns} WHERE id = ?",
        (*fields.values(), session_row_id),
    )
    conn.commit()


def mark_session_available(conn: sqlite3.Connection, session_row_id: int, claude_session_id: str) -> None:
    update_session(
        conn,
        session_row_id,
        claude_session_id=claude_session_id,
        available_for_reuse=1,
    )


def claim_available_session(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT * FROM sessions
        WHERE project_id = ? AND available_for_reuse = 1
        ORDER BY last_activity DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    if row is not None:
        conn.execute("UPDATE sessions SET available_for_reuse = 0 WHERE id = ?", (row["id"],))
        conn.commit()
        row = get_session(conn, row["id"])
    return row


def has_active_implement_session(conn: sqlite3.Connection, project_id: int, prd_number: int) -> bool:
    """True if this project already has a live (non-terminal) implement
    session for `prd_number`. There's no dedicated PRD-number column on
    `sessions`, so this filters the (short) candidate list in Python by
    peeking into each row's `details_json` (seeded at creation with
    `{"prd": {"number": ...}}`)."""
    rows = conn.execute(
        """
        SELECT details_json FROM sessions
        WHERE project_id = ? AND session_type = 'implement' AND phase = 'implementing' AND error_text IS NULL
        """,
        (project_id,),
    ).fetchall()
    for row in rows:
        if not row["details_json"]:
            continue
        details = json.loads(row["details_json"])
        prd = details.get("prd") if isinstance(details, dict) else None
        if prd and prd.get("number") == prd_number:
            return True
    return False


def has_any_active_implement_session(conn: sqlite3.Connection, project_id: int) -> bool:
    """True if this project has any live (non-terminal) implement session at
    all, regardless of which PRD number it's for -- used to gate serial-mode
    queueing (`parallel_implementation = False`)."""
    row = conn.execute(
        """
        SELECT 1 FROM sessions
        WHERE project_id = ? AND session_type = 'implement' AND phase = 'implementing' AND error_text IS NULL
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    return row is not None


def cleanup_sessions_on_shutdown(conn: sqlite3.Connection) -> None:
    """Keep only the single most-recently-used session across all projects."""
    rows = conn.execute(
        "SELECT id FROM sessions WHERE claude_session_id IS NOT NULL ORDER BY last_activity DESC"
    ).fetchall()
    if not rows:
        return
    keep_id = rows[0]["id"]
    conn.execute("DELETE FROM sessions WHERE id != ?", (keep_id,))
    conn.commit()


INTERRUPTED_ERROR_TEXT = (
    "Interrupted: Baton was restarted while this session was implementing."
)


def recover_interrupted_implement_sessions(conn: sqlite3.Connection) -> None:
    """Sweep every project's `sessions` table at process startup for
    implement sessions stuck mid-`implementing` from a prior process that
    died without recording an error. Marks each with a clear "interrupted"
    error message so `has_active_implement_session` /
    `has_any_active_implement_session` (and the frontend's
    `computeImplementingPrdNumbers`) stop treating it as still running --
    mirrors `cleanup_sessions_on_shutdown`'s "sweep once at a process
    boundary, across every project" shape, but marks rows instead of
    deleting them."""
    conn.execute(
        """
        UPDATE sessions
        SET error_text = ?
        WHERE session_type = 'implement' AND phase = 'implementing' AND error_text IS NULL
        """,
        (INTERRUPTED_ERROR_TEXT,),
    )
    conn.commit()
