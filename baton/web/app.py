import asyncio
import contextlib
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from baton import afk_loop, db, live_stream, session_runner
from baton.cli_client import ClaudeCLIError, get_auth_status
from baton.folder_picker import pick_folder
from baton.prd_list import compute_prd_list
from baton.projects import scan_projects
from baton.terminal import open_terminal_running

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def _lifespan(app: FastAPI):
    live_stream.set_loop(asyncio.get_running_loop())
    db.recover_interrupted_implement_sessions(db.get_connection())
    afk_task = asyncio.create_task(
        afk_loop.run_forever(
            get_active_project_id=lambda: _active_project_id,
            get_active_project_cwd=_active_project_cwd,
            fetch_prd_list=lambda cwd: compute_prd_list(_fetch_ready_prds(cwd), _fetch_all_open_issues(cwd)),
        )
    )
    yield
    afk_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await afk_task
    db.cleanup_sessions_on_shutdown(db.get_connection())


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Active project is process-global: Baton is a local, single-user desktop-
# oriented app (one browser tab talking to one backend process), not a
# multi-tenant server, so there's exactly one "current project" at a time.
_active_project_id: int | None = None


def _project_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "path": row["path"],
        "name": row["name"],
        "branch": row["branch"],
        "last_opened": row["last_opened"],
    }


def _rescan_and_cache(conn, root_dir: str) -> None:
    for found in scan_projects(root_dir):
        db.upsert_project(conn, found["path"], found["name"], found["branch"])


@app.get("/")
def index(request: Request):
    try:
        status = get_auth_status()
    except ClaudeCLIError:
        status = {"loggedIn": False}

    if status.get("loggedIn"):
        return RedirectResponse(url="/prompt")

    return templates.TemplateResponse(request, "login.html")


@app.post("/login")
def login():
    open_terminal_running("claude auth login")
    return {"opened": True}


@app.get("/api/auth-status")
def auth_status():
    try:
        return get_auth_status()
    except ClaudeCLIError:
        return {"loggedIn": False}


@app.get("/prompt")
def prompt_page(request: Request):
    status = get_auth_status()
    if not status.get("loggedIn"):
        return RedirectResponse(url="/")

    return templates.TemplateResponse(request, "prompt.html", {"email": status.get("email", "")})


@app.get("/api/app-state")
def app_state():
    conn = db.get_connection()
    root_dir = db.get_root_dir(conn)
    afk_hours = db.get_afk_hours(conn)
    parallel_implementation = db.get_parallel_implementation(conn)
    terminal_view_hidden = db.get_terminal_view_hidden(conn)
    model = db.get_model(conn)
    effort = db.get_effort(conn)

    active_project = None
    if _active_project_id is not None:
        row = db.get_project(conn, _active_project_id)
        if row is not None:
            active_project = _project_to_dict(row)

    projects = []
    if root_dir and active_project is None:
        _rescan_and_cache(conn, root_dir)
        projects = [_project_to_dict(r) for r in db.list_projects(conn)]

    return {
        "root_dir": root_dir,
        "afk_hours": afk_hours,
        "parallel_implementation": parallel_implementation,
        "terminal_view_hidden": terminal_view_hidden,
        "model": model,
        "effort": effort,
        "active_project": active_project,
        "projects": projects,
    }


@app.post("/api/settings/pick-folder")
def pick_folder_endpoint():
    return {"path": pick_folder()}


@app.post("/api/settings/root-dir")
def set_root_dir(body: dict):
    conn = db.get_connection()
    current = db.get_root_dir(conn)
    new_root = body["root_dir"]
    confirm = bool(body.get("confirm"))

    if current and current != new_root and not confirm:
        return {"needs_confirmation": True}

    if current and current != new_root:
        db.clear_projects(conn)
        global _active_project_id
        _active_project_id = None

    db.set_root_dir(conn, new_root)
    _rescan_and_cache(conn, new_root)
    projects = [_project_to_dict(r) for r in db.list_projects(conn)]

    return {"root_dir": new_root, "projects": projects}


@app.post("/api/settings/afk-hours")
def set_afk_hours(body: dict):
    conn = db.get_connection()
    afk_hours = body["afk_hours"]
    db.set_afk_hours(conn, afk_hours)
    return {"afk_hours": afk_hours}


@app.post("/api/settings/parallel-implementation")
def set_parallel_implementation(body: dict):
    conn = db.get_connection()
    parallel_implementation = bool(body["parallel_implementation"])
    db.set_parallel_implementation(conn, parallel_implementation)
    return {"parallel_implementation": parallel_implementation}


@app.post("/api/settings/terminal-view-hidden")
def set_terminal_view_hidden(body: dict):
    conn = db.get_connection()
    terminal_view_hidden = bool(body["terminal_view_hidden"])
    db.set_terminal_view_hidden(conn, terminal_view_hidden)
    return {"terminal_view_hidden": terminal_view_hidden}


@app.post("/api/settings/model")
def set_model(body: dict):
    conn = db.get_connection()
    model = body["model"]
    db.set_model(conn, model)
    return {"model": model}


@app.post("/api/settings/effort")
def set_effort(body: dict):
    conn = db.get_connection()
    effort = body["effort"]
    db.set_effort(conn, effort)
    return {"effort": effort}


@app.post("/api/projects/{project_id}/open")
def open_project(project_id: int):
    global _active_project_id
    conn = db.get_connection()
    row = db.get_project(conn, project_id)
    if row is None:
        return {"error": "Project not found"}

    db.mark_opened(conn, project_id)
    _active_project_id = project_id
    afk_loop.record_activity(project_id)
    row = db.get_project(conn, project_id)

    return {
        "project": _project_to_dict(row),
        "session_state": db.load_session_state(row),
    }


@app.post("/api/projects/{project_id}/close")
def close_project(project_id: int, body: dict):
    global _active_project_id
    conn = db.get_connection()
    db.save_session_state(conn, project_id, body.get("session_state", {}))
    if _active_project_id == project_id:
        _active_project_id = None
    return {"closed": True}


def _active_project_cwd() -> str | None:
    if _active_project_id is None:
        return None
    conn = db.get_connection()
    row = db.get_project(conn, _active_project_id)
    return row["path"] if row is not None else None


def _fetch_ready_prds(cwd: str) -> list[dict]:
    # Filters by the `prd` label (not `ready-for-agent`) so child issues from
    # /to-issues -- which carry `ready-for-agent` only -- never show up here.
    result = subprocess.run(
        ["gh", "issue", "list", "--state", "open", "--label", "prd", "--json", "number,title,body,labels"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def _fetch_all_open_issues(cwd: str) -> list[dict]:
    result = subprocess.run(
        ["gh", "issue", "list", "--state", "open", "--json", "number,title,body"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


@app.get("/api/projects/{project_id}/prds")
def list_prds(project_id: int):
    if project_id != _active_project_id:
        return {"prds": []}

    cwd = _active_project_cwd()
    if cwd is None:
        return {"prds": []}

    prds = _fetch_ready_prds(cwd)
    all_open_issues = _fetch_all_open_issues(cwd)
    return {"prds": compute_prd_list(prds, all_open_issues)}


def _session_to_dict(row) -> dict:
    return {
        "card_id": row["id"],
        "project_id": row["project_id"],
        "session_type": row["session_type"],
        "phase": row["phase"],
        "console_text": row["console_text"],
        "interview": json.loads(row["interview_json"]) if row["interview_json"] else None,
        "details": json.loads(row["details_json"]) if row["details_json"] else None,
        "error": row["error_text"],
        "needs_github_login": bool(row["needs_github_login"]),
        "blocked": json.loads(row["blocked_json"]) if row["blocked_json"] else None,
    }


@app.get("/api/projects/{project_id}/sessions")
def list_sessions(project_id: int):
    conn = db.get_connection()
    return {"sessions": [_session_to_dict(r) for r in db.list_sessions_for_project(conn, project_id)]}


@app.get("/api/projects/{project_id}/afk-notifications")
def get_afk_notifications(project_id: int):
    return {"notifications": afk_loop.get_notifications(project_id)}


@app.post("/api/projects/{project_id}/afk-notifications/dismiss")
def dismiss_afk_notifications(project_id: int):
    afk_loop.dismiss_notifications(project_id)
    return {"dismissed": True}


@app.get("/api/projects/{project_id}/session-error-notifications")
def get_session_error_notifications(project_id: int):
    return {"notifications": session_runner.get_error_notifications(project_id)}


@app.post("/api/projects/{project_id}/session-error-notifications/dismiss")
def dismiss_session_error_notifications(project_id: int):
    session_runner.dismiss_error_notifications(project_id)
    return {"dismissed": True}


@app.get("/api/usage")
def get_usage():
    usage = live_stream.last_usage()
    if usage is None:
        return {"five_hour_pct": None, "seven_day_pct": None}
    return {"five_hour_pct": usage["five_hour_pct"], "seven_day_pct": usage["seven_day_pct"]}


@app.get("/api/pty-tabs/count")
def get_pty_tab_count():
    """How many `PtyEngine` tabs are currently resident across every active
    session (issue #88) -- backs the tab-count indicator next to the
    "Sessions" label in the web UI. Polled rather than pushed over any one
    card's SSE stream since the count is global, not scoped to a card."""
    return {"count": session_runner.open_pty_tab_count()}


@app.get("/api/sessions/{card_id}/stream")
async def stream_session(card_id: int):
    async def event_source():
        # Replay the full history unconditionally -- a session can be retried
        # after reaching `done` once already, appending a fresh run past it.
        history, queue = live_stream.subscribe(card_id)
        try:
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"
            if history and history[-1].get("type") == "done":
                return
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    return
        finally:
            live_stream.unsubscribe(card_id, queue)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/session/start")
async def start_session(body: dict):
    project_id = _active_project_id
    cwd = _active_project_cwd()
    if project_id is None:
        return {"error": "No active project"}

    conn = db.get_connection()
    reused = db.claim_available_session(conn, project_id)
    resume_id = reused["claude_session_id"] if reused is not None else None

    # `effort` in the body is the left-card dropdown's value at the moment
    # "Start" was clicked -- the common case for a per-session override,
    # covered without any race. Falls back to the global default (matching
    # `create_session`'s own fallback) when omitted.
    effort = body.get("effort")
    row_id = db.create_session(conn, project_id, claude_session_id=resume_id, effort=effort)

    asyncio.create_task(session_runner.start_session_job(row_id, body["prompt"], cwd=cwd))

    return {"card_id": row_id}


@app.post("/api/sessions/{card_id}/effort")
def set_session_effort(card_id: int, body: dict):
    """Patch a single session's effort immediately -- used when the left-card
    dropdown changes while that session has been started but hasn't sent its
    first turn yet (see `session_runner.start_session_job`'s docstring for
    why re-reading the row picks this up even given the async-task race)."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return {"error": "Session not found"}

    effort = body["effort"]
    db.update_session(conn, card_id, effort=effort)
    return {"effort": effort}


@app.post("/api/session/continue")
async def continue_session(body: dict):
    conn = db.get_connection()
    row = db.get_session(conn, body["card_id"])
    if row is None:
        return {"error": "Session not found"}

    cwd = _active_project_cwd()
    confirm_advance = body.get("confirm_advance", False)
    asyncio.create_task(
        session_runner.continue_session_job(
            row["id"], body.get("reply", ""), cwd=cwd, confirm_advance=confirm_advance
        )
    )

    return {"card_id": row["id"]}


@app.post("/api/session/start-implement")
async def start_implement(body: dict):
    project_id = _active_project_id
    cwd = _active_project_cwd()
    if project_id is None:
        return {"error": "No active project"}

    number = body["number"]
    title = body.get("title", "")

    afk_loop.record_activity(project_id)
    return await session_runner.start_or_queue_implement(project_id, number, title, cwd)


@app.post("/api/sessions/{card_id}/retry")
async def retry_session(card_id: int):
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return {"error": "Session not found"}

    cwd = _active_project_cwd()
    asyncio.create_task(session_runner.retry_session_job(card_id, cwd))

    return {"card_id": card_id}


@app.post("/api/session/qa-complete")
async def qa_complete(body: dict):
    conn = db.get_connection()
    row = db.get_session(conn, body["card_id"])
    if row is None:
        return {"error": "Session not found"}
    cwd = _active_project_cwd()
    notes = body.get("notes", "")
    asyncio.create_task(session_runner.continue_qa_job(row["id"], notes, cwd=cwd))
    return {"ok": True}


@app.post("/api/sessions/{card_id}/implement-reply")
async def implement_reply(card_id: int, body: dict):
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return {"error": "Session not found"}
    cwd = _active_project_cwd()
    reply = body["reply"]
    asyncio.create_task(session_runner.continue_implement_job(card_id, reply, cwd=cwd))
    return {"ok": True}


@app.post("/api/github-login")
def github_login():
    open_terminal_running("gh auth login")
    return {"opened": True}


@app.get("/api/github-auth-status")
def github_auth_status():
    result = subprocess.run(
        ["gh", "auth", "status"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {"logged_in": result.returncode == 0}
