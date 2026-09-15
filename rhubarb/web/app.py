import asyncio
import contextlib
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rhubarb import afk_loop, caveman_installer, db, error_log, headroom_installer, live_stream, ollama_installer, parser_session, session_runner
from rhubarb.cli_client import ClaudeCLIError, get_auth_status, set_headroom_proxy_active
from rhubarb.folder_picker import pick_folder
from rhubarb.prd_list import compute_prd_list
from rhubarb.projects import scan_projects
from rhubarb.pty_engine import PtyEngineError
from rhubarb.qa_parser import parse_grilling_response
from rhubarb.question_files import read_question_file
from rhubarb.terminal import open_terminal_running

BASE_DIR = Path(__file__).parent

# --- Headroom proxy management (issue #200) ----------------------------------
# One long-lived `headroom proxy` subprocess kept alive for the duration of
# the Rhubarb process when Headroom is enabled and present. Started on app
# startup (if eligible), after a successful install, or when the Settings
# toggle is turned on; stopped when the toggle is turned off or on shutdown.

_headroom_proxy: subprocess.Popen | None = None


def _start_headroom_proxy() -> None:
    """Start a background `headroom proxy` process if one isn't already
    running. Sets the `_headroom_proxy_active` flag in `cli_client` so
    that new `claude` subprocess spawns pick up `ANTHROPIC_BASE_URL`."""
    global _headroom_proxy
    if _headroom_proxy is not None and _headroom_proxy.poll() is None:
        return  # already running
    _headroom_proxy = subprocess.Popen(
        ["headroom", "proxy"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    set_headroom_proxy_active(True)


def _stop_headroom_proxy() -> None:
    """Terminate the background `headroom proxy` process and clear the
    `_headroom_proxy_active` flag so new spawns stop getting the env var."""
    global _headroom_proxy
    set_headroom_proxy_active(False)
    if _headroom_proxy is not None:
        _headroom_proxy.terminate()
        _headroom_proxy = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    live_stream.set_loop(asyncio.get_running_loop())
    db.recover_interrupted_implement_sessions(db.get_connection())
    # Issue #200: start the Headroom proxy on startup if the user hasn't
    # declined it and it's already installed -- so new sessions immediately
    # get `ANTHROPIC_BASE_URL` without needing a toggle round-trip.
    _conn = db.get_connection()
    if not db.get_headroom_declined(_conn) and headroom_installer.check_headroom_presence() == headroom_installer.PRESENCE_PRESENT:
        _start_headroom_proxy()
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
    _stop_headroom_proxy()
    db.cleanup_sessions_on_shutdown(db.get_connection())
    # Issue #189: every live per-project parser session is torn down only
    # here, on Rhubarb's own process exit -- never on a project close/switch
    # (see `open_project`'s wiring above) -- so this shutdown hook is the
    # one and only place that ever happens.
    parser_session.close_all_parser_sessions()


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Active project is process-global: Rhubarb is a local, single-user desktop-
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
def app_state(card_id: int | None = None):
    """`card_id` is an optional hint (issue #139) naming the currently-open
    session card, if any -- the frontend's Model/Effort card passes its
    `leftCardId` here so it can show that card's live `PtyEngine`'s actual
    `(model, effort)` (ground truth for the process actually running)
    instead of always showing the global "next new session" setting. Omitted
    (or naming a card with no live resident engine -- finished/pooled/never
    started) falls back to `db.get_model`/`db.get_effort` exactly as before
    this param existed; `settings.model`/`settings.effort` themselves are
    never read from or written to differently based on this param."""
    conn = db.get_connection()
    root_dir = db.get_root_dir(conn)
    afk_hours = db.get_afk_hours(conn)
    parallel_implementation = db.get_parallel_implementation(conn)
    terminal_view_hidden = db.get_terminal_view_hidden(conn)
    model = db.get_model(conn)
    effort = db.get_effort(conn)

    session_model_effort_live = False
    if card_id is not None:
        engine_model_effort = session_runner.get_engine_model_effort(card_id)
        if engine_model_effort is not None:
            model, effort = engine_model_effort
            session_model_effort_live = True

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
        "session_model_effort_live": session_model_effort_live,
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
    """`card_id` is an optional hint (issue #141), analogous to `GET
    /api/app-state`'s own `card_id` param (#139) -- the frontend's Model/
    Effort card passes its `leftCardId` here so that, if that card has a
    live resident engine, this same request also tears it down and spawns a
    fresh, unresumed one under the new model (see
    `session_runner.respawn_engine_for_model_change`) instead of leaving the
    open Live Terminal running under its old model until some future
    session. Omitted (or naming a card with no live engine) leaves behavior
    exactly as before this param existed -- global setting only, no engine
    touched. `respawned` is only present in the response when `card_id` was
    given, so a caller that never passes it (every existing caller) sees the
    exact same response shape as before."""
    conn = db.get_connection()
    model = body["model"]
    db.set_model(conn, model)

    card_id = body.get("card_id")
    respawned = session_runner.respawn_engine_for_model_change(
        conn, card_id, cwd=_active_project_cwd(), model=model
    )
    if card_id is not None:
        return {"model": model, "respawned": respawned}
    return {"model": model}


@app.post("/api/settings/effort")
def set_effort(body: dict):
    """Symmetric to `set_model` above, for effort -- see its docstring for
    the `card_id`/`respawned` contract, identical here."""
    conn = db.get_connection()
    effort = body["effort"]
    db.set_effort(conn, effort)

    card_id = body.get("card_id")
    respawned = session_runner.respawn_engine_for_effort_change(
        conn, card_id, cwd=_active_project_cwd(), effort=effort
    )
    if card_id is not None:
        return {"effort": effort, "respawned": respawned}
    return {"effort": effort}


@app.get("/api/ollama-status")
def ollama_status():
    """Polled by the first-run gate (and the Settings toggle) to decide
    whether to show the install prompt. Not `async def`: FastAPI/Starlette
    runs a plain `def` route in a worker thread automatically, so
    `check_ollama_presence`'s blocking local HTTP call never blocks the
    event loop -- same reasoning as `pick_folder_endpoint` above."""
    conn = db.get_connection()
    return {
        "presence": ollama_installer.check_ollama_presence(),
        "declined": db.get_ollama_declined(conn),
    }


@app.post("/api/ollama-install")
def start_ollama_install():
    """Runs the install/pull to completion and returns the result. A plain
    `def` route, so FastAPI runs it in a worker thread automatically (same
    as `pick_folder_endpoint`/`ollama_status` above) -- it never blocks the
    event loop or other concurrent sessions, even though a real install can
    take minutes. The gate UI shows a spinner for the duration of this one
    request; no separate progress-polling endpoint is needed since the
    result comes back directly in this response."""
    try:
        presence = ollama_installer.check_ollama_presence()
        if presence == ollama_installer.PRESENCE_NOT_PRESENT:
            ollama_installer.install_and_pull_model()
        elif presence == ollama_installer.PRESENCE_WITHOUT_MODEL:
            ollama_installer.pull_model()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/settings/ollama-declined")
def set_ollama_declined(body: dict):
    conn = db.get_connection()
    declined = bool(body["ollama_declined"])
    db.set_ollama_declined(conn, declined)
    return {"ollama_declined": declined}


@app.get("/api/headroom-status")
def headroom_status():
    """Polled by the first-run gate (and the Settings toggle) to decide
    whether to show the Headroom install prompt. Plain `def` route so
    FastAPI runs it in a worker thread -- `check_headroom_presence`'s
    blocking subprocess call never blocks the event loop."""
    conn = db.get_connection()
    return {
        "presence": headroom_installer.check_headroom_presence(),
        "declined": db.get_headroom_declined(conn),
    }


@app.post("/api/headroom-install")
def start_headroom_install():
    """Runs the install to completion and returns the result. Plain `def`
    route (worker thread) -- a real install can take minutes but never
    blocks the event loop or concurrent sessions. On success, starts the
    Headroom proxy immediately so new sessions pick up ANTHROPIC_BASE_URL
    without a restart."""
    try:
        presence = headroom_installer.check_headroom_presence()
        if presence == headroom_installer.PRESENCE_NOT_PRESENT:
            headroom_installer.install_for_platform()
        _start_headroom_proxy()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/settings/headroom-declined")
def set_headroom_declined_endpoint(body: dict):
    conn = db.get_connection()
    declined = bool(body["headroom_declined"])
    db.set_headroom_declined(conn, declined)
    if declined:
        _stop_headroom_proxy()
    elif headroom_installer.check_headroom_presence() == headroom_installer.PRESENCE_PRESENT:
        _start_headroom_proxy()
    return {"headroom_declined": declined}


@app.get("/api/caveman-status")
def caveman_status():
    """Polled by the first-run gate (and the Settings toggle) to decide
    whether to show the Caveman install prompt. Plain `def` route so
    FastAPI runs it in a worker thread -- `check_caveman_presence`'s
    blocking subprocess call never blocks the event loop."""
    conn = db.get_connection()
    return {
        "presence": caveman_installer.check_caveman_presence(),
        "declined": db.get_caveman_declined(conn),
    }


@app.post("/api/caveman-install")
def start_caveman_install():
    """Runs the skill install to completion and returns the result. Plain
    `def` route (worker thread) -- never blocks the event loop or concurrent
    sessions. No proxy to manage: once installed globally, the skill is
    available to any `claude` invocation Rhubarb spawns automatically."""
    try:
        presence = caveman_installer.check_caveman_presence()
        if presence == caveman_installer.PRESENCE_NOT_PRESENT:
            caveman_installer.install_caveman()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/settings/caveman-declined")
def set_caveman_declined_endpoint(body: dict):
    conn = db.get_connection()
    declined = bool(body["caveman_declined"])
    db.set_caveman_declined(conn, declined)
    if declined:
        # Best-effort: silently ignore if caveman isn't installed or the
        # remove command fails (e.g. the skill was never installed in the
        # first place).
        try:
            caveman_installer.disable_caveman()
        except Exception:
            pass
    return {"caveman_declined": declined}


async def _warm_project_engines(project_id: int, *, cwd: str | None, model: str | None, effort: str | None) -> None:
    """Fire-and-forget engine warm-up for a just-opened project, scheduled
    as ONE background task from `open_project` (rather than two separate
    `asyncio.create_task` calls) so this stays exactly as light a scheduling
    footprint on the event loop as the original single-warm version was.

    - Standby `StreamJsonEngine` (issue #136, adapted to stream-json by
      issue #184): so the first `/do` a user starts doesn't pay the "wait
      for claude to open" spawn cost inline. A brand-new session always
      begins in the grilling phase (`db.create_session`'s own
      `phase="grilling"` default), and grilling now runs on
      `StreamJsonEngine`, matching what `start_session` actually claims for
      a fresh session's first turn. The old `ensure_standby_engine`
      (PtyEngine) standby machinery is left completely intact for any other
      caller -- it's simply not used for this purpose any more, since
      nothing else ever starts a brand-new session.
    - Parser session (issue #189, lifecycle slice of PRD #187 -- `gh issue
      view 187`): lazily creates this project's persistent parser session
      on first open, reused (same subprocess, same session_id) on every
      later open. Unlike the standby above, this session is NOT closed by
      `close_project` below -- its lifetime is independent of project focus
      and it lives until Rhubarb itself exits (see `parser_session.
      close_all_parser_sessions`, wired into `_lifespan`'s shutdown)."""
    await session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model=model, effort=effort)
    await parser_session.ensure_parser_session(project_id, cwd=cwd, model=model, effort=effort)


@app.post("/api/projects/{project_id}/open")
async def open_project(project_id: int):
    global _active_project_id
    conn = db.get_connection()
    row = db.get_project(conn, project_id)
    if row is None:
        return {"error": "Project not found"}

    db.mark_opened(conn, project_id)
    _active_project_id = project_id
    afk_loop.record_activity(project_id)
    row = db.get_project(conn, project_id)

    # Pre-warm this project's engines -- fire-and-forget, never blocks this
    # response (see `_warm_project_engines`'s own docstring for what each
    # warm does and why they're one task, not two).
    asyncio.create_task(
        _warm_project_engines(project_id, cwd=row["path"], model=db.get_model(conn), effort=db.DEFAULT_EFFORT)
    )

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
    # Issue #184: close whichever standby this project actually has warm --
    # in practice this is always the `StreamJsonEngine` one now (see
    # `open_project`), but `close_standby_engine` (PtyEngine) is also called
    # unconditionally, harmlessly, in case anything else ever warms one.
    session_runner.close_standby_engine(project_id)
    session_runner.close_standby_stream_json_engine(project_id)
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
        "stalled": json.loads(row["stalled_json"]) if row["stalled_json"] else None,
    }


@app.get("/api/projects/{project_id}/sessions")
def list_sessions(project_id: int):
    conn = db.get_connection()
    return {"sessions": [_session_to_dict(r) for r in db.list_sessions_for_project(conn, project_id)]}


@app.get("/api/projects/{project_id}/rhubarb-question-file-preview")
def preview_rhubarb_question_file(project_id: int):
    """Debug tool (PRD #123 follow-up): read this project's pending
    `.claude/rhubarb_question.md` right now, parse it exactly the way a real
    grilling turn would, and hand back the resulting interview -- so the
    parse/render path can be checked directly against the file, independent
    of whether a live turn's own file-priority check is reaching it."""
    conn = db.get_connection()
    project = db.get_project(conn, project_id)
    if project is None:
        return {"found": False}

    file_text = read_question_file(project["path"], "rhubarb_question.md")
    if file_text is None:
        return {"found": False}

    return {"found": True, "interview": parse_grilling_response(file_text)}


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


@app.get("/api/projects/{project_id}/parsed-results")
async def get_parsed_results_endpoint(project_id: int, since: int = 0):
    """Issue #193 ("Route parsed results to the focused card or a toast",
    slice of PRD #187 -- `gh issue view 193`/`gh issue view 187`): the poll
    endpoint `prompt.html`'s `pollParsedResults` uses to learn about new
    parser-session results (issue #192's `drain_needs_input_queue`/
    `get_parsed_results`) so it can route each one to the focused left card
    or a toast.

    Nothing else in the app drives issue #192's queue-drain loop end-to-end
    yet, so this request does it inline: fully drains whatever is currently
    queued for `project_id` (a no-op, returning immediately, if the queue is
    empty -- see `drain_needs_input_queue`'s own docstring) before reading
    back the accumulated results list. `get_parsed_results` is purely
    additive/append-only (see its docstring), so a plain integer `since`
    offset is a safe "what's new since my last poll" cursor: nothing already
    returned ever moves, changes, or disappears out from under it. The
    response's `total` is the offset to pass as `since` on the caller's next
    poll.

    Deliberately NOT a persistent background `asyncio.create_task` kicked
    off from `open_project`/`_warm_project_engines` instead: this project's
    own test convention (`tests/test_parser_session.py`'s `_run_and_drain`
    helper, and every test built on it) awaits *every* asyncio task still
    pending after `open_project` returns -- a genuinely never-ending loop
    task scheduled from there would hang that helper, and every test using
    it, forever. Draining once per poll, synchronously within this request,
    gets the same "keeps up with the queue" behavior (the frontend polls
    every few seconds) without needing any background-task lifecycle
    (start-once-per-project bookkeeping, cancellation on shutdown, ...) at
    all.
    """
    async for _ in parser_session.drain_needs_input_queue(project_id):
        pass
    results = parser_session.get_parsed_results(project_id)
    return {"results": results[since:], "total": len(results)}


@app.get("/api/projects/{project_id}/errors")
def get_project_errors(
    project_id: int,
    phase: str | None = None,
    since: str | None = None,
    until: str | None = None,
    card_id: int | None = None,
    q: str | None = None,
):
    return {
        "errors": error_log.query_errors(
            project_id, phase=phase, since=since, until=until, card_id=card_id, q=q
        )
    }


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
    card's SSE stream since the count is global, not scoped to a card.

    `"engines"` (issue #140) is an additive per-engine listing alongside the
    plain `"count"` -- one record per live entry across both resident
    (`_pty_engines`) and pre-warmed standby (`_standby_engines`) engines,
    each carrying its actual `(model, effort)` and a distinguishing
    `card_id` (an int, or the literal string `"standby"`) -- see
    `session_runner.list_live_engines`. Existing consumers that only read
    `"count"` are unaffected."""
    return {"count": session_runner.open_pty_tab_count(), "engines": session_runner.list_live_engines()}


@app.get("/api/sessions/{card_id}/stream")
async def stream_session(card_id: int):
    async def event_source():
        # Replay the full history unconditionally -- a session can be retried
        # after reaching `done` once already, appending a fresh run past it.
        history, queue = live_stream.subscribe(card_id)
        try:
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"
            if history and history[-1].get("type") in ("done", "closed"):
                return
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "closed"):
                    return
        finally:
            live_stream.unsubscribe(card_id, queue)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.websocket("/ws/sessions/{card_id}/pty")
async def pty_passthrough(websocket: WebSocket, card_id: int):
    """Raw interactive passthrough channel (issue #165, child of PRD #162
    "Add raw interactive passthrough mode to the Live Terminal"): accepts a
    WebSocket connection scoped to a single card's resident `PtyEngine` and
    forwards every text frame received over the socket straight into that
    engine's write path (`PtyEngine.write`), byte for byte, exactly as a
    person typing directly into the terminal would.

    This is purely an input channel, additive alongside the existing SSE
    stream (`GET /api/sessions/{card_id}/stream` above) -- it carries no
    output of its own and does not touch, replace, or change that stream's
    behavior in any way; a frontend still reads `terminal_output`/`result`
    events from the SSE stream exactly as before. Frontend wiring that
    actually opens this socket from `prompt.html` is issue #167, out of
    scope here.

    `PtyEngine.write` is guarded by the same `asyncio.Lock` the automated-
    turn write loop (`_stream_chunks_until_marker`) holds for its entire
    paced prompt write, so a passthrough write forwarded here can never
    physically interleave its bytes with an in-flight automated turn's
    writes into the same PTY, in either direction.

    Closes immediately with code 1008 (policy violation) if `card_id` names
    no live resident engine -- there's nothing to forward to. Ends quietly
    (no error) on a normal client disconnect, or if the engine dies/closes
    out from under an open connection (`PtyEngineError` from `write`)."""
    engine = session_runner.get_engine(card_id)
    if engine is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                await engine.write(data)
            except PtyEngineError:
                break
    except WebSocketDisconnect:
        pass


@app.post("/api/sessions/{card_id}/resize")
def resize_session_pty(card_id: int, body: dict):
    """Dynamic PTY resize (issue #166): the frontend's xterm.js fit-addon
    calls this whenever it recomputes the live-terminal panel's actual
    cols/rows (on load, and on every panel resize -- see `resizeTerminalToFit`
    in `prompt.html`) so the REAL pseudoterminal backing this card's
    resident `PtyEngine` is resized to match, not just the on-screen
    xterm.js buffer.

    A plain HTTP endpoint rather than piggybacking on the raw passthrough
    WebSocket (`/ws/sessions/{card_id}/pty`, issue #165) on purpose: that
    channel is documented and built as a byte-for-byte passthrough straight
    into the PTY's stdin, with no control-plane framing of any kind --
    every frame it receives is forwarded to `PtyEngine.write()` verbatim.
    Overloading it with a second, structured message shape would mean
    inventing an escaping/framing scheme to tell a resize control message
    apart from literal keystroke bytes (which can be arbitrary), and would
    contradict that endpoint's own docstring ("carries no output of its
    own" / "byte for byte"). A separate small endpoint keeps that channel's
    contract exactly as simple as it already is.

    A no-op (not an error) if `card_id` names no live resident engine --
    a session can be resized before its first turn ever creates one (or
    after it's already closed); there's simply nothing to resize yet, and
    the size a session eventually spawns at is seeded from `PtyEngine`'s
    own default (`_PTY_ROWS`/`_PTY_COLUMNS`) until a later resize call
    lands against a live engine."""
    rows = body["rows"]
    cols = body["cols"]

    engine = session_runner.get_engine(card_id)
    if engine is None:
        return {"resized": False}

    engine.resize(rows, cols)
    return {"resized": True, "rows": rows, "cols": cols}


@app.post("/api/sessions/{card_id}/stall-reply")
async def reply_to_stalled_session(card_id: int, body: dict):
    """Forward a reply straight into a card's live PTY process (issue #169,
    child of PRD #168 "Recover from a stalled turn instead of hanging the
    turn lock forever"): while `_stream_chunks_until_marker`'s read loop is
    still waiting on the pending read for the completion marker -- surfaced
    to this card's SSE stream as `turn` events carrying `stalled: true` (see
    `session_runner._run_turn`) -- a human may want to nudge the live
    process along (e.g. resend whatever input it seems to have missed)
    without waiting indefinitely for that original turn to either resolve
    on its own or the engine to be torn down.

    Mirrors `resize_session_pty` above in shape: a small, separate HTTP
    endpoint (not piggybacked on the raw passthrough WebSocket) that looks
    up this card's resident engine via `get_engine` and forwards straight
    into its lock-protected `PtyEngine.write()` -- the exact same passthrough
    path raw interactive typing already uses (issue #165). This does NOT go
    through `_run_turn`'s own prompt-write machinery, does NOT start a new
    turn, and does NOT touch the turn lock (`session_runner._get_turn_lock`)
    a second time -- the original turn's marker-wait loop (still running
    underneath, holding that lock) remains the single source of truth for
    when the turn is actually done. This endpoint only ever writes bytes
    into that same live process's stdin, exactly as a person typing directly
    into the terminal would.

    A no-op (`{"replied": False}`, not an error) if `card_id` names no live
    resident engine -- same "nothing to do" shape as `resize_session_pty`.

    Issue #179: an implement-type session parked here because
    `session_runner.classify_needs_input` found the turn needed a human's
    input but nothing shaped like a rich question to extract (see
    `_finish_implement_turn`) is different from this endpoint's original
    mid-turn-nudge case -- that turn has already completed, so nothing is
    still reading the PTY for it, and a raw write here would go nowhere.
    For that case (`session_type == "implement"` and `phase ==
    "implementing"`, the exact state `_finish_implement_turn` leaves such a
    session suspended in), this resumes through `continue_implement_job` --
    a real new turn, run the same way a genuine `implement_blocked` reply
    already is -- instead of writing straight into the PTY."""
    text = body["text"]

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is not None and row["session_type"] == "implement" and row["phase"] == "implementing":
        cwd = _active_project_cwd()
        asyncio.create_task(session_runner.continue_implement_job(card_id, text.rstrip("\r\n"), cwd=cwd))
        return {"replied": True}

    engine = session_runner.get_engine(card_id)
    if engine is None:
        return {"replied": False}

    await engine.write(text)
    return {"replied": True}


@app.post("/api/session/start")
async def start_session(body: dict):
    project_id = _active_project_id
    cwd = _active_project_cwd()
    if project_id is None:
        return {"error": "No active project"}

    conn = db.get_connection()

    # `effort` in the body is the left-card dropdown's value at the moment
    # "Start" was clicked -- the common case for a per-session override,
    # covered without any race. Falls back to the global default (matching
    # `create_session`'s own fallback) when omitted.
    effort = body.get("effort")

    # Try a pre-warmed standby first (issue #136) -- claiming it is only
    # correct if its model/effort match exactly what this session will
    # actually use, so this mirrors create_session's own effort resolution
    # (`effort or DEFAULT_EFFORT`) and start_session_job's own model
    # resolution (`db.get_model(conn)`) precisely, not this row's stored
    # columns (which can differ -- see the comments in session_runner.py).
    #
    # Issue #184: every brand-new session begins in the grilling phase
    # (`db.create_session`'s own `phase="grilling"` default), and grilling
    # now runs on `StreamJsonEngine` -- so the standby claimed/registered
    # here is the `StreamJsonEngine` one, matching what `open_project` warms
    # and what `start_session_job` will actually dispatch this session's
    # first turn to. Unlike `PtyEngine` (which self-assigns a session id up
    # front), a freshly-spawned, never-yet-turned `StreamJsonEngine`'s
    # `session_id` is `None` until its first completed turn -- passing that
    # straight through to `claude_session_id=` here is correct: it's exactly
    # what "no resumable session yet" already means to `create_session`.
    model = db.get_model(conn)
    resolved_effort = effort or db.DEFAULT_EFFORT
    standby = session_runner.claim_standby_stream_json_engine(project_id, model=model, effort=resolved_effort)

    if standby is not None:
        row_id = db.create_session(
            conn, project_id, claude_session_id=standby.session_id, effort=effort
        )
        session_runner.register_stream_json_engine(row_id, standby)
    else:
        reused = db.claim_available_session(conn, project_id)
        resume_id = reused["claude_session_id"] if reused is not None else None
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


@app.post("/api/sessions/{card_id}/continue-do")
async def continue_do(card_id: int, body: dict):
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return {"error": "Session not found"}
    if row["session_type"] != "do" or row["phase"] != "details":
        return {"error": "Session is not at the do-finished pause point"}
    cwd = _active_project_cwd()
    prompt = body.get("prompt", "")
    asyncio.create_task(session_runner.start_do_continue_job(card_id, prompt, cwd=cwd))
    return {"card_id": card_id}


@app.post("/api/sessions/{card_id}/close")
def close_session(card_id: int):
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return {"error": "Session not found"}

    session_runner.close_session(conn, card_id)
    return {"card_id": card_id}


@app.post("/api/session/qa-complete")
async def qa_complete(body: dict):
    conn = db.get_connection()
    row = db.get_session(conn, body["card_id"])
    if row is None:
        return {"error": "Session not found"}
    cwd = _active_project_cwd()
    answers = body.get("answers", {})
    extra_notes = body.get("extra_notes", "")
    asyncio.create_task(session_runner.continue_qa_job(row["id"], answers, extra_notes, cwd=cwd))
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
