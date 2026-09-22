"""Per-session background jobs: run CLI turns off the event loop thread and
stream their translated output live via `rhubarb.live_stream`.

Each job publishes app-level events for a session's card_id: `phase` at the
start of each step, `text`/`action`/`usage` as a turn streams, a richer
`turn` event once a turn's semantics (interview/details/error) are known,
and `done` once the session has reached a terminal state (details or an
unrecoverable error).

Every phase (`grilling`, `to-prd`, `to-issues`, `publish-to-github`,
`implement`, `qa`) drives its turns through a single resident
`StreamJsonEngine` "tab" per `card_id` -- see `_stream_json_engines` below
-- one persistent headless `claude -p --input-format stream-json
--output-format stream-json` subprocess per session, reused across every
turn (issue #87's original "resident tab" model, issue #184's migration off
`PtyEngine`'s interactive-PTY transport once `--resume` reattachment from a
headless-originated session proved unreliable under full interactive mode).
"""

import asyncio
import json
import re
from pathlib import Path

_FENCED_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")


def _rejoin_wrapped_json_strings(candidate: str) -> str:
    """Undo terminal word-wrap inside a fenced JSON block's string literals
    before handing `candidate` to `json.loads` -- an analogous "undo
    word-wrap before parsing" step to the one the now-retired `qa_parser.py`
    regex parsers used to apply via their own `_reflow` (issue #230 removed
    that module entirely; this function is unrelated to it and untouched).

    A real PTY-spawned terminal can word-wrap Claude's rendered text to fit
    whatever width it's reporting (see `pty_engine.PtyEngine`'s own notes on
    this), and that wrapping knows nothing about JSON syntax -- it can just
    as easily inject a raw newline in the middle of a JSON string literal as
    between two structural tokens. A raw, unescaped newline is never valid
    inside a JSON string, so a long `"question"`/`"context"`/etc. value that
    happened to wrap at one terminal width can produce different (and, at
    the point it lands mid-string, invalid) JSON at a different width, even
    though nothing about the semantic content changed.

    Structural whitespace -- the pretty-printer's own newlines and
    indentation between tokens (after `{`, `,`, `:`, etc.) -- is already
    valid JSON exactly as-is and is left untouched; `json.loads` never cared
    about that whitespace in the first place. Only whitespace runs
    containing a newline that fall *inside* a string literal are collapsed
    to a single space, undoing the wrap the same way the original text's
    single space would have read before the terminal broke the line there.

    Tracks string-literal state character by character (honoring `\\"`
    escapes so an escaped quote never toggles it) rather than working
    line-by-line, so this is correct regardless of how many times, or at
    what column, a string got wrapped -- unlike a line-oriented rejoin,
    which would have to guess whether a given physical line starts a new
    logical field. A no-op whenever no string literal contains a raw
    newline, which is true of any already word-wrap-free block (every
    existing fixture predating this function included)."""
    out: list[str] = []
    in_string = False
    escaped = False
    i = 0
    n = len(candidate)
    while i < n:
        ch = candidate[i]
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                i += 1
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                i += 1
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                i += 1
                continue
            if ch == "\n":
                j = i
                while j < n and candidate[j] in " \t\r\n":
                    j += 1
                out.append(" ")
                i = j
                continue
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
        i += 1
    return "".join(out)

from rhubarb import db, error_log, parser_session
from rhubarb.cli_client import ClaudeCLIError
from rhubarb.live_stream import publish
from rhubarb.ollama_rescue import _is_valid_qa_shape, classify_turn_needs_input
from rhubarb.question_files import delete_question_file, read_question_file
from rhubarb.stream_json_engine import StreamJsonEngine, StreamJsonEngineUnrecoverableError
from rhubarb.stream_translate import translate_event

# Filenames under `.claude/` a skill writes its structured question/blocked
# output to (PRD #123) -- read here in preference to scraping the turn's
# rendered terminal text. See `rhubarb/question_files.py`.
_QA_QUESTION_FILE = "rhubarb_qa.md"
_IMPLEMENT_BLOCKED_FILE = "rhubarb_blocked.json"

# The old broad pattern (`auth|login|not logged in|permission denied|401|403`)
# matched any Claude CLI failure that happened to contain one of those common
# words, producing a false "Login with GitHub" prompt for errors that had
# nothing to do with GitHub. This classifier instead looks for `gh`'s own
# remediation text, which only appears in a genuine unauthenticated `gh` CLI
# failure -- not a generic Claude CLI error.
_GH_AUTH_FAILURE_RE = re.compile(r"gh auth login", re.IGNORECASE)


def _is_gh_auth_failure(message: str) -> bool:
    """True only when `message` is shaped like a real unauthenticated `gh`
    CLI failure, grounded in `gh`'s own "gh auth login" remediation hint --
    not any Claude CLI failure that merely contains a generic word like
    "auth", "login", "permission denied", "401", or "403"."""
    return bool(_GH_AUTH_FAILURE_RE.search(message))
_DETAIL_RE = re.compile(r"\b(PRD|Issue)\s*#(\d+)\s*[:\-]\s*(.+)", re.IGNORECASE)

# Context-window budget: before starting the next phase in a session chain,
# `_maybe_clear_for_next_phase` checks the row's last-recorded `context_pct`
# against the relevant cutoff below and starts a fresh `StreamJsonEngine`
# first if it's over. The gate is a pre-phase check only -- a phase already
# running is never interrupted even if it crosses its cutoff while in
# flight. Placeholder pending real /rhubarb:implement and /rhubarb:qa
# context-growth telemetry -- expect this to move.
#
# Every turn now runs through `StreamJsonEngine`'s headless `-p
# --output-format stream-json` transport, which carries real
# `usage`/`modelUsage` fields on its `result` event -- `_context_window_pct`
# computes a real percentage for every phase, not just grilling (issue #225
# removed `PtyEngine`, whose interactive marker-based protocol carried no
# such data). An unknown `context_pct` (e.g. no turn has completed yet) is
# still treated as safe-to-continue by every caller, unchanged.
_IMPLEMENT_TO_QA_CONTEXT_CUTOFF = 0.68

# Context-window gate for the do-finished Continue button (issue #198, child
# of PRD #195): before starting a new grilling round on the same do session,
# `start_do_continue_job` checks whether the row's recorded `context_pct`
# exceeds this and, if so, spawns a fresh `StreamJsonEngine` for a clean
# conversation instead of reusing the old, near-limit one.
_DO_CONTINUE_CONTEXT_CUTOFF = 0.40

# A session that just finished /rhubarb:qa (closed its issues/PRD) is only
# fully closed if its context usage is over this -- under it, it's marked
# available for reuse instead (`db.claim_available_session` picks it up for
# the next fresh /do session on the left card, warm cache and all) rather
# than being discarded with useful headroom still left.
_QA_DONE_RECYCLE_CONTEXT_CUTOFF = 0.50


def _context_window_pct(raw_result_event: dict) -> float | None:
    """Compute how full the context window was for one CLI turn, from the
    raw (untranslated) `result` event's usage fields -- the same formula
    Claude Code's own interactive statusline uses for its pre-calculated
    `context_window.used_percentage` field: `(input_tokens +
    cache_creation_input_tokens + cache_read_input_tokens) / contextWindow`.

    Returns `None` when the event doesn't carry enough to compute this (no
    `usage`/`modelUsage` block, or a zero/missing `contextWindow`) rather
    than raising -- a session with an unknown context usage is
    treated as safe-to-continue by every caller (see
    `_maybe_clear_for_next_phase`), since erring toward "don't gate" only
    risks the growth this budget is meant to catch, not silent data loss.
    """
    usage = raw_result_event.get("usage") or {}
    model_usage = raw_result_event.get("modelUsage") or {}
    first_model_usage = next(iter(model_usage.values()), {})
    context_window = first_model_usage.get("contextWindow")
    if not context_window:
        return None

    used = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )
    return used / context_window


# Per-card_id lock guarding a turn's actual write/read against a duplicate
# overlapping call for the same card_id (issue #144) -- a confirmed race
# where two turn-initiating requests for the same session (e.g. a
# double-clicked button, a retried request) could both reach the same
# resident engine's `stream_turn` at once, interleaving writes and reads on
# a connection that only ever expects one turn in flight. Mirrors
# `_stream_json_engines`'s own per-card_id lifecycle: created on first use
# by `_get_turn_lock`, popped (and discarded) by `_close_stream_json_engine`
# alongside the engine itself so the registry never grows unboundedly over a
# long-running instance.
_turn_locks: dict[int, asyncio.Lock] = {}


def _get_turn_lock(card_id: int) -> asyncio.Lock:
    """Return this card's turn lock, creating one on first use -- mirrors
    `_get_or_create_stream_json_engine`'s "construct on first call, reuse
    after" shape."""
    lock = _turn_locks.get(card_id)
    if lock is None:
        lock = asyncio.Lock()
        _turn_locks[card_id] = lock
    return lock


def close_session(conn, card_id: int) -> None:
    """User-initiated permanent close of a session card (issue #121): tears
    down this card's resident `StreamJsonEngine` tab (`_close_stream_json_engine`
    -- pop and terminate, safe no-op if there isn't one), marks the row
    `phase="closed"` so `db.list_sessions_for_project` excludes it from now
    on, and publishes a terminal `closed` event so any live SSE stream for
    this card (foreground or background) ends the same way a
    naturally-finished session's `done` event does (see `stream_session` in
    `rhubarb/web/app.py`).

    No literal `/clear` turn is sent first -- the process is destroyed
    directly, same rationale as `_spawn_fresh_stream_json_engine`'s
    docstring: the point is to end this conversation for good, not to
    round-trip a slash command into a process that may itself be mid-turn."""
    _close_stream_json_engine(card_id)
    db.update_session(conn, card_id, phase="closed")
    publish(card_id, {"type": "closed", "card_id": card_id})


def get_engine_model_effort(card_id: int) -> tuple[str | None, str | None] | None:
    """Return the `(model, effort)` this card's resident `StreamJsonEngine`
    was actually constructed with (issue #139) -- ground truth for the live
    process's own argv, as opposed to `db.get_model`/`db.get_effort`'s
    global "what the next new session will use" setting. Returns `None` if
    there's no live resident engine for `card_id` -- callers fall back to
    the global settings in that case."""
    engine = _stream_json_engines.get(card_id)
    if engine is None:
        return None
    return engine.model, engine.effort


def _maybe_respawn_for_settings_change(
    conn, card_id: int, *, cwd: str | None, model: str | None, effort: str | None
) -> bool:
    """Shared tail for `respawn_engine_for_model_change`/
    `respawn_engine_for_effort_change` below -- tear down `card_id`'s live
    resident engine and replace it with a fresh, unresumed one (see
    `_spawn_fresh_stream_json_engine`) under `model`/`effort`, keeping the
    row in sync (`claude_session_id`/`model`/`effort`/`context_pct`,
    mirroring `_maybe_clear_for_next_phase`'s own respawn bookkeeping -- a
    fresh conversation has a new session id and nothing yet measured for
    `context_pct`). Only ever touches `_stream_json_engines[card_id]` -- no
    other card's engine, and no project's standby engine, is read or written
    here. Always returns True; callers only call this once they've already
    confirmed (via `get_engine_model_effort`) that `card_id` has a live
    engine to replace."""
    _close_stream_json_engine(card_id)
    engine = _spawn_fresh_stream_json_engine(cwd=cwd, model=model, effort=effort)
    _stream_json_engines[card_id] = engine
    db.update_session(
        conn, card_id, claude_session_id=engine.session_id, model=model, effort=effort, context_pct=None
    )
    return True


def respawn_engine_for_model_change(conn, card_id: int | None, *, cwd: str | None, model: str | None) -> bool:
    """Called from `POST /api/settings/model` (issue #141) with `card_id`
    naming the currently-open/visible session card, if any -- the frontend's
    `leftCardId` at the moment the model selector changed. When that card
    has a live resident engine, tears it down and spawns a fresh, unresumed
    one under the new `model`; `effort` is read straight off the live
    engine's own ground truth (`get_engine_model_effort`) rather than the
    global setting, so a model-only change never silently also changes
    effort.

    Returns False (a pure no-op -- touches no engine, no row) when `card_id`
    is None or names a card with no live engine: the caller's global
    `db.set_model` write already happened either way and is unaffected by
    this function's return value. This is exactly today's behavior for
    every existing caller that doesn't pass a `card_id`."""
    if card_id is None:
        return False
    engine_model_effort = get_engine_model_effort(card_id)
    if engine_model_effort is None:
        return False
    _, current_effort = engine_model_effort
    return _maybe_respawn_for_settings_change(conn, card_id, cwd=cwd, model=model, effort=current_effort)


def respawn_engine_for_effort_change(conn, card_id: int | None, *, cwd: str | None, effort: str | None) -> bool:
    """Symmetric to `respawn_engine_for_model_change` above, for `POST
    /api/settings/effort` -- `model` is read off the live engine's own
    ground truth instead of the global setting, so an effort-only change
    never silently also changes model. See that function's docstring for
    the no-op/return-value contract, which is identical here."""
    if card_id is None:
        return False
    engine_model_effort = get_engine_model_effort(card_id)
    if engine_model_effort is None:
        return False
    current_model, _ = engine_model_effort
    return _maybe_respawn_for_settings_change(conn, card_id, cwd=cwd, model=current_model, effort=effort)


def count_resident_engines() -> int:
    """How many `StreamJsonEngine` "tabs" are currently resident (issue #88)
    -- one per `card_id` with a live entry in `_stream_json_engines`, across
    every active session regardless of phase, plus any pre-warmed standby
    engines (issue #136) -- both are real, live `claude` processes. Backs
    the web UI's tab-count indicator next to the "Sessions" label; polled
    rather than pushed since it's a global count, not scoped to any one
    card's SSE stream."""
    return len(_stream_json_engines) + len(_standby_stream_json_engines)


def list_live_engines() -> list[dict]:
    """One record per live `StreamJsonEngine` "tab" across both
    `_stream_json_engines` and `_standby_stream_json_engines` (issue #140),
    each shaped `{"card_id": int | "standby", "model": str | None, "effort":
    str | None}` -- ground truth `(model, effort)` read straight off the
    resident engine's own attributes for a card (same source
    `get_engine_model_effort` already uses), or off the stored tuple for a
    standby (never touched beyond what
    `ensure_standby_stream_json_engine`/`claim_standby_stream_json_engine`
    already track). A standby's `card_id` is the literal string `"standby"`
    -- it has no card yet, and at most one exists per project, so no further
    disambiguation (e.g. project_id) is included here.

    Backs the tab-count endpoint's additive `"engines"` field -- purely a
    read-only listing for the background-visibility UI; does not stream, or
    otherwise touch, any engine."""
    engines = [
        {"card_id": card_id, "model": engine.model, "effort": engine.effort}
        for card_id, engine in _stream_json_engines.items()
    ]
    engines.extend(
        {"card_id": "standby", "model": model, "effort": effort}
        for engine, model, effort in _standby_stream_json_engines.values()
    )
    return engines


# ---------------------------------------------------------------------------
# StreamJsonEngine wiring -- the sole transport for every phase (issue #184
# introduced it for grilling only; issue #225's live testing found
# `PtyEngine`'s `--resume` reattachment from a headless-originated session
# hangs indefinitely with no output/error on this platform, so every other
# phase migrated here too and `PtyEngine`/`rhubarb/pty_engine.py` was
# removed entirely).
#
# KNOWN GAP -- deliberate, not an oversight: none of this survives a full
# Rhubarb server restart. `_stream_json_engines`/`_standby_stream_json_engines`
# are in-memory only, and nothing here has been built to pick a session back
# up after a restart. If the server restarts mid-session, that session is
# simply lost and the user starts over. Restart-survival is out of scope and
# left for a future issue.
# ---------------------------------------------------------------------------

_stream_json_engines: dict[int, StreamJsonEngine] = {}

# Pre-warmed, unclaimed StreamJsonEngine per project (issue #136) -- so a
# brand-new session's first turn doesn't pay inline spawn latency. Keyed by
# project_id (only one project is ever active at a time -- see
# `_active_project_id` in `rhubarb/web/app.py`), not card_id: a standby is
# never assigned to a card_id or run through a turn until claimed. Stores
# the (model, effort) it was spawned with alongside the engine so a claim
# attempt can tell a match from a stale one.
_standby_stream_json_engines: dict[int, tuple[StreamJsonEngine, str | None, str | None]] = {}


def _get_or_create_stream_json_engine(
    card_id: int, *, cwd: str | None, model: str | None, effort: str | None, resume_session_id: str | None
) -> StreamJsonEngine:
    """Return this card's resident tab, constructing and starting one
    (fresh, or reattached via `--resume resume_session_id` -- e.g. a reused
    pooled session, or a session picked back up after a Rhubarb restart) if
    this is the first turn for this `card_id`. Every later turn for the
    same `card_id` reuses the exact same `StreamJsonEngine` instance --
    never recreated per turn."""
    engine = _stream_json_engines.get(card_id)
    if engine is not None:
        return engine
    engine = StreamJsonEngine(cwd=cwd, model=model, effort=effort, resume_session_id=resume_session_id)
    engine.start()
    _stream_json_engines[card_id] = engine
    return engine


def register_stream_json_engine(card_id: int, engine: StreamJsonEngine) -> None:
    """Register an already-running engine (a claimed standby) as `card_id`'s
    resident tab, so `_get_or_create_stream_json_engine`'s own "already
    resident, don't spawn" check picks it up on the first turn, exactly as
    if it had been spawned for this card_id from the start."""
    _stream_json_engines[card_id] = engine


def _close_stream_json_engine(card_id: int) -> None:
    """Close and forget this card's resident tab, if any -- called whenever
    a card's tab finishes: pooled for reuse, fully done, handed off to a
    different card_id (the /implement -> /qa auto-handoff), or errored out
    (a later retry reattaches a fresh tab via `--resume` instead of
    continuing to drive a process that just raised). Also drops this card's
    turn lock from `_turn_locks`, recreated on first use by the next
    incarnation."""
    engine = _stream_json_engines.pop(card_id, None)
    if engine is not None:
        engine.close()
    _turn_locks.pop(card_id, None)


async def ensure_standby_stream_json_engine(
    project_id: int, *, cwd: str | None, model: str | None, effort: str | None
) -> None:
    """Make sure a live, matching standby exists for `project_id`, spawning
    one if it doesn't (or replacing a dead one). A no-op when a live,
    already-matching standby is already there -- never more than one
    standby per project. Fire-and-forget: callers schedule this via
    `asyncio.create_task` rather than awaiting it, since nothing should
    block on a pre-warm."""
    existing = _standby_stream_json_engines.get(project_id)
    if existing is not None:
        engine, standby_model, standby_effort = existing
        if engine.isalive() and standby_model == model and standby_effort == effort:
            return
        engine.close()
        del _standby_stream_json_engines[project_id]

    engine = await asyncio.to_thread(_spawn_fresh_stream_json_engine, cwd=cwd, model=model, effort=effort)
    _standby_stream_json_engines[project_id] = (engine, model, effort)


def claim_standby_stream_json_engine(
    project_id: int, *, model: str | None, effort: str | None
) -> StreamJsonEngine | None:
    """Pop and return `project_id`'s standby if it's alive and its
    model/effort match what's being requested; otherwise discard whatever
    was there (dead or mismatched -- never left dangling) and return `None`
    so the caller falls back to today's spawn-on-demand/DB-pool path."""
    existing = _standby_stream_json_engines.pop(project_id, None)
    if existing is None:
        return None
    engine, standby_model, standby_effort = existing
    if not engine.isalive() or standby_model != model or standby_effort != effort:
        engine.close()
        return None
    return engine


def close_standby_stream_json_engine(project_id: int) -> None:
    """Close and discard `project_id`'s standby, if any -- called when a
    project is closed or switched away from, so a standby never leaks past
    the project it was warmed for."""
    existing = _standby_stream_json_engines.pop(project_id, None)
    if existing is not None:
        existing[0].close()


def _spawn_fresh_stream_json_engine(*, cwd: str | None, model: str | None, effort: str | None) -> StreamJsonEngine:
    """Construct and start a genuinely fresh `StreamJsonEngine` -- no
    `resume_session_id` -- a brand-new, empty conversation. Used everywhere
    this module needs to reclaim context by starting over rather than
    keeping talking to the same process. Blocking (real spawn is a
    subprocess call) -- callers run this via `asyncio.to_thread`."""
    engine = StreamJsonEngine(cwd=cwd, model=model, effort=effort)
    engine.start()
    return engine


async def _run_stream_json_turn(
    card_id: int,
    prompt: str,
    *,
    session_id: str | None,
    cwd: str | None,
    model: str | None,
    effort: str | None,
    phase: str,
) -> dict | None:
    """Run one turn against this card's resident `StreamJsonEngine` tab (see
    `_get_or_create_stream_json_engine` -- constructed and started on the
    first call for this `card_id`, reattached via `--resume session_id` if
    one is already known, and reused unchanged on every later call for the
    same `card_id`), streaming translated events into the session's live
    buffer as they arrive.

    Every raw event `StreamJsonEngine.stream_turn` yields is translated via
    `stream_translate.translate_event()` and published on this card's
    stream via `publish()`, except the translated `turn` (`result`) event,
    which is returned to the caller instead (the turn's "boundary marker").

    `full_text` on the returned dict is the turn's ENTIRE accumulated
    stream of `text` deltas, not just the CLI's own terse final `result`
    field -- bug found live-testing PRD #222: the CLI's own `result` event
    text is only the LAST assistant text block of the turn -- if the model
    prints a question round, then calls a tool (e.g. writing
    `.claude/rhubarb_question.md`), then closes with a short remark
    ("Waiting on your answers..."), `result` captures only that closing
    remark, silently dropping the earlier content the Live Terminal panel
    already streamed to the user. Every `text` delta is accumulated here
    into `full_text` (a superset of `result`, since every content block --
    including ones before a tool call -- streams via `content_block_delta`
    first) so a caller's parsing/extraction sees exactly what the user
    already saw on screen, not just the turn's final sentence. Callers
    should use `full_text`, not `result`, for any text they parse.

    Issue #144: guarded by this card's turn lock (`_get_turn_lock`) so two
    overlapping calls for the same `card_id` can never both write to and
    read from the same resident engine at once. If the lock is already
    held -- a genuine in-flight turn for this card_id -- this call (issue
    #149) publishes an explicit error `turn` event (`_turn_event`, tagged
    with the caller's own in-flight `phase`) on this card's stream, then
    returns `None` immediately, touching neither the engine nor anything
    else. Every caller must check for `None` and return early rather than
    treat it as a normal completed (or failed) turn -- and must NOT publish
    or log anything further for this case, since the error has already
    been surfaced here.

    Raises `ClaudeCLIError` on an ordinary failure, or propagates
    `StreamJsonEngineUnrecoverableError` (this engine's own internal
    crash-retry-once already gave up) unwrapped -- callers route that into
    the blocked-card flow (see `_route_crash_to_blocked`) instead of
    treating it like a plain `ClaudeCLIError`."""
    lock = _get_turn_lock(card_id)
    if lock.locked():
        lookup_conn = db.get_connection()
        lookup_row = db.get_session(lookup_conn, card_id)
        publish(
            card_id,
            _turn_event(
                phase=phase,
                error="Another turn for this session is already in progress -- please wait for it to finish.",
                needs_github_login=False,
                card_id=card_id,
                project_id=lookup_row["project_id"] if lookup_row is not None else None,
            ),
        )
        return None

    async with lock:
        holder: dict = {}

        async def runner():
            engine = _get_or_create_stream_json_engine(
                card_id, cwd=cwd, model=model, effort=effort, resume_session_id=session_id
            )
            text_chunks: list[str] = []
            async for raw_event in engine.stream_turn(prompt):
                translated = translate_event(raw_event)
                if translated is None:
                    continue
                if translated["type"] == "text":
                    text_chunks.append(translated["text"])
                if translated["type"] == "turn":
                    translated["context_pct"] = _context_window_pct(raw_event)
                    translated["full_text"] = "".join(text_chunks) or translated["result"]
                    holder["turn"] = translated
                    continue
                publish(card_id, translated)

        try:
            await runner()
        except StreamJsonEngineUnrecoverableError:
            raise
        except ClaudeCLIError as e:
            holder["error"] = e
        except Exception as e:
            # Anything unexpected (a malformed raw event, a bug in
            # translation) must still resolve into a recorded session
            # error, not an unhandled exception on the fire-and-forget
            # asyncio task -- that would leave the card stuck in its
            # in-flight phase silently instead of surfacing the failure to
            # the user.
            holder["error"] = ClaudeCLIError(str(e))

        if "error" in holder:
            raise holder["error"]
        return holder["turn"]


async def _extract_questions_via_parser_session(project_id: int, text: str, *, phase: str) -> dict | None:
    """Issue #221 (child of PRD #187/#220), simplified by issue #230's own
    scope-update to a shared helper, and -- PRD #227 follow-up, gap 1,
    discovered during manual testing/design review after #228/#229/#230 were
    already closed out in code -- now delegates to `parser_session.
    extract_with_validation` instead of duplicating its own copy of the
    extraction/validation sequence. Before this change, this function called
    `parser_session.stream_turn`/`_extract_json_object`/
    `_is_valid_grilling_shape` directly, with NO call into issue #229's
    post-extraction mismatch-detection/single-retry/`extraction_incomplete`-
    tagging logic at all -- meaning the original PRD #227 bug (a question's
    options/recommended silently dropped) was NOT actually protected against
    on this path, the live-turn path that produced the bug in the first
    place; only `parser_session._process_one_queued_item`'s async
    needs-input-queue consumer had that protection. `extract_with_validation`
    is the single shared implementation both now call.

    `phase` is passed straight through -- it only labels which phase this
    text came from for the skill's own reference; for every phase this
    function is called with (grilling, implementing), it never changes the
    output shape or the validation applied to it (`extract_with_validation`'s
    default `_is_valid_grilling_shape`/`detect_mismatches=True`).

    Returns a `{header, questions, footer, source: "parser_session"}` dict on
    a successful, schema-valid parse (now possibly carrying
    `extraction_incomplete: true` on individual questions per issue #229).
    Returns `None` on any failure -- a dead/missing parser session, a turn
    that produced no `result` event, or a response that isn't valid JSON
    matching the expected shape -- so a caller treats `None` exactly like
    "the parser session found no questions either". Deliberately no legacy
    regex/Ollama-rescue fallback here (unlike
    `parser_session.drain_needs_input_queue`'s own queue path): a caller with
    no live parser session has nothing else to try for this phase anymore."""
    return await parser_session.extract_with_validation(project_id, text, phase=phase)


async def _extract_grilling_questions_via_parser_session(project_id: int, text: str) -> dict | None:
    """Grilling's own call into the shared `_extract_questions_via_parser_
    session` helper above, with phase `"grilling"` -- kept as its own named
    function (rather than inlining the phase literal at `_run_grilling_turn_
    stream_json`'s own call site) since it's referenced by name in this
    module's docstrings/comments elsewhere."""
    return await _extract_questions_via_parser_session(project_id, text, phase="grilling")


async def _run_grilling_turn_stream_json(
    card_id: int,
    conn,
    row,
    prompt: str,
    *,
    cwd: str | None,
    model: str | None,
    effort: str | None,
    publish_when_empty: bool = False,
) -> dict | None:
    """Run one grilling turn and parse its structured questions -- question
    extraction here is a single synchronous parser-session pass
    (`_extract_grilling_questions_via_parser_session`) against the completed
    turn's own accumulated `full_text` (see `_run_stream_json_turn`'s
    docstring -- NOT just the CLI's own terse `result` sentence, which can
    silently omit an earlier text block), with NO `.claude/
    rhubarb_question.md` file-preference read/delete and NO Ollama
    rescue-classifier fallback, and NO corrective-retry follow-up turn:
    those all existed historically to compensate for `PtyEngine`'s
    PTY-rendering/capture-timing unreliability (word-wrap, ANSI
    cursor-movement redraws, a file write racing a terminal read), which
    doesn't apply to `StreamJsonEngine`'s structured stream-json output --
    see issue #184's acceptance criteria and issue #225's removal of
    `PtyEngine` entirely. QA and implementing keep their own equivalent
    file-preference conventions (`_QA_QUESTION_FILE`, `_IMPLEMENT_BLOCKED_FILE`),
    untouched by this.

    Issue #221 (child of PRD #187/#220), simplified by issue #230: this used
    to try `qa_parser.parse_grilling_response`'s regex parser first and only
    call `_extract_grilling_questions_via_parser_session` when that came back
    empty -- but the grilling skill's real output (`❓ **Qn** - **title**:
    body`) never matches that regex format, so the "try regex first" step
    was confirmed dead (always empty) and removed; this now goes straight to
    the parser-session extraction pass. This replaced the old
    `handle_turn_completed` Ollama needs-input classifier gate for this
    phase specifically, which the issue's acceptance criteria requires be
    gone (grilling always transitions to PRD next; there is no "needs input"
    holding state for it to gate into).

    Issue #223 (child of PRD #222): once the frontier comes back empty (no
    parsed questions), this now calls `advance_past_grilling` itself before
    returning -- no "Yes, proceed" confirmation click required anymore. The
    empty-questions `turn` event is still published first (when
    `publish_when_empty`), so the frontend briefly sees the wrap-up header
    exactly as before; it's the very next thing on this same stream that now
    differs (a `creating_prd` phase event instead of silence).

    Issue #239 (child of PRD #237): the earlier issue #219 fast path here --
    a loose regex check for a PRD/Issue number mentioned anywhere in the
    turn's text, routing straight to `_finish_chain` without ever parsing
    for questions -- is removed. It had turned "the grilling model
    self-answered its own questions and free-ran the whole /do chain in one
    turn" into a silently-accepted, first-class outcome instead of a bug.
    Now every turn's text is parsed for questions exactly the same way
    regardless of whether it happens to mention a PRD/issue number; the
    grilling skill itself is tightened (see its SKILL.md) to never
    self-answer and to always end its turn and wait for a real reply, so a
    turn that still tries to free-run the chain surfaces as a visibly wrong
    state (an unparsed/odd interview shown to the user) instead of being
    absorbed here.

    Issue #225 (child of PRD #222): a `StreamJsonEngineUnrecoverableError`
    (this engine's own crash-retry-once already gave up) now routes into
    the same generic "blocked, reply to retry" flow every other phase uses
    (`_route_crash_to_blocked`), instead of folding into a plain
    `ClaudeCLIError` -- grilling gains blocked-crash-recovery for the first
    time here, removing an asymmetry that only existed because no other
    phase used `StreamJsonEngine` before this migration."""
    publish(card_id, {"type": "phase", "phase": "grilling"})

    try:
        turn = await _run_stream_json_turn(
            card_id, prompt, session_id=row["claude_session_id"], cwd=cwd, model=model, effort=effort, phase="grilling"
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="grilling")
        return None
    except ClaudeCLIError as e:
        _close_stream_json_engine(card_id)
        message = str(e)
        db.update_session(conn, card_id, model=model, effort=effort, error_text=message, needs_github_login=0)
        publish(
            card_id,
            _turn_event(
                phase="grilling",
                error=message,
                needs_github_login=False,
                card_id=card_id,
                project_id=row["project_id"],
            ),
        )
        return None

    if turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the
        # duplicate-call error itself; nothing more to do here.
        return None

    # Use the turn's full accumulated text (every streamed text block, not
    # just the CLI's own final `result` sentence -- see
    # `_run_grilling_stream_json_turn`'s docstring above) for parsing,
    # extraction, and history -- this is what the Live Terminal panel
    # already showed the user, so a round that streamed real questions
    # before a trailing tool call/closing remark is never silently dropped.
    full_text = turn["full_text"]
    console_text = row["console_text"] + "\n\n" + full_text if row["console_text"] else full_text

    # Issue #221 (child of PRD #187/#220), simplified by issue #230: go
    # straight to this project's live parser session -- the SAME
    # synchronous-extraction contract `parser_session._process_one_queued_
    # item` already uses for the needs-input queue, just driven inline here
    # instead of via that queue -- to extract structured questions out of
    # the turn's own free text (e.g. `❓ **Q1** - **title**: body`). Replaces
    # the old `handle_turn_completed` Ollama needs-input classifier gate
    # entirely: grilling always transitions to PRD next, so there is no
    # "needs input" holding state for it to gate into anymore.
    parsed = await _extract_grilling_questions_via_parser_session(row["project_id"], full_text)
    if parsed is None:
        parsed = {"header": full_text, "questions": [], "footer": "", "source": None}

    db.update_session(
        conn,
        card_id,
        model=model,
        effort=effort,
        claude_session_id=turn["session_id"],
        console_text=console_text,
        interview_json=json.dumps(parsed),
        context_pct=turn.get("context_pct"),
    )

    if parsed["questions"] or publish_when_empty:
        publish(card_id, _turn_event(phase="grilling", interview=parsed))

    if parsed["questions"]:
        return parsed

    await advance_past_grilling(card_id, cwd)
    return None


async def _maybe_clear_for_next_phase(card_id: int, conn, row, *, cwd: str | None, cutoff: float) -> str:
    """The context-window budget gate: called right before starting the next
    phase in a session chain (currently `/rhubarb:do` -> `/rhubarb:implement` at
    `_DO_TO_IMPLEMENT_CONTEXT_CUTOFF`, `/rhubarb:implement` -> `/rhubarb:qa` at
    `_IMPLEMENT_TO_QA_CONTEXT_CUTOFF` -- wired in by whichever caller is
    orchestrating that transition).

    Reads `row["context_pct"]` (persisted after the previous phase's last
    turn) rather than measuring anything fresh: an unknown value (`None`,
    e.g. no turn has completed yet) is treated as safe to continue, same as
    `_context_window_pct`'s own None-on-uncertainty behavior.

    At or under `cutoff`: returns the row's existing `claude_session_id`
    unchanged -- the next phase continues in the same tab, no fresh engine.

    Over `cutoff`: tears down this card's resident tab and starts a
    genuinely fresh one (see `_spawn_fresh_stream_json_engine`) *for this
    same card_id* -- the next phase continues right on in the new tab, it's
    just talking to an empty conversation instead of the old one. Persists
    the new session id and resets `context_pct` to `None` on the row (the
    fresh tab starts with an empty, unmeasured context again). This is a
    pre-phase gate only -- once the next phase is running, it is never
    interrupted mid-run even if it goes on to cross `cutoff` itself."""
    context_pct = row["context_pct"]
    if context_pct is None or context_pct <= cutoff:
        return row["claude_session_id"]

    _close_stream_json_engine(card_id)
    engine = await asyncio.to_thread(_spawn_fresh_stream_json_engine, cwd=cwd, model=row["model"], effort=row["effort"])
    _stream_json_engines[card_id] = engine
    new_session_id = engine.session_id
    db.update_session(conn, card_id, claude_session_id=new_session_id, context_pct=None)
    return new_session_id


async def _clear_for_reuse(
    card_id: int, *, project_id: int, cwd: str | None, model: str | None, effort: str | None
) -> str:
    """Reclaim context by starting a genuinely fresh conversation (see
    `_spawn_fresh_stream_json_engine`), for a card whose row is about to be
    pooled (`db.mark_session_available`) for reuse under a *different*,
    future card_id -- this card's own tab is closed for good here, since
    nothing will ever run another turn against this `card_id` again.

    Unlike before issue #136, the freshly-spawned engine is NOT immediately
    closed after reading its id -- it's kept alive as `project_id`'s standby
    (`ensure_standby_stream_json_engine`'s registry), so the next `/do` for
    this project can claim it directly with no spawn at all, instead of
    every pooling cycle paying a full spawn just to throw the process away
    unused. The returned id is still recorded via `db.mark_session_available`
    by every caller, unchanged -- that DB-level pool remains the fallback
    path for whenever this standby isn't claimed (mismatched model/effort,
    or a different project became active first)."""
    _close_stream_json_engine(card_id)
    engine = await asyncio.to_thread(_spawn_fresh_stream_json_engine, cwd=cwd, model=model, effort=effort)
    existing = _standby_stream_json_engines.pop(project_id, None)
    if existing is not None:
        existing[0].close()
    _standby_stream_json_engines[project_id] = (engine, model, effort)
    return engine.session_id


# In-memory, per-project FIFO queue for serial-mode ("parallel_implementation"
# off) PRD implementation requests. Process-lifetime only, same as
# `_active_project_id` in app.py -- nothing here needs to survive a restart.
_implement_queues: dict[int, list[dict]] = {}

# project_id -> list of undismissed background-session-error notifications,
# each {"card_id": int, "phase": str, "message": str} -- an /implement or
# /qa failure while minimized to the background (see `_auto_continue_implement_and_qa`)
# surfaces here instead of only the (possibly unfocused) card's own inline
# error state. Same in-memory, per-project, undismissed-until-dismissed shape
# as `afk_loop`'s self-implement notifications, kept here instead of there to
# avoid an afk_loop <-> session_runner import cycle (afk_loop already imports
# this module).
_error_notifications: dict[int, list[dict]] = {}


def add_error_notification(project_id: int, card_id: int, phase: str, message: str) -> None:
    _error_notifications.setdefault(project_id, []).append(
        {"card_id": card_id, "phase": phase, "message": message}
    )


def get_error_notifications(project_id: int) -> list[dict]:
    return _error_notifications.get(project_id, [])


def dismiss_error_notifications(project_id: int) -> None:
    _error_notifications.pop(project_id, None)


def _enqueue_implement(project_id: int, number: int, title: str) -> None:
    _implement_queues.setdefault(project_id, []).append({"number": number, "title": title})


def _pop_next_implement(project_id: int) -> dict | None:
    queue = _implement_queues.get(project_id)
    if not queue:
        return None
    return queue.pop(0)


def parse_details(text: str) -> dict:
    prd = None
    issues = []
    for match in _DETAIL_RE.finditer(text):
        kind, number, title = match.group(1).lower(), int(match.group(2)), match.group(3).strip()
        if kind == "prd" and prd is None:
            prd = {"number": number, "title": title}
        elif kind == "issue":
            issues.append({"number": number, "title": title})
    return {"prd": prd, "issues": issues, "raw": text}


def _blocked_payload_from_crash(exc: StreamJsonEngineUnrecoverableError) -> dict:
    """Synthesize the same shape of payload a genuine `implement_blocked`
    marker produces (see `_parse_implement_blocked_block`) out of a
    `StreamJsonEngineUnrecoverableError` -- issue #87's crash-routing
    requirement: a turn that dies twice in a row (see that exception's
    docstring) is routed into the exact same suspend-and-wait-for-a-human-
    reply mechanism a genuine blocked marker already uses, rather than a new
    UI/error path."""
    return {
        "phase": "implement_blocked",
        "issue": None,
        "question": (
            "This session's connection to Claude crashed twice in a row and "
            "could not recover automatically. Reply below to try to continue."
        ),
        "context": str(exc),
    }


async def _route_crash_to_blocked(card_id: int, conn, exc: StreamJsonEngineUnrecoverableError, *, phase: str) -> None:
    """A `stream_turn` call raised `StreamJsonEngineUnrecoverableError` (its
    underlying process died twice in a row and gave up). Suspend this
    session exactly like a genuine `implement_blocked` marker would: `phase:
    blocked`, `blocked_json` set, a `turn` event carrying it -- so the same
    reply flow that already resumes a blocked implement session
    (`continue_implement_job`) picks this up too, regardless of which phase
    hit the crash. The dead tab is dropped from the registry so the next
    turn (that reply) constructs a fresh one, reattaching via `--resume` at
    `exc.session_id` -- unless that turn crashed before ever completing a
    single one (`exc.session_id is None`), in which case a later reply
    starts a genuinely fresh conversation rather than truly resuming; an
    accepted, structural limitation of this transport, not a regression."""
    _close_stream_json_engine(card_id)
    blocked = _blocked_payload_from_crash(exc)
    db.update_session(
        conn,
        card_id,
        claude_session_id=exc.session_id,
        phase="blocked",
        blocked_json=json.dumps(blocked),
        stalled_json=None,
    )
    publish(card_id, _turn_event(phase="blocked", blocked=blocked))


def _turn_event(
    *,
    phase: str,
    interview=None,
    details=None,
    error=None,
    needs_github_login=False,
    blocked=None,
    status_message: str | None = None,
    card_id: int | None = None,
    project_id: int | None = None,
    stalled: bool = False,
    stalled_context: str | None = None,
) -> dict:
    """Build the `turn` event every phase publishes on a session's live
    stream. `card_id`/`project_id` are only used here -- not part of the
    published event shape -- to feed issue #153's app-wide error log: every
    caller that reports an error (`error is not None`) already goes through
    this one function, so logging here (rather than at each of the many call
    sites) covers every current and future error `turn` event automatically.

    `status_message` (issue #154) carries a short, human-readable line of
    live progress -- e.g. "PRD draft written.", "PRD published as #5",
    "Created issue #6" -- for phases (`creating_prd`, `creating_issues`,
    `publishing`) that otherwise give no feedback until the chain's final
    `details` turn. The frontend renders it via the same `renderSessionStatus`
    path that already shows the phase label (see `prompt.html`), rather than
    a new display surface.

    `stalled`/`stalled_context` were originally (issue #173, replacing issue
    #169/PRD #168's timer-based version) this same `turn` event shape
    extended for a turn still genuinely in flight, published on every chunk
    read off a live `PtyEngine` process. Issue #175 retired that per-chunk
    mechanism entirely, and issue #225 removed `PtyEngine` itself -- no
    remaining transport ever emits a mid-turn "stall" this way. Both
    `_finish_implement_turn` (issue #179, child of PRD #174) and
    `_run_chain_step` (issue #225) are live callers today: each sets these
    on a *completed* turn that needs a human's input before the
    phase/session can usefully continue on its own -- reviving the existing
    `renderStalledSession`/`sendStallReply` frontend panel for that case.
    Unlike the old mid-turn meaning, `stalled_context` here is a short reason string
    from the classifier, not a live buffer-so-far snapshot -- the turn is
    already over.
    """
    if error is not None:
        error_log.log_error(project_id=project_id, card_id=card_id, phase=phase, message=error)
    return {
        "type": "turn",
        "phase": phase,
        "interview": interview,
        "details": details,
        "error": error,
        "needs_github_login": needs_github_login,
        "blocked": blocked,
        "status_message": status_message,
        "stalled": stalled,
        "stalled_context": stalled_context,
    }


async def classify_needs_input(card_id: int, conn, text: str, phase: str, *, http_post=None) -> dict | None:
    """Classify, via the local Ollama model, whether a just-completed turn's
    rendered `text` (produced during `phase`) needs a human's input before
    this session can usefully continue -- see `ollama_rescue.
    classify_turn_needs_input` for the call itself and its `{needs_input,
    reason}` result shape.

    This is the mechanism only (issue #175, child of PRD #174 "replace the
    per-chunk PTY stall mechanism with a single per-turn Ollama
    classification"): nothing calls this yet from any phase's actual
    turn-handling flow (grilling, QA-grilling, implementing, ...) -- sibling
    issues #177/#178/#179 decide which phases call it and what a `True`
    result does. It is built and directly tested here so that wiring doesn't
    have to happen blind.

    Skipped entirely -- no HTTP call attempted, nothing published -- when
    the user has declined Ollama assistance (`db.get_ollama_declined`,
    issue #119's opt-out), same silent no-op every other Ollama-backed
    fallback in this codebase already gives that preference.

    When Ollama assistance is NOT declined but the classification call still
    fails or times out (`classify_turn_needs_input` returns `None`), that is
    a genuine, unexpected failure -- Ollama was expected to answer and
    didn't -- so this publishes a distinct `{"type": "ollama_unavailable"}`
    event on this card's live stream (`live_stream.publish`) rather than
    silently doing nothing. This is deliberately NOT folded into the
    existing `turn` event's `error` field: that field means the turn ITSELF
    failed, which isn't true here -- the turn already completed fine, only
    this follow-up classification call failed. `prompt.html` shows a minimal
    "Ollama is currently unavailable" modal on this event and takes no other
    action -- a turn's own normal result handling is completely untouched by
    this, on both success and failure."""
    if db.get_ollama_declined(conn):
        return None
    result = await asyncio.to_thread(classify_turn_needs_input, text, phase, http_post=http_post)
    if result is None:
        publish(card_id, {"type": "ollama_unavailable"})
    return result


async def handle_turn_completed(
    card_id: int, conn, row, text: str, phase: str, *, http_post=None
) -> dict | None:
    """The single shared "on turn complete" hook (issue #191, child of PRD
    #187) every phase's turn-handling code now calls once a turn has fully
    resolved, consolidating what used to be separate, duplicated per-phase
    call sites straight to `classify_needs_input`: grilling's own
    `_maybe_extract_needs_input_grilling`, QA-grilling's symmetric
    `_maybe_extract_needs_input_qa`, `_finish_implement_turn`'s direct call
    for implementing (including the parallel per-issue implementation
    sessions `/implement` can run concurrently -- they run through this same
    `_finish_implement_turn` tail, just N at once), and `_run_chain_step`'s
    direct call for the `creating_prd`/`creating_issues` (/to-prd, /to-issues)
    chain. A future session type only needs to call this one function, not
    copy-paste its own `classify_needs_input` wiring.

    Runs the existing Ollama needs-input classifier on `text` exactly as
    `classify_needs_input` always has -- same skip-when-declined/publish-
    `ollama_unavailable`-on-failure behavior, completely unchanged (see that
    function's own docstring).

    ADDITIONALLY -- issue #191's actual new behavior: when the classifier
    flags this turn as possibly needing input, enqueues it (this turn's own
    raw `text`, tagged with `card_id` and `phase`) onto
    `row["project_id"]`'s FIFO needs-input queue
    (`parser_session.enqueue_needs_input_turn`, keyed by project id exactly
    like that module's own parser-session registry -- issue #189). A turn
    the classifier does NOT flag (a negative result, or `None` from a
    decline/failure) is never enqueued.

    This is gating + queueing ONLY (issue #191) -- nothing dequeues/consumes
    that queue yet; that's issue #192, deliberately out of scope here. This
    function also makes no UI decision of its own: it returns the exact same
    classification dict `classify_needs_input` always returned, so every
    existing call site keeps deciding for itself what a positive result
    means for ITS phase (rich question extraction for grilling/QA-grilling/
    implementing, the generic stall-reply panel for creating_prd/
    creating_issues) -- completely unchanged from before this hook existed.
    """
    classification = await classify_needs_input(card_id, conn, text, phase, http_post=http_post)
    if classification is not None and classification.get("needs_input"):
        parser_session.enqueue_needs_input_turn(row["project_id"], card_id=card_id, phase=phase, text=text)
    return classification


async def _run_chain_step(
    card_id: int, conn, row, *, phase: str, prompt: str, cwd: str | None, model: str | None, effort: str | None
) -> tuple[bool, str | None, str]:
    """Run one /to-prd, /to-issues, or /publish-to-github step, live-streamed.
    Returns (ok, claude_session_id, last_result).

    On failure, publishes the error `turn` event and `done` itself -- the
    chain stops here exactly as the old blocking version did.

    Issue #177 (child of PRD #174): right after the turn resolves (and
    before any of the above success bookkeeping), the shared
    `handle_turn_completed` hook (issue #191) checks whether this turn's own
    rendered text suggests a human should weigh in before this phase
    auto-advances -- and, if so, enqueues it onto this project's needs-input
    queue alongside the classification. A `None` result (Ollama declined, or
    genuinely unavailable -- either way already handled/published by
    `classify_needs_input` itself) or `needs_input: False` changes nothing
    here -- this phase completes exactly as it always has.

    Issue #225 (child of PRD #222): a `needs_input: True` result persists
    `stalled_json`/publishes the stalled `turn` event and returns `(False,
    None, "")` -- treated by every caller exactly like a genuine failure
    (`if not ok: return`), WITHOUT publishing `done` (the SSE stream stays
    open). This function does NOT await inline for the human's reply the
    way it used to (`PtyEngine`'s raw keystroke passthrough let a reply be
    written directly into a still-open process's stdin mid-wait --
    `StreamJsonEngine` has no such passthrough, headless turns are a
    complete request/response each). `continue_stalled_chain_step_job`
    resumes this phase later as a genuinely new turn, with the human's
    reply as `prompt` -- mirroring the pattern `_finish_implement_turn`'s
    own needs-input handling already uses successfully for `implementing`.
    Re-stalling on that resumed turn needs no special handling: this
    function re-runs `handle_turn_completed` on every turn it drives,
    including a reply-resumed one, so it naturally re-persists/re-returns
    not-ok if the reply still doesn't satisfy the classifier."""
    db.update_session(conn, row["id"], phase=phase, error_text=None, needs_github_login=0)
    publish(card_id, {"type": "phase", "phase": phase})

    try:
        turn = await _run_stream_json_turn(
            card_id,
            prompt,
            session_id=row["claude_session_id"],
            cwd=cwd,
            model=model,
            effort=effort,
            phase=phase,
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase=phase)
        return False, None, ""
    except ClaudeCLIError as e:
        _close_stream_json_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, row["id"], error_text=message, needs_github_login=needs_login)
        publish(
            card_id,
            _turn_event(
                phase=phase,
                error=message,
                needs_github_login=bool(needs_login),
                card_id=card_id,
                project_id=row["project_id"],
            ),
        )
        publish(card_id, {"type": "done"})
        return False, None, ""

    if turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the
        # duplicate-call error itself; nothing more to do here.
        return False, None, ""

    full_text = turn["full_text"]
    console_text = row["console_text"] + "\n\n" + full_text

    classification = await handle_turn_completed(card_id, conn, row, full_text, phase)
    if classification is not None and classification.get("needs_input"):
        db.update_session(
            conn,
            row["id"],
            claude_session_id=turn["session_id"],
            console_text=console_text,
            context_pct=turn.get("context_pct"),
            stalled_json=json.dumps({"phase": phase, "context": full_text}),
        )
        publish(card_id, _turn_event(phase=phase, stalled=True, stalled_context=full_text))
        return False, None, ""

    db.update_session(
        conn,
        row["id"],
        claude_session_id=turn["session_id"],
        console_text=console_text,
        context_pct=turn.get("context_pct"),
        stalled_json=None,
    )

    return True, turn["session_id"], full_text



async def _finish_chain(card_id: int, conn, claude_session_id: str, cwd: str | None, *, summary: str = "") -> None:
    """`/rhubarb:do` just reached `details` (PRD + issues published). Publishes
    the `details` turn event then stops -- the resident engine stays alive and
    idle, waiting on the user's next action (Continue or Close session via the
    do-finished modal). Implementation is never started automatically from
    here; the user either clicks a PRD in "To be implemented," the AFK
    self-implement timer picks it up, or they click Continue to start a fresh
    grilling round on the same card.

    `summary` is Claude's own closing paragraph from the final chain step
    (the `creating_issues` turn result). Stored in `details` so the
    do-finished modal can display it verbatim without an extra fetch."""
    row = db.get_session(conn, card_id)
    details = parse_details(row["console_text"])
    if summary:
        details["summary"] = summary
    db.update_session(
        conn, card_id, phase="details", details_json=json.dumps(details), claude_session_id=claude_session_id
    )
    publish(card_id, _turn_event(phase="details", details=details))


# The creating_prd -> creating_issues -> publishing chain, in order, paired
# with each phase's own default skill prompt -- shared by `_run_chain_from`
# below (issue #225, child of PRD #222): `/rhubarb:to-prd`/`/rhubarb:to-issues`
# are pure drafting steps with no GitHub side effects of their own;
# `/rhubarb:publish-to-github` runs last, in the same resumed conversation
# (so both drafts are still in its context), and is the one place that
# actually calls `gh issue create` for the PRD and every child issue.
_CHAIN_PHASES: list[tuple[str, str]] = [
    ("creating_prd", "/rhubarb:to-prd"),
    ("creating_issues", "/rhubarb:to-issues"),
    ("publishing", "/rhubarb:publish-to-github"),
]


async def _run_chain_from(
    card_id: int,
    conn,
    cwd: str | None,
    *,
    start_phase: str,
    start_prompt: str,
    model: str | None,
    effort: str | None,
) -> None:
    """Run the `_CHAIN_PHASES` cascade starting at `start_phase`, using
    `start_prompt` for that first step and each later phase's own default
    prompt thereafter -- shared by `advance_past_grilling` (start_phase=
    "creating_prd", the original `/rhubarb:to-prd` skill prompt),
    `retry_session_job` (a plain retry -- the same start_phase/prompt
    pairing a fresh run would use), and `continue_stalled_chain_step_job`
    (start_phase=the paused phase, start_prompt=the human's reply text).
    Stops the moment one step comes back not-ok -- already fully handled
    (an error, a crash routed to blocked, or a fresh stall) by
    `_run_chain_step` itself, nothing further to do here. Calls
    `_finish_chain` once every remaining phase succeeds."""
    start_index = next(i for i, (phase, _) in enumerate(_CHAIN_PHASES) if phase == start_phase)
    claude_session_id = None
    last_result = ""
    for i, (phase, default_prompt) in enumerate(_CHAIN_PHASES[start_index:], start=start_index):
        prompt = start_prompt if i == start_index else default_prompt
        row = db.get_session(conn, card_id)
        ok, claude_session_id, last_result = await _run_chain_step(
            card_id, conn, row, phase=phase, prompt=prompt, cwd=cwd, model=model, effort=effort
        )
        if not ok:
            return

    await _finish_chain(card_id, conn, claude_session_id, cwd, summary=last_result)


async def advance_past_grilling(card_id: int, cwd: str | None) -> None:
    """Grilling just finished: run /to-prd, /to-issues, /publish-to-github
    (live-streamed) via `_run_chain_from`. Uses the model already recorded
    on the row (set back when the session started grilling) -- this is a
    continuation of that same session, not a fresh one, so the configured
    model is not re-read here even if it's changed since.

    This card's resident `StreamJsonEngine` (if any; a card that finished
    grilling with zero questions on its very first turn never had one) is
    closed here first -- `_run_chain_step` then constructs a fresh one for
    `creating_prd`, reattaching via `--resume` at this row's
    `claude_session_id`, the real session id grilling's own `result` event
    handed back."""
    _close_stream_json_engine(card_id)
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    await _run_chain_from(
        card_id,
        conn,
        cwd,
        start_phase="creating_prd",
        start_prompt="/rhubarb:to-prd",
        model=row["model"],
        effort=row["effort"],
    )


async def continue_stalled_chain_step_job(card_id: int, reply: str, *, cwd: str | None) -> None:
    """Called from `POST /api/sessions/{card_id}/stall-reply` when the
    paused phase is `creating_prd`/`creating_issues`/`publishing` (issue
    #225, child of PRD #222). Resumes the chain at whichever phase paused
    (read straight off the row, not off `stalled_json`, since `phase` is
    already exactly that -- `_run_chain_step` never advances the row's
    `phase` past the step it's currently running), sending the human's
    `reply` as that step's turn prompt -- reattached via the row's existing
    `claude_session_id`, same resumable-turn pattern grilling's own reply
    flow already uses -- then cascades forward through the remaining phases
    exactly like a fresh run would."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    await _run_chain_from(
        card_id,
        conn,
        cwd,
        start_phase=row["phase"],
        start_prompt=reply,
        model=row["model"],
        effort=row["effort"],
    )


async def start_do_continue_job(card_id: int, prompt: str, *, cwd: str | None) -> None:
    """Start a fresh grilling round on a /do session sitting at
    `phase="details"` -- the Continue button on the do-finished banner
    (issue #198, child of PRD #195). Keeps the model/effort already recorded
    on the row unchanged (no re-prompt).

    Context-budget gate: if the row's `context_pct` is over
    `_DO_CONTINUE_CONTEXT_CUTOFF`, tears down the resident `StreamJsonEngine`
    and spawns a fresh one (persisting the new session id and resetting
    `context_pct` to None) so the new grilling round starts in a clean
    conversation. At or under the cutoff the existing engine/conversation is
    reused unchanged."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    if row is None:
        return

    model = row["model"]
    effort = row["effort"]

    context_pct = row["context_pct"]
    if context_pct is not None and context_pct > _DO_CONTINUE_CONTEXT_CUTOFF:
        _close_stream_json_engine(card_id)
        engine = await asyncio.to_thread(
            _spawn_fresh_stream_json_engine, cwd=cwd, model=model, effort=effort
        )
        _stream_json_engines[card_id] = engine
        db.update_session(conn, card_id, claude_session_id=engine.session_id, context_pct=None)
        row = db.get_session(conn, card_id)

    db.update_session(conn, card_id, phase="grilling")
    row = db.get_session(conn, card_id)

    await _run_grilling_turn_stream_json(
        card_id, conn, row, f"/rhubarb:grilling {prompt}", cwd=cwd, model=model, effort=effort,
        publish_when_empty=True,
    )


async def start_session_job(card_id: int, prompt: str, *, cwd: str | None) -> None:
    """A brand-new grilling session begins here -- read the currently
    configured model once, now, and use it for this session's entire
    lifetime (later turns read it back off the row instead of re-checking
    the setting).

    Effort is read off the row instead (not a fresh global lookup like
    model): `create_session` already seeded it from the request body or the
    global default, and re-fetching the row here (rather than trusting a
    value captured before this task was scheduled) picks up a same-session
    override made via `POST /api/session/{card_id}/effort` in the brief
    window between the row's creation and this task actually running --
    exactly the "still no turns sent yet" case that should apply immediately.

    `db.create_session` defaults every brand-new row's `phase` to
    `"grilling"`, and this function only ever runs on a session's very
    first turn -- so `row["phase"]` is always `"grilling"` here. Every
    grilling turn drives through `StreamJsonEngine`
    (`_run_grilling_turn_stream_json`) -- issue #225's migration off
    `PtyEngine` removed the only other transport this could ever have
    dispatched to."""
    conn = db.get_connection()
    model = db.get_model(conn)
    row = db.get_session(conn, card_id)
    await _run_grilling_turn_stream_json(
        card_id, conn, row, f"/rhubarb:grilling {prompt}", cwd=cwd, model=model, effort=row["effort"],
        publish_when_empty=True,
    )


async def continue_session_job(card_id: int, reply: str, *, cwd: str | None) -> None:
    """A grilling reply.

    Runs one more grilling CLI turn and publishes its `turn` event. Issue
    #223 (child of PRD #222): when that turn's parsed questions come back
    empty, `_run_grilling_turn_stream_json` itself now auto-advances
    straight into `advance_past_grilling` before returning -- there is no
    confirmation click in between anymore, and no `confirm_advance` flag on
    this function. A turn that still has open questions is unaffected: it
    publishes and waits for the user's next reply, exactly as before."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    await _run_grilling_turn_stream_json(
        card_id, conn, row, reply, cwd=cwd, model=row["model"], effort=row["effort"], publish_when_empty=True
    )


def _parse_qa_grilling_block(text: str) -> dict | None:
    """Extract the first JSON code block with phase=='qa_grilling' from a
    CLI turn result, as emitted by the /qa skill Phase 2. Returns None when
    no such block is found (normal /implement run without the /qa auto-handoff).

    Each candidate is run through `_rejoin_wrapped_json_strings` before
    `json.loads` -- see that function's docstring -- so a block that a
    terminal word-wrapped at some column (splitting a long string value
    across physical lines) still parses the same as it would unwrapped. A
    genuinely malformed/incomplete block still fails `json.loads` and is
    still skipped exactly as before."""
    for match in _FENCED_JSON_BLOCK_RE.finditer(text):
        try:
            data = json.loads(_rejoin_wrapped_json_strings(match.group(1)))
            if isinstance(data, dict) and data.get("phase") == "qa_grilling":
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    # Fallback: bare JSON (no code fence)
    try:
        data = json.loads(_rejoin_wrapped_json_strings(text.strip()))
        if isinstance(data, dict) and data.get("phase") == "qa_grilling":
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_implement_blocked_block(text: str) -> dict | None:
    """Extract the first JSON code block with phase=='implement_blocked' from
    a CLI turn result, as emitted by /rhubarb:implement when it genuinely
    cannot proceed without user-only information (see the skill's top-level
    "never pause to ask" directive). Returns None for the normal,
    not-blocked case -- structurally identical to `_parse_qa_grilling_block`,
    including running each candidate through `_rejoin_wrapped_json_strings`
    first (see that function's docstring) so a word-wrapped block still
    parses, while a genuinely malformed/incomplete one still returns None."""
    for match in _FENCED_JSON_BLOCK_RE.finditer(text):
        try:
            data = json.loads(_rejoin_wrapped_json_strings(match.group(1)))
            if isinstance(data, dict) and data.get("phase") == "implement_blocked":
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    try:
        data = json.loads(_rejoin_wrapped_json_strings(text.strip()))
        if isinstance(data, dict) and data.get("phase") == "implement_blocked":
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _read_tracker_file(cwd: str | None) -> dict | None:
    """Read `.claude/implement-tracker.json` from the project's working
    directory, written by the `/implement` skill's Phase 4. Returns `None`
    if the file is missing or isn't valid JSON -- a run that errored before
    Phase 4, or that finished but never wrote it, shouldn't crash the job."""
    if not cwd:
        return None
    path = Path(cwd) / ".claude" / "implement-tracker.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _launch_implement(conn, project_id: int, number: int, title: str, cwd: str | None) -> int:
    """Claim a pooled session (if any), create the session row, and fire the
    background `/implement` job -- the shared plumbing behind both an
    immediate PRD click and a queued PRD's turn coming up in serial mode.

    A fresh implement session begins here -- read the currently configured
    model once, now, and record it on the new row so `start_implement_job`
    (and any later retry of *this* row) uses it for the row's lifetime."""
    reused = db.claim_available_session(conn, project_id)
    resume_id = reused["claude_session_id"] if reused is not None else None
    model = db.get_model(conn)
    effort = db.get_effort(conn)

    row_id = db.create_session(
        conn,
        project_id,
        claude_session_id=resume_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": number, "title": title}},
        model=model,
        effort=effort,
    )

    asyncio.create_task(start_implement_job(row_id, number, cwd=cwd))

    return row_id


async def start_or_queue_implement(
    project_id: int, number: int, title: str, cwd: str | None, *, allow_queue: bool = True
) -> dict:
    """Decide whether a clicked PRD starts implementing right away or, in
    serial mode with another implement session already live for this
    project, gets enqueued to auto-start once that slot frees up.

    `allow_queue=False` (used by `afk_loop.check_once`) skips queuing
    entirely when another implement session is already active: the AFK
    opportunity is dropped outright rather than deferred, so a background
    timer decision never chains through a backlog the instant a slot frees
    up. A manually-clicked PRD (the default, `allow_queue=True`) is
    unaffected -- it still queues and drains as soon as the running session
    finishes.

    `db.get_parallel_implementation`/`_implement_queues` remain the only
    concurrency policy: with it on, every PRD (this one included) starts
    its own independent `StreamJsonEngine` tab immediately; with it off,
    only one implement tab runs per project at a time and everything else
    queues here, same as before issue #87 -- only the underlying per-session
    process model changed."""
    conn = db.get_connection()
    if db.has_active_implement_session(conn, project_id, number):
        return {"error": "Already implementing"}

    if not db.get_parallel_implementation(conn) and db.has_any_active_implement_session(conn, project_id):
        if not allow_queue:
            return {"skipped": True}
        _enqueue_implement(project_id, number, title)
        return {"queued": True}

    card_id = _launch_implement(conn, project_id, number, title, cwd)
    return {"card_id": card_id}


async def _drain_implement_queue(project_id: int, cwd: str | None) -> None:
    """Called right as a running implement session frees its "one running at
    a time" slot (serial mode). Starts at most one queued PRD -- that job's
    own completion will drain the one after it, in turn."""
    entry = _pop_next_implement(project_id)
    if entry is None:
        return

    conn = db.get_connection()
    _launch_implement(conn, project_id, entry["number"], entry["title"], cwd)


async def start_implement_job(card_id: int, prd_number: int, *, cwd: str | None) -> None:
    """Run a single `/implement prd: N` turn end to end: `implementing` while
    it's in flight, then `implemented` on success with details replaced by
    the tracker file's contents (falling back to the seeded PRD stub if the
    tracker file is missing), then pool the session exactly like the /do
    chain does. If the turn reports it's blocked instead (see
    `_finish_implement_turn`), the session is left suspended in `phase:
    blocked` for `continue_implement_job` to pick up."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    model = row["model"]
    effort = row["effort"]
    publish(card_id, {"type": "phase", "phase": "implementing"})

    try:
        turn = await _run_stream_json_turn(
            card_id,
            f"/rhubarb:implement prd: {prd_number}",
            session_id=row["claude_session_id"],
            cwd=cwd,
            model=model,
            effort=effort,
            phase="implementing",
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="implementing")
        return
    except ClaudeCLIError as e:
        # /implement's PRD-selection step calls `gh issue list` directly, so
        # a genuine gh auth failure is possible here -- classify it.
        _close_stream_json_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(
            card_id,
            _turn_event(
                phase="implementing",
                error=message,
                needs_github_login=bool(needs_login),
                card_id=card_id,
                project_id=row["project_id"],
            ),
        )
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "implementing", message)
        await _drain_implement_queue(row["project_id"], cwd)
        return

    if turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the
        # duplicate-call error itself; nothing more to do here.
        return

    await _finish_implement_turn(card_id, conn, row, turn, cwd=cwd, model=model, effort=effort)


# Issue #159: mirrors grilling's own corrective-retry prompt (see the
# sibling fix for `_run_grilling_turn`) -- hands the model its own
# unparseable QA-grilling round back verbatim and asks it to rewrite that
# same round in the exact required format, rather than silently treating an
# unparseable round as "no more QA questions".
_QA_CORRECTIVE_RETRY_PROMPT_TEMPLATE = (
    "Your last QA-grilling round did not parse into the required structured "
    "format. Here is exactly what you sent:\n\n{broken_text}\n\n"
    "Please rewrite that same round of QA questions in the exact required "
    'format (a `QA session for PRD N: "..."` header, one `Issue N: "..."` '
    'group per issue in tracker order, `Question N: "..."` lines numbered '
    'sequentially within each issue, and an optional `Recommended text: '
    '"..."` per question), and write it to `.claude/rhubarb_qa.md` exactly '
    "as the qa-grilling skill instructs."
)


# Issue #230: `ollama_rescue.should_attempt_qa_rescue`'s own trigger check,
# now local to this module -- it was only ever a cheap "does this look like
# it was trying" fingerprint gating whether to spend a follow-up call, never
# itself an extraction attempt, so it survives the retirement of the
# regex-parser + Ollama-rescue chain it used to gate.
_QA_ATTEMPT_TRIGGER = "qa session for prd"


def _looks_like_qa_attempt(text: str) -> bool:
    """True when `text` contains the `QA session for PRD` fingerprint
    (case-insensitive) the qa-grilling skill always emits when it's
    genuinely trying to produce a QA-grilling round."""
    return _QA_ATTEMPT_TRIGGER in text.lower()


async def _extract_qa_issues_via_skill(project_id: int, text: str) -> dict | None:
    """PRD #227 follow-up, gap 2 (`gh issue view 227`; discovered during
    manual testing/design review after #228/#229/#230 were already closed
    out in code): the QA-grilling analogue of
    `_extract_grilling_questions_via_parser_session` above. This used to be
    `_extract_qa_issues_via_parser_session`, which sent its own hand-rolled
    inline prompt (`_QA_EXTRACTION_PROMPT_TEMPLATE`, now deleted) directly
    via `parser_session.stream_turn`, bypassing the `/rhubarb:parse-
    interview` skill every other phase already went through -- a real
    inconsistency, flagged during design review and confirmed by the user to
    unify: the skill should handle both shapes. The skill's instructions
    (`rhubarb/claude_plugin/skills/parse-interview/SKILL.md`) now have their
    own `qa_grilling_issues`-phase section, ported over from
    `_QA_EXTRACTION_PROMPT_TEMPLATE`'s own rules (issue grouping/detection,
    the PRD header, the "no recognizable QA content" fallback) with no logic
    lost, plus the SAME Recommended:-line-mapping/options rules the flat
    shape already had -- so this is now just `parser_session.
    extract_with_validation` called with that phase and `ollama_rescue.
    _is_valid_qa_shape` as its validator, mirroring
    `_extract_grilling_questions_via_parser_session`'s own one-line shape.

    Deliberately its own phase string, `"qa_grilling_issues"`, NOT the
    `"qa_grilling"` phase name `_maybe_extract_needs_input_qa`'s `handle_
    turn_completed` call already uses to enqueue this exact same raw text
    onto this project's separate needs-input queue
    (`parser_session.enqueue_needs_input_turn`/`drain_needs_input_queue`).
    That queue's own consumer (`parser_session._process_one_queued_item`)
    expects the FLAT `{header, questions, footer}` shape for every phase it
    handles, `"qa_grilling"` included (see `tests/test_parser_session.py`'s
    `test_drain_respects_the_drain_then_clear_ceiling_mid_queue`, which
    asserts a flat-shape result for a `"qa_grilling"`-phase queued item) --
    reusing that same phase string here would have made the skill's own
    `phase:` branching ambiguous for one identical phase name used by two
    call sites wanting two different shapes. `"qa_grilling_issues"` is a new,
    unambiguous phase string that only this call site ever sends.

    `detect_mismatches=False`: issue #229's mismatch-detection/retry/tagging
    logic is built entirely around the flat shape's own single `"questions"`
    list and has not (yet) been generalized to this nested `issues[].
    questions` structure -- see `extract_with_validation`'s own docstring for
    why, and this PRD's report for why this is a deliberate, narrower scope
    rather than an oversight.

    Returns the parsed `{prd, issues, source: "parser_session"}` dict on a
    successful, schema-valid parse. Returns `None` on any failure -- a
    dead/missing parser session, a turn that produced no `result` event, or a
    response that isn't valid JSON matching the expected shape -- exactly
    like `_extract_questions_via_parser_session`'s own failure contract.
    Callers must treat `None` as "extraction failed", not as "found
    nothing"."""
    return await parser_session.extract_with_validation(
        project_id, text, phase="qa_grilling_issues", validator=_is_valid_qa_shape, detect_mismatches=False
    )


def _qa_result_is_suspicious(qa_parsed: dict, qa_file_text: str | None, terminal_text: str) -> bool:
    """True when extraction (`_extract_qa_issues_via_skill`) came
    back with no issues, but either the QA question file's raw content or
    the raw terminal text looks like the model was genuinely trying to
    produce a QA-grilling round rather than this being some other, unrelated
    turn -- `_looks_like_qa_attempt`'s trigger check, against *both* sources
    (issue #159 extends that check, which today only ever looks at terminal
    text, to the file too) rather than just one. A round with neither source
    showing this fingerprint is never retried -- that's the "no QA
    questions" case, indistinguishable from a genuine wrap-up, so it's left
    exactly as before this existed."""
    if qa_parsed["issues"]:
        return False
    if qa_file_text is not None and _looks_like_qa_attempt(qa_file_text):
        return True
    return _looks_like_qa_attempt(terminal_text)


async def _fail_qa_corrective_retry(card_id: int, conn, row, cwd: str | None, message: str) -> None:
    """Shared explicit-error tail for `_attempt_qa_corrective_retry`'s
    failure paths (see its docstring): closes this card's engine, persists
    `error_text`, publishes the error `turn` event (which feeds the
    app-wide error log via `_turn_event`/`error_log.log_error`), publishes
    `done`, records a background error notification, and drains this
    project's queued implement jobs -- the same shape every other
    `ClaudeCLIError` handler in this module already uses, since this
    implement session is ending here instead of handing off to QA."""
    _close_stream_json_engine(card_id)
    db.update_session(conn, card_id, error_text=message, needs_github_login=0)
    publish(
        card_id,
        _turn_event(
            phase="qa_grilling",
            error=message,
            needs_github_login=False,
            card_id=card_id,
            project_id=row["project_id"],
        ),
    )
    publish(card_id, {"type": "done"})
    add_error_notification(row["project_id"], card_id, "qa_grilling", message)
    await _drain_implement_queue(row["project_id"], cwd)


async def _maybe_extract_needs_input_qa(card_id: int, conn, row, text: str) -> dict | None:
    """Issue #178: the QA-grilling equivalent of
    `_maybe_extract_needs_input_grilling` -- the last-resort check run right
    before treating a QA round as genuinely having no more questions, after
    the existing file/terminal-text/parser-session-extraction chain (and,
    where attempted, the PRD #157-style corrective retry --
    `_attempt_qa_corrective_retry` here) has already concluded `text` carries
    no parseable issues.

    Asks the local Ollama needs-input classifier, via the shared
    `handle_turn_completed` hook (issue #191, phase `"qa_grilling"`), whether
    a human's input is actually still needed -- that hook also enqueues this
    turn onto the project's needs-input queue when it says yes. If so, makes
    one more attempt at structured extraction from the same `text` via
    `_extract_qa_issues_via_skill` -- the same parser-session/skill
    extraction the rest of this chain uses (issue #230 retired the Ollama-
    rescue extraction attempt this used to make here).

    Returns the extracted `{prd, issues}` dict only when *both* the
    classifier says input is needed *and* extraction actually produced at
    least one issue. Returns `None` in every other case, which callers
    treat exactly like a genuine "no more QA questions" conclusion,
    unchanged."""
    classification = await handle_turn_completed(card_id, conn, row, text, "qa_grilling")
    if classification is None or not classification.get("needs_input"):
        return None
    rescued = await _extract_qa_issues_via_skill(row["project_id"], text)
    if rescued is None or not rescued["issues"]:
        return None
    return rescued


async def _attempt_qa_corrective_retry(
    card_id: int, conn, row, *, broken_text: str, session_id: str, cwd: str | None, model, effort
) -> tuple[dict, dict] | None:
    """One corrective follow-up turn (issue #159), run once a QA-grilling
    round's result looked like it was genuinely trying to contain questions
    (`_qa_result_is_suspicious`) but failed to parse even via the
    parser-session extraction pipeline (issue #230 -- `qa_parser.
    parse_qa_response`/`ollama_rescue.rescue_qa_response` are gone; every
    extraction attempt in this function goes through `_extract_qa_issues_
    via_parser_session` instead). Hands the model its own unparseable output
    back (`broken_text`) and asks it to rewrite that round in the required
    format, then re-extracts the retry turn's own result the same way this
    module always has (file first, then terminal text).

    Returns `(qa_parsed, qa_turn)` -- the freshly re-parsed dict (guaranteed
    to carry at least one issue) and the completed retry turn -- on success.
    On any failure (the retry turn itself erroring or crashing, a duplicate
    in-flight turn, or a retry result that *still* fails to parse), this
    function has already reported an explicit error and fully wound down
    this card (see `_fail_qa_corrective_retry`/`_route_crash_to_blocked`) --
    callers must treat a `None` return as fully handled and simply return,
    exactly like every other `_run_stream_json_turn`-wrapping call site in
    this module."""
    prompt = _QA_CORRECTIVE_RETRY_PROMPT_TEMPLATE.format(broken_text=broken_text)

    try:
        retry_turn = await _run_stream_json_turn(
            card_id, prompt, session_id=session_id, cwd=cwd, model=model, effort=effort, phase="qa_grilling"
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="qa_grilling")
        return None
    except ClaudeCLIError as e:
        await _fail_qa_corrective_retry(card_id, conn, row, cwd, str(e))
        return None

    if retry_turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the duplicate-call error
        # itself; nothing more to do here.
        return None

    retry_file_text = read_question_file(cwd, _QA_QUESTION_FILE)
    retry_parsed = await _extract_qa_issues_via_skill(row["project_id"], retry_file_text) if retry_file_text is not None else None
    if retry_file_text is not None and (retry_parsed is None or not retry_parsed["issues"]):
        delete_question_file(cwd, _QA_QUESTION_FILE)
        retry_parsed = None
    if retry_parsed is None:
        retry_parsed = await _extract_qa_issues_via_skill(row["project_id"], retry_turn["full_text"])
    if retry_parsed is None:
        retry_parsed = {"prd": None, "issues": []}

    if not retry_parsed["issues"]:
        # Issue #178: the corrective retry also failed to produce anything
        # parseable via the existing chain -- before giving up, ask
        # Ollama's needs-input classifier whether this retry's own result
        # actually still needs a human's input and, if so, make one more
        # parser-session extraction attempt on it (issue #230 -- no more
        # Ollama rescue here either).
        rescued_via_classifier = await _maybe_extract_needs_input_qa(card_id, conn, row, retry_turn["full_text"])
        if rescued_via_classifier is not None:
            return rescued_via_classifier, retry_turn

        # The corrective retry ran, but its own result still didn't parse --
        # stop trying automatically and surface this loudly instead of
        # silently handing off a broken/empty QA round.
        message = (
            "A QA-grilling round could not be parsed into structured "
            "questions, even after asking the model to rewrite it in the "
            "required format. Check the session's console output for what "
            "the model actually sent."
        )
        await _fail_qa_corrective_retry(card_id, conn, row, cwd, message)
        return None

    return retry_parsed, retry_turn


async def _extract_implementing_question(project_id: int, text: str) -> dict | None:
    """Try to pull recognizable question/option content out of an
    implementing turn's rendered `text`, once `classify_needs_input` (issue
    #179, child of PRD #174) has already decided a human needs to read this
    turn before the session can usefully continue.

    Implementing has no structured question format of its own -- unlike
    grilling/QA-grilling, it never asks the model to write anything in the
    `Question N: "..."` protocol (its only structured output is the
    separate, pre-existing `implement_blocked` JSON marker, untouched by
    this function). So this reuses the exact same parser-session extraction
    pipeline grilling's own turn handling uses
    (`_extract_questions_via_parser_session`, phase `"implementing"` --
    matching this file's own established phase-name convention for implement
    turns, e.g. `_turn_event(phase="implementing", ...)`/
    `handle_turn_completed(..., "implementing")` elsewhere in this module) on
    the chance the model's prose still happens to contain one. Issue #230
    retired the `qa_parser.parse_grilling_response`-regex-then-Ollama-rescue
    chain this used to try first -- there is no regex pre-check and no
    Ollama-rescue fallback here anymore, identical to grilling's own
    extraction.

    Returns the parsed `{header, questions, footer, ...}` dict only when it
    actually carries at least one question -- `None` otherwise (a dead/
    missing parser session, an unparseable response, or a response with no
    recognizable question at all), so the caller can fall back to the
    generic reply panel instead of rendering an empty rich-question UI."""
    parsed = await _extract_questions_via_parser_session(project_id, text, phase="implementing")
    return parsed if parsed and parsed["questions"] else None


async def _finish_implement_turn(card_id: int, conn, row, turn: dict, *, cwd: str | None, model, effort) -> None:
    """Shared tail for both `start_implement_job`'s first turn and
    `continue_implement_job`'s resume turn: persist the turn's console
    text/context usage, check for the `implement_blocked` marker (leaving
    the session suspended in `phase: blocked` if found, with no `done` --
    same "suspended, not finished" shape as a QA session awaiting Perfect),
    and otherwise run the existing tracker-file/QA-handoff/pooling logic
    exactly as before this function existed."""
    console_text = row["console_text"] + "\n\n" + turn["full_text"] if row["console_text"] else turn["full_text"]
    db.update_session(conn, card_id, console_text=console_text, context_pct=turn.get("context_pct"))

    file_text = read_question_file(cwd, _IMPLEMENT_BLOCKED_FILE)
    blocked = None
    if file_text is not None:
        try:
            file_data = json.loads(file_text.strip())
        except (json.JSONDecodeError, ValueError):
            file_data = None
        if isinstance(file_data, dict) and file_data.get("phase") == "implement_blocked":
            blocked = file_data
        else:
            # Present but unparseable/wrong shape -- never let a corrupt
            # file wedge future turns; drop it and fall back to the
            # terminal-text scan below exactly as if it had never existed.
            delete_question_file(cwd, _IMPLEMENT_BLOCKED_FILE)
    if blocked is None:
        blocked = _parse_implement_blocked_block(turn["full_text"])
    if blocked is not None:
        db.update_session(
            conn,
            card_id,
            claude_session_id=turn["session_id"],
            phase="blocked",
            blocked_json=json.dumps(blocked),
            # Clear any stray needs-input state (issue #179) a prior turn in
            # this same session may have left behind -- this genuine
            # implement_blocked marker supersedes it.
            stalled_json=None,
            interview_json=None,
        )
        publish(card_id, _turn_event(phase="blocked", blocked=blocked))
        return

    # Issue #179 (child of PRD #174): a pure ADDITION after the existing
    # implement_blocked marker check above -- that check, and its own
    # marker-parsing logic, are completely untouched. This only runs once
    # the turn's result carries no blocked marker at all, exactly where
    # today the turn would just continue automatically. Ask the local
    # Ollama classifier, via the shared `handle_turn_completed` hook (issue
    # #191 -- which also enqueues this turn onto the project's needs-input
    # queue when it says yes), whether this turn actually needs a human's
    # input anyway (a question buried in ordinary prose, an approval
    # request, ...); a negative classification, a decline, or an
    # unavailable Ollama (`classify_needs_input` returns `None` for the
    # latter two, same as every other caller) all fall straight through to
    # today's unchanged automatic continuation below. This same path (and
    # its enqueue) also covers the parallel per-issue implementation
    # sessions `/implement` can run concurrently -- they run through this
    # exact `_finish_implement_turn` tail, just N at once.
    #
    # Also skipped entirely when this turn's result carries a `qa_grilling`
    # handoff marker (`_parse_qa_grilling_block`, checked below at its own
    # existing call site) -- that marker is itself a legitimate, already-
    # structured signal, and the QA-handoff logic further down (its own
    # regex/rescue/corrective-retry chain, including issue #178's own
    # `classify_needs_input` call on a genuine QA wrap-up) must get the
    # first and only look at this turn's text. Classifying here first would
    # otherwise intercept a real QA handoff before that logic ever runs.
    classification = None
    if _parse_qa_grilling_block(turn["full_text"]) is None:
        classification = await handle_turn_completed(card_id, conn, row, turn["full_text"], "implementing")
    if classification is not None and classification.get("needs_input"):
        extracted = await _extract_implementing_question(row["project_id"], turn["full_text"])
        if extracted is not None:
            # Recognizable question/option content -- render it with the
            # exact same rich question UI grilling's own rounds use.
            db.update_session(
                conn,
                card_id,
                claude_session_id=turn["session_id"],
                interview_json=json.dumps(extracted),
                stalled_json=None,
            )
            publish(card_id, _turn_event(phase="implementing", interview=extracted))
            return

        # Doesn't fit that structured shape -- fall back to the pre-existing
        # generic reply panel from PRD #168/#172 (`renderStalledSession`/
        # `sendStallReply` in prompt.html, backed by the `stalled`/
        # `stalled_context` fields on the `turn` event and the `stalled_json`
        # row column -- both already exist and are otherwise unused today
        # since `PtyEngine` retired its per-chunk stall mechanism, issue
        # #175). Reviving that dead path for this call site, the same way
        # issue #177 revives it for non-Q&A phases: this turn has already
        # completed (unlike the old mid-turn nudge this panel originally
        # served), so `/api/sessions/{card_id}/stall-reply` resumes an
        # implement-type session parked here through `continue_implement_job`
        # -- a real new turn -- instead of its original raw-PTY-write
        # behavior, which would go nowhere with no turn still reading the
        # PTY for it.
        reason = classification.get("reason") or "This turn may need your input before continuing."
        db.update_session(
            conn,
            card_id,
            claude_session_id=turn["session_id"],
            stalled_json=json.dumps({"phase": "implementing", "context": reason}),
            interview_json=None,
        )
        publish(card_id, _turn_event(phase="implementing", stalled=True, stalled_context=reason))
        return

    tracker = _read_tracker_file(cwd)
    if tracker is not None:
        details = tracker
    else:
        details = json.loads(row["details_json"]) if row["details_json"] else None

    db.update_session(
        conn,
        card_id,
        phase="implemented",
        blocked_json=None,
        # Clear any stray needs-input state (issue #179) a prior turn in
        # this same session may have left behind -- this turn resolved
        # cleanly, superseding it.
        stalled_json=None,
        interview_json=None,
        details_json=json.dumps(details) if details is not None else None,
    )

    qa_data = _parse_qa_grilling_block(turn["full_text"])
    if qa_data is not None:
        # /implement Phase 5 ran /qa, which replied with the nested
        # "QA session for PRD N: ..." question format (see the qa-grilling
        # skill and `_extract_qa_issues_via_skill` above) -- the
        # qa_grilling JSON block itself is now just a lightweight signal
        # ({phase, prd}) that this handoff happened; the actual issues/
        # questions are parsed from the turn's own free text.
        # Hand the session_id to the QA session instead of pooling it here
        # -- this card's own tab is done; the new QA row's own tab starts
        # fresh (reattached via --resume) on its own first turn.
        qa_prd = qa_data.get("prd")
        qa_file_text = read_question_file(cwd, _QA_QUESTION_FILE)
        qa_parsed = (
            await _extract_qa_issues_via_skill(row["project_id"], qa_file_text)
            if qa_file_text is not None
            else None
        )
        if qa_file_text is not None and (qa_parsed is None or not qa_parsed["issues"]):
            # Present but unparseable -- never let a corrupt file wedge
            # every future round; drop it and fall back to the terminal-text
            # path below exactly as if it had never been written.
            delete_question_file(cwd, _QA_QUESTION_FILE)
            qa_parsed = None
        if qa_parsed is None:
            qa_parsed = await _extract_qa_issues_via_skill(row["project_id"], turn["full_text"])
        if qa_parsed is None:
            # Issue #230: the parser-session extraction pipeline is the sole
            # extraction mechanism now -- no regex pre-check, no Ollama-
            # rescue fallback. A failed/empty extraction here is treated
            # exactly like a genuine "no issues yet" result; the suspicious-
            # result corrective retry right below is what decides whether
            # this is worth one more attempt.
            qa_parsed = {"prd": None, "issues": []}

        qa_turn = turn
        if _qa_result_is_suspicious(qa_parsed, qa_file_text, turn["full_text"]):
            # Issue #159: this round looks like it was genuinely trying to
            # contain QA questions -- rather than silently treating this as
            # "no more QA questions" (indistinguishable from a genuine
            # wrap-up otherwise), give the model one corrective follow-up
            # turn with its own unparseable output before giving up for
            # real.
            broken_text = qa_file_text if qa_file_text is not None else turn["full_text"]
            retry_result = await _attempt_qa_corrective_retry(
                card_id,
                conn,
                row,
                broken_text=broken_text,
                session_id=turn["session_id"],
                cwd=cwd,
                model=model,
                effort=effort,
            )
            if retry_result is None:
                # Already fully handled (explicit error published/persisted,
                # engine closed, queue drained) -- see
                # `_attempt_qa_corrective_retry`'s docstring.
                return
            qa_parsed, qa_turn = retry_result
            retried_console_text = console_text + "\n\n" + qa_turn["full_text"]
            db.update_session(conn, card_id, console_text=retried_console_text, context_pct=qa_turn.get("context_pct"))
        elif not qa_parsed["issues"]:
            # Issue #178: the chain concluded no QA questions without even
            # attempting the corrective retry above (not suspicious) -- same
            # last-resort Ollama needs-input check before treating this as a
            # genuine "no more QA questions" wrap-up.
            rescued_via_classifier = await _maybe_extract_needs_input_qa(card_id, conn, row, turn["full_text"])
            if rescued_via_classifier is not None:
                qa_parsed = rescued_via_classifier

        qa_issues = qa_parsed["issues"]
        qa_row_id = db.create_session(
            conn,
            row["project_id"],
            claude_session_id=qa_turn["session_id"],
            session_type="qa",
            phase="qa_grilling",
            details={"prd": qa_prd},
            model=model,
            effort=effort,
        )
        _close_stream_json_engine(card_id)
        publish(card_id, {"type": "qa_started", "qa_card_id": qa_row_id})
        publish(card_id, _turn_event(phase="implemented", details=details))
        publish(card_id, {"type": "done"})
        await _drain_implement_queue(row["project_id"], cwd)
        asyncio.create_task(start_qa_job(qa_row_id, qa_prd, qa_issues, cwd=cwd))
    else:
        # See the matching comment in _auto_continue_implement_and_qa --
        # model/effort here match a brand-new /do's own resolution, not
        # this finishing implement session's own model/effort.
        new_session_id = await _clear_for_reuse(
            card_id, project_id=row["project_id"], cwd=cwd, model=db.get_model(conn), effort=db.DEFAULT_EFFORT
        )
        db.mark_session_available(conn, card_id, new_session_id)
        publish(card_id, _turn_event(phase="implemented", details=details))
        publish(card_id, {"type": "done"})
        await _drain_implement_queue(row["project_id"], cwd)


async def continue_implement_job(card_id: int, reply: str, *, cwd: str | None) -> None:
    """Called from POST /api/sessions/{card_id}/implement-reply. Resumes a
    `phase: blocked` session with the user's reply as the next turn's
    prompt, then runs it through the exact same `_finish_implement_turn`
    tail as the original turn -- if the reply doesn't fully unblock it, the
    result is another `implement_blocked` marker and the session stays
    suspended for another reply (a natural loop: each reply is an
    independent call, nothing here needs an explicit loop construct).

    This is the blocked question being answered -- delete its
    `.claude/rhubarb_blocked.json` (PRD #123) if one is still there, now
    that it's served its purpose."""
    delete_question_file(cwd, _IMPLEMENT_BLOCKED_FILE)

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    model = row["model"]
    effort = row["effort"]

    db.update_session(conn, card_id, phase="implementing", error_text=None)
    publish(card_id, {"type": "phase", "phase": "implementing"})

    try:
        turn = await _run_stream_json_turn(
            card_id, reply, session_id=row["claude_session_id"], cwd=cwd, model=model, effort=effort,
            phase="implementing",
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="implementing")
        return
    except ClaudeCLIError as e:
        _close_stream_json_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(
            card_id,
            _turn_event(
                phase="implementing",
                error=message,
                needs_github_login=bool(needs_login),
                card_id=card_id,
                project_id=row["project_id"],
            ),
        )
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "implementing", message)
        await _drain_implement_queue(row["project_id"], cwd)
        return

    if turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the
        # duplicate-call error itself; nothing more to do here.
        return

    row = db.get_session(conn, card_id)
    await _finish_implement_turn(card_id, conn, row, turn, cwd=cwd, model=model, effort=effort)


async def start_qa_job(card_id: int, prd: dict | None, issues: list[dict], *, cwd: str | None) -> None:
    """Publish the qa_grilling turn event for a QA session created by the
    /implement auto-handoff. `prd`/`issues` were already parsed from the
    implement turn's result (the qa_grilling JSON marker for `prd`,
    `_extract_qa_issues_via_skill` for the nested `issues`/
    `questions`); emit them and leave the session suspended (no 'done')
    until POST /api/session/qa-complete is called."""
    publish(card_id, {"type": "phase", "phase": "qa_grilling"})
    publish(card_id, {
        "type": "turn",
        "phase": "qa_grilling",
        "prd": prd,
        "issues": issues,
        "interview": None,
        "details": None,
        "error": None,
        "needs_github_login": False,
        "blocked": None,
    })


async def continue_qa_job(card_id: int, answers: dict, extra_notes: str, *, cwd: str | None) -> None:
    """Called from POST /api/session/qa-complete. Unblocks the QA session by
    running Phase 3+ with the user's per-question answers (keyed by question
    id, as published in start_qa_job's `issues`) and the trailing free-text
    box's content forwarded as context.

    This is the QA round being answered -- delete its `.claude/rhubarb_qa.md`
    (PRD #123) if one is still there, now that it's served its purpose."""
    delete_question_file(cwd, _QA_QUESTION_FILE)

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    model = row["model"]
    effort = row["effort"]

    db.update_session(conn, card_id, phase="qa_closing", error_text=None)
    publish(card_id, {"type": "phase", "phase": "qa_closing"})

    answer_lines = "\n".join(f"- {qid}: {text}" for qid, text in answers.items() if text and text.strip())
    context_parts = []
    if answer_lines:
        context_parts.append(f"Answers:\n{answer_lines}")
    if extra_notes.strip():
        context_parts.append(f"Additional notes: {extra_notes}")
    note_ctx = "\n" + "\n".join(context_parts) if context_parts else ""

    prompt = (
        f"The user reviewed the implementation and clicked Perfect.{note_ctx}\n\n"
        "Please continue from Phase 3: close all child issues, close the parent PRD, "
        "clear the tracker, commit, push, and run the final /clear."
    )

    try:
        turn = await _run_stream_json_turn(
            card_id, prompt, session_id=row["claude_session_id"], cwd=cwd, model=model, effort=effort,
            phase="qa_closing",
        )
    except StreamJsonEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="qa_closing")
        return
    except ClaudeCLIError as e:
        # /qa's closing step calls `gh issue close`/`gh issue edit` directly,
        # so a genuine gh auth failure is possible here -- classify it.
        _close_stream_json_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(
            card_id,
            _turn_event(
                phase="qa_closing",
                error=message,
                needs_github_login=bool(needs_login),
                card_id=card_id,
                project_id=row["project_id"],
            ),
        )
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "qa_closing", message)
        return

    if turn is None:
        # Issue #144/#149: a genuine turn for this card_id is already in
        # flight -- `_run_stream_json_turn` already published the duplicate-call error
        # itself; nothing more to do here.
        return

    context_pct = turn.get("context_pct")
    db.update_session(conn, card_id, claude_session_id=turn["session_id"], context_pct=context_pct)

    # Recycle a low-usage finished session instead of leaving it dead weight:
    # an unknown context_pct is treated as "has headroom" (same
    # safe-by-default stance as `_context_window_pct`/`_maybe_clear_for_next_phase`),
    # so it's marked available too. Either way, this card's tab is done --
    # a recycled row's `claude_session_id` is picked up later by a brand-new
    # card_id's own fresh tab, not this one.
    if context_pct is None or context_pct < _QA_DONE_RECYCLE_CONTEXT_CUTOFF:
        db.mark_session_available(conn, card_id, turn["session_id"])
        # Unlike the other two pooling sites, this one never spawns a fresh
        # engine to reuse as a standby (it just recycles this card's own
        # already-resident tab's id) -- ensure one gets warmed up separately.
        asyncio.create_task(
            ensure_standby_stream_json_engine(
                row["project_id"], cwd=cwd, model=db.get_model(conn), effort=db.DEFAULT_EFFORT
            )
        )
    _close_stream_json_engine(card_id)

    publish(card_id, _turn_event(phase="qa_closing"))
    publish(card_id, {"type": "done"})


async def retry_session_job(card_id: int, cwd: str | None) -> None:
    """Resume a session left in `creating_prd` or `creating_issues` -- either
    because that phase errored (e.g. GitHub auth) or because the app process
    restarted mid-phase. Resumes from that phase and completes the rest live,
    matching what a fresh run through the chain would have done.

    An errored `implement` session is handled differently: its own
    `claude_session_id` may belong to a CLI turn that died mid-flight, so
    resuming it isn't safe. Instead this reads the PRD it was implementing
    out of `details_json` and hands off to `start_or_queue_implement`, the
    same entry point a fresh PRD-list click uses -- a brand-new session row
    is created (and queued instead of started immediately if serial mode is
    on and another implement session is already live), while the original
    errored row is left exactly as it is, kept around as history."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)

    if row["session_type"] == "implement":
        details = json.loads(row["details_json"]) if row["details_json"] else None
        prd = details.get("prd") if details else None
        if prd is None:
            return
        await start_or_queue_implement(row["project_id"], prd["number"], prd.get("title", ""), cwd)
        return

    if row["phase"] == "creating_prd":
        # `advance_past_grilling` also closes this card's resident
        # `StreamJsonEngine` first -- correct here too: a row stuck at
        # `creating_prd` may still have a leftover grilling-phase engine
        # resident (e.g. an app restart caught it before that close ever
        # ran), and `_run_chain_step` always constructs its own fresh tab
        # for the phase it's about to run regardless.
        await advance_past_grilling(card_id, cwd)
    elif row["phase"] in ("creating_issues", "publishing"):
        default_prompt = dict(_CHAIN_PHASES)[row["phase"]]
        await _run_chain_from(
            card_id, conn, cwd, start_phase=row["phase"], start_prompt=default_prompt,
            model=row["model"], effort=row["effort"],
        )
