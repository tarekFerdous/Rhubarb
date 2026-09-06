"""Per-session background jobs: run CLI turns off the event loop thread and
stream their translated output live via `rhubarb.live_stream`.

Each job publishes app-level events for a session's card_id: `phase` at the
start of each step, `text`/`action`/`usage` as a turn streams, a richer
`turn` event once a turn's semantics (interview/details/error) are known,
and `done` once the session has reached a terminal state (details or an
unrecoverable error).

Every phase (`do`, `to-prd`, `to-issues`, `implement`, `qa`) drives its turns
through a single resident `PtyEngine` "tab" per `card_id` -- see
`_pty_engines` below -- instead of the old subprocess-per-turn
`cli_client.run_prompt`/`stream_prompt` model (issue #87).
"""

import asyncio
import json
import re
from pathlib import Path

_FENCED_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")

from rhubarb import db
from rhubarb.cli_client import ClaudeCLIError
from rhubarb.github_publisher import GithubPublishError, publish_draft
from rhubarb.live_stream import publish
from rhubarb.pty_engine import PtyEngine, PtyEngineUnrecoverableError
from rhubarb.qa_parser import parse_grilling_response, parse_qa_response
from rhubarb.stream_translate import translate_event

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
# against the relevant cutoff below and starts a fresh PtyEngine first if
# it's over. Both are pre-phase gates only -- a phase already running is
# never interrupted even if it crosses its cutoff while in flight.
# Placeholders pending real /rhubarb:implement and /rhubarb:qa context-growth
# telemetry (this repo's own measurements only ever covered the /do chain)
# -- expect these to move.
#
# NOTE (issue #87): `PtyEngine`'s turn-complete-marker protocol carries no
# usage/token-count data (interactive mode has no `--output-format
# stream-json`-style `usage`/`modelUsage` fields the way headless `-p` did),
# so `_context_window_pct` now always returns `None` for a turn driven
# through `PtyEngine`, and every check below always falls into its
# treat-as-safe/has-headroom branch. This is an inherent consequence of the
# marker-based interactive protocol, not something this issue changes the
# shape of -- the gate and the recycle cutoff are left in place exactly as
# they already handle an unknown `context_pct` (safe-by-default), so a
# future engine enhancement that recovers usage data would make them live
# again with no further changes needed here.
_DO_TO_IMPLEMENT_CONTEXT_CUTOFF = 0.40
_IMPLEMENT_TO_QA_CONTEXT_CUTOFF = 0.68

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
    `usage`/`modelUsage` block, or a zero/missing `contextWindow` -- always
    true of a `PtyEngine`-driven turn, see the module-level note above)
    rather than raising -- a session with an unknown context usage is
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


# One resident PtyEngine ("tab") per card_id, kept alive across every turn
# for that card's session -- across grilling -> to-prd -> to-issues ->
# implement, since those all continue the same claude conversation on the
# same card_id today (see `_auto_continue_implement_and_qa`). Replaces
# cli_client's old `_persistent_processes` pool: that pool was also
# card_id-keyed, but only ever used by the /do chain, and only kept a
# `-p`-style subprocess open (not a true interactive PTY) with its own
# ad hoc fallback/respawn-with-`--resume` logic on death. Every phase now
# shares this one mechanism instead -- see `_get_or_create_engine`.
_pty_engines: dict[int, PtyEngine] = {}


def _get_or_create_engine(
    card_id: int, *, cwd: str | None, model: str | None, effort: str | None, resume_session_id: str | None
) -> PtyEngine:
    """Return this card's resident tab, constructing and starting one (fresh,
    or reattached via `--resume resume_session_id` -- e.g. a reused pooled
    session, or a session picked back up after a Rhubarb restart) if this is
    the first turn for this `card_id`. Every later turn for the same
    `card_id` reuses the exact same `PtyEngine` instance -- never recreated
    per turn."""
    engine = _pty_engines.get(card_id)
    if engine is not None:
        return engine
    engine = PtyEngine(cwd=cwd, model=model, effort=effort, resume_session_id=resume_session_id)
    engine.start()
    _pty_engines[card_id] = engine
    return engine


def _close_engine(card_id: int) -> None:
    """Close and forget this card's resident tab, if any -- called whenever
    a card's tab finishes: pooled for reuse, fully done, handed off to a
    different card_id (the /implement -> /qa auto-handoff), or errored out
    (a later retry reattaches a fresh tab via `--resume` instead of
    continuing to drive a process that just raised)."""
    engine = _pty_engines.pop(card_id, None)
    if engine is not None:
        engine.close()


def open_pty_tab_count() -> int:
    """How many `PtyEngine` "tabs" are currently resident (issue #88) --
    one per `card_id` with a live entry in `_pty_engines`, across every
    active session regardless of phase. Backs the web UI's tab-count
    indicator next to the "Sessions" label (`GET /api/pty-tabs/count` in
    `rhubarb/web/app.py`); polled rather than pushed since it's a global
    count, not scoped to any one card's SSE stream."""
    return len(_pty_engines)


def _spawn_fresh_engine(*, cwd: str | None, model: str | None, effort: str | None) -> PtyEngine:
    """Construct and start a genuinely fresh `PtyEngine` -- no
    `resume_session_id` -- a brand-new, empty conversation. Used everywhere
    this module used to call `cli_client.clear_session`: issue #87 replaces
    sending the literal text "/clear" to a resident process with tearing
    that tab down and starting an actually-fresh one, since the whole point
    of clearing is to reclaim context by starting over, not to keep talking
    to the same process. Blocking (real spawn is a subprocess call) --
    callers run this via `asyncio.to_thread`."""
    engine = PtyEngine(cwd=cwd, model=model, effort=effort)
    engine.start()
    return engine


async def _maybe_clear_for_next_phase(card_id: int, conn, row, *, cwd: str | None, cutoff: float) -> str:
    """The context-window budget gate: called right before starting the next
    phase in a session chain (currently `/rhubarb:do` -> `/rhubarb:implement` at
    `_DO_TO_IMPLEMENT_CONTEXT_CUTOFF`, `/rhubarb:implement` -> `/rhubarb:qa` at
    `_IMPLEMENT_TO_QA_CONTEXT_CUTOFF` -- wired in by whichever caller is
    orchestrating that transition).

    Reads `row["context_pct"]` (persisted after the previous phase's last
    turn) rather than measuring anything fresh: an unknown value (`None`,
    e.g. no turn has completed yet, or a `PtyEngine`-driven turn's usage
    fields are unavailable -- see the module-level note above) is treated
    as safe to continue, same as `_context_window_pct`'s own
    None-on-uncertainty behavior.

    At or under `cutoff`: returns the row's existing `claude_session_id`
    unchanged -- the next phase continues in the same tab, no fresh engine.

    Over `cutoff`: tears down this card's resident tab and starts a
    genuinely fresh one (see `_spawn_fresh_engine`) *for this same
    card_id* -- the next phase continues right on in the new tab, it's just
    talking to an empty conversation instead of the old one. Persists the
    new session id and resets `context_pct` to `None` on the row (the fresh
    tab starts with an empty, unmeasured context again). This is a
    pre-phase gate only -- once the next phase is running, it is never
    interrupted mid-run even if it goes on to cross `cutoff` itself."""
    context_pct = row["context_pct"]
    if context_pct is None or context_pct <= cutoff:
        return row["claude_session_id"]

    _close_engine(card_id)
    engine = await asyncio.to_thread(_spawn_fresh_engine, cwd=cwd, model=row["model"], effort=row["effort"])
    _pty_engines[card_id] = engine
    new_session_id = engine.claude_session_id
    db.update_session(conn, card_id, claude_session_id=new_session_id, context_pct=None)
    return new_session_id


async def _clear_for_reuse(card_id: int, *, cwd: str | None, model: str | None, effort: str | None) -> str:
    """Reclaim context by starting a genuinely fresh conversation (see
    `_spawn_fresh_engine`), for a card whose row is about to be pooled
    (`db.mark_session_available`) for reuse under a *different*, future
    card_id -- this card's own tab is closed for good here, since nothing
    will ever run another turn against this `card_id` again. The future
    reuse constructs its own fresh tab, reattaching via `--resume` at
    whatever id this returns."""
    _close_engine(card_id)
    engine = await asyncio.to_thread(_spawn_fresh_engine, cwd=cwd, model=model, effort=effort)
    engine.close()
    return engine.claude_session_id


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


def _blocked_payload_from_crash(exc: PtyEngineUnrecoverableError) -> dict:
    """Synthesize the same shape of payload a genuine `implement_blocked`
    marker produces (see `_parse_implement_blocked_block`) out of a
    `PtyEngineUnrecoverableError` -- issue #87's crash-routing requirement:
    a turn that dies twice in a row (see that exception's docstring) is
    routed into the exact same suspend-and-wait-for-a-human-reply mechanism
    a genuine blocked marker already uses, rather than a new UI/error path."""
    return {
        "phase": "implement_blocked",
        "issue": None,
        "question": (
            "This session's connection to Claude crashed twice in a row and "
            "could not recover automatically. Reply below to try to continue."
        ),
        "context": str(exc),
    }


async def _route_crash_to_blocked(card_id: int, conn, exc: PtyEngineUnrecoverableError, *, phase: str) -> None:
    """A `stream_turn` call raised `PtyEngineUnrecoverableError` (its
    underlying process died twice in a row and gave up). Suspend this
    session exactly like a genuine `implement_blocked` marker would: `phase:
    blocked`, `blocked_json` set, a `turn` event carrying it -- so the same
    reply flow that already resumes a blocked implement session
    (`continue_implement_job`) picks this up too, regardless of which phase
    hit the crash. The dead tab is dropped from the registry so the next
    turn (that reply) constructs a fresh one, reattaching via `--resume` at
    `exc.claude_session_id`."""
    _close_engine(card_id)
    blocked = _blocked_payload_from_crash(exc)
    db.update_session(
        conn, card_id, claude_session_id=exc.claude_session_id, phase="blocked", blocked_json=json.dumps(blocked)
    )
    publish(card_id, _turn_event(phase="blocked", blocked=blocked))


async def _run_turn(
    card_id: int,
    prompt: str,
    *,
    session_id: str | None,
    cwd: str | None,
    model: str | None = None,
    effort: str | None = None,
) -> dict:
    """Run one turn against this card's resident `PtyEngine` tab (see
    `_get_or_create_engine` -- constructed and started on the first call for
    this `card_id`, reattached via `--resume session_id` if one is already
    known, and reused unchanged on every later call for the same
    `card_id`), streaming translated events into the session's live buffer
    as they arrive. Returns the raw `result` event's translated boundary
    marker ({"result", "session_id", "is_error"}) once the turn finishes;
    raises `ClaudeCLIError` on an ordinary failure, or propagates
    `PtyEngineUnrecoverableError` unwrapped -- callers route that into the
    blocked-card flow (see `_route_crash_to_blocked`) instead of treating it
    like a plain `ClaudeCLIError`.
    """
    holder: dict = {}

    async def runner():
        engine = _get_or_create_engine(card_id, cwd=cwd, model=model, effort=effort, resume_session_id=session_id)
        async for raw_event in engine.stream_turn(prompt):
            translated = translate_event(raw_event)
            if translated is None:
                continue
            if translated["type"] == "turn":
                translated["context_pct"] = _context_window_pct(raw_event)
                holder["turn"] = translated
                continue
            publish(card_id, translated)

    try:
        await runner()
    except PtyEngineUnrecoverableError:
        raise
    except ClaudeCLIError as e:
        holder["error"] = e
    except Exception as e:
        # Anything unexpected (a malformed raw event, a bug in translation)
        # must still resolve into a recorded session error, not an
        # unhandled exception on the fire-and-forget asyncio task -- that
        # would leave the card stuck in its in-flight phase silently
        # instead of surfacing the failure to the user.
        holder["error"] = ClaudeCLIError(str(e))

    if "error" in holder:
        raise holder["error"]
    return holder["turn"]


def _turn_event(
    *, phase: str, interview=None, details=None, error=None, needs_github_login=False, blocked=None
) -> dict:
    return {
        "type": "turn",
        "phase": phase,
        "interview": interview,
        "details": details,
        "error": error,
        "needs_github_login": needs_github_login,
        "blocked": blocked,
    }


async def _run_grilling_turn(
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
    """Run one grilling-phase turn and publish its `turn` event. Returns the
    parsed interview dict if grilling is still ongoing, `None` if this turn
    finished grilling (no more questions) or failed.

    `model` is the model this *session* (not just this turn) was created
    with -- see `start_session_job`/`continue_session_job` for where it comes
    from. It's persisted back onto the row alongside the other per-turn
    fields so a later call (a follow-up reply, a retry) can keep reading it
    off the row instead of re-checking the current setting.

    `publish_when_empty` covers `start_session_job`: a brand-new session's
    very first turn must always render (even a bare header with no
    structured questions), matching the old behavior of always surfacing
    `parse_grilling_response`'s result on start. `continue_session_job` now
    always passes `True` too -- zero questions there means grilling is done
    and the turn's header is the assistant's wrap-up message, which the
    frontend renders as a "ready to proceed?" gate rather than the chain
    auto-advancing on its own."""
    publish(card_id, {"type": "phase", "phase": "grilling"})

    try:
        turn = await _run_turn(
            card_id,
            prompt,
            session_id=row["claude_session_id"],
            cwd=cwd,
            model=model,
            effort=effort,
        )
    except PtyEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="grilling")
        return None
    except ClaudeCLIError as e:
        # Grilling's Claude turn never calls `gh` (see .claude/skills' own
        # instructions), so any match here would be a false positive --
        # always report no GitHub login is needed.
        _close_engine(card_id)
        message = str(e)
        db.update_session(
            conn, card_id, model=model, effort=effort, error_text=message, needs_github_login=0
        )
        publish(card_id, _turn_event(phase="grilling", error=message, needs_github_login=False))
        return None

    parsed = parse_grilling_response(turn["result"])
    console_text = row["console_text"] + "\n\n" + turn["result"] if row["console_text"] else turn["result"]
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

    return parsed if parsed["questions"] else None


async def _run_chain_step(
    card_id: int, conn, row, *, phase: str, prompt: str, cwd: str | None, model: str | None, effort: str | None
) -> tuple[bool, str | None]:
    """Run one /to-prd or /to-issues step, live-streamed. Returns (ok, claude_session_id).

    On failure, publishes the error `turn` event and `done` itself -- the
    chain stops here exactly as the old blocking version did.
    """
    db.update_session(conn, row["id"], phase=phase, error_text=None, needs_github_login=0)
    publish(card_id, {"type": "phase", "phase": phase})

    try:
        turn = await _run_turn(
            card_id,
            prompt,
            session_id=row["claude_session_id"],
            cwd=cwd,
            model=model,
            effort=effort,
        )
    except PtyEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase=phase)
        return False, None
    except ClaudeCLIError as e:
        # /to-prd and /to-issues are explicitly forbidden from calling `gh`
        # (see their skill instructions), so any match here would be a
        # false positive -- always report no GitHub login is needed.
        _close_engine(card_id)
        message = str(e)
        db.update_session(conn, row["id"], error_text=message, needs_github_login=0)
        publish(card_id, _turn_event(phase=phase, error=message, needs_github_login=False))
        publish(card_id, {"type": "done"})
        return False, None

    db.update_session(
        conn,
        row["id"],
        claude_session_id=turn["session_id"],
        console_text=row["console_text"] + "\n\n" + turn["result"],
        context_pct=turn.get("context_pct"),
    )
    return True, turn["session_id"]


async def _run_publish_step(card_id: int, conn, row, *, cwd: str | None) -> bool:
    """Run the publishing step: reads `.claude/prd_draft.json` and calls
    `github_publisher.publish_draft()` in a background thread -- a purely
    scripted `gh` operation, no Claude CLI turn involved. Returns True on
    success, having appended the publisher's result text to `console_text`
    so `_finish_chain`'s `parse_details()` can parse PRD/issue numbers
    unchanged. Returns False on failure, having published the error `turn`
    and `done` itself here -- exactly like `_run_chain_step` does."""
    db.update_session(conn, row["id"], phase="publishing", error_text=None, needs_github_login=0)
    publish(card_id, {"type": "phase", "phase": "publishing"})

    draft_path = Path(cwd) / ".claude" / "prd_draft.json" if cwd else Path(".claude/prd_draft.json")

    try:
        result_text = await asyncio.to_thread(publish_draft, draft_path, cwd)
    except GithubPublishError as e:
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, row["id"], error_text=message, needs_github_login=needs_login)
        publish(card_id, _turn_event(phase="publishing", error=message, needs_github_login=bool(needs_login)))
        publish(card_id, {"type": "done"})
        return False

    db.update_session(
        conn,
        row["id"],
        console_text=row["console_text"] + "\n\n" + result_text if row["console_text"] else result_text,
    )
    return True


async def _finish_chain(card_id: int, conn, claude_session_id: str, cwd: str | None) -> None:
    """`/rhubarb:do` just reached `details` (PRD + issues published). Publishes
    the `details` turn event, then hands off to `_auto_continue_implement_and_qa`
    instead of the old immediate `/clear`-and-pool -- that function decides
    whether to continue in this same tab or start a fresh one (the
    context-window budget gate), and starts `/rhubarb:implement` automatically."""
    row = db.get_session(conn, card_id)
    details = parse_details(row["console_text"])
    db.update_session(
        conn, card_id, phase="details", details_json=json.dumps(details), claude_session_id=claude_session_id
    )

    publish(card_id, _turn_event(phase="details", details=details))

    await _auto_continue_implement_and_qa(card_id, conn, cwd)


async def _auto_continue_implement_and_qa(card_id: int, conn, cwd: str | None) -> None:
    """Continue a session past `details` straight into `/rhubarb:implement`
    (and, if that phase's own Phase 5 hands off to `/qa`, into that too --
    already-automatic today via `_parse_qa_grilling_block`/`start_qa_job`,
    unchanged here) instead of pooling the session for a later manual PRD
    click.

    Publishes `minimize` before doing anything else so the frontend frees
    the left card for a new `/do` immediately -- this session's `phase`/
    `turn`/`done` events keep flowing exactly as they do for a manually
    started `/implement`, so the existing right-side panel plumbing renders
    it with no special case beyond handling `minimize` itself.

    Unlike the old subprocess-per-turn model, this card's resident tab (see
    `_pty_engines`) is left running across this transition when a PRD was
    found -- `/rhubarb:implement` continues in the exact same tab as the /do
    chain that led here, just under a new `session_type`/`phase` on the row;
    only the context-window budget gate (`_maybe_clear_for_next_phase`) ever
    tears it down and starts fresh, same as for any other phase transition.

    Falls back to the old pool-and-wait behavior if `parse_details` in
    `_finish_chain` came up with no PRD number to implement, rather than
    getting the session stuck mid-transition -- this path pools the session
    (and does close this card's tab; see `_clear_for_reuse`), since nothing
    else is going to run on this `card_id` until a human picks a PRD."""
    row = db.get_session(conn, card_id)
    details = json.loads(row["details_json"]) if row["details_json"] else None
    prd = details.get("prd") if details else None

    if prd is None:
        new_session_id = await _clear_for_reuse(card_id, cwd=cwd, model=row["model"], effort=row["effort"])
        db.mark_session_available(conn, card_id, new_session_id)
        publish(card_id, {"type": "done"})
        return

    publish(card_id, {"type": "minimize"})

    session_id = await _maybe_clear_for_next_phase(
        card_id, conn, row, cwd=cwd, cutoff=_DO_TO_IMPLEMENT_CONTEXT_CUTOFF
    )
    db.update_session(
        conn, card_id, phase="implementing", session_type="implement", claude_session_id=session_id
    )
    await start_implement_job(card_id, prd["number"], cwd=cwd)


async def advance_past_grilling(card_id: int, cwd: str | None) -> None:
    """Grilling just finished: run /to-prd, /to-issues (live-streamed), then
    auto-/clear and pool the session. Uses the model already recorded on the
    row (set back when the session started grilling) -- this is a
    continuation of that same session, not a fresh one, so the configured
    model is not re-read here even if it's changed since."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    model = row["model"]
    effort = row["effort"]
    ok, _ = await _run_chain_step(
        card_id, conn, row, phase="creating_prd", prompt="/rhubarb:to-prd", cwd=cwd, model=model, effort=effort
    )
    if not ok:
        return

    row = db.get_session(conn, card_id)
    ok, claude_session_id = await _run_chain_step(
        card_id, conn, row, phase="creating_issues", prompt="/rhubarb:to-issues", cwd=cwd, model=model, effort=effort
    )
    if not ok:
        return

    row = db.get_session(conn, card_id)
    ok = await _run_publish_step(card_id, conn, row, cwd=cwd)
    if not ok:
        return

    await _finish_chain(card_id, conn, claude_session_id, cwd)


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
    exactly the "still no turns sent yet" case that should apply immediately."""
    conn = db.get_connection()
    model = db.get_model(conn)
    row = db.get_session(conn, card_id)
    await _run_grilling_turn(
        card_id, conn, row, f"/rhubarb:do {prompt}", cwd=cwd, model=model, effort=row["effort"], publish_when_empty=True
    )


async def continue_session_job(card_id: int, reply: str, *, cwd: str | None, confirm_advance: bool = False) -> None:
    """A grilling reply, or an explicit "yes, proceed" confirmation.

    Normal replies (`confirm_advance=False`, the default) always run one
    more grilling CLI turn and publish its `turn` event -- whether or not
    `questions` comes back empty -- and then stop; there is no auto-advance
    into the PRD/issues chain anymore. When grilling has no more questions,
    the frontend shows the turn's `header` with "Yes, proceed" / "No, keep
    discussing" buttons, and "Yes" is what re-invokes this function with
    `confirm_advance=True`.

    `confirm_advance=True` skips running a grilling CLI turn entirely and
    goes straight to `advance_past_grilling`, resuming the session's existing
    `claude_session_id` -- exactly like `retry_session_job` does for a
    `creating_prd` row. Both paths use the model already recorded on the row
    -- this is a continuation of an existing session, not a fresh one."""
    if confirm_advance:
        await advance_past_grilling(card_id, cwd)
        return

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    await _run_grilling_turn(
        card_id, conn, row, reply, cwd=cwd, model=row["model"], effort=row["effort"], publish_when_empty=True
    )


def _parse_qa_grilling_block(text: str) -> dict | None:
    """Extract the first JSON code block with phase=='qa_grilling' from a
    CLI turn result, as emitted by the /qa skill Phase 2. Returns None when
    no such block is found (normal /implement run without the /qa auto-handoff)."""
    for match in _FENCED_JSON_BLOCK_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and data.get("phase") == "qa_grilling":
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    # Fallback: bare JSON (no code fence)
    try:
        data = json.loads(text.strip())
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
    not-blocked case -- structurally identical to `_parse_qa_grilling_block`."""
    for match in _FENCED_JSON_BLOCK_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and data.get("phase") == "implement_blocked":
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    try:
        data = json.loads(text.strip())
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
    its own independent `PtyEngine` tab immediately; with it off, only one
    implement tab runs per project at a time and everything else queues
    here, same as before issue #87 -- only the underlying per-session
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
        turn = await _run_turn(
            card_id,
            f"/rhubarb:implement prd: {prd_number}",
            session_id=row["claude_session_id"],
            cwd=cwd,
            model=model,
            effort=effort,
        )
    except PtyEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="implementing")
        return
    except ClaudeCLIError as e:
        # /implement's PRD-selection step calls `gh issue list` directly, so
        # a genuine gh auth failure is possible here -- classify it.
        _close_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(card_id, _turn_event(phase="implementing", error=message, needs_github_login=bool(needs_login)))
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "implementing", message)
        await _drain_implement_queue(row["project_id"], cwd)
        return

    await _finish_implement_turn(card_id, conn, row, turn, cwd=cwd, model=model, effort=effort)


async def _finish_implement_turn(card_id: int, conn, row, turn: dict, *, cwd: str | None, model, effort) -> None:
    """Shared tail for both `start_implement_job`'s first turn and
    `continue_implement_job`'s resume turn: persist the turn's console
    text/context usage, check for the `implement_blocked` marker (leaving
    the session suspended in `phase: blocked` if found, with no `done` --
    same "suspended, not finished" shape as a QA session awaiting Perfect),
    and otherwise run the existing tracker-file/QA-handoff/pooling logic
    exactly as before this function existed."""
    console_text = row["console_text"] + "\n\n" + turn["result"] if row["console_text"] else turn["result"]
    db.update_session(conn, card_id, console_text=console_text, context_pct=turn.get("context_pct"))

    blocked = _parse_implement_blocked_block(turn["result"])
    if blocked is not None:
        db.update_session(
            conn, card_id, claude_session_id=turn["session_id"], phase="blocked", blocked_json=json.dumps(blocked)
        )
        publish(card_id, _turn_event(phase="blocked", blocked=blocked))
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
        details_json=json.dumps(details) if details is not None else None,
    )

    qa_data = _parse_qa_grilling_block(turn["result"])
    if qa_data is not None:
        # /implement Phase 5 ran /qa, which replied with the new nested
        # "QA session for PRD N: ..." question format (see qa_parser.py) --
        # the qa_grilling JSON block itself is now just a lightweight signal
        # ({phase, prd}) that this handoff happened; the actual issues/
        # questions are parsed from the turn's own free text.
        # Hand the session_id to the QA session instead of pooling it here
        # -- this card's own tab is done; the new QA row's own tab starts
        # fresh (reattached via --resume) on its own first turn.
        qa_prd = qa_data.get("prd")
        qa_issues = parse_qa_response(turn["result"])["issues"]
        qa_row_id = db.create_session(
            conn,
            row["project_id"],
            claude_session_id=turn["session_id"],
            session_type="qa",
            phase="qa_grilling",
            details={"prd": qa_prd},
            model=model,
            effort=effort,
        )
        _close_engine(card_id)
        publish(card_id, {"type": "qa_started", "qa_card_id": qa_row_id})
        publish(card_id, _turn_event(phase="implemented", details=details))
        publish(card_id, {"type": "done"})
        await _drain_implement_queue(row["project_id"], cwd)
        asyncio.create_task(start_qa_job(qa_row_id, qa_prd, qa_issues, cwd=cwd))
    else:
        new_session_id = await _clear_for_reuse(card_id, cwd=cwd, model=model, effort=effort)
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
    independent call, nothing here needs an explicit loop construct)."""
    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    model = row["model"]
    effort = row["effort"]

    db.update_session(conn, card_id, phase="implementing", error_text=None)
    publish(card_id, {"type": "phase", "phase": "implementing"})

    try:
        turn = await _run_turn(
            card_id, reply, session_id=row["claude_session_id"], cwd=cwd, model=model, effort=effort
        )
    except PtyEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="implementing")
        return
    except ClaudeCLIError as e:
        _close_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(card_id, _turn_event(phase="implementing", error=message, needs_github_login=bool(needs_login)))
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "implementing", message)
        await _drain_implement_queue(row["project_id"], cwd)
        return

    row = db.get_session(conn, card_id)
    await _finish_implement_turn(card_id, conn, row, turn, cwd=cwd, model=model, effort=effort)


async def start_qa_job(card_id: int, prd: dict | None, issues: list[dict], *, cwd: str | None) -> None:
    """Publish the qa_grilling turn event for a QA session created by the
    /implement auto-handoff. `prd`/`issues` were already parsed from the
    implement turn's result (the qa_grilling JSON marker for `prd`,
    `qa_parser.parse_qa_response` for the nested `issues`/`questions`); emit
    them and leave the session suspended (no 'done') until POST
    /api/session/qa-complete is called."""
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
    box's content forwarded as context."""
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
        turn = await _run_turn(
            card_id, prompt, session_id=row["claude_session_id"], cwd=cwd, model=model, effort=effort
        )
    except PtyEngineUnrecoverableError as e:
        await _route_crash_to_blocked(card_id, conn, e, phase="qa_closing")
        return
    except ClaudeCLIError as e:
        # /qa's closing step calls `gh issue close`/`gh issue edit` directly,
        # so a genuine gh auth failure is possible here -- classify it.
        _close_engine(card_id)
        message = str(e)
        needs_login = 1 if _is_gh_auth_failure(message) else 0
        db.update_session(conn, card_id, error_text=message, needs_github_login=needs_login)
        publish(card_id, _turn_event(phase="qa_closing", error=message, needs_github_login=bool(needs_login)))
        publish(card_id, {"type": "done"})
        add_error_notification(row["project_id"], card_id, "qa_closing", message)
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
    _close_engine(card_id)

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
        await advance_past_grilling(card_id, cwd)
    elif row["phase"] == "creating_issues":
        ok, claude_session_id = await _run_chain_step(
            card_id,
            conn,
            row,
            phase="creating_issues",
            prompt="/rhubarb:to-issues",
            cwd=cwd,
            model=row["model"],
            effort=row["effort"],
        )
        if ok:
            row = db.get_session(conn, card_id)
            ok = await _run_publish_step(card_id, conn, row, cwd=cwd)
            if ok:
                await _finish_chain(card_id, conn, claude_session_id, cwd)
    elif row["phase"] == "publishing":
        ok = await _run_publish_step(card_id, conn, row, cwd=cwd)
        if ok:
            await _finish_chain(card_id, conn, row["claude_session_id"], cwd)
