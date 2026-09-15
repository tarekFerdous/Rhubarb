"""Persistent per-project "parser session" lifecycle (issue #189, the
lifecycle slice of PRD #187 -- `gh issue view 187`/`gh issue view 189` for
full context).

PRD #187 wants one long-lived `claude` subprocess per PROJECT, dedicated to
extracting needs-input questions out of every session running under that
project, so the existing per-phase regex/Ollama-rescue extraction can be
retired in favor of one shared, reliable pipeline. This module started as
only the lifecycle slice of that (spawn-on-first-open, reuse-on-reopen,
independent concurrent sessions across projects, and clean shutdown -- issue
#189), then grew issue #190's token-ceiling handling: tracking each parser
session's own context-window usage from its turn results, and draining +
`/clear`-ing it in place once usage crosses 60% (see `stream_turn`/
`_CONTEXT_CLEAR_CUTOFF` below). It now also owns issue #191's needs-input
queue (see "Needs-input queue" below) -- the gating/enqueue half of the
pipeline, fed by `session_runner.handle_turn_completed` -- and issue #192's
actual dispatch/extraction half (see "Queue draining and structured
extraction (issue #192)" below): `drain_needs_input_queue` pops a project's
queue one item at a time, sends each item's raw turn text through
`stream_turn` (transparently riding out any mid-drain `/clear` issue #190
triggers), asks the parser session to extract a structured question/options
payload, and tags the (normalized, validated) result with the originating
item's `card_id` as `source_session_id` so a consumer (issue #193's frontend
routing) can route it back to the right card. It also owns issue #194's
regex/Ollama-rescue fallback (see "Legacy fallback on parse failure" below):
when the primary parser-session extraction fails for one queued item, that
one item -- and only that item -- falls back to the pre-existing extraction
pipeline instead of just returning an untagged failure. Routing itself
(focused-card update vs. toast, including the failure-variant toast issue
#194 also needs) is frontend work, out of scope for this module -- see
`rhubarb/web/templates/prompt.html`'s `routeParsedResult`.

## Why a new module, not `session_runner._stream_json_engines`

`session_runner.py` already holds two engine registries
(`_pty_engines`/`_stream_json_engines`), but both are keyed by `card_id` --
one resident engine per SESSION CARD, claimed/closed alongside that card's
own lifecycle (`register_engine`/`_close_engine`, `ensure_standby_engine`/
`claim_standby_engine`/`close_standby_engine`). A parser session is a
different kind of thing entirely: it belongs to a PROJECT, not a session
card, has no "claim" step (it's never handed off/reassigned the way a
standby engine is), and is never closed just because a project is closed or
switched away from -- per PRD #187, its lifetime is independent of UI focus
and it is torn down only when Rhubarb itself exits. Reusing the card-keyed
registries/functions for this would conflate two different keys (card_id vs
project_id) and two different lifecycles (per-card churn vs.
process-lifetime), so this lives in its own small module instead.

## Reuse of `StreamJsonEngine`

`StreamJsonEngine` (`rhubarb/stream_json_engine.py`) is already fully
generic -- its constructor only takes `cwd`/`model`/`effort`/
`resume_session_id`/`process_factory`, nothing grilling- or session-card-
specific -- so this module uses it completely unmodified, one instance per
project id, rather than needing any wrapper or subclass.
"""

import asyncio
import json
import re
from collections import deque
from collections.abc import AsyncIterator

from rhubarb.ollama_rescue import _is_valid_grilling_shape, rescue_grilling_response, should_attempt_grilling_rescue
from rhubarb.qa_parser import parse_grilling_response
from rhubarb.stream_json_engine import StreamJsonEngine

# One live `StreamJsonEngine` per project id -- in-memory only, exactly like
# `session_runner`'s own engine registries (no restart-survival: if Rhubarb's
# own process restarts, every parser session is gone and the next project
# open simply spawns a fresh one). Never popped on project close/switch --
# per PRD #187, a parser session's lifetime is independent of which project
# is currently UI-active; the only two ways an entry ever leaves this dict
# are (a) `close_all_parser_sessions()` at process shutdown, or (b) being
# replaced here by a fresh spawn if it's found dead on a later
# `ensure_parser_session` call.
_parser_sessions: dict[int, StreamJsonEngine] = {}

# One `asyncio.Lock` per project id, created lazily -- guards the
# check-then-spawn sequence in `ensure_parser_session` so two overlapping
# opens of the SAME project (e.g. a double-click, or two browser tabs) can
# never race into spawning two subprocesses for one project id. Never
# cleaned up (a handful of leftover empty-ish `Lock` objects for project ids
# that existed at some point in this process's life is not worth the
# bookkeeping to prune), mirroring this module's own "in-memory, process-
# lifetime only" contract.
_locks: dict[int, asyncio.Lock] = {}


def _lock_for(project_id: int) -> asyncio.Lock:
    lock = _locks.get(project_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[project_id] = lock
    return lock


def get_parser_session(project_id: int) -> StreamJsonEngine | None:
    """Look up `project_id`'s parser session without creating one. Returns
    `None` if no session is registered, or if the registered one's
    subprocess has died -- never hands back a dead engine."""
    engine = _parser_sessions.get(project_id)
    if engine is None or not engine.isalive():
        return None
    return engine


async def ensure_parser_session(
    project_id: int, *, cwd: str | None, model: str | None = None, effort: str | None = None
) -> StreamJsonEngine:
    """Make sure `project_id` has a live parser session, and return it --
    the entry point the project-open flow calls on every single open
    (`rhubarb/web/app.py`'s `open_project`).

    - First open (no entry yet, or a previously-registered one whose
      subprocess has since died): spawns a fresh `StreamJsonEngine` and
      registers it under `project_id`.
    - Every later open of the SAME project: returns the exact SAME engine
      instance already registered -- same subprocess, same `session_id` --
      no new spawn.
    - A different `project_id` is a completely independent entry: its own
      spawn, its own `session_id`, spawned/torn down independently of any
      other project's.

    `model`/`effort` only matter for a fresh spawn -- unlike the standby-
    engine registries in `session_runner.py`, a live parser session is never
    respawned just because a later call passes different settings (it has
    no "claim" step to mismatch against; it is the same long-lived session
    for as long as it stays alive).

    Double-checked-locking against `project_id`'s own lock so two calls
    racing for the same brand-new project (e.g. two overlapping opens)
    can't both observe "nothing registered yet" and each spawn their own
    subprocess -- exactly one spawn happens, and the loser of the race gets
    the winner's engine back, never its own.

    Fire-and-forget friendly: safe to call via `asyncio.create_task` without
    awaiting, same as `session_runner.ensure_standby_stream_json_engine`.
    """
    existing = get_parser_session(project_id)
    if existing is not None:
        return existing

    async with _lock_for(project_id):
        existing = get_parser_session(project_id)
        if existing is not None:
            return existing

        engine = await asyncio.to_thread(_spawn_parser_session, cwd=cwd, model=model, effort=effort)
        _parser_sessions[project_id] = engine
        return engine


def _spawn_parser_session(*, cwd: str | None, model: str | None, effort: str | None) -> StreamJsonEngine:
    """Construct and start a genuinely fresh `StreamJsonEngine` -- no
    `resume_session_id`, a brand-new conversation. Blocking (a real spawn is
    a subprocess call) -- `ensure_parser_session` runs this via
    `asyncio.to_thread`, mirroring `session_runner._spawn_fresh_engine`/
    `_spawn_fresh_stream_json_engine`."""
    engine = StreamJsonEngine(cwd=cwd, model=model, effort=effort)
    engine.start()
    return engine


def close_all_parser_sessions() -> None:
    """Terminate every live parser-session subprocess and clear the
    registry. Called exactly once, from the web app's own shutdown hook
    (`rhubarb/web/app.py`'s `_lifespan`, the same place
    `db.cleanup_sessions_on_shutdown` already runs at process exit) -- a
    parser session has no explicit per-project close action (unlike a
    standby engine, closed when its project is closed/switched away from:
    see `session_runner.close_standby_stream_json_engine`), so this is the
    ONLY place a parser session is ever torn down, per issue #189/#187.
    Safe to call even with an empty registry."""
    for engine in _parser_sessions.values():
        engine.close()
    _parser_sessions.clear()


# ---------------------------------------------------------------------------
# Token-ceiling handling (issue #190, `gh issue view 190` for full context).
#
# A parser session's whole point is to stay resident for a project's entire
# lifetime (issue #189 above) rather than being respawned per turn the way
# `session_runner._spawn_fresh_engine` replaces a card's PtyEngine to
# "clear" it. That means its own conversation history only ever grows, so
# left unchecked it would eventually fill its 1M-token context window and
# start failing turns. PRD #187's fix: once its tracked usage crosses 60% of
# that window, let whatever parse is already running finish untouched, then
# send one `/clear` turn into the SAME still-open subprocess (same
# `session_id`, only its conversation history resets -- NOT the
# `_spawn_fresh_engine`-style respawn `PtyEngine` clearing uses, which this
# module deliberately does not follow here; see #187's PRD body for why a
# parser session's identity must survive a clear).
# ---------------------------------------------------------------------------

# Claude Code's own interactive statusline threshold this mirrors is a UI
# nicety; this is a hard gate PRD #187 fixed at 60% of the model's 1M-token
# context window. Not user-configurable (see PRD #187's Out of Scope list).
_CONTEXT_CLEAR_CUTOFF = 0.60

# Last-known context-window usage fraction (0.0-1.0) per project id, updated
# from every turn's own raw `result` event that carried enough usage data to
# compute it (see `_context_window_pct`). Absent/`None` for a project whose
# parser session has never completed a turn with usage data yet -- treated
# as "not over the cutoff" everywhere this is read, same safe-by-default
# stance `session_runner._context_window_pct`'s own callers already take.
# In-memory only, process-lifetime, exactly like `_parser_sessions` above.
_context_pct: dict[int, float | None] = {}


def get_context_pct(project_id: int) -> float | None:
    """`project_id`'s parser session's last-known context-window usage
    fraction (0.0-1.0), or `None` if no turn has completed yet (or none so
    far carried usage data -- e.g. every turn so far went through a fake/
    test engine whose scripted `result` events don't include `usage`/
    `modelUsage`)."""
    return _context_pct.get(project_id)


def _context_window_pct(raw_result_event: dict) -> float | None:
    """How full the context window was for one turn, from its raw
    (untranslated) `result` event's usage fields -- the exact same formula
    `session_runner._context_window_pct` already uses (Claude Code's own
    interactive statusline formula: `(input_tokens +
    cache_creation_input_tokens + cache_read_input_tokens) / contextWindow`),
    deliberately duplicated here rather than imported from there:
    `session_runner.py` is the heavier, card_id-keyed orchestration module
    (imports `github_publisher`, `ollama_rescue`, `qa_parser`, ...) that a
    future issue (#191/#192) is expected to make call INTO this module for
    the real needs-input queue, not the other way around -- importing
    `session_runner` from here would risk exactly the circular dependency
    that future direction needs to avoid (see this module's own docstring,
    "Why a new module, not `session_runner._stream_json_engines`").

    Returns `None` when the event doesn't carry enough to compute this (no
    `usage`/`modelUsage` block, or a zero/missing `contextWindow`)."""
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


def _record_usage(project_id: int, result_event: dict) -> None:
    """Update `project_id`'s tracked usage from one turn's `result` event,
    if it carried enough data to compute a fraction -- a turn whose event
    didn't (`_context_window_pct` returned `None`) leaves the previously
    tracked value untouched rather than clobbering a real, already-recorded
    reading with an unknown one."""
    pct = _context_window_pct(result_event)
    if pct is not None:
        _context_pct[project_id] = pct


async def _drain_clear_turn(project_id: int, engine: StreamJsonEngine) -> None:
    """Send one `/clear` turn into `engine` and drain it to completion --
    same subprocess, same `session_id` (an ordinary `StreamJsonEngine.
    stream_turn` call; the CLI itself interprets the literal prompt text
    `/clear` as the reset command, same as a user typing it interactively).
    Updates the tracked usage from the `/clear` turn's own `result` event
    too, rather than hardcoding a fresh reading back to `0.0`: a cleared
    conversation still costs a little context up front (system prompt,
    tool definitions), so the real post-clear number -- whatever the CLI
    reports -- is more honest than an assumed zero."""
    async for event in engine.stream_turn("/clear"):
        if event.get("type") == "result":
            _record_usage(project_id, event)


async def stream_turn(project_id: int, prompt: str) -> AsyncIterator[dict]:
    """Drive one turn through `project_id`'s already-registered parser
    session (call `ensure_parser_session` first -- this looks up, but never
    creates, a session; raises `LookupError` if none is currently live for
    `project_id`), tracking context-window usage from the turn's own
    `result` event, and handling the 60%-ceiling `/clear` per issue #190.

    Yields every event the underlying `StreamJsonEngine.stream_turn`
    yields for `prompt`, unmodified and in order -- a parse already in
    flight is NEVER interrupted by crossing the ceiling, structurally: the
    ceiling is only ever checked AFTER `prompt`'s own turn has fully
    finished (its `result` event already yielded), never mid-turn.

    If usage is now at/over `_CONTEXT_CLEAR_CUTOFF` once `prompt`'s turn
    has completed, this drains one `/clear` turn through the SAME engine
    (`_drain_clear_turn` -- same subprocess, same `session_id`, only
    history resets) before returning, so the very next `stream_turn` call
    for this project already starts against a cleared session -- "resume
    normal processing afterward" per the issue's spec. A synthetic
    `{"type": "context_cleared", ...}` event is yielded once that finishes,
    mirroring `stream_translate.py`'s own precedent for injecting
    synthetic (non-CLI-native) event types alongside the CLI's real ones,
    so a caller (or a test) can observe the clear happened without having
    to inspect the engine's own internals."""
    engine = get_parser_session(project_id)
    if engine is None:
        raise LookupError(f"no live parser session registered for project {project_id}")

    async for event in engine.stream_turn(prompt):
        yield event
        if event.get("type") == "result":
            _record_usage(project_id, event)

    if (_context_pct.get(project_id) or 0.0) >= _CONTEXT_CLEAR_CUTOFF:
        await _drain_clear_turn(project_id, engine)
        yield {"type": "context_cleared", "project_id": project_id, "session_id": engine.session_id}


# ---------------------------------------------------------------------------
# Needs-input queue (issue #191, `gh issue view 191` for full context).
#
# The gating half of PRD #187's parser-session pipeline: `session_runner.
# handle_turn_completed` -- the shared "on turn complete" hook issue #191
# consolidates every phase's needs-input classification behind (grilling,
# qa_grilling, implementing -- including the parallel per-issue implement
# sessions, which run the exact same `_finish_implement_turn` tail -- and
# creating_prd/creating_issues) -- enqueues a completed turn here whenever
# the existing cheap Ollama needs-input classifier
# (`session_runner.classify_needs_input`, issue #175) flags it as possibly
# needing a human's input. One FIFO queue per project id, exactly matching
# `_parser_sessions`'s own project-id keying above, so turns from different
# concurrent sessions under the SAME project queue together in FIFO order
# while turns under a DIFFERENT project never mix in.
#
# This is gating + queueing ONLY. Nothing here (or anywhere else yet)
# automatically drains a project's queue into its actual parser session
# (`stream_turn` above) -- that dispatch is issue #192's job, deliberately
# not built here. `dequeue_needs_input_turn`/`get_needs_input_queue` exist
# now purely so tests -- and #192's future real consumer -- can inspect/
# drain a queue deterministically.
# ---------------------------------------------------------------------------

# project_id -> FIFO deque of {"project_id", "card_id", "phase", "text"}
# items, oldest first. In-memory, process-lifetime only, exactly like
# `_parser_sessions`/`_locks` above -- no restart-survival, nothing here
# needs any.
_needs_input_queues: dict[int, deque] = {}


def enqueue_needs_input_turn(project_id: int, *, card_id: int, phase: str, text: str) -> dict:
    """Append one completed, needs-input-flagged turn onto `project_id`'s
    FIFO queue. `card_id`/`phase` identify which session/phase this turn
    came from -- so a future consumer (issue #192) can route a parsed result
    back to the right card -- and `text` is that turn's own raw, rendered
    output, unmodified. Returns the enqueued item."""
    item = {"project_id": project_id, "card_id": card_id, "phase": phase, "text": text}
    _needs_input_queues.setdefault(project_id, deque()).append(item)
    return item


def dequeue_needs_input_turn(project_id: int) -> dict | None:
    """Pop and return the oldest still-queued item for `project_id`, or
    `None` if its queue is empty or nothing has ever been enqueued for it.
    Not called by anything in this slice -- exposed for tests, and for issue
    #192's real consumer to build on."""
    queue = _needs_input_queues.get(project_id)
    if not queue:
        return None
    return queue.popleft()


def get_needs_input_queue(project_id: int) -> list[dict]:
    """Read-only snapshot of `project_id`'s queue, oldest first -- does not
    mutate it. For inspection (tests; a future dashboard/debug view)."""
    return list(_needs_input_queues.get(project_id, ()))


# ---------------------------------------------------------------------------
# Queue draining and structured extraction (issue #192, `gh issue view 192`
# for full context).
#
# The actual dispatch half of PRD #187's pipeline: drains `project_id`'s
# needs-input queue (populated above by `enqueue_needs_input_turn`, fed by
# `session_runner.handle_turn_completed`) one item at a time, in FIFO order,
# sending each queued item's raw turn text through this project's parser
# session (`stream_turn` above) with a prompt asking it to extract the
# question(s)/options it contains as JSON. The parser session is a real
# `claude` subprocess turn, not Ollama's schema-constrained generation, so
# there is no way to force valid JSON out of it structurally -- instead this
# asks for JSON in the prompt, then validates whatever text comes back
# (`_extract_json_object` + `_is_valid_grilling_shape`, reused unmodified
# from `ollama_rescue.py`'s own rescue-parsing pattern) and rejects anything
# that doesn't match, exactly like `ollama_rescue.rescue_grilling_response`
# already does for its own (schema-constrained, but still separately
# validated) Ollama responses.
#
# Output schema: reuses the existing grilling-rescue shape (`header`,
# `questions: [{id, text, kind, options, recommended, recommended_text}]`,
# `footer` -- see `ollama_rescue._GRILLING_SCHEMA`/`_is_valid_grilling_shape`
# and `qa_parser.parse_grilling_response`, which this deliberately mirrors),
# extended with `source_session_id`. That tag is NOT trusted from the
# model's own response text -- it is always set here, deterministically,
# from the queued item's own `card_id` (per issue #191's queue entry shape)
# after a successful parse, so a routing bug or a hallucinated id in the
# model's output can never mis-tag a result. This is also why a failed
# parse still returns a tagged (if minimal) result rather than `None`: a
# future consumer (issue #193 for routing, #194 for the regex/Ollama-rescue
# fallback) needs to know WHICH session's turn failed to parse, not just
# that something did.
#
# What this exposes for #193 to build on: `drain_needs_input_queue` is an
# async generator -- the same shape `stream_turn` above already is -- that
# dequeues and yields one tagged result dict per queued item as it finishes,
# so a caller can react to each result as it arrives (e.g. to route it to a
# focused card or raise a toast) without waiting for the whole queue to
# drain. Every yielded result is also appended to an in-memory per-project
# list (`get_parsed_results`), mirroring `get_needs_input_queue`'s own
# read-only-snapshot shape above, so a caller that isn't actively iterating
# the generator (e.g. a test, or a future polling consumer) can still
# inspect what's been produced so far. Neither of these two vantage points
# is authoritative over the other -- both are fed from the exact same
# `_process_one_queued_item` call per item.
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT_TEMPLATE = (
    "The following text is the complete, final reply a Claude Code assistant "
    'just gave for one turn during Rhubarb\'s "{phase}" phase. It has '
    "already been flagged as needing a human's input before that session "
    "can usefully continue. Extract the question(s) it is actually asking "
    "(or the choice(s) it is waiting on) as JSON matching exactly this "
    "shape, and respond with ONLY that JSON object -- no other prose, no "
    "code fences, no markdown:\n\n"
    '{{"header": <free text before the first question, or the whole text '
    'if there is no clear question in it>, "questions": [{{"id": <a short '
    'stable string id, e.g. "q1">, "text": <the question itself, verbatim '
    'or lightly cleaned up>, "kind": "single" | "multi" | "open", '
    '"options": <array of option strings verbatim, or null if this '
    'question is open-ended>, "recommended": <array of 1-based indexes '
    'into "options" that were recommended, or null>, "recommended_text": '
    "<free-text recommendation for an open-ended question, or null>}}, "
    '...], "footer": <free text after the last question, or "" if '
    "none>}}\n\n"
    'Use "single" for a pick-one choice, "multi" for a pick-several '
    'choice, and "open" (with "options" and "recommended" both null) for '
    "anything without discrete options -- free-text questions still go in "
    '"questions" with "kind": "open". If the text genuinely contains no '
    "question at all (this should be rare, since it was already flagged "
    'as needing input), return "questions": [] with the whole text as '
    '"header" and "" as "footer".\n\n'
    "Text:\n{text}"
)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort pull of one JSON object out of a parser-session turn's
    raw reply text. Tries, in order: the whole (stripped) text as-is; the
    contents of a fenced ```json ... ``` (or bare ``` ... ```) code block,
    in case the model wrapped its answer despite being asked not to; and
    the substring between the first `{` and the last `}` in the text, in
    case it added any stray prose before/after the object. Returns the
    first candidate that parses as a JSON *object* (a JSON array, string,
    etc. at the top level doesn't count -- `_is_valid_grilling_shape`
    requires a dict), or `None` if nothing in the text parses at all."""
    stripped = text.strip()
    candidates = []

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    candidates.append(stripped)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _tagged_success(item: dict, data: dict) -> dict:
    return {
        "source_session_id": item["card_id"],
        "ok": True,
        "header": data["header"],
        "questions": data["questions"],
        "footer": data["footer"],
        "source": "parser_session",
    }


# ---------------------------------------------------------------------------
# Legacy fallback on parse failure (issue #194, `gh issue view 194` for full
# context).
#
# When a queued item's primary extraction (via this project's parser
# session, above) errors, times out, or comes back not matching the
# expected schema, that one turn -- and ONLY that turn -- falls back to the
# pre-existing regex-parser + Ollama-rescue extraction pipeline
# (`qa_parser.parse_grilling_response` / `ollama_rescue.
# rescue_grilling_response`) instead of just giving up. This is the exact
# same chain `session_runner.py`'s own grilling-turn handling already uses
# (see e.g. `session_runner._run_grilling_turn`): the free, deterministic
# regex parser first, and only if THAT comes back with no questions AND the
# raw text still looks like it was trying to contain one
# (`should_attempt_grilling_rescue`'s trigger check) does this spend an
# Ollama call on it.
#
# Retry-per-turn, not a permanent downgrade (PRD #187's explicit
# requirement): nothing here reads or writes any project-level or
# session-level state. Every call to `_process_one_queued_item` decides
# primary-vs-fallback fresh, purely from whether THIS item's own primary
# attempt succeeded -- so a project that just had a failure is attempted via
# the parser session completely normally on its very next queued item, and
# a string of failures never accumulates into any kind of sticky "this
# project is on the fallback pipeline now" state.
#
# Tagging: a successful fallback extraction is returned as an `ok: True`
# result (`_tagged_fallback_success`) shaped exactly like a primary success
# (see `_tagged_success` above) so it flows through issue #193's existing
# routing (focused-card update or a held-result toast) completely
# unmodified -- the only difference is `"source": "fallback"` instead of
# `"source": "parser_session"`, which is what `prompt.html`'s routing keys
# off of to ALSO raise a distinct failure-variant toast alongside that
# routing, telling the user parsing degraded for this turn (issue #188's
# toast component). If the fallback extraction ALSO fails to produce a
# usable question (the regex parser and, if attempted, Ollama rescue both
# come up empty), there is genuinely nothing to route -- this returns an
# `ok: False` result, still tagged `"source": "fallback"` so the frontend
# still raises the failure-variant toast (visibility over silence is the
# whole point of this issue) even though there is no question to show.
# ---------------------------------------------------------------------------


def _tagged_fallback_success(item: dict, data: dict) -> dict:
    return {
        "source_session_id": item["card_id"],
        "ok": True,
        "header": data["header"],
        "questions": data["questions"],
        "footer": data["footer"],
        "source": "fallback",
    }


def _tagged_fallback_failure(item: dict, *, error: str) -> dict:
    return {"source_session_id": item["card_id"], "ok": False, "error": error, "source": "fallback"}


async def _legacy_fallback_extract(text: str) -> dict | None:
    """The pre-existing regex-parser + Ollama-rescue extraction chain, run
    on one failed item's raw turn `text` -- mirrors `session_runner.py`'s
    own grilling-turn extraction order exactly (regex first, Ollama rescue
    only if the regex came back empty AND the text still looks like it was
    trying to contain a question). `rescue_grilling_response` is a blocking
    HTTP call, so it's run via `asyncio.to_thread`, same as every other
    call site of it in `session_runner.py`.

    Returns the parsed `{header, questions, footer, source}` dict only when
    it actually carries at least one question, `None` otherwise (a genuine
    "nothing extractable" outcome -- the regex parser found nothing and
    either the trigger check said not to bother with Ollama, or Ollama
    rescue was tried and also came back empty/unavailable)."""
    parsed = parse_grilling_response(text)
    if not parsed["questions"] and should_attempt_grilling_rescue(parsed, text):
        rescued = await asyncio.to_thread(rescue_grilling_response, text)
        if rescued is not None:
            parsed = rescued
    return parsed if parsed["questions"] else None


async def _fallback_or_failure(item: dict, *, primary_error: str) -> dict:
    """Called from every failure branch of `_process_one_queued_item` below
    once the primary parser-session extraction has failed -- attempts issue
    #194's legacy fallback extraction on `item`'s own raw text, and returns
    whichever tagged result that produces (`_tagged_fallback_success` if it
    found a usable question, `_tagged_fallback_failure` -- still carrying
    `primary_error` -- if it didn't). Always tagged `"source": "fallback"`
    one way or the other, so the frontend can always tell a primary failure
    happened and raise its failure-variant toast, whether or not the
    fallback itself managed to recover a question."""
    fallback = await _legacy_fallback_extract(item["text"])
    if fallback is not None:
        return _tagged_fallback_success(item, fallback)
    return _tagged_fallback_failure(item, error=primary_error)


async def _process_one_queued_item(project_id: int, item: dict) -> dict:
    """Send one dequeued item's raw turn `text` through `project_id`'s
    parser session and return a tagged result -- `_tagged_success` on a
    clean primary-path parse, or (issue #194) whatever `_fallback_or_failure`
    produces once the primary path has failed. Every possible result always
    carries `source_session_id` (`item["card_id"]`) and `ok`, so a caller
    can branch on `ok` without needing to know anything about why a failure
    happened.

    Never raises: a dead/missing parser session (`stream_turn`'s
    `LookupError`), a subprocess crash that exhausts `stream_turn`'s own
    retry (`StreamJsonEngineUnrecoverableError`), a turn that ends without
    ever producing a `result` event, unparseable JSON, or JSON that parses
    but doesn't match the expected shape are all just different reasons the
    primary path is considered failed -- each one hands off to
    `_fallback_or_failure` rather than raising, so one bad turn (even one
    whose fallback also comes up empty) can never take down
    `drain_needs_input_queue`'s processing of the rest of the queue."""
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(phase=item["phase"], text=item["text"])

    try:
        result_text = None
        async for event in stream_turn(project_id, prompt):
            if event.get("type") == "result":
                result_text = event.get("result")
    except Exception as exc:  # noqa: BLE001 -- any subprocess/engine failure is a per-item failure, not a crash
        return await _fallback_or_failure(item, primary_error=str(exc))

    if result_text is None:
        return await _fallback_or_failure(item, primary_error="parser session turn produced no result event")

    data = _extract_json_object(result_text)
    if data is None:
        return await _fallback_or_failure(item, primary_error="parser session response was not valid JSON")

    if not _is_valid_grilling_shape(data):
        return await _fallback_or_failure(
            item, primary_error="parser session response did not match the expected question schema"
        )

    return _tagged_success(item, data)


# project_id -> list of every tagged result `drain_needs_input_queue` has
# produced so far, oldest first -- in-memory, process-lifetime only, exactly
# like every other registry in this module. Purely additive (nothing ever
# pops from this list); a future consumer that wants "only what's new" is
# expected to track its own read offset, the same way a caller iterating
# `drain_needs_input_queue` itself naturally only ever sees each result once.
_parsed_results: dict[int, list[dict]] = {}


async def drain_needs_input_queue(project_id: int) -> AsyncIterator[dict]:
    """Pop and process `project_id`'s needs-input queue (issue #191) one
    item at a time, oldest first, until it's empty -- the actual FIFO
    dispatch loop issue #192 adds. Yields each item's tagged result
    (`_tagged_success`, or -- issue #194 -- `_tagged_fallback_success`/
    `_tagged_fallback_failure` once the primary path has failed) as soon as
    that item finishes, so a caller can react per-result rather than
    waiting for the whole queue.

    FIFO and non-conflation: items are dequeued one at a time via
    `dequeue_needs_input_turn`, and the NEXT item is only ever dequeued
    after the current item's `_process_one_queued_item` call has fully
    resolved -- so two queued items, even from different source sessions
    under the same project, are always processed strictly in order and
    each produces its own independently-tagged result; neither can ever
    observe or overwrite the other's.

    Ceiling-crossing safety (issue #190): `_process_one_queued_item` calls
    `stream_turn`, which -- transparently, internally -- drains a `/clear`
    turn through the SAME still-open subprocess/session if this item's own
    turn pushed usage over the 60% cutoff, before `stream_turn`'s async
    generator itself finishes. Since this loop fully awaits that generator
    (via the `async for` inside `_process_one_queued_item`) before ever
    calling `dequeue_needs_input_turn` again, a mid-drain `/clear` can never
    cause an item to be skipped (the next `dequeue` only happens once the
    clear, if any, has already completed) or double-processed (each item is
    popped exactly once, before it is sent).

    A queue with nothing enqueued (or already fully drained) yields nothing
    and returns immediately -- safe to call speculatively."""
    while True:
        item = dequeue_needs_input_turn(project_id)
        if item is None:
            return
        result = await _process_one_queued_item(project_id, item)
        _parsed_results.setdefault(project_id, []).append(result)
        yield result


def get_parsed_results(project_id: int) -> list[dict]:
    """Read-only snapshot of every tagged result `drain_needs_input_queue`
    has produced so far for `project_id`, oldest first -- does not mutate
    it. Mirrors `get_needs_input_queue`'s own read-only-snapshot shape;
    for inspection (tests, and a future consumer that polls instead of
    iterating the generator directly)."""
    return list(_parsed_results.get(project_id, ()))
