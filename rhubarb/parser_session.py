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
fallback-on-parse-failure tagging (see "Legacy fallback on parse failure"
below): when the primary parser-session extraction fails for one queued
item, that failure is tagged distinctly (`"source": "fallback"`) instead of
just returning an untagged failure -- issue #230 retired the regex/Ollama-
rescue extraction chain that tag used to actually attempt as a second try,
so a primary failure is now simply a tagged failure, with the
parser-session pipeline as the one and only extraction mechanism. Routing
itself (focused-card update vs. toast, including the failure-variant toast
issue #194 also needs) is frontend work, out of scope for this module -- see
`rhubarb/web/templates/prompt.html`'s `routeParsedResult`.

## Why a new module, not `session_runner._stream_json_engines`

`session_runner.py` already holds an engine registry (`_stream_json_engines`),
keyed by `card_id` -- one resident engine per SESSION CARD, claimed/closed
alongside that card's own lifecycle (`register_stream_json_engine`/
`_close_stream_json_engine`, `ensure_standby_stream_json_engine`/
`claim_standby_stream_json_engine`/`close_standby_stream_json_engine`). A
parser session is a
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

from rhubarb.ollama_rescue import _is_valid_grilling_shape
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
# `session_runner._spawn_fresh_stream_json_engine` replaces a card's engine to
# "clear" it. That means its own conversation history only ever grows, so
# left unchecked it would eventually fill its 1M-token context window and
# start failing turns. PRD #187's fix: once its tracked usage crosses 60% of
# that window, let whatever parse is already running finish untouched, then
# send one `/clear` turn into the SAME still-open subprocess (same
# `session_id`, only its conversation history resets -- NOT the
# `_spawn_fresh_stream_json_engine`-style respawn-clearing `session_runner`
# uses, which this module deliberately does not follow here; see #187's PRD
# body for why a parser session's identity must survive a clear).
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
# session (`stream_turn` above), invoking the `rhubarb` plugin's
# `/rhubarb:parse-interview` skill (issue #228, `gh issue view 228`) to
# extract the question(s)/options it contains as JSON -- see
# `_build_extraction_prompt` below. The parser session is a real `claude`
# subprocess turn, not Ollama's schema-constrained generation, so there is
# no way to force valid JSON out of it structurally -- instead this asks
# for JSON via the skill's own instructions, then validates whatever text
# comes back (`_extract_json_object` + `_is_valid_grilling_shape`, reused
# unmodified from `ollama_rescue.py`'s own rescue-parsing pattern) and
# rejects anything that doesn't match, exactly like `ollama_rescue.
# rescue_grilling_response` already does for its own (schema-constrained,
# but still separately validated) Ollama responses.
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

def _build_extraction_prompt(*, phase: str, text: str) -> str:
    """Build the turn-extraction prompt sent to this project's parser
    session -- invokes the `rhubarb` plugin's `/rhubarb:parse-interview`
    skill (`rhubarb/claude_plugin/skills/parse-interview/SKILL.md`) rather
    than inlining the extraction rules here (issue #228, `gh issue view
    228`/`gh issue view 227`). The skill is discoverable by every
    `claude` subprocess Rhubarb spawns via `--plugin-dir`
    (`stream_json_engine.py`'s `_build_args`/`cli_client._plugin_args`),
    regardless of the target project's own `cwd`, exactly like the
    existing `/rhubarb:grilling`/`/rhubarb:implement` invocations
    `session_runner.py` already sends.

    `phase`/`text` are passed the same way for every phase that routes
    through this extraction path (grilling, QA, QA-grilling, implement) --
    no phase-specific branching -- as a labeled `phase:` line followed by
    a `Text:` block carrying the turn's raw text verbatim, mirroring the
    `prd: <N>` labeled-argument convention `/rhubarb:implement` already
    uses for its own structured argument."""
    return f"/rhubarb:parse-interview phase: {phase}\n\nText:\n{text}"


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


# ---------------------------------------------------------------------------
# Post-extraction validation + single corrective retry (issue #229, `gh
# issue view 229` for full context; parent PRD #227, `gh issue view 227`).
#
# The `/rhubarb:parse-interview` skill invocation above (issue #228) still
# runs as an ordinary free-form `claude` turn -- there is no structural way
# to force it to preserve every `Recommended:` line or bulleted options
# list, only prose instructions asking it to. This section adds a cheap,
# local (no extra LLM call) sanity check comparing the skill's returned JSON
# against simple textual signals in the turn's own raw reply, so a silent
# drop (the root cause PRD #227 describes) gets caught and given one chance
# to self-correct before being accepted as-is.
#
# Two independent signals, both heuristic/approximate by design (PRD #227's
# explicit call: "this doesn't need to be perfectly precise, just catch the
# 'options went from present to null/empty' case"):
#   1. How many `Recommended:`/`Recommended text:` lines the raw text
#      contains vs. how many questions in the JSON actually ended up with a
#      populated `recommended`/`recommended_text` field.
#   2. Per question, how many bulleted option-looking lines appear near that
#      question's own text vs. the length of its returned `options` array.
#
# On a mismatch, exactly one retry (`_retry_extraction_once`) is sent to the
# SAME parser session, naming the specific mismatch(es) found, asking for a
# corrected full JSON object. If the retry's own result still fails the same
# check, the affected question(s) -- and only those -- are tagged
# `"extraction_incomplete": True` (additive; absent, never `False`, on
# questions that don't need it) and the result is returned as-is. Nothing
# in this section ever raises -- a retry-turn failure (dead session,
# unparseable/invalid-shape response) is treated exactly like "the retry
# didn't fix it", never a hard error, per PRD #227's "never error, never
# loop further" requirement.
# ---------------------------------------------------------------------------

_RECOMMENDED_TEXT_LINE_RE = re.compile(r"recommended\s+text\s*:", re.IGNORECASE)
_RECOMMENDED_PLAIN_LINE_RE = re.compile(r"recommended\s*:", re.IGNORECASE)
_BULLET_LINE_RE = re.compile(r"(?m)^[ \t]*[-*•][ \t]+\S")
_QUESTION_MARKER_RE = re.compile(r"❓")  # the "❓" emoji marking a `❓ **Qn**` question header


def _count_recommended_signals(text: str) -> int:
    """How many `Recommended:`/`Recommended text:` lines appear anywhere in
    `text`, case-insensitively. The two patterns never double-count the same
    occurrence -- `Recommended text:` has ` text` between `Recommended` and
    the colon, so it never also matches the plain `Recommended:` pattern
    (which requires the colon immediately after `Recommended`, only
    whitespace allowed in between)."""
    return len(_RECOMMENDED_TEXT_LINE_RE.findall(text)) + len(_RECOMMENDED_PLAIN_LINE_RE.findall(text))


def _question_has_populated_recommendation(question: dict) -> bool:
    """True if `question`'s `recommended` (a non-empty list) or
    `recommended_text` (a non-empty/non-whitespace string) is actually
    populated -- `None`, `[]`, and `""`/whitespace-only all count as "not
    populated", matching what a real dropped recommendation looks like."""
    recommended = question.get("recommended")
    recommended_text = question.get("recommended_text")
    return bool(recommended) or bool(recommended_text and recommended_text.strip())


def _count_bullet_lines(block: str) -> int:
    """How many lines in `block` look like a bulleted option (`- `, `* `, or
    `• ` at the start of a line, ignoring leading whitespace)."""
    return len(_BULLET_LINE_RE.findall(block))


def _split_into_question_blocks(raw_text: str, n_questions: int) -> list[str]:
    """Best-effort split of `raw_text` into one substring per question, in
    order, so each question's own nearby-bullets/recommended-line check
    looks at roughly the right slice of text rather than the whole turn.

    Splits on `❓ **Qn**`-style question markers (the format the
    grilling/QA/implement skills actually emit -- see `parse-interview`'s
    own `SKILL.md`) when there are at least as many markers as questions,
    each block running from one marker up to the next (or end of text).
    When there aren't enough markers to reliably attribute one to each
    question (a turn with no markers at all, or fewer markers than
    questions -- e.g. plain prose, or a format this heuristic doesn't
    recognize), falls back to handing every question the ENTIRE raw text --
    a strictly more permissive (never false-negative-inducing on a truly
    missing signal) fallback than guessing wrong boundaries; `n_questions`
    of 0 returns an empty list."""
    if n_questions <= 0:
        return []

    marker_starts = [m.start() for m in _QUESTION_MARKER_RE.finditer(raw_text)]
    if len(marker_starts) < n_questions:
        return [raw_text] * n_questions

    bounds = marker_starts[:n_questions] + [len(raw_text)]
    return [raw_text[bounds[i] : bounds[i + 1]] for i in range(n_questions)]


def _detect_extraction_mismatches(raw_text: str, data: dict) -> list[dict]:
    """Compare `data` (an already schema-valid grilling-shape payload)
    against `raw_text` and return a list of `{"index", "id", "reasons"}`
    dicts, one per question that looks like it lost something -- empty when
    nothing looks wrong. `index` is the question's position in
    `data["questions"]`, `id` is its own `"id"` field (for a human-readable
    retry prompt), and `reasons` is a list of short strings naming exactly
    what looked off.

    Per-question checks (against that question's own slice from
    `_split_into_question_blocks`): a `Recommended:`/`Recommended text:`
    line present in the slice but no populated `recommended`/
    `recommended_text` on the question, and/or bulleted lines present in
    the slice but an empty/null `"options"` array.

    Falls back to one aggregate, turn-wide check -- attributed to question 0
    as a best-effort target -- only when the per-question checks above found
    nothing AND the raw text's total `Recommended:`/`Recommended text:`
    count still doesn't match the number of questions with a populated
    recommendation; this catches a real drop in a turn whose format the
    per-question splitter couldn't reliably attribute (e.g. no `❓`
    markers at all)."""
    questions = data.get("questions") or []
    blocks = _split_into_question_blocks(raw_text, len(questions))

    mismatches = []
    for index, question in enumerate(questions):
        block = blocks[index] if index < len(blocks) else raw_text
        reasons = []

        if _count_recommended_signals(block) > 0 and not _question_has_populated_recommendation(question):
            reasons.append(
                "raw text has a Recommended:/Recommended text: line for this question, but its "
                "'recommended'/'recommended_text' field came back empty"
            )

        bullet_count = _count_bullet_lines(block)
        options = question.get("options") or []
        if bullet_count > 0 and not options:
            reasons.append(
                f"raw text appears to list {bullet_count} bulleted option line(s) near this question, "
                "but its 'options' array came back empty/null"
            )

        if reasons:
            mismatches.append({"index": index, "id": question.get("id"), "reasons": reasons})

    if not mismatches and questions:
        raw_count = _count_recommended_signals(raw_text)
        populated_count = sum(1 for q in questions if _question_has_populated_recommendation(q))
        # One-directional deliberately: only `raw_count > populated_count`
        # (the raw text names more recommendations than the JSON reflects)
        # is "something was dropped." `populated_count > raw_count` is NOT
        # flagged -- it means the JSON reflects a recommendation this cheap
        # substring count didn't literally see (e.g. paraphrased text, or a
        # test/caller-supplied `text` that's shorter than a real turn's raw
        # reply), which is not the drop this check exists to catch and would
        # otherwise misfire a retry on perfectly good extractions.
        if raw_count > populated_count:
            mismatches.append(
                {
                    "index": 0,
                    "id": questions[0].get("id"),
                    "reasons": [
                        f"raw text has {raw_count} Recommended:/Recommended text: occurrence(s) overall, "
                        f"but only {populated_count} question(s) in the returned JSON have a populated "
                        "recommendation"
                    ],
                }
            )

    return mismatches


def _build_retry_prompt(*, phase: str, text: str, mismatches: list[dict]) -> str:
    """Build the corrective follow-up prompt sent for issue #229's single
    retry -- re-invokes the same `/rhubarb:parse-interview` skill (so the
    parser session re-reads its own extraction rules) but leads with exactly
    what `_detect_extraction_mismatches` found, naming each affected
    question's index/id and reason, so the second attempt has a concrete
    signal to act on instead of a blind "try again."""
    mismatch_lines = []
    for mismatch in mismatches:
        id_suffix = f' (id "{mismatch["id"]}")' if mismatch.get("id") else ""
        for reason in mismatch["reasons"]:
            mismatch_lines.append(f"- question index {mismatch['index']}{id_suffix}: {reason}")

    return (
        f"/rhubarb:parse-interview phase: {phase}\n\n"
        "Your previous JSON extraction of the text below appears to have dropped information. "
        "Specifically:\n" + "\n".join(mismatch_lines) + "\n\n"
        "Please re-extract the SAME text and reply with a corrected, complete JSON object -- fixing "
        "only the issue(s) named above; every other question/field should stay exactly as it should "
        "already be. Reply with ONLY the JSON object, no other prose.\n\n"
        f"Text:\n{text}"
    )


def _tag_extraction_incomplete(data: dict, mismatches: list[dict]) -> dict:
    """Return a copy of `data` with `"extraction_incomplete": True` added to
    every question at an index named in `mismatches` -- every other
    question, and every other field on the affected ones, is left exactly
    as `data` already had it. Additive/backward-compatible: a question with
    no mismatch never gets the field at all (never explicitly `False`), and
    `data` itself is never mutated in place. A `mismatches` of `[]` (the
    retry actually fixed everything) returns `data` completely unchanged."""
    if not mismatches:
        return data

    affected_indices = {mismatch["index"] for mismatch in mismatches}
    questions = data.get("questions") or []
    tagged_questions = []
    for index, question in enumerate(questions):
        if index in affected_indices:
            question = dict(question)
            question["extraction_incomplete"] = True
        tagged_questions.append(question)

    tagged_data = dict(data)
    tagged_data["questions"] = tagged_questions
    return tagged_data


async def _retry_extraction_once(
    project_id: int,
    item: dict,
    first_data: dict,
    first_mismatches: list[dict],
    *,
    validator=_is_valid_grilling_shape,
) -> dict:
    """Send exactly one corrective retry turn to `project_id`'s parser
    session for `item`, naming `first_mismatches` (already found in
    `first_data`), and return whichever data should be treated final:

    - The retry's own response, re-validated the same way as a primary
      extraction (`_extract_json_object` + `validator`) and re-checked with
      `_detect_extraction_mismatches` against the SAME raw text -- if that
      response is usable, it wins even if it's still imperfect (tagged with
      `_tag_extraction_incomplete` for whatever the retry itself still got
      wrong, which may differ from `first_mismatches`), since it's "whatever
      the last attempt produced."
    - `first_data` itself, tagged with `first_mismatches`, if the retry
      turn fails outright (dead/missing session, or any other exception) or
      comes back unparseable/schema-invalid -- there is nothing better to
      prefer, and this function must never raise or attempt a second retry.

    `validator` defaults to `_is_valid_grilling_shape` (the flat
    grilling/QA/implement shape) -- the only shape `extract_with_validation`
    below ever actually calls this with a retry for today (its nested
    QA-grilling shape opts out of mismatch-detection/retry entirely, see that
    function's own docstring), but this stays parameterized rather than
    hardcoded so a future caller for a different shape isn't structurally
    blocked from reusing it.

    Exactly one `stream_turn` call happens here, no matter which branch is
    taken -- this function is only ever invoked once per queued item, from
    `_process_one_queued_item`/`extract_with_validation`, and never calls
    itself or loops."""
    retry_prompt = _build_retry_prompt(phase=item["phase"], text=item["text"], mismatches=first_mismatches)

    try:
        retry_result_text = None
        async for event in stream_turn(project_id, retry_prompt):
            if event.get("type") == "result":
                retry_result_text = event.get("result")
    except Exception:  # noqa: BLE001 -- a failed retry falls back to the first attempt, never raises
        return _tag_extraction_incomplete(first_data, first_mismatches)

    if retry_result_text is None:
        return _tag_extraction_incomplete(first_data, first_mismatches)

    retry_data = _extract_json_object(retry_result_text)
    if retry_data is None or not validator(retry_data):
        return _tag_extraction_incomplete(first_data, first_mismatches)

    retry_mismatches = _detect_extraction_mismatches(item["text"], retry_data)
    return _tag_extraction_incomplete(retry_data, retry_mismatches)


# ---------------------------------------------------------------------------
# Shared extraction entry point (PRD #227 follow-up, gap 1 -- discovered
# during manual testing/design review after #228/#229/#230 were already
# closed out in code: `gh issue view 227` for the parent PRD; #229's own
# validation/retry/flagging logic above was wired into `_process_one_queued_
# item` below (the async needs-input-queue consumer) but NOT into
# `session_runner._extract_questions_via_parser_session` -- the function that
# drives a turn completing LIVE in the open UI card, which is exactly the
# path PRD #227's original bug (a dropped Recommended:/options list) was
# reported on. `extract_with_validation` is the fix: the one implementation
# of "call the skill, validate, retry once on mismatch, tag
# extraction_incomplete", composed from the pieces above, that BOTH
# `_process_one_queued_item` and `session_runner._extract_questions_via_
# parser_session` now call, so there is never a second, divergent copy of
# this behavior.
# ---------------------------------------------------------------------------


async def extract_with_validation(
    project_id: int,
    text: str,
    *,
    phase: str,
    validator=_is_valid_grilling_shape,
    detect_mismatches: bool = True,
) -> dict | None:
    """Drive `project_id`'s already-running parser session to extract
    structured question/issue data out of `text` via the `/rhubarb:parse-
    interview` skill (`_build_extraction_prompt`), validate the result
    against `validator`, and -- when `detect_mismatches` is true -- run issue
    #229's local Recommended:/bulleted-options mismatch check with its single
    corrective retry before returning. This is the ONE shared implementation
    both `_process_one_queued_item` (the async needs-input queue) and
    `session_runner._extract_questions_via_parser_session` (a live turn's
    synchronous extraction) call -- see the section header above for why this
    exists as its own function.

    `phase`/`text` pass straight through to `_build_extraction_prompt`, the
    same uniform (no Python-side phase branching) invocation every phase
    already used -- the skill's OWN instructions are what may branch on
    `phase` now (see `SKILL.md`'s `qa_grilling_issues` section, PRD #227's
    gap 2).

    `validator` picks which schema the skill's response must match:
    `_is_valid_grilling_shape` (default) for the flat `{header, questions,
    footer}` shape grilling/QA/implement all use, or `ollama_rescue.
    _is_valid_qa_shape` for the nested `{prd, issues: [{questions}]}` shape
    QA-grilling's own issue-grouped display needs (see
    `session_runner._extract_qa_issues_via_skill`).

    `detect_mismatches` (default `True`) is issue #229's own validation/
    retry/flagging step, which is built entirely around the flat shape's
    single `"questions"` list (nearby-bullet/Recommended:-line counting
    keyed to individual questions in that one list -- see
    `_detect_extraction_mismatches`'s own docstring). It has deliberately NOT
    been generalized to the nested QA-grilling `issues[].questions`
    structure as part of this change: a caller using the nested shape
    (`validator=_is_valid_qa_shape`) passes `detect_mismatches=False`,
    skipping straight from a valid parse to a returned result with no
    mismatch-retry safety net for that shape yet. This narrower scope (vs.
    generalizing the mismatch detector itself) is a deliberate judgment call
    given the size of that generalization -- a known follow-up gap, not
    something silently dropped.

    Returns, on a schema-valid response: for the flat shape,
    `{header, questions, footer, source: "parser_session"}` (now possibly
    carrying `extraction_incomplete: true` on individual questions); for the
    nested shape (`detect_mismatches=False`), the skill's own parsed `{prd,
    issues}` dict with `source: "parser_session"` added, unchanged otherwise.
    Returns `None` on any failure -- a dead/missing parser session, a turn
    that produced no `result` event, or a response that isn't valid JSON
    matching `validator` -- the same failure contract `_extract_questions_
    via_parser_session` already had before this refactor."""
    prompt = _build_extraction_prompt(phase=phase, text=text)

    try:
        result_text = None
        async for event in stream_turn(project_id, prompt):
            if event.get("type") == "result":
                result_text = event.get("result")
    except Exception:  # noqa: BLE001 -- a dead/failed parser session is just "extraction failed"
        return None

    if result_text is None:
        return None

    data = _extract_json_object(result_text)
    if data is None or not validator(data):
        return None

    if not detect_mismatches:
        return {**data, "source": "parser_session"}

    mismatches = _detect_extraction_mismatches(text, data)
    if mismatches:
        data = await _retry_extraction_once(
            project_id, {"phase": phase, "text": text}, data, mismatches, validator=validator
        )

    return {"header": data["header"], "questions": data["questions"], "footer": data["footer"], "source": "parser_session"}


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
# Legacy fallback on parse failure (issue #194, `gh issue view 194` for
# original context; retired by issue #230, `gh issue view 230`).
#
# When a queued item's primary extraction (via this project's parser
# session, above) errors, times out, or comes back not matching the
# expected schema, that one turn used to fall back to a regex-parser +
# Ollama-rescue extraction pipeline (`qa_parser.parse_grilling_response` /
# `ollama_rescue.rescue_grilling_response`) instead of just giving up.
# Issue #230 confirmed that chain fully dead/retired across every phase that
# uses this pipeline (grilling, QA, QA-grilling, implement all now go
# through the parser-session/`/rhubarb:parse-interview` skill uniformly, with
# no per-phase fallback machinery) and removed it -- `_legacy_fallback_
# extract` below is kept only as the named call site `_fallback_or_failure`
# already has, now always returning `None` (there is nothing left to fall
# back TO), so a primary extraction failure is just a tagged failure.
#
# Retry-per-turn, not a permanent downgrade (PRD #187's explicit
# requirement): nothing here reads or writes any project-level or
# session-level state. Every call to `_process_one_queued_item` decides
# primary-vs-failure fresh, purely from whether THIS item's own primary
# attempt succeeded -- so a project that just had a failure is attempted via
# the parser session completely normally on its very next queued item, and
# a string of failures never accumulates into any kind of sticky "this
# project is on the fallback pipeline now" state.
#
# Tagging: every result from a failed primary extraction is tagged
# `"source": "fallback"` (`_tagged_fallback_failure`) -- distinct from a
# clean primary success's `"source": "parser_session"` -- so `prompt.html`'s
# routing can still raise its failure-variant toast (issue #188's toast
# component) telling the user parsing failed for this turn, even though
# there is no longer any second extraction attempt that might recover a
# usable question from it.
# ---------------------------------------------------------------------------


def _tagged_fallback_failure(item: dict, *, error: str) -> dict:
    return {"source_session_id": item["card_id"], "ok": False, "error": error, "source": "fallback"}


async def _legacy_fallback_extract(text: str) -> dict | None:
    """Issue #230: the pre-existing regex-parser + Ollama-rescue extraction
    chain (`qa_parser.parse_grilling_response` / `ollama_rescue.
    rescue_grilling_response`) that used to run here on one failed item's raw
    turn `text` is fully retired -- the parser-session/`/rhubarb:parse-
    interview` skill pipeline is the ONLY extraction mechanism now, uniformly
    across every phase, so there is nothing left to fall back to. Always
    returns `None` (a genuine "nothing extractable" outcome, matching how
    `session_runner._extract_questions_via_parser_session` already reports
    its own failures) -- kept as a named function only so `_fallback_or_
    failure` below has a stable call site documenting where a future
    fallback mechanism, if any is ever added, would plug back in."""
    return None


async def _fallback_or_failure(item: dict, *, primary_error: str) -> dict:
    """Called from every failure branch of `_process_one_queued_item` below
    once the primary parser-session extraction has failed. Issue #230
    retired the legacy regex/Ollama-rescue fallback `_legacy_fallback_extract`
    used to attempt here -- it now always returns `None`, so this always
    returns `_tagged_fallback_failure` (still carrying `primary_error`),
    tagged `"source": "fallback"` so the frontend can always tell a primary
    failure happened and raise its failure-variant toast."""
    fallback = await _legacy_fallback_extract(item["text"])
    if fallback is not None:
        raise AssertionError("unreachable: _legacy_fallback_extract always returns None (issue #230)")
    return _tagged_fallback_failure(item, error=primary_error)


async def _process_one_queued_item(project_id: int, item: dict) -> dict:
    """Send one dequeued item's raw turn `text` through `project_id`'s
    parser session and return a tagged result -- `_tagged_success` on a
    clean primary-path parse, or (issue #194) whatever `_fallback_or_failure`
    produces once the primary path has failed. Every possible result always
    carries `source_session_id` (`item["card_id"]`) and `ok`, so a caller
    can branch on `ok` without needing to know anything about why a failure
    happened.

    The actual extraction (build the skill prompt, send the turn, validate,
    retry once on a detected mismatch, tag `extraction_incomplete`) is
    `extract_with_validation` above -- the shared implementation this
    function no longer duplicates (PRD #227 follow-up, gap 1: this used to
    inline all of that here, which is exactly why `session_runner.
    _extract_questions_via_parser_session`'s own copy of the live-turn path
    could silently drift out of sync with it). This function's only jobs now
    are queue-specific: turning a `None` (any failure -- dead/missing parser
    session, no `result` event, unparseable JSON, or a response that doesn't
    match the expected schema; `extract_with_validation` itself never raises)
    into `_fallback_or_failure`'s tagged-failure shape, and wrapping a
    successful extraction in `_tagged_success`'s `source_session_id`/`ok`
    envelope.

    Never raises: `extract_with_validation` swallows every failure mode
    itself and returns `None` rather than propagating, and the defensive
    `try`/`except` here exists only so a bug in that contract still can't
    take down `drain_needs_input_queue`'s processing of the rest of the
    queue."""
    try:
        data = await extract_with_validation(project_id, item["text"], phase=item["phase"])
    except Exception as exc:  # noqa: BLE001 -- any subprocess/engine failure is a per-item failure, not a crash
        return await _fallback_or_failure(item, primary_error=str(exc))

    if data is None:
        return await _fallback_or_failure(
            item,
            primary_error=(
                "parser session extraction failed: no result event, unparseable JSON, or a response that "
                "did not match the expected question schema"
            ),
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
    (`_tagged_success`, or -- issue #194, retired by #230 --
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
