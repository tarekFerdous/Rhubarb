"""Interactive, PTY-backed engine for driving `claude` (see issue #84).

`cli_client.run_prompt`/`stream_prompt` spawn `claude -p` (headless/print
mode) as a one-shot subprocess per turn: the process exits when the turn is
done, which is how those functions know to stop reading. This module spawns
`claude` in full *interactive* mode (no `-p`) instead, inside a pseudoterminal
(a Windows ConPTY, or a Unix pty on macOS/Linux -- see "Cross-platform
backends" below) that this engine owns and drives directly, and keeps that
one process alive across turns.

This is a NEW engine, additive only -- `cli_client.run_prompt`/`stream_prompt`
are untouched and remain what every current caller (`session_runner.py`)
uses. `PtyEngine` is not wired into `session_runner.py` in this change; a
later issue decides whether/how to switch callers over. Its public shape is
deliberately close to `stream_prompt`'s, though, so that switch is a drop-in
later: `stream_turn()` is an async generator of turn-event dicts carrying a
`session_id`, mirroring the `{"type": ..., "session_id": ...}` events
`stream_prompt` already yields and `session_runner.py` already parses.

Same auth rule as `cli_client`: `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`
are stripped from the child environment so the subprocess authenticates via
the user's Claude subscription login, never pay-per-token API billing (see
CLAUDE.md). Reused straight from `cli_client` rather than duplicated.

## The turn-complete marker

Headless `-p` mode has a free "turn is done" signal: the process exits.
Interactive mode has no such signal -- the process stays up waiting for the
next line of input forever. Something has to tell this engine, from inside
the PTY's own output stream, "the assistant is done responding to this
turn, hand control back."

The approach chosen here: `--append-system-prompt` (present in `claude
--help` on this build) appends an instruction to the assistant's system
prompt telling it to print a single, distinctive marker line -- exactly
`<<<RHUBARB_TURN_COMPLETE>>>` and nothing else -- immediately after finishing
its response to every user turn. `stream_turn()` accumulates PTY output
into a buffer and stops reading as soon as that literal string appears in
it, then splits the buffer on the marker and yields everything before it as
the turn's captured text.

Why this over alternatives:
- There is no flag to make interactive mode behave like `-p` (exit after
  one turn) without giving up the persistent-process/PTY model this issue
  asks for.
- The marker is plain assistant *output*, not a control-plane signal, so it
  costs nothing beyond one line of text and needs no CLI/protocol support
  beyond a flag that already exists (`--append-system-prompt`).
- `<<<RHUBARB_TURN_COMPLETE>>>` is chosen to be extremely unlikely to appear
  in ordinary Claude output (code, prose, or the fenced-JSON markers Rhubarb
  already parses elsewhere -- `implement_blocked`, `qa_grilling`; see
  `session_runner._parse_implement_blocked_block`/`_parse_qa_grilling_block`)
  while still being trivially greppable. Those fenced-JSON blocks are left
  completely alone by this engine -- it only ever looks for its own marker
  string and otherwise passes captured text through unmodified, so they
  keep parsing unchanged out of the engine's `result` text exactly as they
  do out of `stream_prompt`'s.
- A session-scoped system-prompt append (rather than, say, asking the user
  to type a magic phrase, or trying to detect the interactive prompt box
  redraw via ANSI parsing) survives `--resume` reattachment too, since it's
  passed again on every spawn -- including the fresh PTY spawned to
  reattach to an existing `claude_session_id`.

## Session id

Interactive mode doesn't hand back a `session_id` in a parsable final
event the way `--output-format json`/`stream-json` do for `-p`. To
avoid having to scrape one out of terminal UI chrome, this engine instead
*assigns* the session id itself: on a fresh (non-resume) spawn it generates
a UUID and passes it via `--session-id <uuid>` (documented in `claude
--help` as "Use a specific session ID for the conversation"), so the id is
known up front rather than discovered after the fact. Reattaching passes
that same id back via `--resume <uuid>`.

## Cross-platform backends (issue #85)

`PtyEngine` itself is fully OS-agnostic -- turn-marker detection,
`--resume` reattachment, and `--session-id`-on-first-spawn all live here and
never touch the OS directly. Only *spawning* the pseudoterminal is
platform-specific, and that's isolated behind the `PtyBackend` Protocol via
dependency injection (`pty_factory`):

- Windows: `_spawn_winpty`, using `pywinpty` (wraps ConPTY).
- macOS/Linux: `_spawn_unix_pty`, using `ptyprocess` (a standard, widely
  used Unix `pty` wrapper -- also what `pexpect` builds on). Its
  `PtyProcessUnicode.spawn(...)` is used specifically (rather than the
  plain bytes-oriented `PtyProcess`) so `write`/`read` deal in `str`,
  matching `PtyBackend` exactly the same way `winpty.PtyProcess` already
  does, with no bytes<->str adapter needed in this module.

`_default_pty_factory()` picks between them based on `platform.system()` so
`PtyEngine(...)` (no explicit `pty_factory`) does the right thing
automatically on whichever OS it runs on; tests keep injecting a fake
`PtyBackend` directly and never exercise this selection against a real
process.

## Crash/restart recovery (issue #86)

The PTY/process backing a `PtyEngine` can die mid-turn for reasons that
have nothing to do with the conversation itself: a transient PTY hiccup,
the `claude` binary crashing, or -- notably -- Rhubarb's own process having
been restarted while a phase was mid-flight, so a brand-new `PtyEngine` is
constructed with `resume_session_id=` for a conversation whose original
backing process is long gone before a single turn is ever sent through
*this* instance. All of these look identical from `stream_turn`'s point of
view: the backend raises `EOFError` on `read()`, or `read()` returns
nothing and `isalive()` says the process is gone.

`stream_turn` handles exactly one such death by transparently respawning
the process -- reattaching with `--resume <claude_session_id>` regardless
of whether this engine's *first* spawn used `--resume` or a fresh
`--session-id`, since by the time a restart is needed the conversation
already exists under that id server-side -- and resending the same prompt.
If that retried turn ALSO dies the same way, `stream_turn` gives up rather
than retrying again, and raises `PtyEngineUnrecoverableError` (see its
docstring for the failure shape a future caller should key off of).

## Live console trace (issue #160)

There was previously no way to *watch* a PTY-driven turn happen short of
attaching a live-terminal-view consumer to the `terminal_output` events, or
waiting for the single `[pty result]` line `stream_turn` already printed
once a turn finished. That makes an unattended run started with `python -m
uvicorn rhubarb.web.app:app --reload` opaque while it's in flight, for
every phase that goes through `PtyEngine` (grilling, creating_prd,
creating_issues, implementing, qa, ...), not just grilling.

`_stream_chunks_until_marker` now re-resolves the accumulated buffer-so-far
through `_render_terminal_text` -- the exact same function that later
produces `stream_turn`'s final `result` text -- after every single raw
chunk it reads off the PTY, and prints that resolved state straight to this
process's stdout. This is unconditional: no environment variable or config
flag gates it, and it applies uniformly to every PTY-driven turn regardless
of phase. It is purely a console side effect -- it changes nothing about
what `_stream_chunks_until_marker`/`stream_turn` yield to callers.
"""

import asyncio
import platform
import re
import unicodedata
import uuid
from collections.abc import AsyncIterator
from typing import Protocol

import pyte
from wcwidth import wcwidth

from rhubarb.cli_client import _clean_env, _effort_args, _plugin_args

# Printed by the assistant (via --append-system-prompt, below) as the last
# line of every turn's output. Chosen to be inert as ordinary Claude output
# (code fences, prose, or Rhubarb's own fenced-JSON markers like
# `implement_blocked`/`qa_grilling` never produce this literal string) while
# staying trivially detectable in a raw PTY byte/text stream.
TURN_COMPLETE_MARKER = "<<<RHUBARB_TURN_COMPLETE>>>"

_MARKER_INSTRUCTION = (
    "After you finish your ENTIRE response to a user turn (including any "
    "tool use), print a new line containing exactly this text and nothing "
    f"else on that line: {TURN_COMPLETE_MARKER}\n"
    "Always print this line, for every turn, with no exceptions -- even if "
    "you errored, were interrupted, or have nothing else to say. Never "
    "print it anywhere except as the very last line of a turn."
)

# Read chunk size for polling the PTY. Small enough not to over-buffer,
# large enough that a normal turn doesn't need many round trips.
_READ_CHUNK = 4096

# Quiet-period timeout (issue #169, child of PRD #168 "Recover from a
# stalled turn instead of hanging the turn lock forever"): `_stream_chunks_
# until_marker`'s read loop previously issued one plain blocking read per
# iteration with no timeout at all -- if the live CLI process ever went
# quiet without printing `TURN_COMPLETE_MARKER` (suspected trigger: the
# raw-passthrough typing feature, issue #165, interfering with the CLI
# mid-turn -- though the fix below does not depend on knowing the exact
# cause), that read blocked forever, and the per-card turn lock it's held
# under in `session_runner._run_turn` never released -- every later submit
# for that card then permanently failed with "Another turn for this session
# is already in progress."
#
# `_stream_chunks_until_marker` now wraps each read in a `Task` and awaits it
# via `asyncio.wait(..., timeout=_STALL_QUIET_PERIOD_SECONDS)` instead of a
# bare `await`. Critically, `asyncio.wait` does NOT cancel a task that times
# out -- unlike `asyncio.wait_for`, which would cancel (and therefore lose)
# the pending read -- so on a timeout this only yields a `stall` event and
# loops back to `asyncio.wait` on the exact SAME still-pending read task,
# never issuing a second, concurrent read against the same PTY and never
# losing whatever data eventually arrives. A stall can recur any number of
# times on the same pending read before it finally resolves.
#
# A module-level constant (not hardcoded inline) so a test can monkeypatch
# it down to a tiny value (see `tests/test_pty_engine.py`'s stall tests) and
# exercise this path without a real multi-second sleep -- referenced by bare
# name at call time (like `_render_terminal_text` already is elsewhere in
# this module), so monkeypatching `pty_engine._STALL_QUIET_PERIOD_SECONDS`
# takes effect immediately, with no reload needed.
_STALL_QUIET_PERIOD_SECONDS = 5

# Virtual screen size for `_render_terminal_text`'s terminal emulation.
# `_VIRTUAL_SCREEN_COLUMNS` is the DEFAULT column width only -- issue #166
# (dynamic PTY resize) made the real PTY's width per-session state (see
# `PtyEngine._cols` below, seeded from this default but changed at runtime
# by `PtyEngine.resize()`), so `_render_terminal_text` now takes a
# `columns` argument instead of always reading this module constant
# directly; every call site inside `PtyEngine` passes `self._cols` so the
# virtual re-render screen width can never drift from whatever the real
# PTY was last resized to (see `resize()`'s docstring for how that
# invariant is preserved). `_VIRTUAL_SCREEN_LINES` (the scrollback depth of
# the virtual re-render, unrelated to the real PTY's row count) stays a
# plain module constant -- deliberately far larger than any real terminal
# Claude Code itself would have been given, so a normal turn's output is
# never truncated and never forced to re-wrap differently than the source
# terminal already did.
_VIRTUAL_SCREEN_COLUMNS = 200
_VIRTUAL_SCREEN_LINES = 4000

# Paced-write tuning (issue #148): writing an entire prompt to the PTY in
# one atomic `write(prompt + "\r")` call can trip the `claude` CLI's own
# paste-detection heuristic -- a large block of text arriving in a single
# write looks like a pasted block, not human typing, and the CLI can treat
# the trailing "\r" as part of that pasted content rather than as an Enter
# keypress submitting it, leaving the prompt sitting unsubmitted in the
# input box. Splitting the prompt into small chunks, written with a short
# delay between each, mimics the arrival pattern of real keystrokes closely
# enough to avoid that misdetection; writing the trailing "\r" as its own
# separate write (after a slightly longer pause) keeps it from ever being
# glued onto the last text chunk, so it reads unambiguously as a distinct
# Enter keystroke. This is applied uniformly to every submission regardless
# of length or content -- paste-detection reacts to how bytes arrive at the
# PTY, not what they say, so there is no "risky prompt" heuristic to gate
# it on. Values are implementation-tuned constants, not user-configurable.
_WRITE_CHUNK_SIZE = 32
_WRITE_CHUNK_DELAY_SECONDS = 0.02
_WRITE_FINAL_DELAY_SECONDS = 0.05

# DEFAULT window size a new `PtyEngine` seeds itself with (issue #166: these
# are no longer the only size a session ever runs at -- see `PtyEngine.__init__`'s
# `rows`/`cols` parameters and `PtyEngine.resize()`, which make the real
# per-session size mutable at runtime). `pywinpty`/`ptyprocess` both default
# to a plain 80x24 if not told otherwise, which is narrow enough that Claude
# Code word-wraps its own question/option text across multiple physical
# lines -- breaking `qa_parser.py`'s single-line field matching.
# `_PTY_COLUMNS` intentionally reuses `_VIRTUAL_SCREEN_COLUMNS` (rather than
# a separately hardcoded number) so a freshly-constructed engine's real PTY
# width and its `_render_terminal_text` virtual re-render screen width start
# out equal; `resize()` is what keeps them equal from then on as either one
# changes at runtime.
_PTY_ROWS = 50
_PTY_COLUMNS = _VIRTUAL_SCREEN_COLUMNS


# Matches a CSI parameter block (digits/semicolons/colons) immediately
# followed by its final byte -- used by `_desubparameterize_csi_sequences`
# below to rewrite colon-delimited SGR subparameters (`CSI 4:3 m`, the
# ISO-8613-6 style some terminal UI libraries emit for e.g. curly
# underlines, and 24-bit colors written `CSI 38:2::r:g:b m`) to the
# semicolon-delimited form pyte's parser actually understands.
_CSI_COLON_PARAMS_RE = re.compile(r"(\x1b\[[0-9:;]*):([0-9:;]*[A-Za-z])")


def _desubparameterize_csi_sequences(raw: str) -> str:
    """Rewrite `:`-delimited CSI parameters to the `;`-delimited form.

    pyte's CSI parser (`pyte.streams.Stream._parser_fsm`) only recognizes
    `;` as a parameter separator. On an unrecognized separator character --
    which includes `:`, valid per ECMA-48/ISO-8613-6 and emitted by some
    terminal UI libraries for SGR subparameters (curly-underline styles,
    24-bit color `CSI 38:2::r:g:bm`) -- it treats that character as if it
    were the sequence's OWN final byte: it dispatches immediately (to a
    no-op debug handler, since e.g. `:` isn't a real CSI final byte) and
    returns to plain-text mode. Every character after the `:` up to the
    real final byte (`m`, etc.) is then read back out of CSI mode and drawn
    onto the screen as literal, visible text -- e.g. `\\x1b[4:3mgate` renders
    as `3mgate`, not `gate`. Repeated application handles more than one
    colon-delimited sequence (and more than one colon within a single
    sequence) in the same input."""
    rewritten = _CSI_COLON_PARAMS_RE.sub(r"\1;\2", raw)
    while rewritten != raw:
        raw = rewritten
        rewritten = _CSI_COLON_PARAMS_RE.sub(r"\1;\2", raw)
    return rewritten


def _strip_unadvancing_format_characters(raw: str) -> str:
    """Drop Unicode format/zero-width characters (e.g. a variation
    selector like U+FE0F completing an emoji, a zero-width joiner joining
    two emoji into one, a zero-width space) that `pyte.Screen.draw` cannot
    place on the screen.

    `Screen.draw` advances the cursor by each character's `wcwidth()` and
    only knows how to handle three cases: a normal (width 1 or 2) character,
    or a *combining* (`unicodedata.combining(char)` truthy) zero-width
    character, which it merges into the previous cell. Anything else with
    zero or negative width -- a non-combining format character, or a
    codepoint `wcwidth` doesn't recognize at all (-1) -- hits `draw`'s
    `else: break`, which silently abandons the REST of that `draw()` call's
    text, not just the offending character. Since a format character never
    occupies a screen cell on a real terminal either, dropping it here
    (rather than handing it to pyte) changes nothing about what a person
    watching the real terminal would see, while avoiding that crash-stop.
    Only codepoints above the C0/C1 control range are considered, so
    control/escape bytes pyte's own parser (not `draw`) is responsible for
    (`\\x1b`, `\\x9b`, etc.) are never touched here."""
    return "".join(
        char
        for char in raw
        if not (ord(char) > 0x9F and wcwidth(char) <= 0 and not unicodedata.combining(char))
    )


def _render_terminal_text(raw: str, columns: int = _VIRTUAL_SCREEN_COLUMNS) -> str:
    """Resolve `raw` -- true terminal-emulator input (ANSI escape codes,
    cursor movement, and all), exactly as a real PTY produced it -- to the
    plain text a person watching a real terminal would see once every
    redraw/animation frame has settled to its final state.

    Naively regex-stripping escape codes is not enough: an animated spinner
    redrawing the same line in place would still concatenate every frame's
    literal characters end-to-end. Feeding the raw bytes through a real
    terminal emulator (`pyte`) and reading back the resolved screen content
    is what correctly collapses that down to the final line, the same way
    the live `xterm.js` terminal view already renders this same raw stream
    correctly.

    `columns` (issue #166) is the virtual screen's width -- every caller
    inside this module passes the calling `PtyEngine`'s own `self._cols`,
    the SAME number that engine's real PTY was last resized to (see
    `PtyEngine.resize()`), so this virtual re-render can never drift out of
    sync with the real terminal width that produced `raw` in the first
    place. It defaults to `_VIRTUAL_SCREEN_COLUMNS` only so this function
    remains callable on its own (e.g. in a test) without a `PtyEngine`.

    Only rows up to (and including) the one the cursor ended on are kept --
    `pyte.Screen.display` always returns exactly `_VIRTUAL_SCREEN_LINES`
    rows regardless of how much was actually written, and every row is
    padded to `columns` with spaces, so both are trimmed back off to
    recover the real content and its original line count.

    Before `raw` reaches pyte, two pyte parser/screen defects that
    otherwise corrupt or drop ordinary text are worked around --
    see `_desubparameterize_csi_sequences` and
    `_strip_unadvancing_format_characters` for exactly what each one fixes
    and why. Both are no-ops on input that doesn't trigger them, so
    ordinary text/ANSI resolves exactly as already documented above."""
    raw = _desubparameterize_csi_sequences(raw)
    raw = _strip_unadvancing_format_characters(raw)
    screen = pyte.Screen(columns, _VIRTUAL_SCREEN_LINES)
    pyte.Stream(screen).feed(raw)
    rows = screen.display[: screen.cursor.y + 1]
    return "\n".join(row.rstrip() for row in rows)


class PtyEngineError(RuntimeError):
    pass


class PtyEngineUnrecoverableError(PtyEngineError):
    """Raised by `stream_turn` when the underlying process dies mid-turn a
    SECOND time in a row -- once, and then again after this engine's one
    automatic `--resume` restart attempt (see the module docstring's
    "Crash/restart recovery" section and issue #86). A plain
    `PtyEngineError` no longer reaches a caller for the ordinary
    "process died mid-turn" case -- that is now always retried once,
    transparently, inside `stream_turn` itself. This subclass is what's
    left over once the retry doesn't help either: no longer a transient PTY
    hiccup, but an infra-level failure a human needs to see and act on.

    This is the "distinguishable failure shape" issue #86 asks for, sized
    for a future caller (issue #87, not this change) to catch this specific
    subclass around a `stream_turn` call and route it into the existing
    `implement_blocked`-style blocked-card flow (see
    `session_runner._parse_implement_blocked_block`) instead of a generic
    turn-error path -- e.g. by synthesizing a blocked-card payload from
    `claude_session_id` plus `str(exception)`, the same way a genuine
    `implement_blocked` JSON block already suspends a session today.

    `claude_session_id` is preserved on the instance (same value as
    `PtyEngine.claude_session_id` at the moment of failure) so a caller can
    still offer to resume this conversation later, even though this engine
    instance itself gives up rather than retrying indefinitely.
    """

    def __init__(self, message: str, *, claude_session_id: str):
        super().__init__(message)
        self.claude_session_id = claude_session_id


class PtyBackend(Protocol):
    """The minimal surface `PtyEngine` needs from a spawned PTY process --
    matches `winpty.PtyProcess`'s (Windows) and `ptyprocess.PtyProcessUnicode`'s
    (macOS/Linux) public APIs closely enough that both real backends
    (`_spawn_winpty`, `_spawn_unix_pty`) are thin adapters, and a test
    double only needs to implement this much."""

    def write(self, data: str) -> int: ...

    def read(self, size: int = _READ_CHUNK) -> str:
        """Return the next chunk of output. Implementations should raise
        `EOFError` when the process has exited and no more output remains
        (this is `winpty.PtyProcess.read`'s and `ptyprocess.PtyProcess`'s
        own behavior)."""
        ...

    def isalive(self) -> bool: ...

    def terminate(self, force: bool = False) -> None: ...

    def resize(self, rows: int, cols: int) -> None:
        """Resize the underlying PTY's window size at runtime (issue #166:
        dynamic PTY resize, so a live session's real terminal can be kept in
        sync with the frontend xterm.js panel's actual pixel size instead of
        staying fixed at whatever size it was first spawned with).

        Neither real backend names this method `resize` itself --
        `winpty.PtyProcess` and `ptyprocess.PtyProcessUnicode` both expose
        the equivalent as `setwinsize(rows, cols)` (`_spawn_winpty`/
        `_spawn_unix_pty` below alias `.resize` to that method right after
        spawning, so both conform to this Protocol without a wrapper
        class)."""
        ...


def _spawn_winpty(
    argv: list[str], *, cwd: str | None, env: dict, rows: int = _PTY_ROWS, cols: int = _PTY_COLUMNS
) -> PtyBackend:
    """Real backend: spawn `argv` inside a Windows ConPTY via `pywinpty`.

    Imported lazily so importing this module (e.g. for tests, which always
    inject a fake `pty_factory`) never requires `pywinpty` to be installed
    on non-Windows dev/CI machines.

    Passes an explicit `dimensions=` (`rows`/`cols`, defaulting to
    `_PTY_ROWS`/`_PTY_COLUMNS` -- see those constants' docstring) instead of
    relying on `winpty.PtyProcess.spawn`'s own 80x24 default. `PtyEngine`
    passes its own current `self._rows`/`self._cols` here when it resolved
    this function itself as the default backend (see `PtyEngine.__init__`
    and `start()`); a directly-injected test `pty_factory` never calls this
    function at all, so those defaults are what a direct unit test of this
    function alone (no `PtyEngine` involved) exercises.

    `winpty.PtyProcess` has no method literally named `resize` -- it's
    aliased here, right after spawn, to the object's own `setwinsize`
    (issue #166) so the returned object satisfies `PtyBackend.resize`."""
    import winpty

    proc = winpty.PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=(rows, cols))
    proc.resize = proc.setwinsize
    return proc


def _spawn_unix_pty(
    argv: list[str], *, cwd: str | None, env: dict, rows: int = _PTY_ROWS, cols: int = _PTY_COLUMNS
) -> PtyBackend:
    """Real backend: spawn `argv` inside a standard Unix pty via
    `ptyprocess` (macOS/Linux).

    Uses `PtyProcessUnicode` rather than the base `PtyProcess` so
    `write`/`read` operate on `str` (UTF-8 decoded/encoded internally),
    matching `PtyBackend` without a bytes<->str adapter -- the same shape
    `winpty.PtyProcess` already provides on Windows.

    Imported lazily so importing this module (e.g. for tests, which always
    inject a fake `pty_factory`) never requires `ptyprocess` to be installed
    on Windows dev/CI machines, and so this module stays importable there
    even though `ptyprocess` is a Unix-only package.

    Passes an explicit `dimensions=` (`rows`/`cols`, defaulting to
    `_PTY_ROWS`/`_PTY_COLUMNS`) instead of relying on
    `ptyprocess.PtyProcessUnicode.spawn`'s own 80x24 default -- see
    `_spawn_winpty`'s docstring above for who actually supplies non-default
    values.

    `ptyprocess.PtyProcessUnicode` has no method literally named `resize`
    either -- aliased here to its own `setwinsize`, same as `_spawn_winpty`
    above."""
    import ptyprocess

    proc = ptyprocess.PtyProcessUnicode.spawn(argv, cwd=cwd, env=env, dimensions=(rows, cols))
    proc.resize = proc.setwinsize
    return proc


def _default_pty_factory():
    """Pick the real PTY backend for the platform this process is running
    on: ConPTY (`pywinpty`) on Windows, a standard Unix pty (`ptyprocess`)
    on macOS/Linux. Callers never need to know or care which OS they're on
    -- `PtyEngine()` with no explicit `pty_factory` just does the right
    thing.

    A function (not a module-level constant) so the `platform.system()`
    check happens at `PtyEngine()` construction time, not at import time --
    tests can monkeypatch `platform.system` and observe the effect without
    reloading this module.
    """
    if platform.system() == "Windows":
        return _spawn_winpty
    return _spawn_unix_pty


class PtyEngine:
    """One interactive `claude` process, owned and driven turn by turn
    through a PTY this engine controls directly (a ConPTY on Windows, a
    standard Unix pty on macOS/Linux -- see `_default_pty_factory`).

    `pty_factory(argv, *, cwd, env) -> PtyBackend` is injectable so tests
    never spawn a real `claude` process; it defaults to
    `_default_pty_factory()`, which selects the right real backend for the
    current platform automatically -- and, only in that default case, is
    also called with `rows=`/`cols=` (see `start()`) so the two real
    backends spawn at this engine's actual current size (issue #166).

    `rows`/`cols` (issue #166) seed this session's PTY size -- per-instance
    state now, not a fixed module-wide constant -- and can be changed at any
    time via `resize()`, which also keeps `_render_terminal_text`'s virtual
    re-render screen width (used by `stream_turn`'s `result` text and the
    live console trace) in lockstep with whatever the real PTY was last
    resized to.

    Mirrors `cli_client.stream_prompt`'s external shape on purpose: a
    prompt goes in, an async iterable of turn-event dicts comes out, and
    `claude_session_id` is available afterward for a future `--resume`.
    Unlike `stream_prompt`, one `PtyEngine` instance IS the long-lived
    process (no separate `card_id`-keyed pool) -- construct one per
    session and keep it around across turns, or drop it and construct a
    fresh one with `resume_session_id=` to reattach later.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        resume_session_id: str | None = None,
        pty_factory=None,
        rows: int = _PTY_ROWS,
        cols: int = _PTY_COLUMNS,
    ):
        self.cwd = cwd
        self.model = model
        self.effort = effort

        # Per-session PTY size (issue #166) -- seeded from `_PTY_ROWS`/
        # `_PTY_COLUMNS` by default, but no longer fixed for this engine's
        # whole lifetime: `resize()` updates these at runtime, and `self._cols`
        # is what every `_render_terminal_text` call below is resolved
        # against, so the virtual re-render width tracks the real PTY width
        # by construction -- there is only ever this one stored number, never
        # a second copy that could drift out of sync with it.
        self._rows = rows
        self._cols = cols

        # `pty_factory` is the test-injection seam: every existing caller
        # (all of `tests/test_pty_engine.py`) passes an explicit fake here
        # with the plain `(argv, *, cwd, env)` signature and expects it
        # called with exactly that -- no `rows`/`cols` kwargs. Only when NO
        # `pty_factory` is injected (real production use, and the
        # `_default_pty_factory` selection tests) does `start()` below pass
        # this engine's current `self._rows`/`self._cols` through, since only
        # the two real backends (`_spawn_winpty`/`_spawn_unix_pty`) accept
        # them.
        if pty_factory is not None:
            self._pty_factory = pty_factory
            self._pty_factory_takes_dimensions = False
        else:
            self._pty_factory = _default_pty_factory()
            self._pty_factory_takes_dimensions = True

        self._proc: PtyBackend | None = None

        # Known immediately -- see module docstring's "Session id" section --
        # rather than only after the child process starts talking.
        self.claude_session_id: str = resume_session_id or str(uuid.uuid4())
        self._is_resume = resume_session_id is not None

        # Guards every write to `self._proc` -- both the paced automated-turn
        # prompt writes below (`_stream_chunks_until_marker`) and the public
        # `write()` passthrough path (issue #165, raw interactive passthrough
        # -- see PRD #162) -- so the two writers can never physically
        # interleave bytes into the same PTY. `_stream_chunks_until_marker`
        # holds this lock for its ENTIRE paced write (every chunk plus the
        # trailing "\r"), not just per-chunk, precisely because a passthrough
        # write landing in the gap between two paced chunks would already be
        # byte-level interleaving from the PTY's point of view.
        self._write_lock = asyncio.Lock()

    def _build_args(self) -> list[str]:
        args = ["claude", "--dangerously-skip-permissions"]
        args += _plugin_args()
        args += ["--append-system-prompt", _MARKER_INSTRUCTION]
        if self._is_resume:
            args += ["--resume", self.claude_session_id]
        else:
            args += ["--session-id", self.claude_session_id]
        if self.model:
            args += ["--model", self.model]
        args += _effort_args(self.effort)
        return args

    def start(self) -> "PtyEngine":
        """Spawn (or, for a reattach, respawn) the PTY-backed `claude`
        process. Idempotent no-op if already started.

        Passes this engine's current `self._rows`/`self._cols` (issue #166)
        to the factory ONLY when it's the real default backend selection
        (see `__init__`'s `_pty_factory_takes_dimensions`) -- an injected
        test `pty_factory` is always called with just `(argv, cwd, env)`,
        matching every existing test's fake signature. Reading `self._rows`/
        `self._cols` here (rather than snapshotting them in `__init__`)
        means a `resize()` call made before this engine's first spawn --
        or before a crash-triggered respawn via `_restart_after_death` --
        is picked up: the fresh process comes up already sized to the
        latest known size, not this engine's construction-time default."""
        if self._proc is not None:
            return self
        env = _clean_env()
        if self._pty_factory_takes_dimensions:
            self._proc = self._pty_factory(
                self._build_args(), cwd=self.cwd, env=env, rows=self._rows, cols=self._cols
            )
        else:
            self._proc = self._pty_factory(self._build_args(), cwd=self.cwd, env=env)
        return self

    def resize(self, rows: int, cols: int) -> None:
        """Resize this session's PTY to `(rows, cols)` at runtime (issue
        #166) -- called when the frontend's xterm.js fit-addon recomputes
        the live-terminal panel's actual cols/rows (on load, and on every
        panel resize) and signals the new size to the backend, so the REAL
        pseudoterminal is resized to match what the user now sees, not just
        the on-screen xterm.js buffer.

        Always updates `self._rows`/`self._cols` first, even if no process
        is currently running (nothing left to forward the resize to yet --
        `start()` above will spawn at this new size instead) -- so a
        `resize()` called between sessions, or right after a mid-turn crash
        and before `_restart_after_death`'s respawn completes, is never
        silently lost.

        This is also what keeps `_render_terminal_text`'s virtual re-render
        screen width in lockstep with the real PTY width: `stream_turn`/
        `_stream_chunks_until_marker` always read `self._cols` fresh for
        every render call (see their `_render_terminal_text(..., self._cols)`
        calls below), so as soon as this method updates `self._cols`, the
        very next render -- even one already in flight for the current
        turn -- resolves against the new width. There is only ever this one
        stored width, never a second copy that could drift out of sync with
        it.

        Forwards to the live backend's own `resize()` (see `PtyBackend.resize`)
        only when a process is actually running; otherwise a no-op on the
        process side, since there's no process to resize yet."""
        self._rows = rows
        self._cols = cols
        if self._proc is not None:
            self._proc.resize(rows, cols)

    def _restart_after_death(self) -> None:
        """Tear down the dead process and respawn once, reattaching via
        `--resume <claude_session_id>` -- see the module docstring's
        "Crash/restart recovery" section. Forces `--resume` even if this
        engine's original spawn used `--session-id` (a fresh session that
        died before ever restarting still needs `--resume` to continue the
        same conversation on the second spawn, not another fresh
        `--session-id`, which would start an unrelated new conversation)."""
        self.close()
        self._is_resume = True
        self.start()

    async def _stream_chunks_until_marker(self, prompt: str) -> AsyncIterator[str | dict]:
        """Write `prompt` to the current process and yield each raw output
        chunk exactly as read from the PTY (issue #88 -- this is what lets a
        caller relay the real, unmodified terminal byte stream -- ANSI
        escapes, control characters, and all -- to a live-terminal-view
        consumer), stopping once `TURN_COMPLETE_MARKER` has appeared in the
        accumulated output. The marker line itself is yielded like any other
        chunk (this is genuinely what the PTY printed) -- callers that only
        want the turn's text strip it back out themselves, the same way
        `stream_turn` does for its `result` event below.

        Live console trace (issue #160): after every raw chunk is folded
        into the accumulated buffer, that buffer-so-far is re-resolved
        through `_render_terminal_text` -- the SAME function `stream_turn`
        uses to produce its final `result` text -- and printed to this
        process's stdout. This is unconditional (no debug flag gates it)
        and applies to every PTY-driven turn, for any phase (grilling,
        creating_prd, creating_issues, implementing, qa, ...), not just
        grilling -- so a person watching the console `python -m uvicorn
        rhubarb.web.app:app --reload` runs in can see the interactive
        session's screen resolve live, turn by turn, the same way a real
        terminal watching the raw PTY stream would, without needing to
        attach a separate live-terminal-view consumer.

        Stall detection (issue #169): each pending read is wrapped in a
        `Task` and awaited via `asyncio.wait(..., timeout=
        _STALL_QUIET_PERIOD_SECONDS)` -- NOT `asyncio.wait_for`, which would
        cancel (and lose) the pending read on a timeout. Every time that
        timeout elapses with the read still unresolved, this yields a
        `{"type": "stall", "data": <buffer-so-far rendered through
        _render_terminal_text>}` event and goes right back to waiting on the
        SAME still-pending task -- never a second, concurrent read against
        the same PTY, and never cancelled/replaced. This can recur any
        number of times before the read finally resolves. Once it does, this
        resumes exactly as before: the chunk (even an empty one, from a
        completed-but-empty read) is folded into the buffer/marker check
        below, with no special handling left over from having stalled.

        Raises `PtyEngineError` if the process ends first -- callers decide
        what to do with that (see `stream_turn`, which retries this once via
        `_restart_after_death` before giving up).

        Writes `prompt` as a paced sequence of small chunks (see issue #148
        and the `_WRITE_CHUNK_SIZE`/`_WRITE_CHUNK_DELAY_SECONDS` constants
        above) rather than one atomic write, with the trailing `"\\r"` sent
        as its own separate write after `_WRITE_FINAL_DELAY_SECONDS` -- see
        those constants' docstring for why. Applied unconditionally, for
        every prompt regardless of length or content."""
        assert self._proc is not None

        # Held for the whole paced sequence -- see `_write_lock`'s docstring
        # in `__init__` for why per-chunk locking wouldn't be enough.
        async with self._write_lock:
            for start in range(0, len(prompt), _WRITE_CHUNK_SIZE):
                text_chunk = prompt[start : start + _WRITE_CHUNK_SIZE]
                await asyncio.to_thread(self._proc.write, text_chunk)
                await asyncio.sleep(_WRITE_CHUNK_DELAY_SECONDS)

            await asyncio.sleep(_WRITE_FINAL_DELAY_SECONDS)
            await asyncio.to_thread(self._proc.write, "\r")

        buffer = ""
        while TURN_COMPLETE_MARKER not in buffer:
            read_task = asyncio.ensure_future(asyncio.to_thread(self._proc.read, _READ_CHUNK))
            while True:
                done, _pending = await asyncio.wait({read_task}, timeout=_STALL_QUIET_PERIOD_SECONDS)
                if read_task in done:
                    break
                # The read is still pending after a full quiet period --
                # surface a stall event carrying what's been seen so far, and
                # go back to waiting on this EXACT same task (not a new
                # read).
                yield {"type": "stall", "data": _render_terminal_text(buffer, self._cols)}

            try:
                chunk = read_task.result()
            except EOFError:
                raise PtyEngineError(
                    "claude PTY process ended before printing the turn-complete marker"
                ) from None
            if not chunk:
                if not self._proc.isalive():
                    raise PtyEngineError(
                        "claude PTY process ended before printing the turn-complete marker"
                    )
                continue
            buffer += chunk
            print(f"[pty live]\n{_render_terminal_text(buffer, self._cols)}")
            yield chunk

    async def stream_turn(self, prompt: str) -> AsyncIterator[dict]:
        """Send one prompt to the interactive session and yield turn-event
        dicts as output streams in, same shape as `stream_prompt`'s events
        (each carries `type` and `session_id`), PLUS (issue #88) a new
        `{"type": "terminal_output", "data": <chunk text>}` event interleaved
        BEFORE the final `result` event for every raw chunk read off the
        PTY as it arrives -- the real, unmodified terminal stream (ANSI
        escapes, control characters, and all), for a caller to relay live to
        a terminal-emulator UI. This is purely additive: the existing
        `system`/`result` event shapes, and the turn-completion/marker
        logic below that produces `result`, are unchanged.

        Starts the process on first use if `start()` wasn't already called.
        Reads the PTY in a background thread (the backend's `read()`
        blocks -- true of both `pywinpty` and `ptyprocess`) via
        `asyncio.to_thread` so this stays a well-behaved async generator
        instead of blocking the event loop.

        Stops reading as soon as `TURN_COMPLETE_MARKER` appears in the
        accumulated output -- that is this turn's end-of-turn signal, since
        (unlike `-p`) the process never exits on its own. Text after the
        marker (there normally isn't any -- the instruction asks for it to
        be the very last line) is dropped; text before it is resolved by
        `_render_terminal_text` (ANSI escapes, cursor movement, and redraws
        collapsed to their final rendered form) before becoming the turn's
        `result`, fenced-JSON markers like `implement_blocked`/`qa_grilling`
        included, readable exactly as a person watching the real terminal
        would see them. (The `terminal_output` chunks themselves are NOT
        touched by this -- they mirror the real PTY stream byte-for-byte,
        marker line included, same as a human watching the actual terminal
        would see, for the live-terminal-view consumer.)

        Crash/restart recovery (issue #86): if the process dies before
        printing the marker, this method automatically restarts it exactly
        ONCE -- reattaching via `--resume <claude_session_id>` -- and
        resends the same `prompt`, transparently to the caller (no event is
        emitted for the death or the restart; the caller just sees the
        turn take a little longer). If that retried attempt ALSO fails,
        `stream_turn` raises `PtyEngineUnrecoverableError` instead of
        retrying again -- see that class's docstring for the shape a
        caller should route into the blocked-card flow.

        Stall events (issue #169): `_stream_chunks_until_marker` can also
        yield a `{"type": "stall", ...}` dict (instead of a raw string
        chunk) whenever the pending read has gone quiet for
        `_STALL_QUIET_PERIOD_SECONDS`. Those are passed through here
        unmodified, exactly like `terminal_output` -- they carry no bearing
        on `buffer`/marker detection (nothing new was actually read) and do
        not end this loop; the underlying read this turn is waiting on is
        still pending underneath.
        """
        self.start()
        assert self._proc is not None

        yield {"type": "system", "subtype": "init", "session_id": self.claude_session_id}

        buffer = ""
        try:
            async for item in self._stream_chunks_until_marker(prompt):
                if isinstance(item, dict):
                    yield item
                    continue
                buffer += item
                yield {"type": "terminal_output", "data": item}
        except PtyEngineError as first_error:
            try:
                self._restart_after_death()
                buffer = ""
                async for item in self._stream_chunks_until_marker(prompt):
                    if isinstance(item, dict):
                        yield item
                        continue
                    buffer += item
                    yield {"type": "terminal_output", "data": item}
            except Exception as second_error:
                raise PtyEngineUnrecoverableError(
                    "claude PTY process died twice in a row for the same turn "
                    f"(first: {first_error}; after one automatic --resume restart: "
                    f"{second_error}) -- giving up after exactly one retry",
                    claude_session_id=self.claude_session_id,
                ) from second_error

        text, _marker, _trailing = buffer.partition(TURN_COMPLETE_MARKER)
        clean_text = _render_terminal_text(text, self._cols)
        # Debug visibility into this turn's rendered (not raw) PTY output --
        # what a round of questions/options resolves to after
        # `_render_terminal_text` collapses ANSI/cursor-movement redraws to
        # their final plain text, same text `qa_parser.py` parses.
        print(f"[pty result]\n{clean_text}")
        yield {"type": "result", "result": clean_text, "session_id": self.claude_session_id, "is_error": False}

    async def write(self, data: str) -> None:
        """Public write path for raw interactive passthrough (issue #165,
        child of PRD #162 -- "raw interactive passthrough mode"): forwards
        `data` straight to the underlying PTY process, exactly as a person
        typing directly into the terminal would produce, with no marker/
        prompt handling of any kind layered on top.

        Guarded by the same `_write_lock` the automated-turn write loop
        (`_stream_chunks_until_marker`) holds for its entire paced prompt
        write, so a passthrough write can never land in the middle of an
        in-flight automated turn's chunks (or vice versa) -- whichever
        writer gets the lock first completes its ENTIRE write before the
        other's bytes reach the PTY. Neither writer's bytes are ever
        physically interleaved into the underlying process's input stream.

        Raises `PtyEngineError` if the process hasn't been started yet (or
        was already closed) -- there is nothing to write to."""
        if self._proc is None:
            raise PtyEngineError("PtyEngine has not been started")
        async with self._write_lock:
            await asyncio.to_thread(self._proc.write, data)

    def isalive(self) -> bool:
        """Whether the underlying PTY process is still running -- False if
        it was never started, already closed, or has died on its own (e.g.
        a standby engine that crashed while sitting unclaimed; see issue
        #136). Never raises."""
        if self._proc is None:
            return False
        try:
            return self._proc.isalive()
        except Exception:
            return False

    def close(self) -> None:
        """Terminate the underlying PTY process, if one was started. Safe
        to call more than once or when never started."""
        if self._proc is None:
            return
        try:
            if self._proc.isalive():
                self._proc.terminate(force=True)
        except Exception:
            pass
        self._proc = None
