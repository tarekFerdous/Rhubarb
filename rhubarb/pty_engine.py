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
"""

import asyncio
import platform
import uuid
from collections.abc import AsyncIterator
from typing import Protocol

import pyte

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

# Virtual screen size for `_render_terminal_text`'s terminal emulation --
# deliberately far larger than any real terminal Claude Code itself would
# have been given, so a normal turn's output is never truncated and never
# forced to re-wrap differently than the source terminal already did.
_VIRTUAL_SCREEN_COLUMNS = 200
_VIRTUAL_SCREEN_LINES = 4000

# Window size the real PTY itself is spawned with. `pywinpty`/`ptyprocess`
# both default to a plain 80x24 if not told otherwise, which is narrow
# enough that Claude Code word-wraps its own question/option text across
# multiple physical lines -- breaking `qa_parser.py`'s single-line field
# matching. `_PTY_COLUMNS` intentionally reuses `_VIRTUAL_SCREEN_COLUMNS`
# (rather than a separately hardcoded number) so the real PTY's width and
# `_render_terminal_text`'s virtual re-render screen width can never drift
# out of sync with each other.
_PTY_ROWS = 50
_PTY_COLUMNS = _VIRTUAL_SCREEN_COLUMNS


def _render_terminal_text(raw: str) -> str:
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

    Only rows up to (and including) the one the cursor ended on are kept --
    `pyte.Screen.display` always returns exactly `_VIRTUAL_SCREEN_LINES`
    rows regardless of how much was actually written, and every row is
    padded to `_VIRTUAL_SCREEN_COLUMNS` with spaces, so both are trimmed
    back off to recover the real content and its original line count."""
    screen = pyte.Screen(_VIRTUAL_SCREEN_COLUMNS, _VIRTUAL_SCREEN_LINES)
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


def _spawn_winpty(argv: list[str], *, cwd: str | None, env: dict) -> PtyBackend:
    """Real backend: spawn `argv` inside a Windows ConPTY via `pywinpty`.

    Imported lazily so importing this module (e.g. for tests, which always
    inject a fake `pty_factory`) never requires `pywinpty` to be installed
    on non-Windows dev/CI machines.

    Passes an explicit `dimensions=` (see `_PTY_ROWS`/`_PTY_COLUMNS`)
    instead of relying on `winpty.PtyProcess.spawn`'s own 80x24 default --
    see those constants' docstring for why.
    """
    import winpty

    return winpty.PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=(_PTY_ROWS, _PTY_COLUMNS))


def _spawn_unix_pty(argv: list[str], *, cwd: str | None, env: dict) -> PtyBackend:
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

    Passes an explicit `dimensions=` (see `_PTY_ROWS`/`_PTY_COLUMNS`)
    instead of relying on `ptyprocess.PtyProcessUnicode.spawn`'s own 80x24
    default -- see those constants' docstring for why.
    """
    import ptyprocess

    return ptyprocess.PtyProcessUnicode.spawn(argv, cwd=cwd, env=env, dimensions=(_PTY_ROWS, _PTY_COLUMNS))


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
    current platform automatically.

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
    ):
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self._pty_factory = pty_factory or _default_pty_factory()
        self._proc: PtyBackend | None = None

        # Known immediately -- see module docstring's "Session id" section --
        # rather than only after the child process starts talking.
        self.claude_session_id: str = resume_session_id or str(uuid.uuid4())
        self._is_resume = resume_session_id is not None

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
        process. Idempotent no-op if already started."""
        if self._proc is not None:
            return self
        env = _clean_env()
        self._proc = self._pty_factory(self._build_args(), cwd=self.cwd, env=env)
        return self

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

    async def _stream_chunks_until_marker(self, prompt: str) -> AsyncIterator[str]:
        """Write `prompt` to the current process and yield each raw output
        chunk exactly as read from the PTY (issue #88 -- this is what lets a
        caller relay the real, unmodified terminal byte stream -- ANSI
        escapes, control characters, and all -- to a live-terminal-view
        consumer), stopping once `TURN_COMPLETE_MARKER` has appeared in the
        accumulated output. The marker line itself is yielded like any other
        chunk (this is genuinely what the PTY printed) -- callers that only
        want the turn's text strip it back out themselves, the same way
        `stream_turn` does for its `result` event below.

        Raises `PtyEngineError` if the process ends first -- callers decide
        what to do with that (see `stream_turn`, which retries this once via
        `_restart_after_death` before giving up)."""
        assert self._proc is not None

        await asyncio.to_thread(self._proc.write, prompt + "\r")

        buffer = ""
        while TURN_COMPLETE_MARKER not in buffer:
            try:
                chunk = await asyncio.to_thread(self._proc.read, _READ_CHUNK)
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
        """
        self.start()
        assert self._proc is not None

        yield {"type": "system", "subtype": "init", "session_id": self.claude_session_id}

        buffer = ""
        try:
            async for chunk in self._stream_chunks_until_marker(prompt):
                buffer += chunk
                yield {"type": "terminal_output", "data": chunk}
        except PtyEngineError as first_error:
            try:
                self._restart_after_death()
                buffer = ""
                async for chunk in self._stream_chunks_until_marker(prompt):
                    buffer += chunk
                    yield {"type": "terminal_output", "data": chunk}
            except Exception as second_error:
                raise PtyEngineUnrecoverableError(
                    "claude PTY process died twice in a row for the same turn "
                    f"(first: {first_error}; after one automatic --resume restart: "
                    f"{second_error}) -- giving up after exactly one retry",
                    claude_session_id=self.claude_session_id,
                ) from second_error

        text, _marker, _trailing = buffer.partition(TURN_COMPLETE_MARKER)
        clean_text = _render_terminal_text(text)
        yield {"type": "result", "result": clean_text, "session_id": self.claude_session_id, "is_error": False}

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
