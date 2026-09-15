"""Single-process, stream-json-backed engine for driving `claude` (see issue
#183, step 1 of PRD #182 -- `gh issue view 182` for full context).

`PtyEngine` (`rhubarb/pty_engine.py`) keeps one `claude` process alive across
turns by spawning it in full interactive mode inside a real pseudoterminal
and simulating keystrokes into it -- built on the belief that headless `-p`
mode has no way to stay alive across turns. That belief was wrong: PRD #182
empirically proved that `claude -p --input-format stream-json --output-format
stream-json` DOES support a single persistent subprocess handling multiple
turns over one still-open stdin pipe, with no PTY involved at all --
confirmed directly: spawn once, write one JSON turn to stdin, read a full
turn's NDJSON events ending in a `result` event (carrying `session_id`), then
write a SECOND JSON turn to the SAME still-open stdin -- identical
`session_id` both turns, process stayed alive, and the second turn's
`cache_creation_input_tokens` dropped drastically (served from cache) versus
the first turn's full prefix write.

This module is that mechanism, built as a standalone, self-contained unit --
it does not modify or import from `PtyEngine`, and nothing outside this
module (and its own tests) references it yet. Wiring a caller (`session_runner
.py`) up to this engine is issue #184, a separate future task.

## Why this transport has no PTY, no marker, no rendering

`PtyEngine` needs an injected "turn is complete" marker instruction (`
--append-system-prompt`, `TURN_COMPLETE_MARKER`) because interactive mode has
no built-in signal for "done with this turn, hand control back" -- something
has to be scraped out of the PTY's own output stream. It also needs a real
terminal emulator (`pyte`, `_render_terminal_text`) to resolve ANSI escapes
and cursor-movement redraws into plain text, because a PTY is a byte stream
meant for a human's eyes, not a structured protocol.

`--input-format stream-json --output-format stream-json` sidesteps both
problems structurally: turns are written as one JSON object per line on
stdin, output arrives as one JSON object per line on stdout, and the CLI
itself emits a native `{"type": "result", ...}` line -- carrying `session_id`,
`is_error`, and the turn's final text in a `result` field -- to mark a turn
complete. No marker string to inject or scan for, no ANSI/cursor-movement
rendering to reverse: `stream_json_engine.stream_turn` simply reads NDJSON
lines until it sees `"type": "result"`.

## Multi-line prompts (PRD #180/#181 regression)

`PtyEngine`'s keystroke-simulation transport suffered a real, hard-to-fix bug
class: a multi-line composed reply (e.g. a grilling round's answer to 2+
questions, joined with embedded newlines) could fail to submit reliably,
because simulating keystrokes into a live terminal risks an embedded `\n`
being misread as something other than plain text (paste-detection heuristics,
bracketed-paste-mode wrapping, etc. -- see PRD #180/#181, closed stale after a
real attempted fix still failed under real conditions).

This transport has no such failure mode, structurally: a prompt is written as
one JSON object's `content` string field (`{"type": "user", "message":
{"role": "user", "content": <text>}}\n`), and JSON string fields don't
interpret an embedded `\n` as anything but two literal characters inside the
string -- there is no keystroke stream for a newline to be misread in. See
`test_stream_json_engine.py`'s explicit regression test for this.

## Session id

Unlike `PtyEngine` (which assigns its own UUID up front via `--session-id`
since interactive mode never hands one back), this transport's native
`result` event carries a real `session_id` straight from the CLI itself, so
`StreamJsonEngine.session_id` starts as whatever `resume_session_id` the
caller passed in (or `None` for a brand-new session) and is simply updated
from each turn's `result` event as turns complete.

## Crash/restart recovery (mirrors PtyEngine's issue #86, adapted transport)

If the subprocess dies mid-turn -- `read_line()` raises `EOFError`, or a
blank read coincides with the process no longer being alive -- `stream_turn`
retries exactly once: tear down the dead process, respawn (reattaching via
`--resume <session_id>` if a session id is already known from a prior
completed turn or an explicit `resume_session_id`; otherwise a fresh spawn,
since there is no established session yet to resume), and resend the exact
same prompt, transparent to the caller. If that retried attempt ALSO fails,
`stream_turn` raises `StreamJsonEngineUnrecoverableError` instead of retrying
again -- mirroring `PtyEngineUnrecoverableError`'s "give up after exactly one
retry" contract.

## Environment / billing rule

Same hard project rule as `cli_client`/`PtyEngine` (see CLAUDE.md):
`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are stripped from the child
process's environment (`cli_client._clean_env`, reused unmodified) so the
subprocess always authenticates via the user's Claude subscription login,
never pay-per-token API billing.
"""

import asyncio
import json
import platform
import subprocess
from collections.abc import AsyncIterator
from typing import Protocol

from rhubarb.cli_client import _clean_env, _effort_args, _isolated_process_group, _plugin_args


class StreamJsonEngineError(RuntimeError):
    pass


class StreamJsonEngineUnrecoverableError(StreamJsonEngineError):
    """Raised by `stream_turn` when the underlying process dies mid-turn a
    SECOND time in a row -- once, and then again after this engine's one
    automatic `--resume` restart attempt (mirrors `PtyEngineUnrecoverableError`
    -- see `pty_engine.py` for the original issue #86 shape this follows).

    `session_id` is preserved (whatever was known at the moment of failure --
    possibly `None`, if this engine never completed a single turn) so a
    caller can still offer to resume this conversation later if a session id
    is available, even though this engine instance itself gives up rather
    than retrying indefinitely."""

    def __init__(self, message: str, *, session_id: str | None):
        super().__init__(message)
        self.session_id = session_id


class StreamJsonBackend(Protocol):
    """The minimal surface `StreamJsonEngine` needs from a spawned `claude`
    subprocess -- a thin line-oriented wrapper over stdin/stdout pipes, no
    PTY involved. `_spawn_subprocess` (the real backend) is a thin adapter
    over `subprocess.Popen`; a test double only needs to implement this
    much (mirrors `pty_engine.PtyBackend`'s test-injection shape, adapted
    for a line-based, not raw-chunk-based, transport)."""

    def write_line(self, line: str) -> None:
        """Write one line (WITHOUT a trailing newline -- the implementation
        adds it) to the process's stdin and flush immediately so the child
        sees it without waiting for a buffer to fill."""
        ...

    def read_line(self) -> str:
        """Block for, and return, the next full line of stdout (without its
        trailing newline). Raise `EOFError` once the process has exited and
        no more output remains -- this is `subprocess.Popen`'s own
        `.stdout.readline()` behavior translated into the same "raise
        EOFError at end of stream" convention `pty_engine.PtyBackend.read`
        already uses, so both engines' crash-detection logic reads the
        same way."""
        ...

    def is_alive(self) -> bool: ...

    def terminate(self, force: bool = False) -> None: ...


class _PopenBackend:
    """Real backend: a thin `StreamJsonBackend` adapter over a plain
    `subprocess.Popen` with text-mode stdin/stdout pipes -- no PTY, no
    `pywinpty`/`ptyprocess` involved anywhere in this class."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen

    def write_line(self, line: str) -> None:
        assert self._popen.stdin is not None
        self._popen.stdin.write(line + "\n")
        self._popen.stdin.flush()

    def read_line(self) -> str:
        assert self._popen.stdout is not None
        line = self._popen.stdout.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\n")

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def terminate(self, force: bool = False) -> None:
        if force:
            self._popen.kill()
        else:
            self._popen.terminate()


def _spawn_subprocess(argv: list[str], *, cwd: str | None, env: dict) -> StreamJsonBackend:
    """Spawn `argv` as a plain subprocess with text-mode stdin/stdout pipes
    -- no PTY of any kind. Reuses the same Windows `shell=True`/`.cmd`
    invocation quirk and `_isolated_process_group()` (CTRL_C_EVENT immunity)
    `cli_client.run_prompt`/`get_auth_status` already use for spawning
    `claude`, for the same reasons documented there.

    `stderr` is discarded (`DEVNULL`) rather than piped: this process is
    kept open indefinitely across many turns, and an unread `PIPE` would
    eventually deadlock once the child's stderr buffer fills -- there is no
    background reader thread draining it in this module. Diagnostic stderr
    output is out of scope for this issue.
    """
    popen = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,  # line-buffered
        shell=platform.system() == "Windows",
        **_isolated_process_group(),
    )
    return _PopenBackend(popen)


class StreamJsonEngine:
    """One `claude -p --input-format stream-json --output-format
    stream-json` process, spawned once and kept alive across turns over its
    still-open stdin/stdout pipes -- no PTY, no keystroke simulation, no
    turn-complete marker.

    `process_factory(argv, *, cwd, env) -> StreamJsonBackend` is injectable
    so tests never spawn a real `claude` process -- it defaults to
    `_spawn_subprocess`, the real backend. Mirrors `PtyEngine`'s
    `pty_factory` injection seam and constructor shape (`cwd`, `model`,
    `effort`, `resume_session_id`) on purpose, for the same reason: tests
    stay fast and deterministic, and a future caller's shape is familiar.

    Construct one instance per session, call `stream_turn(prompt)` for each
    turn (async generator of raw NDJSON event dicts, ending in a `result`
    event carrying `session_id`/`is_error`/`result` text -- exactly the
    shape `stream_translate.translate_event()` already has branches for),
    and `close()` when the session ends.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        resume_session_id: str | None = None,
        process_factory=None,
    ):
        self.cwd = cwd
        self.model = model
        self.effort = effort

        # Unlike `PtyEngine`, this transport's native `result` event hands
        # back a real session id from the CLI itself -- there is nothing to
        # assign up front. Starts as whatever the caller already knows
        # (`resume_session_id`, or `None` for a brand-new session) and is
        # updated from every turn's `result` event as turns complete.
        self.session_id: str | None = resume_session_id
        self._is_resume = resume_session_id is not None

        self._process_factory = process_factory or _spawn_subprocess
        self._proc: StreamJsonBackend | None = None

    def _build_args(self) -> list[str]:
        args = [
            "claude",
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
        ]
        args += _plugin_args()
        if self._is_resume and self.session_id:
            args += ["--resume", self.session_id]
        if self.model:
            args += ["--model", self.model]
        args += _effort_args(self.effort)
        return args

    def start(self) -> "StreamJsonEngine":
        """Spawn (or, for a reattach, respawn) the subprocess. Idempotent
        no-op if already started."""
        if self._proc is not None:
            return self
        env = _clean_env()
        self._proc = self._process_factory(self._build_args(), cwd=self.cwd, env=env)
        return self

    def _restart_after_death(self) -> None:
        """Tear down the dead process and respawn once -- see the module
        docstring's "Crash/restart recovery" section. Reattaches via
        `--resume <session_id>` when a session id is already known (either
        this engine was constructed with `resume_session_id=`, or an
        earlier turn through this same instance already completed and
        recorded one); otherwise there is no established session yet to
        resume, so the respawn is a fresh one, same as the original."""
        self.close()
        if self.session_id:
            self._is_resume = True
        self.start()

    async def _write_turn(self, prompt: str) -> None:
        """Write `prompt` to the subprocess's stdin as the exact shape
        confirmed working in the empirical spike this issue is built from:
        `{"type": "user", "message": {"role": "user", "content": <text>}}`.
        A single `json.dumps` call handles the whole object in one write --
        `prompt`'s own embedded newlines (if any) become the literal
        characters `\\n` inside the JSON string, never interpreted as
        anything else (see the module docstring's "Multi-line prompts"
        section -- this is the structural reason this transport has no
        PRD #180/#181-style submission failure mode)."""
        assert self._proc is not None
        message = {"type": "user", "message": {"role": "user", "content": prompt}}
        await asyncio.to_thread(self._proc.write_line, json.dumps(message))

    async def _read_events_until_result(self) -> AsyncIterator[dict]:
        """Read NDJSON lines off stdout, one per line, YIELDING each parsed
        event as soon as it's read (this is the actual live-streaming
        behavior `stream_turn` promises its caller -- see the bug this
        replaces, below) until a `{"type": "result", ...}` line appears --
        that is this turn's completion signal (no marker string, no buffer
        scanning: the CLI's own output format already delimits turns).

        Bug fixed here (found live, not caught by this module's own unit
        tests -- they only ever asserted on the final set/order of yielded
        events, never on WHEN each was yielded relative to the others,
        since a fake backend's reads never actually block): this method
        used to accumulate every event into a `list[dict]` and `return` it
        only once the `result` line arrived, and `stream_turn` then did
        `for event in events: yield event` over that already-complete list
        -- meaning nothing ever reached a caller (and therefore nothing
        ever reached `publish()`/the SSE stream/the Live Terminal panel)
        until the ENTIRE turn had already finished. A grilling card would
        show a completely empty panel for the whole turn, then the full
        response would appear all at once the instant it resolved -- no
        live feedback at all, defeating the entire point of this engine
        driving a block feed instead of a raw terminal. Yielding here,
        directly off each `read_line()`, is what actually fixes that.

        Raises `StreamJsonEngineError` if the process ends before a `result`
        line ever arrives -- `stream_turn` decides what to do with that
        (retries once via `_restart_after_death`, mirroring `PtyEngine.
        stream_turn`)."""
        assert self._proc is not None
        while True:
            try:
                line = await asyncio.to_thread(self._proc.read_line)
            except EOFError:
                raise StreamJsonEngineError(
                    "claude process ended before printing a result event"
                ) from None
            stripped = line.strip()
            if not stripped:
                if not self._proc.is_alive():
                    raise StreamJsonEngineError(
                        "claude process ended before printing a result event"
                    )
                continue
            event = json.loads(stripped)
            yield event
            if event.get("type") == "result":
                return

    async def stream_turn(self, prompt: str) -> AsyncIterator[dict]:
        """Write one turn to the live subprocess and yield every raw NDJSON
        event read back for it AS IT ARRIVES (true streaming -- see
        `_read_events_until_result`'s docstring for the bug this fixes), in
        order, ending in a `result` event (carries `session_id`, `is_error`,
        and the turn's final text in `result` -- per the real CLI's shape,
        cross-referenced against `stream_translate.translate_event()`'s
        existing `result` branch and `tests/test_stream_translate.py`'s
        captures). Starts the process on first use if `start()` wasn't
        already called.

        `self.session_id` is updated from the `result` event once the turn
        completes, so a second call reusing this same engine instance (the
        core behavior this whole module exists to prove -- see the module
        docstring) reuses the same subprocess and returns the identical
        `session_id` both times.

        Crash/restart recovery: if the process dies before a `result` event
        arrives, this method automatically restarts it exactly ONCE --
        reattaching via `--resume <session_id>` when a session id is already
        known -- and resends the same `prompt`, transparently to the caller
        (no event emitted for the death/restart). If that retried attempt
        ALSO fails, raises `StreamJsonEngineUnrecoverableError` instead of
        retrying again. Note: if the first attempt had already yielded some
        events to the caller before dying mid-turn, those already-published
        events are NOT retracted -- the retry's events simply follow them.
        This is a cosmetic tradeoff (a rare crash could briefly show a
        partial response before the retry's full one arrives) accepted in
        exchange for genuine live streaming on the overwhelmingly common
        non-crash path."""
        self.start()
        assert self._proc is not None

        last_event: dict | None = None

        async def _one_attempt() -> AsyncIterator[dict]:
            nonlocal last_event
            await self._write_turn(prompt)
            async for event in self._read_events_until_result():
                last_event = event
                yield event

        try:
            async for event in _one_attempt():
                yield event
        except StreamJsonEngineError as first_error:
            try:
                self._restart_after_death()
                async for event in _one_attempt():
                    yield event
            except Exception as second_error:
                raise StreamJsonEngineUnrecoverableError(
                    "claude process died twice in a row for the same turn "
                    f"(first: {first_error}; after one automatic --resume restart: "
                    f"{second_error}) -- giving up after exactly one retry",
                    session_id=self.session_id,
                ) from second_error

        if last_event is not None:
            self.session_id = last_event.get("session_id", self.session_id)

    def isalive(self) -> bool:
        """Whether the underlying subprocess is still running -- False if
        it was never started, already closed, or has died on its own. Never
        raises."""
        if self._proc is None:
            return False
        try:
            return self._proc.is_alive()
        except Exception:
            return False

    def close(self) -> None:
        """Terminate the underlying subprocess, if one was started. Safe to
        call more than once or when never started."""
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._proc.terminate(force=True)
        except Exception:
            pass
        self._proc = None
