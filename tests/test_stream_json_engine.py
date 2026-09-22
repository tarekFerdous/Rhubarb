import asyncio
import json

import pytest

from rhubarb.stream_json_engine import (
    StreamJsonEngine,
    StreamJsonEngineError,
    StreamJsonEngineUnrecoverableError,
)
from rhubarb.stream_translate import translate_event


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    return [event async for event in agen]


class FakeStreamJsonBackend:
    """Stand-in for `subprocess.Popen`'s stdin/stdout pipes (via the real
    `_PopenBackend` adapter): `lines` is served one already-serialized NDJSON
    line per `read_line()` call (mimicking output arriving one line at a
    time), then either raises `EOFError` (process exited) or returns ""
    forever (process still alive, nothing new yet) once exhausted, per
    `eof_after` -- mirrors `test_pty_engine.py`'s `FakePtyBackend` pattern,
    adapted for a line-based (not raw-chunk-based) transport."""

    def __init__(self, lines, *, eof_after=True):
        self._lines = list(lines)
        self._eof_after = eof_after
        self.written_lines = []
        self.terminated = False
        self.reads = 0

    def write_line(self, line):
        self.written_lines.append(line)

    def read_line(self):
        self.reads += 1
        if self._lines:
            return self._lines.pop(0)
        if self._eof_after:
            raise EOFError
        return ""

    def is_alive(self):
        return not self.terminated and (bool(self._lines) or not self._eof_after)

    def terminate(self, force=False):
        self.terminated = True


def _fake_factory(backend):
    captured = {}

    def factory(argv, *, cwd, env):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["env"] = env
        return backend

    return factory, captured


def _sequenced_factory(backends):
    """Factory that hands out one backend per call, in order, and records
    every spawn's argv (unlike `_fake_factory`, which only keeps the most
    recent) -- needed to inspect what a *restart* respawn passed, and to
    directly prove a second turn does NOT trigger a second spawn."""
    backends = list(backends)
    calls = []

    def factory(argv, *, cwd, env):
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return backends.pop(0)

    return factory, calls


def _result_line(text, session_id, is_error=False):
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "result": text,
            "session_id": session_id,
        }
    )


# ---------------------------------------------------------------------------
# Spawn args: no PTY, plain subprocess, stream-json flags, skip-permissions,
# plugin-dir, env-stripping
# ---------------------------------------------------------------------------


def test_start_spawns_plain_subprocess_with_stream_json_flags_and_skip_permissions():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    engine = StreamJsonEngine(process_factory=factory)
    engine.start()

    argv = captured["argv"]
    assert "claude" in argv
    assert "-p" in argv
    assert "--input-format" in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert "--include-partial-messages" in argv
    assert "--dangerously-skip-permissions" in argv


def test_start_passes_the_plugin_dir_argument():
    from rhubarb import stream_json_engine

    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(process_factory=factory).start()

    assert "--plugin-dir" in captured["argv"]
    idx = captured["argv"].index("--plugin-dir")
    assert captured["argv"][idx + 1] == stream_json_engine._plugin_args()[1]


def test_start_omits_lean_ctx_args_when_disabled():
    from rhubarb import cli_client

    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)
    cli_client.set_lean_ctx_enabled(False)

    StreamJsonEngine(process_factory=factory).start()

    assert "--mcp-config" not in captured["argv"]
    assert "--settings" not in captured["argv"]


def test_start_passes_lean_ctx_args_when_enabled():
    from rhubarb import cli_client
    from rhubarb.lean_ctx_installer import LEAN_CTX_HOOKS_SETTINGS_PATH, LEAN_CTX_MCP_CONFIG_PATH

    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)
    cli_client.set_lean_ctx_enabled(True)
    try:
        StreamJsonEngine(process_factory=factory).start()
    finally:
        cli_client.set_lean_ctx_enabled(False)

    argv = captured["argv"]
    assert "--mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == str(LEAN_CTX_MCP_CONFIG_PATH)
    assert "--settings" in argv
    assert argv[argv.index("--settings") + 1] == str(LEAN_CTX_HOOKS_SETTINGS_PATH)


def test_start_strips_api_key_and_auth_token_from_child_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-inherited")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-be-inherited")
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(process_factory=factory).start()

    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]


def test_start_is_idempotent_and_does_not_respawn():
    backend = FakeStreamJsonBackend([])
    calls = []

    def factory(argv, *, cwd, env):
        calls.append(argv)
        return backend

    engine = StreamJsonEngine(process_factory=factory)
    engine.start()
    engine.start()

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Fresh session (no session id yet) vs. --resume reattachment
# ---------------------------------------------------------------------------


def test_fresh_engine_has_no_session_id_and_omits_resume_flag():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    engine = StreamJsonEngine(process_factory=factory)
    assert engine.session_id is None

    engine.start()

    assert "--resume" not in captured["argv"]


def test_engine_constructed_with_resume_session_id_passes_resume_flag():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    engine = StreamJsonEngine(resume_session_id="existing-session-123", process_factory=factory)
    assert engine.session_id == "existing-session-123"

    engine.start()

    assert "--resume" in captured["argv"]
    idx = captured["argv"].index("--resume")
    assert captured["argv"][idx + 1] == "existing-session-123"


# ---------------------------------------------------------------------------
# --model/--effort argv-level coverage
# ---------------------------------------------------------------------------


def test_fresh_spawn_passes_the_configured_model_flag():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(model="claude-opus-4-8", process_factory=factory).start()

    assert "--model" in captured["argv"]
    idx = captured["argv"].index("--model")
    assert captured["argv"][idx + 1] == "claude-opus-4-8"


def test_fresh_spawn_omits_model_flag_when_none_configured():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(model=None, process_factory=factory).start()

    assert "--model" not in captured["argv"]


def test_fresh_spawn_passes_the_effort_flag_when_a_real_value_is_configured():
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(effort="high", process_factory=factory).start()

    assert "--effort" in captured["argv"]
    idx = captured["argv"].index("--effort")
    assert captured["argv"][idx + 1] == "high"


@pytest.mark.parametrize("effort", [None, "auto"])
def test_fresh_spawn_omits_the_effort_flag_for_none_or_auto(effort):
    backend = FakeStreamJsonBackend([])
    factory, captured = _fake_factory(backend)

    StreamJsonEngine(effort=effort, process_factory=factory).start()

    assert "--effort" not in captured["argv"]


# ---------------------------------------------------------------------------
# Writing a turn: exact JSON message shape confirmed in the empirical spike
# ---------------------------------------------------------------------------


def test_stream_turn_writes_the_confirmed_json_message_shape():
    backend = FakeStreamJsonBackend([_result_line("Hi there.", "session-abc")])
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    run(_collect(engine.stream_turn("hello")))

    assert len(backend.written_lines) == 1
    written = json.loads(backend.written_lines[0])
    assert written == {"type": "user", "message": {"role": "user", "content": "hello"}}


# ---------------------------------------------------------------------------
# Reading a turn's result: NDJSON lines up to and including `result`
# ---------------------------------------------------------------------------


def test_stream_turn_yields_every_raw_event_ending_in_result_and_updates_session_id():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "session-abc"}),
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hi"},
                },
                "session_id": "session-abc",
            }
        ),
        _result_line("Hi there.", "session-abc"),
    ]
    backend = FakeStreamJsonBackend(lines)
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    events = run(_collect(engine.stream_turn("hello")))

    assert [e["type"] for e in events] == ["system", "stream_event", "result"]
    assert events[-1]["result"] == "Hi there."
    assert events[-1]["session_id"] == "session-abc"
    assert events[-1]["is_error"] is False
    assert engine.session_id == "session-abc"


def test_stream_turn_events_are_compatible_with_translate_event():
    """The raw events this engine yields must already be the exact shape
    `stream_translate.translate_event()` expects -- cross-referenced against
    `tests/test_stream_translate.py`'s real-CLI-captured samples -- since a
    later issue (#184, not this one) wires this engine's events through that
    exact function unmodified."""
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "npm test"}}]},
            }
        ),
        _result_line("Ran the tests.", "session-abc"),
    ]
    backend = FakeStreamJsonBackend(lines)
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    events = run(_collect(engine.stream_turn("run the tests")))

    translated = [translate_event(e) for e in events]
    assert {"type": "action", "summary": "$ npm test"} in translated
    assert {
        "type": "turn",
        "result": "Ran the tests.",
        "session_id": "session-abc",
        "is_error": False,
    } in translated


def test_stream_turn_starts_the_process_automatically_if_not_already_started():
    backend = FakeStreamJsonBackend([_result_line("hi", "session-abc")])
    factory, captured = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    run(_collect(engine.stream_turn("hello")))

    assert captured["argv"]  # start() was called implicitly


# ---------------------------------------------------------------------------
# Core behavior the whole experiment depends on: the same subprocess stays
# alive and is reused across a second (and further) turn, with an identical
# session_id both times.
# ---------------------------------------------------------------------------


def test_second_turn_reuses_the_same_subprocess_not_a_fresh_spawn():
    backend = FakeStreamJsonBackend(
        [
            _result_line("first reply", "session-abc"),
            _result_line("second reply", "session-abc"),
        ],
        eof_after=False,
    )
    factory, calls = _sequenced_factory([backend])
    engine = StreamJsonEngine(process_factory=factory)

    first_events = run(_collect(engine.stream_turn("turn one")))
    second_events = run(_collect(engine.stream_turn("turn two")))

    # Only ONE spawn happened across both turns -- the same fake-subprocess
    # instance handled both.
    assert len(calls) == 1
    assert engine._proc is backend

    assert first_events[-1]["session_id"] == "session-abc"
    assert second_events[-1]["session_id"] == "session-abc"
    assert first_events[-1]["session_id"] == second_events[-1]["session_id"]

    # Both turns were written to that same backend, second on the still-open
    # stdin, no fresh --resume/--session-id respawn in between.
    written = [json.loads(line)["message"]["content"] for line in backend.written_lines]
    assert written == ["turn one", "turn two"]


# ---------------------------------------------------------------------------
# Multi-line regression test (PRD #180/#181 bug class)
# ---------------------------------------------------------------------------


def test_multiline_prompt_transmits_and_completes_as_a_single_turn():
    """Direct regression check: a prompt containing embedded newlines
    (simulating a multi-answer composed grilling reply) must be sent and
    complete as a single turn with no special handling. This transport has
    no keystroke-simulation failure mode structurally -- a JSON string field
    doesn't interpret an embedded "\\n" as anything but a literal two-byte
    escape sequence inside the string -- so this passes by construction, but
    the behavior must be asserted explicitly."""
    multiline_prompt = "1. answer one\n2. answer two"
    backend = FakeStreamJsonBackend([_result_line("Got both answers.", "session-abc")])
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    events = run(_collect(engine.stream_turn(multiline_prompt)))

    # Exactly one write for the whole prompt -- not split/retried/handled
    # specially because of the embedded newline.
    assert len(backend.written_lines) == 1
    written = json.loads(backend.written_lines[0])
    assert written["message"]["content"] == multiline_prompt
    assert "\n" in written["message"]["content"]

    # The turn still completed normally, as a single turn.
    assert events[-1]["type"] == "result"
    assert events[-1]["result"] == "Got both answers."
    assert events[-1]["is_error"] is False


# ---------------------------------------------------------------------------
# Crash/restart recovery
# ---------------------------------------------------------------------------


def test_stream_turn_restarts_once_via_resume_and_retries_the_same_prompt_after_a_mid_turn_death():
    """First backend dies mid-turn (EOFError before a result event). The
    engine should transparently respawn -- reattaching with `--resume
    <session_id>`, using the session id from a PRIOR completed turn through
    this same engine -- resend the same prompt, and succeed on the second
    (fresh) backend, all without the caller seeing any error."""
    first_backend = FakeStreamJsonBackend(
        [_result_line("first reply", "session-abc")],
        eof_after=True,
    )
    second_backend = FakeStreamJsonBackend([_result_line("recovered reply", "session-abc")])
    factory, calls = _sequenced_factory([first_backend, second_backend])

    engine = StreamJsonEngine(process_factory=factory)

    # Turn 1 succeeds normally and establishes the session id.
    run(_collect(engine.stream_turn("turn one")))
    assert engine.session_id == "session-abc"

    # Turn 2: first_backend has nothing left queued and is exhausted, so its
    # very next read raises EOFError -- simulating a mid-turn death.
    events = run(_collect(engine.stream_turn("turn two")))

    # Exactly two spawns total: the original, and the one restart.
    assert len(calls) == 2

    # The restart respawn reattaches via --resume with the session id learned
    # from turn one.
    assert "--resume" in calls[1]["argv"]
    idx = calls[1]["argv"].index("--resume")
    assert calls[1]["argv"][idx + 1] == "session-abc"

    # The same prompt was resent to the fresh backend.
    assert json.loads(second_backend.written_lines[0])["message"]["content"] == "turn two"

    # Caller sees a perfectly normal, successful turn -- no error surfaced.
    assert events[-1]["type"] == "result"
    assert events[-1]["is_error"] is False
    assert events[-1]["result"] == "recovered reply"
    assert events[-1]["session_id"] == "session-abc"


def test_restart_respawn_is_a_fresh_spawn_when_no_session_id_is_known_yet():
    """A death on the very FIRST turn ever sent through a brand-new engine
    (no `resume_session_id`, no prior completed turn) has no session id yet
    to reattach with -- the retried respawn must be a fresh spawn (no
    --resume), not crash trying to resume an unknown session."""
    dead_backend = FakeStreamJsonBackend([], eof_after=True)
    healthy_backend = FakeStreamJsonBackend([_result_line("all good now", "session-xyz")])
    factory, calls = _sequenced_factory([dead_backend, healthy_backend])

    engine = StreamJsonEngine(process_factory=factory)

    events = run(_collect(engine.stream_turn("first ever turn")))

    assert len(calls) == 2
    assert "--resume" not in calls[0]["argv"]
    assert "--resume" not in calls[1]["argv"]
    assert events[-1]["is_error"] is False
    assert events[-1]["result"] == "all good now"
    assert engine.session_id == "session-xyz"


def test_stream_turn_gives_up_after_one_failed_restart_with_a_distinguishable_error():
    """Both the original AND the restarted backend die -- the engine must
    give up after exactly one restart attempt (not retry indefinitely) and
    raise the distinguishable `StreamJsonEngineUnrecoverableError` subclass."""
    first_dead = FakeStreamJsonBackend([], eof_after=True)
    second_dead = FakeStreamJsonBackend([], eof_after=True)
    factory, calls = _sequenced_factory([first_dead, second_dead])

    engine = StreamJsonEngine(resume_session_id="known-session", process_factory=factory)

    with pytest.raises(StreamJsonEngineUnrecoverableError) as excinfo:
        run(_collect(engine.stream_turn("go")))

    assert len(calls) == 2
    assert excinfo.value.session_id == "known-session"
    # It's still a StreamJsonEngineError for any caller only checking the base class.
    assert isinstance(excinfo.value, StreamJsonEngineError)


def test_stream_turn_raises_plain_error_when_process_dies_and_no_retry_is_possible_but_still_only_retries_once():
    """No fresh backend is available to retry into here (`_fake_factory`
    always hands back the very same dead backend on respawn), so the one
    automatic restart also dies immediately -- the caller still ultimately
    sees a `StreamJsonEngineError` (its subclass
    `StreamJsonEngineUnrecoverableError`), it just takes one extra internal
    attempt to get there."""
    backend = FakeStreamJsonBackend([], eof_after=True)
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    with pytest.raises(StreamJsonEngineError):
        run(_collect(engine.stream_turn("go")))


# ---------------------------------------------------------------------------
# isalive() / close()
# ---------------------------------------------------------------------------


def test_isalive_reflects_backend_state():
    backend = FakeStreamJsonBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)

    assert engine.isalive() is False  # never started

    engine.start()
    assert engine.isalive() is True

    backend.terminated = True
    assert engine.isalive() is False


def test_close_terminates_a_live_process():
    backend = FakeStreamJsonBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = StreamJsonEngine(process_factory=factory)
    engine.start()

    engine.close()

    assert backend.terminated is True
    assert engine.isalive() is False


def test_close_is_a_noop_when_never_started():
    engine = StreamJsonEngine(process_factory=lambda *a, **kw: FakeStreamJsonBackend([]))
    engine.close()  # must not raise
