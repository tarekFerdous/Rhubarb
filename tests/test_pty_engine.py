import asyncio
import re

import pytest

from rhubarb import pty_engine
from rhubarb.pty_engine import (
    TURN_COMPLETE_MARKER,
    PtyEngine,
    PtyEngineError,
    PtyEngineUnrecoverableError,
)


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    return [event async for event in agen]


class FakePtyBackend:
    """Stand-in for `winpty.PtyProcess`/`ptyprocess.PtyProcessUnicode`:
    `chunks` is served one item per `read()` call (mimicking a real PTY's
    output arriving incrementally), then either raises EOFError (process
    exited) or returns "" forever (process still alive, nothing new yet)
    once exhausted, per `eof_after`.

    `resize`/`setwinsize` (issue #166) both record every call into
    `self.resizes` -- `setwinsize` is the name the two real backends
    (`winpty.PtyProcess`, `ptyprocess.PtyProcessUnicode`) actually expose;
    `resize` is the `PtyBackend` Protocol's own name, which `_spawn_winpty`/
    `_spawn_unix_pty` alias to the real object's `setwinsize` right after
    spawning (see their docstrings) -- defining both here the same way lets
    this one fake stand in for either call path.
    """

    def __init__(self, chunks, *, eof_after=True):
        self._chunks = list(chunks)
        self._eof_after = eof_after
        self.writes = []
        self.terminated = False
        self.reads = 0
        self.resizes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def read(self, size=4096):
        self.reads += 1
        if self._chunks:
            return self._chunks.pop(0)
        if self._eof_after:
            raise EOFError
        return ""

    def isalive(self):
        return not self.terminated and (bool(self._chunks) or not self._eof_after)

    def terminate(self, force=False):
        self.terminated = True

    def setwinsize(self, rows, cols):
        self.resizes.append((rows, cols))

    def resize(self, rows, cols):
        self.setwinsize(rows, cols)


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
    recent) -- needed to inspect what a *restart* respawn passed."""
    backends = list(backends)
    calls = []

    def factory(argv, *, cwd, env):
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return backends.pop(0)

    return factory, calls


# ---------------------------------------------------------------------------
# Spawn args: skip-permissions, plugin-dir, env-stripping, marker instruction
# ---------------------------------------------------------------------------


def test_start_spawns_interactive_claude_with_skip_permissions_and_plugin_dir():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    engine = PtyEngine(pty_factory=factory)
    engine.start()

    assert "claude" in captured["argv"]
    assert "-p" not in captured["argv"]  # interactive, not headless
    assert "--dangerously-skip-permissions" in captured["argv"]
    assert "--plugin-dir" in captured["argv"]
    idx = captured["argv"].index("--plugin-dir")
    assert captured["argv"][idx + 1] == pty_engine._plugin_args()[1]


def test_start_appends_the_turn_complete_marker_instruction_to_the_system_prompt():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(pty_factory=factory).start()

    assert "--append-system-prompt" in captured["argv"]
    idx = captured["argv"].index("--append-system-prompt")
    assert TURN_COMPLETE_MARKER in captured["argv"][idx + 1]


def test_start_strips_api_key_and_auth_token_from_child_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-inherited")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-be-inherited")
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(pty_factory=factory).start()

    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]


def test_start_is_idempotent_and_does_not_respawn():
    backend = FakePtyBackend([])
    calls = []

    def factory(argv, *, cwd, env):
        calls.append(argv)
        return backend

    engine = PtyEngine(pty_factory=factory)
    engine.start()
    engine.start()

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Fresh session id vs. --resume reattachment
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_fresh_engine_generates_and_passes_a_session_id_up_front():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    engine = PtyEngine(pty_factory=factory)
    assert _UUID_RE.match(engine.claude_session_id)

    engine.start()

    assert "--session-id" in captured["argv"]
    idx = captured["argv"].index("--session-id")
    assert captured["argv"][idx + 1] == engine.claude_session_id
    assert "--resume" not in captured["argv"]


def test_resume_reattaches_with_the_given_session_id_against_a_fresh_pty():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    engine = PtyEngine(resume_session_id="existing-session-123", pty_factory=factory)
    assert engine.claude_session_id == "existing-session-123"

    engine.start()

    assert "--resume" in captured["argv"]
    idx = captured["argv"].index("--resume")
    assert captured["argv"][idx + 1] == "existing-session-123"
    assert "--session-id" not in captured["argv"]


# ---------------------------------------------------------------------------
# --model/--effort argv-level coverage (issue #138): every prior test only
# asserted a Python-level kwarg made it to `PtyEngine(...)` (see
# tests/test_sessions.py's `_mock_engine`) -- nothing exercised what actually
# lands in the argv list handed to the real subprocess spawn. These assert
# directly against `_build_args()`'s output (via the injected `pty_factory`,
# exactly like the --resume/--session-id tests above) for a fresh spawn, a
# --resume spawn, and a death-triggered restart-respawn.
# ---------------------------------------------------------------------------


def test_fresh_spawn_passes_the_configured_model_flag():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(model="claude-opus-4-8", pty_factory=factory).start()

    assert "--model" in captured["argv"]
    idx = captured["argv"].index("--model")
    assert captured["argv"][idx + 1] == "claude-opus-4-8"


def test_fresh_spawn_omits_model_flag_when_none_configured():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(model=None, pty_factory=factory).start()

    assert "--model" not in captured["argv"]


def test_resume_spawn_passes_the_configured_model_flag():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(model="claude-opus-4-8", resume_session_id="existing-session-123", pty_factory=factory).start()

    assert "--resume" in captured["argv"]
    assert "--model" in captured["argv"]
    idx = captured["argv"].index("--model")
    assert captured["argv"][idx + 1] == "claude-opus-4-8"


def test_restart_respawn_passes_the_same_model_flag_as_the_original_spawn():
    """A mid-turn death respawns via `_restart_after_death` (see the
    crash/restart recovery tests further below) -- the model this engine was
    constructed with must still be passed on the retried spawn, not silently
    dropped just because that spawn takes the --resume branch instead of the
    original --session-id one."""
    dead_backend = FakePtyBackend(["the process dies before the marker\n"], eof_after=True)
    healthy_backend = FakePtyBackend([f"all good now\n{TURN_COMPLETE_MARKER}\n"])
    factory, calls = _sequenced_factory([dead_backend, healthy_backend])

    engine = PtyEngine(model="claude-opus-4-8", pty_factory=factory)

    run(_collect(engine.stream_turn("what is 2+2?")))

    assert len(calls) == 2
    for call in calls:
        assert "--model" in call["argv"]
        idx = call["argv"].index("--model")
        assert call["argv"][idx + 1] == "claude-opus-4-8"


def test_fresh_spawn_passes_the_effort_flag_when_a_real_value_is_configured():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(effort="high", pty_factory=factory).start()

    assert "--effort" in captured["argv"]
    idx = captured["argv"].index("--effort")
    assert captured["argv"][idx + 1] == "high"


@pytest.mark.parametrize("effort", [None, "auto"])
def test_fresh_spawn_omits_the_effort_flag_for_none_or_auto(effort):
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(effort=effort, pty_factory=factory).start()

    assert "--effort" not in captured["argv"]


def test_resume_spawn_passes_the_effort_flag_when_a_real_value_is_configured():
    backend = FakePtyBackend([])
    factory, captured = _fake_factory(backend)

    PtyEngine(effort="high", resume_session_id="existing-session-123", pty_factory=factory).start()

    assert "--resume" in captured["argv"]
    assert "--effort" in captured["argv"]
    idx = captured["argv"].index("--effort")
    assert captured["argv"][idx + 1] == "high"


def test_restart_respawn_passes_the_same_effort_flag_as_the_original_spawn():
    dead_backend = FakePtyBackend(["the process dies before the marker\n"], eof_after=True)
    healthy_backend = FakePtyBackend([f"all good now\n{TURN_COMPLETE_MARKER}\n"])
    factory, calls = _sequenced_factory([dead_backend, healthy_backend])

    engine = PtyEngine(effort="high", pty_factory=factory)

    run(_collect(engine.stream_turn("what is 2+2?")))

    assert len(calls) == 2
    for call in calls:
        assert "--effort" in call["argv"]
        idx = call["argv"].index("--effort")
        assert call["argv"][idx + 1] == "high"


# ---------------------------------------------------------------------------
# Turn-event contract + marker detection
# ---------------------------------------------------------------------------


def test_stream_turn_yields_init_then_result_events_carrying_session_id():
    backend = FakePtyBackend([f"Hello there.\n{TURN_COMPLETE_MARKER}\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    assert events[0]["type"] == "system"
    assert events[0]["session_id"] == engine.claude_session_id
    assert events[-1]["type"] == "result"
    assert events[-1]["session_id"] == engine.claude_session_id
    assert events[-1]["is_error"] is False


def test_stream_turn_writes_the_prompt_to_the_pty():
    """Issue #148: the prompt must reach the PTY as a paced sequence of
    small text chunks followed by a SEPARATE trailing "\\r" write -- not one
    atomic `write(prompt + "\\r")` call, which is what let the CLI's own
    paste-detection heuristic swallow the Enter keystroke. Joining every
    write back together must still reproduce the original prompt text plus
    the trailing carriage return, and the very last write must be exactly
    "\\r" on its own."""
    backend = FakePtyBackend([f"ok\n{TURN_COMPLETE_MARKER}\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    run(_collect(engine.stream_turn("what is 2+2?")))

    assert len(backend.writes) > 1
    assert backend.writes[-1] == "\r"
    assert "".join(backend.writes) == "what is 2+2?\r"
    # No single write bundles the trailing "\r" onto prompt text.
    for chunk in backend.writes[:-1]:
        assert not chunk.endswith("\r")


def test_stream_turn_writes_the_prompt_in_small_paced_chunks():
    """The prompt text itself must be split into chunks no larger than
    `pty_engine._WRITE_CHUNK_SIZE` (not handed to the PTY as one atomic
    string), with the trailing "\\r" arriving as its own final write."""
    long_prompt = "x" * (pty_engine._WRITE_CHUNK_SIZE * 3 + 5)
    backend = FakePtyBackend([f"ok\n{TURN_COMPLETE_MARKER}\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    run(_collect(engine.stream_turn(long_prompt)))

    text_writes = backend.writes[:-1]
    assert len(text_writes) > 1
    for chunk in text_writes:
        assert len(chunk) <= pty_engine._WRITE_CHUNK_SIZE
    assert "".join(text_writes) == long_prompt
    assert backend.writes[-1] == "\r"


def test_stream_turn_writes_a_short_prompt_with_a_separate_trailing_carriage_return():
    """Even a prompt shorter than one chunk still gets its "\\r" written
    separately -- the paced-write behavior is unconditional, not gated on
    prompt length."""
    backend = FakePtyBackend([f"ok\n{TURN_COMPLETE_MARKER}\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    run(_collect(engine.stream_turn("hi")))

    assert backend.writes == ["hi", "\r"]


def test_stream_turn_stops_reading_as_soon_as_marker_appears_across_chunks():
    """The marker can arrive split across separate PTY reads (a slow
    terminal write, a chunk boundary mid-line) -- detection must work on
    the accumulated buffer, not a single chunk, and must stop pulling
    further chunks once satisfied."""
    backend = FakePtyBackend(
        [
            "Working on it...\n",
            "Here is the answer: 4\n<<<RHUBARB_TURN_",
            "COMPLETE>>>\n",
            "SOMETHING THAT SHOULD NEVER BE READ",
        ]
    )
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("what is 2+2?")))

    result_text = events[-1]["result"]
    assert "Here is the answer: 4" in result_text
    assert TURN_COMPLETE_MARKER not in result_text
    assert "SHOULD NEVER BE READ" not in result_text
    # Exactly the three chunks up to and including the marker were consumed.
    assert backend.reads == 3


def test_stream_turn_yields_terminal_output_events_for_every_raw_chunk_before_result():
    """Issue #88: each raw chunk read off the PTY must be relayed as a
    `{"type": "terminal_output", "data": ...}` event, in order, interleaved
    BEFORE the final `result` event -- additive only, the existing
    `system`/`result` event shapes and their positions are unchanged."""
    chunks = ["Working", " on it...\n", f"Here you go.\n{TURN_COMPLETE_MARKER}\n"]
    backend = FakePtyBackend(chunks)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    assert events[0]["type"] == "system"
    assert events[-1]["type"] == "result"

    terminal_events = [e for e in events if e["type"] == "terminal_output"]
    assert [e["data"] for e in terminal_events] == chunks
    # Every terminal_output event comes after the init event and before the
    # final result event.
    assert events.index(terminal_events[0]) > 0
    assert events.index(terminal_events[-1]) < len(events) - 1


def test_stream_turn_passes_through_fenced_json_marker_blocks_unmodified():
    """Rhubarb's existing markers (implement_blocked, qa_grilling -- see
    session_runner._parse_implement_blocked_block/_parse_qa_grilling_block)
    must survive unmangled inside the captured turn text; this engine only
    ever looks for its own completion marker.

    Line endings are real PTY output (`\\r\\n`), not bare `\\n` -- a PTY's
    own line discipline performs that LF -> CRLF translation, so that's
    what a real raw buffer actually contains, and it's what the terminal
    emulation this engine now runs the buffer through (`_render_terminal_text`)
    needs to correctly resolve column position across lines."""
    fenced_block = (
        "Some preamble text.\r\n"
        "```json\r\n"
        '{"phase": "implement_blocked", "reason": "need a decision"}\r\n'
        "```\r\n"
        "Trailing text.\r\n"
    )
    backend = FakePtyBackend([fenced_block + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("go")))

    assert (
        'Some preamble text.\n```json\n{"phase": "implement_blocked", "reason": "need a decision"}\n```\nTrailing text.\n'
        in events[-1]["result"]
    )


def test_stream_turn_resolves_ansi_color_codes_to_clean_text():
    """A turn whose raw buffer includes 24-bit ANSI color codes must resolve
    to plain, readable text in `result` -- no escape-sequence artifacts."""
    raw = "\x1b[38;2;215;119;87mHello there.\x1b[0m\r\n"
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    assert events[-1]["result"] == "Hello there.\n"
    assert "\x1b" not in events[-1]["result"]


def test_stream_turn_resolves_cursor_positioning_sequences_to_clean_text():
    """Cursor-positioning/erase-line escape sequences (used to redraw a
    status line or the `>` prompt box in place) must resolve to the final
    rendered text, not leak through as literal escape artifacts."""
    raw = (
        "\x1b[2J\x1b[H"  # clear screen, home cursor
        "Line one.\r\n"
        "\x1b[1;1H"  # reposition cursor back to the top row
        "\x1b[2K"  # erase that line
        "Replaced line one.\r\n"
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    result_text = events[-1]["result"]
    assert "\x1b" not in result_text
    assert "Replaced line one." in result_text
    assert "Line one." not in result_text


def test_stream_turn_collapses_an_animated_spinner_redraw_to_its_final_frame():
    """An animated spinner (the same line redrawn in place via carriage
    return + erase-line, as Claude Code's own 'thinking...' indicator does)
    must collapse to its final settled frame, not concatenate every
    intermediate frame's characters end-to-end -- reproducing this exact
    bug's original failure mode (a manual copy of a spinner-redraw stream
    shows every frame glued together)."""
    raw = (
        "\x1b[38;2;215;119;87mWorking\x1b[0m\r"
        "\x1b[38;2;215;119;87mWorking.\x1b[0m\r"
        "\x1b[38;2;215;119;87mWorking..\x1b[0m\r"
        "\x1b[2K\rDone thinking.\r\n"
        'Question 1: "Should this be Python or Node?"\r\n'
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    result_text = events[-1]["result"]
    assert result_text == 'Done thinking.\nQuestion 1: "Should this be Python or Node?"\n'
    assert "Working" not in result_text


# ---------------------------------------------------------------------------
# Character-level corruption fix (issue #117): pyte's CSI parser terminating
# early on a `:`-delimited SGR subparameter, and pyte's `Screen.draw`
# silently abandoning the rest of a draw call on a zero-width/format
# character that isn't a combining mark.
# ---------------------------------------------------------------------------


def test_stream_turn_resolves_multibyte_unicode_adjacent_to_ansi_formatting_with_no_replacement_character():
    """An em dash sitting directly against ANSI/control sequences on both
    sides -- a bold SGR pair, and a colon-delimited SGR subparameter
    sequence (`CSI 4:3 m`, the ISO-8613-6 curly-underline/24-bit-color
    style some terminal UI libraries emit) -- must resolve to the exact
    plain text, with no `�` replacement character and no bytes
    dropped."""
    raw = (
        "before \x1b[1m—\x1b[0m after\r\n"
        "styled \x1b[4:3m—\x1b[24m done\r\n"
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    result_text = events[-1]["result"]
    assert result_text == "before — after\nstyled — done\n"
    assert "�" not in result_text


def test_stream_turn_resolves_ascii_adjacent_to_the_same_control_sequences_with_no_dropped_character():
    """The same two control-sequence classes -- a bold SGR pair, and a
    colon-delimited SGR subparameter sequence -- plus a zero-width joiner
    sitting right next to plain text (a format character pyte's
    `Screen.draw` otherwise chokes on, silently dropping everything after
    it) -- must not drop or mangle any of the surrounding plain ASCII text.
    The zero-width joiner itself carries no width and is dropped by design
    (see `_strip_unadvancing_format_characters`) -- a real terminal's own
    font shaping is what would otherwise fuse it with its neighbors into
    one glyph, which this plain-text rendering never attempted anyway."""
    raw = (
        "before \x1b[1mgate\x1b[0m after\r\n"
        "styled \x1b[4:3mgate\x1b[24m done\r\n"
        "joined \U0001f9d1‍\U0001f4bb gate open\r\n"
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    result_text = events[-1]["result"]
    assert result_text == (
        "before gate after\n"
        "styled gate done\n"
        "joined \U0001f9d1\U0001f4bb gate open\n"
    )


def test_stream_turn_reproduces_the_production_corruption_pattern_gate_and_em_dash():
    """Regression for the real corrupted `console_text` this issue was
    filed from: a dropped character in the word "gate" (caused by a
    zero-width joiner landing next to it -- see
    `_strip_unadvancing_format_characters`), an em dash sitting against a
    colon-delimited SGR subparameter sequence (see
    `_desubparameterize_csi_sequences`), and a `Question 2:` header line
    immediately after both -- all in the same surrounding multi-line,
    ANSI-colored transcript a real interactive turn would produce. None of
    the three may be corrupted or go missing."""
    raw = (
        "\x1b[38;2;215;119;87mPassing the next g​ate\x1b[0m requires review—\r\n"
        "\x1b[4:3msign-off—\x1b[24m confirmed\r\n"
        'Question 2: "Should this ship behind a flag?"\r\n'
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    result_text = events[-1]["result"]
    assert "gate" in result_text
    assert "g te" not in result_text
    assert "�" not in result_text
    assert "—" in result_text
    assert 'Question 2: "Should this ship behind a flag?"' in result_text


def test_stream_turn_prints_a_live_rendered_trace_for_every_chunk_read(monkeypatch, capsys):
    """Issue #160: after every raw chunk is read off the PTY, the
    accumulated buffer-so-far must be re-rendered through the SAME
    `_render_terminal_text` function `stream_turn` uses to produce its
    final `result` text, and printed to the console -- unconditionally, no
    debug flag needed, for every phase (this test doesn't care which
    phase -- `PtyEngine` has no notion of phase at all, so "applies to
    every phase" just falls out of it applying to every turn)."""
    chunks = ["Working", " on it...\n", f"Here you go.\n{TURN_COMPLETE_MARKER}\n"]
    backend = FakePtyBackend(chunks)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    calls = []
    real_render = pty_engine._render_terminal_text

    def counting_render(raw, columns=pty_engine._VIRTUAL_SCREEN_COLUMNS):
        calls.append(raw)
        return real_render(raw, columns)

    monkeypatch.setattr(pty_engine, "_render_terminal_text", counting_render)

    run(_collect(engine.stream_turn("hi")))

    # One live re-render per chunk read during the loop, plus the one
    # existing final render (over the marker-stripped text) that already
    # produces `result` -- i.e. exactly len(chunks) + 1 calls total, not
    # just once at the end.
    assert len(calls) == len(chunks) + 1

    # Each live call saw the buffer accumulated so far, not just the latest
    # chunk in isolation.
    accumulated = ""
    for expected_chunk, call_arg in zip(chunks, calls):
        accumulated += expected_chunk
        assert call_arg == accumulated

    # And it was actually printed to the console, not just computed.
    printed = capsys.readouterr().out
    assert "Working on it..." in printed
    assert "Here you go." in printed


def test_stream_turn_terminal_output_events_still_carry_the_raw_unmodified_bytes():
    """The live-terminal-view consumer must keep receiving the exact raw
    PTY bytes, ANSI and all -- only the separate `result` text is resolved
    through the terminal emulator."""
    raw_chunk = "\x1b[38;2;215;119;87mHello there.\x1b[0m\r\n"
    backend = FakePtyBackend([raw_chunk, TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("hi")))

    terminal_events = [e for e in events if e["type"] == "terminal_output"]
    assert terminal_events[0]["data"] == raw_chunk
    assert "\x1b[38;2;215;119;87m" in terminal_events[0]["data"]


def test_stream_turn_extracts_fenced_json_markers_through_realistic_ansi_noise():
    """implement_blocked/qa_grilling JSON markers must still be extractable
    from the resolved `result` text even when real ANSI color/cursor noise
    surrounds them -- not just plain text."""
    from rhubarb.session_runner import _parse_implement_blocked_block

    raw = (
        "\x1b[38;2;215;119;87mSome preamble text.\x1b[0m\r\n"
        "```json\r\n"
        '{"phase": "implement_blocked", "issue": 8, "question": "Which provider?", "context": "ambiguous"}\r\n'
        "```\r\n"
    )
    backend = FakePtyBackend([raw + TURN_COMPLETE_MARKER + "\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    events = run(_collect(engine.stream_turn("go")))

    blocked = _parse_implement_blocked_block(events[-1]["result"])
    assert blocked == {
        "phase": "implement_blocked",
        "issue": 8,
        "question": "Which provider?",
        "context": "ambiguous",
    }


def test_stream_turn_raises_if_process_exits_before_printing_the_marker():
    """No fresh backend is available to retry into here (`_fake_factory`
    always hands back the very same dead backend on respawn), so the one
    automatic restart also dies immediately -- the caller still ultimately
    sees a `PtyEngineError` (its subclass `PtyEngineUnrecoverableError`),
    it just takes one extra internal attempt to get there."""
    backend = FakePtyBackend(["partial output, then the process dies\n"], eof_after=True)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    with pytest.raises(PtyEngineError):
        run(_collect(engine.stream_turn("go")))


def test_stream_turn_starts_the_process_automatically_if_not_already_started():
    backend = FakePtyBackend([f"hi\n{TURN_COMPLETE_MARKER}\n"])
    factory, captured = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)

    run(_collect(engine.stream_turn("hello")))

    assert captured["argv"]  # start() was called implicitly


# ---------------------------------------------------------------------------
# Crash/restart recovery (issue #86)
# ---------------------------------------------------------------------------


def test_stream_turn_restarts_once_and_retries_the_same_prompt_after_a_mid_turn_death():
    """First backend dies mid-turn (EOFError before the marker). The engine
    should transparently respawn -- reattaching with `--resume
    <claude_session_id>` -- resend the same prompt, and succeed on the
    second (fresh) backend, all without the caller seeing any error."""
    dead_backend = FakePtyBackend(["the process dies before the marker\n"], eof_after=True)
    healthy_backend = FakePtyBackend([f"all good now\n{TURN_COMPLETE_MARKER}\n"])
    factory, calls = _sequenced_factory([dead_backend, healthy_backend])

    engine = PtyEngine(pty_factory=factory)
    session_id = engine.claude_session_id

    events = run(_collect(engine.stream_turn("what is 2+2?")))

    # Exactly two spawns: the original, and the one restart.
    assert len(calls) == 2

    # First spawn was fresh (no prior resume requested).
    assert "--session-id" in calls[0]["argv"]
    idx = calls[0]["argv"].index("--session-id")
    assert calls[0]["argv"][idx + 1] == session_id

    # The restart respawn reattaches via --resume with the SAME session id,
    # even though the original spawn used --session-id, not --resume.
    assert "--resume" in calls[1]["argv"]
    idx = calls[1]["argv"].index("--resume")
    assert calls[1]["argv"][idx + 1] == session_id

    # The same prompt was written to both backends (retry resends it),
    # each as paced chunk(s) followed by a separate trailing "\r" write.
    assert dead_backend.writes == ["what is 2+2?", "\r"]
    assert healthy_backend.writes == ["what is 2+2?", "\r"]

    # Caller sees a perfectly normal, successful turn -- no error surfaced.
    assert events[0]["type"] == "system"
    assert events[-1]["type"] == "result"
    assert events[-1]["is_error"] is False
    assert "all good now" in events[-1]["result"]
    assert events[-1]["session_id"] == session_id


def test_restart_respawn_resumes_even_when_original_spawn_was_already_a_resume():
    """Covers the "Rhubarb itself restarted mid-phase" scenario: a fresh
    `PtyEngine` constructed with `resume_session_id=` for a conversation
    whose original process is already gone dies on its very first turn
    through this instance, and must still retry via --resume (not give up
    just because the *first* attempt was already a resume)."""
    dead_backend = FakePtyBackend([], eof_after=True)
    healthy_backend = FakePtyBackend([f"back online\n{TURN_COMPLETE_MARKER}\n"])
    factory, calls = _sequenced_factory([dead_backend, healthy_backend])

    engine = PtyEngine(resume_session_id="orphaned-session-456", pty_factory=factory)

    events = run(_collect(engine.stream_turn("continue")))

    assert len(calls) == 2
    for call in calls:
        assert "--resume" in call["argv"]
        idx = call["argv"].index("--resume")
        assert call["argv"][idx + 1] == "orphaned-session-456"

    assert events[-1]["is_error"] is False
    assert "back online" in events[-1]["result"]


def test_stream_turn_gives_up_after_one_failed_restart_with_a_distinguishable_error():
    """Both the original AND the restarted backend die -- the engine must
    give up after exactly one restart attempt (not retry indefinitely) and
    raise the distinguishable `PtyEngineUnrecoverableError` subclass, per
    the failure shape documented in pty_engine.py for a future caller to
    route into the blocked-card flow."""
    first_dead = FakePtyBackend(["dying...\n"], eof_after=True)
    second_dead = FakePtyBackend(["dying again...\n"], eof_after=True)
    factory, calls = _sequenced_factory([first_dead, second_dead])

    engine = PtyEngine(pty_factory=factory)
    session_id = engine.claude_session_id

    with pytest.raises(PtyEngineUnrecoverableError) as excinfo:
        run(_collect(engine.stream_turn("go")))

    # Exactly one restart attempt: two spawns total, never a third.
    assert len(calls) == 2
    assert excinfo.value.claude_session_id == session_id
    # It's still a PtyEngineError for any caller only checking the base class.
    assert isinstance(excinfo.value, PtyEngineError)


# ---------------------------------------------------------------------------
# Cross-platform backend selection (issue #85)
# ---------------------------------------------------------------------------


def test_default_pty_factory_picks_winpty_backend_on_windows(monkeypatch):
    monkeypatch.setattr(pty_engine.platform, "system", lambda: "Windows")

    assert pty_engine._default_pty_factory() is pty_engine._spawn_winpty


def test_default_pty_factory_picks_unix_backend_on_linux(monkeypatch):
    monkeypatch.setattr(pty_engine.platform, "system", lambda: "Linux")

    assert pty_engine._default_pty_factory() is pty_engine._spawn_unix_pty


def test_default_pty_factory_picks_unix_backend_on_macos(monkeypatch):
    monkeypatch.setattr(pty_engine.platform, "system", lambda: "Darwin")

    assert pty_engine._default_pty_factory() is pty_engine._spawn_unix_pty


def test_engine_uses_platform_selected_factory_when_none_is_injected(monkeypatch):
    """`PtyEngine()` with no explicit `pty_factory` should resolve to
    whatever `_default_pty_factory()` selects for the current platform,
    without the caller having to know or care which OS it's running on."""
    sentinel_backend = FakePtyBackend([])

    def fake_unix_factory(argv, *, cwd, env, **kwargs):
        # `**kwargs` absorbs the `rows`/`cols` PtyEngine passes through to
        # the real default-selected factory (issue #166, see
        # `PtyEngine.start`) -- this test only cares that the platform
        # selection itself resolved to `_spawn_unix_pty`, not what dims it
        # was called with.
        return sentinel_backend

    monkeypatch.setattr(pty_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pty_engine, "_spawn_unix_pty", fake_unix_factory)
    monkeypatch.setattr(pty_engine, "_spawn_winpty", None)  # must not be used

    engine = PtyEngine()
    engine.start()

    assert engine._proc is sentinel_backend


def test_spawn_unix_pty_uses_ptyprocess_unicode_spawn(monkeypatch):
    """`_spawn_unix_pty` should be a thin adapter over
    `ptyprocess.PtyProcessUnicode.spawn`, imported lazily (so importing
    `pty_engine` never requires `ptyprocess` to be installed on Windows).
    A fake `ptyprocess` module is injected into `sys.modules` so this runs
    without the real (Unix-only) dependency installed."""
    import sys
    import types

    calls = {}

    class FakePtyProcessUnicode:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=None):
            calls["argv"] = argv
            calls["cwd"] = cwd
            calls["env"] = env
            calls["dimensions"] = dimensions
            return FakePtyBackend([])

    fake_module = types.SimpleNamespace(PtyProcessUnicode=FakePtyProcessUnicode)
    monkeypatch.setitem(sys.modules, "ptyprocess", fake_module)

    backend = pty_engine._spawn_unix_pty(["claude", "--foo"], cwd="/tmp", env={"A": "B"})

    assert calls["argv"] == ["claude", "--foo"]
    assert calls["cwd"] == "/tmp"
    assert calls["env"] == {"A": "B"}
    assert isinstance(backend, FakePtyBackend)


def test_spawn_unix_pty_passes_an_explicit_wide_dimensions_instead_of_the_80_column_default(monkeypatch):
    """Issue #109/#110: `ptyprocess`'s own default is a narrow 80x24, which
    is what causes Claude Code to word-wrap its question/option text across
    multiple physical lines. `_spawn_unix_pty` must override that default
    with an explicit, wide `dimensions=`, and its `cols` must be exactly
    `_PTY_COLUMNS` (== `_VIRTUAL_SCREEN_COLUMNS`, the same width
    `_render_terminal_text`'s virtual re-render screen uses) so the two
    can never drift out of sync."""
    import sys
    import types

    calls = {}

    class FakePtyProcessUnicode:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=None):
            calls["dimensions"] = dimensions
            return FakePtyBackend([])

    fake_module = types.SimpleNamespace(PtyProcessUnicode=FakePtyProcessUnicode)
    monkeypatch.setitem(sys.modules, "ptyprocess", fake_module)

    pty_engine._spawn_unix_pty(["claude"], cwd=None, env={})

    assert calls["dimensions"] == (pty_engine._PTY_ROWS, pty_engine._PTY_COLUMNS)
    assert calls["dimensions"][1] == pty_engine._VIRTUAL_SCREEN_COLUMNS
    assert calls["dimensions"] != (24, 80)


def test_spawn_winpty_passes_an_explicit_wide_dimensions_instead_of_the_80_column_default(monkeypatch):
    """Same as the `_spawn_unix_pty` case above, for the Windows backend:
    `winpty.PtyProcess.spawn`'s own default is a narrow 80x24, which
    `_spawn_winpty` must override with the same explicit, wide
    `dimensions=` (`_PTY_ROWS`, `_PTY_COLUMNS`)."""
    import sys
    import types

    calls = {}

    class FakePtyProcess:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=None, backend=None):
            calls["dimensions"] = dimensions
            return FakePtyBackend([])

    fake_module = types.SimpleNamespace(PtyProcess=FakePtyProcess)
    monkeypatch.setitem(sys.modules, "winpty", fake_module)

    pty_engine._spawn_winpty(["claude"], cwd=None, env={})

    assert calls["dimensions"] == (pty_engine._PTY_ROWS, pty_engine._PTY_COLUMNS)
    assert calls["dimensions"][1] == pty_engine._VIRTUAL_SCREEN_COLUMNS
    assert calls["dimensions"] != (24, 80)


def test_unix_pty_backend_conforms_to_pty_backend_protocol_surface():
    """A backend produced by `_spawn_unix_pty` (real or faked) only needs
    to support write/read/isalive/terminate to satisfy `PtyBackend` --
    verify a stand-in shaped like `ptyprocess.PtyProcessUnicode` round-trips
    through `PtyEngine` exactly like the Windows fake does."""
    backend = FakePtyBackend([f"hi from unix\n{TURN_COMPLETE_MARKER}\n"])

    def factory(argv, *, cwd, env):
        return backend

    engine = PtyEngine(pty_factory=factory)
    events = run(_collect(engine.stream_turn("hello")))

    assert "hi from unix" in events[-1]["result"]
    assert backend.writes == ["hello", "\r"]


# ---------------------------------------------------------------------------
# Dynamic PTY resize (issue #166): `PtyEngine.resize()` forwards to the live
# backend, keeps `self._cols` (and therefore `_render_terminal_text`'s
# virtual re-render screen width) in lockstep with the real PTY width, and
# is picked up by a later spawn/respawn when made against the real
# default-selected backend.
# ---------------------------------------------------------------------------


def test_resize_forwards_to_the_running_backends_resize_method():
    backend = FakePtyBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)
    engine.start()

    engine.resize(30, 100)

    assert backend.resizes == [(30, 100)]


def test_resize_before_start_does_not_raise_and_updates_stored_dimensions():
    """A resize signal can legitimately arrive before this card's engine has
    ever spawned a process (or after it's died) -- must not raise, and must
    still remember the new size for whenever a process does exist."""
    engine = PtyEngine(pty_factory=lambda *a, **kw: FakePtyBackend([], eof_after=False))

    engine.resize(10, 40)  # must not raise -- no process yet

    assert engine._rows == 10
    assert engine._cols == 40


def test_resize_updates_stored_dimensions_used_by_a_later_start(monkeypatch):
    """A resize made before `start()` is ever called must be honored by the
    eventual spawn, not silently dropped in favor of this engine's
    construction-time default -- verified against the REAL default-selected
    factory (not an injected fake), since only that path threads `rows`/
    `cols` through to the spawn call at all (see `PtyEngine.start`)."""
    calls = {}

    def fake_spawn(argv, *, cwd, env, rows=None, cols=None):
        calls["rows"] = rows
        calls["cols"] = cols
        return FakePtyBackend([])

    monkeypatch.setattr(pty_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pty_engine, "_spawn_unix_pty", fake_spawn)

    engine = PtyEngine()
    engine.resize(40, 140)
    engine.start()

    assert calls == {"rows": 40, "cols": 140}


def test_resize_keeps_the_virtual_render_screen_width_in_lockstep_with_the_pty_width(monkeypatch):
    """After a resize, every subsequent `_render_terminal_text` call this
    engine makes (both the per-chunk live trace and the final `result`
    render) must use the NEW column width, not the width this engine was
    constructed with -- the whole point of storing a single `self._cols`
    rather than two separately-tracked numbers (see `resize()`'s
    docstring)."""
    backend = FakePtyBackend([f"hi\n{TURN_COMPLETE_MARKER}\n"])
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)
    engine.start()

    engine.resize(30, 120)

    seen_columns = []
    real_render = pty_engine._render_terminal_text

    def capturing_render(raw, columns=pty_engine._VIRTUAL_SCREEN_COLUMNS):
        seen_columns.append(columns)
        return real_render(raw, columns)

    monkeypatch.setattr(pty_engine, "_render_terminal_text", capturing_render)

    run(_collect(engine.stream_turn("hi")))

    assert backend.resizes == [(30, 120)]
    assert seen_columns  # at least one render call happened
    assert all(columns == 120 for columns in seen_columns)


def test_a_default_constructed_engine_seeds_its_dimensions_from_the_module_defaults():
    engine = PtyEngine(pty_factory=lambda *a, **kw: FakePtyBackend([]))

    assert engine._rows == pty_engine._PTY_ROWS
    assert engine._cols == pty_engine._PTY_COLUMNS


def test_spawn_winpty_aliases_resize_to_the_real_objects_setwinsize(monkeypatch):
    """`winpty.PtyProcess` has no method literally named `resize` -- `_spawn_winpty`
    must alias it to the real object's own `setwinsize` right after spawn so the
    returned backend satisfies `PtyBackend.resize`."""
    import sys
    import types

    class FakeWinptyProcess:
        def __init__(self):
            self.resizes = []

        def setwinsize(self, rows, cols):
            self.resizes.append((rows, cols))

    class FakePtyProcess:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=None):
            return FakeWinptyProcess()

    fake_module = types.SimpleNamespace(PtyProcess=FakePtyProcess)
    monkeypatch.setitem(sys.modules, "winpty", fake_module)

    backend = pty_engine._spawn_winpty(["claude"], cwd=None, env={})
    backend.resize(50, 200)

    assert backend.resizes == [(50, 200)]


def test_spawn_unix_pty_aliases_resize_to_the_real_objects_setwinsize(monkeypatch):
    """Same as the winpty case above, for `ptyprocess.PtyProcessUnicode`."""
    import sys
    import types

    class FakeUnixProcess:
        def __init__(self):
            self.resizes = []

        def setwinsize(self, rows, cols):
            self.resizes.append((rows, cols))

    class FakePtyProcessUnicode:
        @classmethod
        def spawn(cls, argv, cwd=None, env=None, dimensions=None):
            return FakeUnixProcess()

    fake_module = types.SimpleNamespace(PtyProcessUnicode=FakePtyProcessUnicode)
    monkeypatch.setitem(sys.modules, "ptyprocess", fake_module)

    backend = pty_engine._spawn_unix_pty(["claude"], cwd=None, env={})
    backend.resize(50, 200)

    assert backend.resizes == [(50, 200)]


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_terminates_a_live_process():
    backend = FakePtyBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)
    engine.start()

    engine.close()

    assert backend.terminated is True


def test_close_is_a_noop_when_never_started():
    engine = PtyEngine(pty_factory=lambda *a, **kw: FakePtyBackend([]))
    engine.close()  # must not raise


# ---------------------------------------------------------------------------
# Public write() passthrough path and the shared write lock (issue #165)
# ---------------------------------------------------------------------------


def test_write_forwards_data_straight_to_the_backend():
    backend = FakePtyBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)
    engine.start()

    run(engine.write("raw passthrough keystrokes"))

    assert backend.writes == ["raw passthrough keystrokes"]


def test_write_raises_if_the_engine_was_never_started():
    engine = PtyEngine(pty_factory=lambda *a, **kw: FakePtyBackend([]))

    with pytest.raises(PtyEngineError):
        run(engine.write("x"))


def test_write_raises_after_close():
    backend = FakePtyBackend([], eof_after=False)
    factory, _ = _fake_factory(backend)
    engine = PtyEngine(pty_factory=factory)
    engine.start()
    engine.close()

    with pytest.raises(PtyEngineError):
        run(engine.write("x"))


class _SignalingPtyBackend(FakePtyBackend):
    """Same as `FakePtyBackend`, but sets `started_writing` the moment the
    first write lands -- used below to deterministically let a concurrent
    passthrough write attempt only AFTER the automated-turn write loop has
    already acquired `_write_lock` and is mid-sequence (rather than racing
    against a bare `asyncio.sleep`, which would be flaky)."""

    def __init__(self, chunks, started_writing, *, eof_after=False):
        super().__init__(chunks, eof_after=eof_after)
        self._started_writing = started_writing

    def write(self, data):
        result = super().write(data)
        if not self._started_writing.is_set():
            self._started_writing.set()
        return result


def test_concurrent_passthrough_write_never_interleaves_with_paced_automated_write():
    """Issue #165's core guarantee: a passthrough write arriving while an
    automated turn's paced prompt write is in flight must never land in the
    middle of that write's chunks -- it must be forced to wait for the
    ENTIRE paced sequence (every chunk plus the trailing "\\r") to finish,
    since anything else would physically interleave bytes into the same PTY
    from the fake backend's point of view."""

    async def scenario():
        started_writing = asyncio.Event()
        backend = _SignalingPtyBackend([TURN_COMPLETE_MARKER], started_writing)
        factory, _ = _fake_factory(backend)
        engine = PtyEngine(pty_factory=factory)
        engine.start()

        long_prompt = "x" * (pty_engine._WRITE_CHUNK_SIZE * 5)

        async def run_turn():
            return [event async for event in engine.stream_turn(long_prompt)]

        async def run_passthrough():
            # Wait until the automated write loop has already started (and
            # therefore already holds `_write_lock`) before racing our own
            # write in -- deterministic instead of a timing guess.
            await started_writing.wait()
            await engine.write("PASSTHROUGH")

        await asyncio.gather(run_turn(), run_passthrough())
        return backend.writes, long_prompt

    writes, long_prompt = run(scenario())

    passthrough_indices = [i for i, w in enumerate(writes) if w == "PASSTHROUGH"]
    assert len(passthrough_indices) == 1

    carriage_return_indices = [i for i, w in enumerate(writes) if w == "\r"]
    assert len(carriage_return_indices) == 1

    # The passthrough write started only once the automated write loop had
    # already begun (and thus already held `_write_lock`), so it must land
    # entirely AFTER the automated turn's whole paced sequence -- never
    # spliced in between its chunks or between the last chunk and the
    # trailing "\r".
    assert passthrough_indices[0] > carriage_return_indices[0]

    # Reconstructing everything up to (and including) the "\r" must
    # reproduce the original prompt exactly, with no foreign bytes mixed in
    # -- i.e. no corrupted/interleaved byte sequence.
    prompt_writes = writes[: carriage_return_indices[0]]
    assert "".join(prompt_writes) == long_prompt
