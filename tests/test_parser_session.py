"""Tests for the persistent per-project parser-session lifecycle (issue
#189, lifecycle slice of PRD #187 -- `gh issue view 187`/`gh issue view
189`). Follows `tests/test_stream_json_engine.py`'s conventions (a fake
`process_factory` standing in for a real `claude` subprocess) for the
module-level unit tests, and `tests/test_sessions.py`'s fake-engine-class +
`TestClient` conventions for the project-open wiring tests."""

import asyncio
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from rhubarb import db, parser_session
from rhubarb.stream_json_engine import StreamJsonEngine
from rhubarb.web import app as app_module


def run(coro):
    return asyncio.run(coro)


async def _run_and_drain(coro):
    """Await `coro`, then let any `asyncio.create_task(...)` it scheduled
    (e.g. `open_project`'s fire-and-forget parser-session warm) run to
    completion too -- mirrors `tests/test_sessions.py`'s own
    `_run_and_drain` helper, for the same reason: a fire-and-forget task's
    end state must be observed deterministically, not raced against."""
    result = await coro
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)
    return result


# ---------------------------------------------------------------------------
# Module-level unit tests: `ensure_parser_session`/`get_parser_session`/
# `close_all_parser_sessions` against a real `StreamJsonEngine`, driven by a
# fake `process_factory` (no real `claude` subprocess spawned) -- same
# injection seam `test_stream_json_engine.py` itself uses.
# ---------------------------------------------------------------------------


class FakeStreamJsonBackend:
    """Minimal stand-in for `StreamJsonEngine`'s `StreamJsonBackend`
    protocol -- mirrors `test_stream_json_engine.py`'s own fake of the same
    name, trimmed to what these lifecycle tests need (a scripted `result`
    line, `is_alive`/`terminate` bookkeeping, and -- for issue #190's
    `/clear`-turn assertions -- a `written_lines` log of every line written
    to it, same as `test_stream_json_engine.py`'s own fake already keeps)."""

    def __init__(self, lines, *, eof_after=True):
        self._lines = list(lines)
        self._eof_after = eof_after
        self.terminated = False
        self.written_lines = []

    def write_line(self, line):
        self.written_lines.append(line)

    def read_line(self):
        if self._lines:
            return self._lines.pop(0)
        if self._eof_after:
            raise EOFError
        return ""

    def is_alive(self):
        return not self.terminated and (bool(self._lines) or not self._eof_after)

    def terminate(self, force=False):
        self.terminated = True


def _result_line(session_id):
    import json

    return json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": "ok", "session_id": session_id}
    )


def _sequenced_process_factory(backends):
    """Hands out one backend per call, in order -- used to prove exactly how
    many real subprocess spawns happened across several `ensure_parser_
    session` calls."""
    backends = list(backends)
    calls = []

    def factory(argv, *, cwd, env):
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return backends.pop(0)

    return factory, calls


def _patch_stream_json_engine_process_factory(monkeypatch, factory):
    """`parser_session._spawn_parser_session` constructs a plain
    `StreamJsonEngine(cwd=..., model=..., effort=...)` with no
    `process_factory` kwarg exposed -- so these tests patch
    `StreamJsonEngine.__init__`'s default via a thin subclass-free shim:
    monkeypatch the `StreamJsonEngine` name `parser_session` itself imported,
    to a constructor that always injects `factory`."""

    class _InjectedStreamJsonEngine(StreamJsonEngine):
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            super().__init__(
                cwd=cwd, model=model, effort=effort, resume_session_id=resume_session_id, process_factory=factory
            )

    monkeypatch.setattr(parser_session, "StreamJsonEngine", _InjectedStreamJsonEngine)


def test_ensure_parser_session_spawns_exactly_one_subprocess_on_first_open(monkeypatch):
    backend = FakeStreamJsonBackend([_result_line("session-abc")], eof_after=False)
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    engine = run(parser_session.ensure_parser_session(1, cwd="/repo"))

    assert len(calls) == 1
    assert engine.isalive() is True
    assert 1 in parser_session._parser_sessions
    assert parser_session._parser_sessions[1] is engine


def test_reopening_the_same_project_reuses_the_existing_session_no_new_subprocess(monkeypatch):
    backend = FakeStreamJsonBackend([_result_line("session-abc"), _result_line("session-abc")], eof_after=False)
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    first = run(parser_session.ensure_parser_session(1, cwd="/repo"))
    # A real first turn establishes `session_id` the way the real project-
    # open -> later-turn flow eventually will (issue #190+); do it here so
    # the "same session_id both times" assertion below is meaningful rather
    # than trivially None == None.
    run(_drive_one_turn(first, "hello"))

    second = run(parser_session.ensure_parser_session(1, cwd="/repo"))

    assert len(calls) == 1  # only the original spawn -- reopening did not respawn
    assert second is first
    assert second.session_id == "session-abc"


async def _drive_one_turn(engine, prompt):
    async for _event in engine.stream_turn(prompt):
        pass


def test_opening_a_second_different_project_creates_an_independent_session(monkeypatch):
    backend_1 = FakeStreamJsonBackend([_result_line("session-proj1")], eof_after=False)
    backend_2 = FakeStreamJsonBackend([_result_line("session-proj2")], eof_after=False)
    factory, calls = _sequenced_process_factory([backend_1, backend_2])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    engine_1 = run(parser_session.ensure_parser_session(1, cwd="/repo-1"))
    engine_2 = run(parser_session.ensure_parser_session(2, cwd="/repo-2"))

    run(_drive_one_turn(engine_1, "hello project 1"))
    run(_drive_one_turn(engine_2, "hello project 2"))

    assert len(calls) == 2  # two independent spawns, one per project
    assert engine_1 is not engine_2
    assert engine_1.session_id == "session-proj1"
    assert engine_2.session_id == "session-proj2"
    assert parser_session._parser_sessions[1] is engine_1
    assert parser_session._parser_sessions[2] is engine_2


def test_both_parser_sessions_can_be_driven_concurrently_without_interfering(monkeypatch):
    """Two different projects' parser sessions, driven with a turn at the
    same time (via `asyncio.gather`), must each see only their own prompt
    and return their own distinct result/session_id -- no cross-talk."""
    backend_1 = FakeStreamJsonBackend([_result_line("session-proj1")], eof_after=False)
    backend_2 = FakeStreamJsonBackend([_result_line("session-proj2")], eof_after=False)
    factory, calls = _sequenced_process_factory([backend_1, backend_2])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    engine_1 = run(parser_session.ensure_parser_session(1, cwd="/repo-1"))
    engine_2 = run(parser_session.ensure_parser_session(2, cwd="/repo-2"))

    async def _turn(engine, prompt):
        events = [event async for event in engine.stream_turn(prompt)]
        return events[-1]

    async def _both():
        return await asyncio.gather(_turn(engine_1, "turn for project 1"), _turn(engine_2, "turn for project 2"))

    result_1, result_2 = run(_both())

    assert result_1["session_id"] == "session-proj1"
    assert result_2["session_id"] == "session-proj2"
    assert result_1["session_id"] != result_2["session_id"]


def test_ensure_parser_session_respawns_when_the_registered_one_has_died(monkeypatch):
    dead_backend = FakeStreamJsonBackend([], eof_after=False)
    dead_backend.terminated = True  # already dead before anyone asks
    fresh_backend = FakeStreamJsonBackend([_result_line("session-fresh")], eof_after=False)
    factory, calls = _sequenced_process_factory([dead_backend, fresh_backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    first = run(parser_session.ensure_parser_session(1, cwd="/repo"))
    assert first.isalive() is False

    second = run(parser_session.ensure_parser_session(1, cwd="/repo"))

    assert len(calls) == 2
    assert second is not first
    assert second.isalive() is True


def test_get_parser_session_returns_none_when_nothing_registered():
    assert parser_session.get_parser_session(999) is None


def test_get_parser_session_returns_none_for_a_dead_registered_session(monkeypatch):
    backend = FakeStreamJsonBackend([], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    engine = run(parser_session.ensure_parser_session(1, cwd="/repo"))
    backend.terminated = True

    assert parser_session.get_parser_session(1) is None
    assert engine.isalive() is False


def test_concurrent_ensure_calls_for_the_same_project_spawn_exactly_once(monkeypatch):
    """Two overlapping `ensure_parser_session` calls for the SAME project
    (e.g. a double-clicked project open) must not race into spawning two
    subprocesses -- the loser of the race gets the winner's engine back."""
    backend = FakeStreamJsonBackend([], eof_after=False)
    calls = []

    def slow_factory(argv, *, cwd, env):
        import time

        time.sleep(0.05)  # widen the race window inside `asyncio.to_thread`
        calls.append(argv)
        return backend

    def slow_spawn(*, cwd, model, effort):
        engine = StreamJsonEngine(cwd=cwd, model=model, effort=effort, process_factory=slow_factory)
        engine.start()
        return engine

    monkeypatch.setattr(parser_session, "_spawn_parser_session", lambda **kw: slow_spawn(**kw))

    async def _both():
        return await asyncio.gather(
            parser_session.ensure_parser_session(1, cwd="/repo"),
            parser_session.ensure_parser_session(1, cwd="/repo"),
        )

    engine_a, engine_b = run(_both())

    assert len(calls) == 1
    assert engine_a is engine_b


def test_close_all_parser_sessions_terminates_every_live_session_and_clears_registry(monkeypatch):
    backend_1 = FakeStreamJsonBackend([], eof_after=False)
    backend_2 = FakeStreamJsonBackend([], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend_1, backend_2])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)

    engine_1 = run(parser_session.ensure_parser_session(1, cwd="/repo-1"))
    engine_2 = run(parser_session.ensure_parser_session(2, cwd="/repo-2"))
    assert engine_1.isalive() is True
    assert engine_2.isalive() is True

    parser_session.close_all_parser_sessions()

    assert backend_1.terminated is True
    assert backend_2.terminated is True
    assert engine_1.isalive() is False
    assert engine_2.isalive() is False
    assert parser_session._parser_sessions == {}


def test_close_all_parser_sessions_is_a_noop_on_an_empty_registry():
    parser_session.close_all_parser_sessions()  # must not raise


# ---------------------------------------------------------------------------
# Project-open wiring: `rhubarb/web/app.py`'s `open_project` handler lazily
# creates a project's parser session on first open, and reuses it on later
# opens -- mirrors `tests/test_sessions.py`'s `_make_fake_engine_class` +
# `TestClient` conventions for exercising the real HTTP-facing handler.
# ---------------------------------------------------------------------------


def _init_repo(path, remote_url):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=path, check=True)


def _create_project(client, tmp_path, name):
    """Same as `test_sessions.py`'s `_open_project`, minus the final
    `/open` POST -- these tests drive `open_project` directly (via
    `_run_and_drain`) instead, so the fire-and-forget parser-session warm
    it schedules is deterministically observed rather than raced against."""
    root = tmp_path / name
    root.mkdir()
    _init_repo(root / "repo", f"https://github.com/x/{name}.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root), "confirm": True})
    return client.get("/api/app-state").json()["projects"][0]["id"]


def _create_two_projects(client, tmp_path):
    """`scan_projects` scans ONE root dir, one level deep, for every git
    repo with a GitHub remote directly under it (`rhubarb/projects.py`) --
    unlike `_create_project` above (one project per its own root-dir call,
    which wipes/rescans the whole `projects` table each time -- see
    `db.set_root_dir`), this creates TWO sibling repos under one shared
    root and scans once, so both projects coexist in the db at the same
    time, each with its own id -- what's actually needed to test two
    concurrently-open projects."""
    _init_repo(tmp_path / "proj-a", "https://github.com/x/proj-a.git")
    _init_repo(tmp_path / "proj-b", "https://github.com/x/proj-b.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(tmp_path), "confirm": True})
    projects = {p["name"]: p["id"] for p in client.get("/api/app-state").json()["projects"]}
    return projects["proj-a"], projects["proj-b"]


def _make_fake_parser_engine_class():
    class FakeEngine:
        instances: list["FakeEngine"] = []

        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.cwd = cwd
            self.model = model
            self.effort = effort
            self.session_id = resume_session_id or f"generated-{len(FakeEngine.instances) + 1}"
            self.started = False
            self.closed = False
            FakeEngine.instances.append(self)

        def start(self):
            self.started = True
            return self

        def close(self):
            self.closed = True

        def isalive(self):
            return self.started and not self.closed

        async def stream_turn(self, prompt):
            yield {"type": "result", "subtype": "success", "is_error": False, "result": "ok", "session_id": self.session_id}

    return FakeEngine


def _mock_parser_engine(monkeypatch):
    fake_class = _make_fake_parser_engine_class()
    monkeypatch.setattr(parser_session, "StreamJsonEngine", fake_class)
    return fake_class


def test_opening_a_project_for_the_first_time_creates_exactly_one_parser_session(client, tmp_path, monkeypatch):
    fake_class = _mock_parser_engine(monkeypatch)
    project_id = _create_project(client, tmp_path, "proj")

    run(_run_and_drain(app_module.open_project(project_id)))

    assert len(fake_class.instances) == 1
    assert fake_class.instances[0].started is True
    assert parser_session._parser_sessions[project_id] is fake_class.instances[0]


def test_reopening_the_same_project_via_the_endpoint_reuses_the_parser_session(client, tmp_path, monkeypatch):
    fake_class = _mock_parser_engine(monkeypatch)
    project_id = _create_project(client, tmp_path, "proj")

    run(_run_and_drain(app_module.open_project(project_id)))
    first = parser_session._parser_sessions[project_id]
    run(_run_and_drain(app_module.open_project(project_id)))
    second = parser_session._parser_sessions[project_id]

    assert len(fake_class.instances) == 1  # still only ever spawned once
    assert second is first
    assert second.session_id == first.session_id


def test_opening_a_second_project_via_the_endpoint_gets_its_own_parser_session(client, tmp_path, monkeypatch):
    fake_class = _mock_parser_engine(monkeypatch)
    project_a, project_b = _create_two_projects(client, tmp_path)

    run(_run_and_drain(app_module.open_project(project_a)))
    run(_run_and_drain(app_module.open_project(project_b)))

    assert len(fake_class.instances) == 2
    engine_a = parser_session._parser_sessions[project_a]
    engine_b = parser_session._parser_sessions[project_b]
    assert engine_a is not engine_b
    assert engine_a.session_id != engine_b.session_id


def test_closing_a_project_does_not_tear_down_its_parser_session(client, tmp_path, monkeypatch):
    """Per PRD #187, a parser session's lifetime is independent of project
    UI focus -- closing (or switching away from) a project must leave its
    parser session alive, unlike the standby engine `close_project` does
    explicitly close."""
    fake_class = _mock_parser_engine(monkeypatch)
    project_id = _create_project(client, tmp_path, "proj")

    run(_run_and_drain(app_module.open_project(project_id)))
    engine = parser_session._parser_sessions[project_id]

    client.post(f"/api/projects/{project_id}/close", json={"session_state": {}})

    assert parser_session._parser_sessions[project_id] is engine
    assert engine.closed is False


def test_app_shutdown_terminates_all_live_parser_session_subprocesses(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_path / "rhubarb.db")
    monkeypatch.setattr(db, "OLD_DB_PATH", tmp_path / "not-a-real-legacy-db" / "baton.db")
    monkeypatch.setattr(app_module, "_active_project_id", None)
    fake_class = _mock_parser_engine(monkeypatch)

    with TestClient(app_module.app) as c:
        project_id = _create_project(c, tmp_path, "proj")
        run(_run_and_drain(app_module.open_project(project_id)))
        engine = parser_session._parser_sessions[project_id]
        assert engine.closed is False

    assert engine.closed is True
    assert parser_session._parser_sessions == {}


# ---------------------------------------------------------------------------
# Token-ceiling handling (issue #190, `gh issue view 190` for full context):
# `parser_session.stream_turn` tracks context usage from each turn's raw
# `result` event and, once usage crosses 60% of the model's 1M-token
# context window, drains one `/clear` turn through the SAME still-open
# subprocess/session before returning. Driven directly with synthetic
# turns against a fake `process_factory`, per the issue's own "buildable
# and testable independently of the real needs-input queue" framing --
# same injection seam the lifecycle tests above already use.
# ---------------------------------------------------------------------------


def _result_line_with_usage(
    session_id,
    *,
    input_tokens=0,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
    context_window=1_000_000,
    model="claude-parser-model",
    text="ok",
):
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": text,
            "session_id": session_id,
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            },
            "modelUsage": {model: {"contextWindow": context_window}},
        }
    )


async def _consume(agen):
    return [event async for event in agen]


def _written_prompts(backend):
    """The `content` field of every turn written to `backend`, in order --
    what a test checks to prove exactly which turns (the caller's own
    prompts vs. an auto-issued `/clear`) actually reached the subprocess."""
    return [json.loads(line)["message"]["content"] for line in backend.written_lines]


def test_stream_turn_tracks_context_pct_from_result_event_usage(monkeypatch):
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", input_tokens=100_000)], eof_after=False
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    assert parser_session.get_context_pct(1) is None  # nothing completed yet

    run(_consume(parser_session.stream_turn(1, "parse this turn")))

    assert parser_session.get_context_pct(1) == pytest.approx(0.10)
    assert _written_prompts(backend) == ["parse this turn"]  # well under 60%, no /clear issued


def test_result_event_without_usage_data_leaves_tracked_pct_unchanged(monkeypatch):
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", input_tokens=100_000), _result_line("session-abc")],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    run(_consume(parser_session.stream_turn(1, "first turn")))
    assert parser_session.get_context_pct(1) == pytest.approx(0.10)

    run(_consume(parser_session.stream_turn(1, "second turn, no usage in its result")))

    assert parser_session.get_context_pct(1) == pytest.approx(0.10)  # unchanged, not clobbered to None/0


def test_crossing_60_percent_does_not_interrupt_the_in_flight_parse(monkeypatch):
    """The turn that itself pushes usage over the cutoff must still yield
    every one of its own events normally -- the ceiling is only ever acted
    on AFTER this turn's `result` event, never mid-turn."""
    backend = FakeStreamJsonBackend(
        [
            json.dumps({"type": "system", "subtype": "init", "session_id": "session-abc"}),
            json.dumps({"type": "stream_event", "event": {"delta": "partial text"}}),
            _result_line_with_usage("session-abc", input_tokens=650_000, text="the full parse result"),
            _result_line_with_usage("session-abc", input_tokens=5_000, text="cleared"),  # the auto /clear's result
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    events = run(_consume(parser_session.stream_turn(1, "parse this turn")))

    kinds = [e["type"] for e in events]
    # The in-flight parse's own events all made it through untouched, in
    # order, INCLUDING its own result -- nothing was truncated or dropped
    # to react to the crossing.
    assert kinds[:3] == ["system", "stream_event", "result"]
    assert events[2]["result"] == "the full parse result"


def test_crossing_60_percent_sends_clear_turn_to_the_same_subprocess_after_parse_completes(monkeypatch):
    backend = FakeStreamJsonBackend(
        [
            _result_line_with_usage("session-abc", input_tokens=650_000),  # crosses 60%
            _result_line_with_usage("session-abc", input_tokens=5_000),  # the auto /clear's own result
        ],
        eof_after=False,
    )
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    engine = run(parser_session.ensure_parser_session(1, cwd="/repo"))

    events = run(_consume(parser_session.stream_turn(1, "parse this turn")))

    # Exactly two turns reached the subprocess: the caller's own prompt,
    # then -- and only then -- a `/clear` turn, automatically.
    assert _written_prompts(backend) == ["parse this turn", "/clear"]
    # Same subprocess, same session_id -- not a respawn.
    assert len(calls) == 1
    assert engine.session_id == "session-abc"
    assert engine.isalive() is True
    # The caller can observe the clear happened via the synthetic event.
    assert events[-1] == {"type": "context_cleared", "project_id": 1, "session_id": "session-abc"}
    # Tracked usage reflects the /clear turn's own (much lower) reading,
    # not a hardcoded/assumed zero.
    assert parser_session.get_context_pct(1) == pytest.approx(0.005)


def test_usage_under_60_percent_never_triggers_a_clear_turn(monkeypatch):
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", input_tokens=300_000)], eof_after=False
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    events = run(_consume(parser_session.stream_turn(1, "parse this turn")))

    assert _written_prompts(backend) == ["parse this turn"]
    assert all(e["type"] != "context_cleared" for e in events)


def test_turn_submitted_after_the_clear_is_processed_normally(monkeypatch):
    """A `stream_turn` call made AFTER a prior call already crossed the
    ceiling and cleared must be a completely ordinary turn: no extra
    `/clear`, same live subprocess, its own result returned."""
    backend = FakeStreamJsonBackend(
        [
            _result_line_with_usage("session-abc", input_tokens=650_000),  # crosses 60%
            _result_line_with_usage("session-abc", input_tokens=5_000),  # the auto /clear's own result
            _result_line_with_usage("session-abc", input_tokens=8_000, text="post-clear parse"),
        ],
        eof_after=False,
    )
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    engine = run(parser_session.ensure_parser_session(1, cwd="/repo"))

    run(_consume(parser_session.stream_turn(1, "turn that crosses the ceiling")))
    events = run(_consume(parser_session.stream_turn(1, "turn after the clear")))

    assert _written_prompts(backend) == ["turn that crosses the ceiling", "/clear", "turn after the clear"]
    assert len(calls) == 1  # still the same one subprocess throughout
    assert engine.session_id == "session-abc"
    assert events[-1]["type"] == "result"
    assert events[-1]["result"] == "post-clear parse"
    assert all(e["type"] != "context_cleared" for e in events)  # this turn alone didn't cross anything
    assert parser_session.get_context_pct(1) == pytest.approx(0.008)


def test_stream_turn_raises_lookup_error_when_no_parser_session_is_registered():
    async def _try():
        return await _consume(parser_session.stream_turn(999, "hi"))

    with pytest.raises(LookupError):
        run(_try())


# ---------------------------------------------------------------------------
# Needs-input queue (issue #191, `gh issue view 191` for full context). Pure
# module-level tests of the FIFO queue primitives themselves
# (`enqueue_needs_input_turn`/`dequeue_needs_input_turn`/
# `get_needs_input_queue`) -- no engine/subprocess involved at all.
# `session_runner`'s own tests (`tests/test_sessions.py`) cover the actual
# gating hook (`handle_turn_completed`) that calls these.
# ---------------------------------------------------------------------------


def test_enqueue_needs_input_turn_returns_the_enqueued_item():
    item = parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="Which database?")

    assert item == {"project_id": 1, "card_id": 5, "phase": "grilling", "text": "Which database?"}
    assert parser_session.get_needs_input_queue(1) == [item]


def test_enqueue_preserves_fifo_order_within_one_project():
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="first")
    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="second")
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="third")

    queue = parser_session.get_needs_input_queue(1)
    assert [item["text"] for item in queue] == ["first", "second", "third"]
    assert [item["card_id"] for item in queue] == [5, 6, 5]


def test_enqueue_keeps_different_projects_independent():
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="project one")
    parser_session.enqueue_needs_input_turn(2, card_id=9, phase="creating_prd", text="project two")

    assert [item["text"] for item in parser_session.get_needs_input_queue(1)] == ["project one"]
    assert [item["text"] for item in parser_session.get_needs_input_queue(2)] == ["project two"]


def test_dequeue_needs_input_turn_pops_the_oldest_item_first():
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="first")
    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="second")

    first = parser_session.dequeue_needs_input_turn(1)
    second = parser_session.dequeue_needs_input_turn(1)

    assert first["text"] == "first"
    assert second["text"] == "second"
    assert parser_session.get_needs_input_queue(1) == []


def test_dequeue_needs_input_turn_returns_none_when_empty_or_unknown():
    assert parser_session.dequeue_needs_input_turn(999) is None

    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="only one")
    parser_session.dequeue_needs_input_turn(1)
    assert parser_session.dequeue_needs_input_turn(1) is None


def test_get_needs_input_queue_is_a_snapshot_that_does_not_mutate_the_queue():
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="only one")

    snapshot = parser_session.get_needs_input_queue(1)
    snapshot.append({"fake": "entry"})

    assert len(parser_session.get_needs_input_queue(1)) == 1


def test_get_needs_input_queue_returns_empty_list_for_unknown_project():
    assert parser_session.get_needs_input_queue(999) == []


# ---------------------------------------------------------------------------
# Queue draining and structured extraction (issue #192, `gh issue view 192`
# for full context): `drain_needs_input_queue` pops a project's needs-input
# queue one item at a time, sends each item's raw text through the parser
# session, and yields a tagged (`source_session_id`) normalized result.
# Driven against a fake `process_factory`, same injection seam the lifecycle/
# token-ceiling tests above already use -- no real `claude` subprocess.
# ---------------------------------------------------------------------------


_VALID_PAYLOAD_1 = {
    "header": "",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Should this be Python or Node?",
            "kind": "single",
            "options": ["Python", "Node"],
            "recommended": [1],
            "recommended_text": None,
        }
    ],
}

_VALID_PAYLOAD_2 = {
    "header": "Almost done.",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which environments should this support?",
            "kind": "multi",
            "options": ["Dev", "Staging", "Prod"],
            "recommended": [1, 3],
            "recommended_text": None,
        }
    ],
}

_VALID_PAYLOAD_3 = {
    "header": "",
    "footer": "Thanks!",
    "questions": [
        {
            "id": "q1",
            "text": "Where should this run?",
            "kind": "open",
            "options": None,
            "recommended": None,
            "recommended_text": "On the existing droplet.",
        }
    ],
}


def _extraction_result_line(session_id, payload, **usage_kwargs):
    return _result_line_with_usage(session_id, text=json.dumps(payload), **usage_kwargs)


async def _drain_all(project_id):
    return [item async for item in parser_session.drain_needs_input_queue(project_id)]


def test_drain_processes_a_single_queued_item_and_tags_it(monkeypatch):
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", _VALID_PAYLOAD_1)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="Python or Node?")

    results = run(_drain_all(1))

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["source_session_id"] == 5
    assert results[0]["header"] == _VALID_PAYLOAD_1["header"]
    assert results[0]["questions"] == _VALID_PAYLOAD_1["questions"]
    assert results[0]["footer"] == _VALID_PAYLOAD_1["footer"]
    assert parser_session.get_needs_input_queue(1) == []
    assert parser_session.get_parsed_results(1) == results


def test_drain_processes_items_in_fifo_order_across_different_source_sessions(monkeypatch):
    """Two queued items from different source sessions (`card_id`) in the
    SAME project must come back as two independently-tagged results, in the
    order they were enqueued -- never conflated with each other."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="turn from card 5")
    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="turn from card 6")

    results = run(_drain_all(1))

    assert len(results) == 2
    assert results[0]["source_session_id"] == 5
    assert results[0]["questions"] == _VALID_PAYLOAD_1["questions"]
    assert results[1]["source_session_id"] == 6
    assert results[1]["questions"] == _VALID_PAYLOAD_2["questions"]
    # Not conflated: each result carries only its own item's own payload.
    assert results[0]["header"] != results[1]["header"] or results[0]["questions"] != results[1]["questions"]


def test_drain_sends_items_to_the_correct_project_and_leaves_other_projects_alone(monkeypatch):
    backend_1 = FakeStreamJsonBackend([_extraction_result_line("session-p1", _VALID_PAYLOAD_1)], eof_after=False)
    backend_2 = FakeStreamJsonBackend([_extraction_result_line("session-p2", _VALID_PAYLOAD_2)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend_1, backend_2])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo-1"))
    run(parser_session.ensure_parser_session(2, cwd="/repo-2"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="project one's turn")
    parser_session.enqueue_needs_input_turn(2, card_id=9, phase="creating_prd", text="project two's turn")

    results_1 = run(_drain_all(1))
    results_2 = run(_drain_all(2))

    assert len(results_1) == 1 and results_1[0]["source_session_id"] == 5
    assert len(results_2) == 1 and results_2[0]["source_session_id"] == 9
    assert parser_session.get_needs_input_queue(1) == []
    assert parser_session.get_needs_input_queue(2) == []


def test_drain_respects_the_drain_then_clear_ceiling_mid_queue(monkeypatch):
    """Three items queued; the FIRST item's own turn crosses the 60%
    ceiling, triggering an automatic `/clear` (issue #190) before the queue
    keeps going. All three must still be processed, in order, none skipped
    or double-processed, and the `/clear` must reach the subprocess between
    the first and second items' own extraction turns."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1, input_tokens=650_000),  # crosses 60%
            _result_line_with_usage("session-abc", input_tokens=5_000, text="cleared"),  # the auto /clear's result
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2, input_tokens=8_000),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_3, input_tokens=8_000),
        ],
        eof_after=False,
    )
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    engine = run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="first")
    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="second")
    parser_session.enqueue_needs_input_turn(1, card_id=7, phase="qa_grilling", text="third")

    results = run(_drain_all(1))

    assert [r["source_session_id"] for r in results] == [5, 6, 7]
    assert [r["ok"] for r in results] == [True, True, True]
    assert results[0]["questions"] == _VALID_PAYLOAD_1["questions"]
    assert results[1]["questions"] == _VALID_PAYLOAD_2["questions"]
    assert results[2]["questions"] == _VALID_PAYLOAD_3["questions"]
    # Same subprocess throughout -- the ceiling crossing never respawned it.
    assert len(calls) == 1
    assert engine.isalive() is True
    # The `/clear` reached the subprocess exactly once, between the first
    # and second items' own extraction prompts -- nothing skipped, nothing
    # sent twice.
    written = _written_prompts(backend)
    assert len(written) == 4
    assert written[1] == "/clear"
    assert parser_session.get_needs_input_queue(1) == []


def test_drain_marks_a_malformed_non_json_response_as_a_failed_but_tagged_result(monkeypatch):
    """"turn text" (the queued item's raw text) has no "question " trigger
    and doesn't parse via the regex parser either, so issue #194's fallback
    (attempted automatically once the primary parse fails) also comes up
    empty here -- the result stays `ok: False`, but is now additionally
    tagged `"source": "fallback"` to record that the fallback WAS attempted
    (see the "Legacy fallback on parse failure" tests further down for the
    fallback-succeeds cases)."""
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", text="Sorry, I can't help with that request.")],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="turn text")

    results = run(_drain_all(1))

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert results[0]["source_session_id"] == 5  # still tagged, even on failure
    assert results[0]["source"] == "fallback"  # issue #194: fallback was attempted (and also failed)
    assert "error" in results[0]
    assert parser_session.get_needs_input_queue(1) == []  # item still consumed, not stuck


def test_drain_marks_a_response_with_an_invalid_schema_shape_as_a_failed_result(monkeypatch):
    invalid_payload = {"header": "", "questions": []}  # missing required "footer"
    backend = FakeStreamJsonBackend(
        [_extraction_result_line("session-abc", invalid_payload)],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="turn text")

    results = run(_drain_all(1))

    assert results[0]["ok"] is False
    assert results[0]["source_session_id"] == 5
    assert results[0]["source"] == "fallback"  # issue #194: fallback attempted here too, also came up empty


def test_drain_extracts_json_wrapped_in_a_markdown_code_fence(monkeypatch):
    """The parser session is asked to respond with bare JSON, but a model
    might still wrap it in a fenced code block -- this must still parse."""
    fenced_text = "Here you go:\n```json\n" + json.dumps(_VALID_PAYLOAD_1) + "\n```\n"
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", text=fenced_text)],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="turn text")

    results = run(_drain_all(1))

    assert results[0]["ok"] is True
    assert results[0]["questions"] == _VALID_PAYLOAD_1["questions"]


def test_drain_yields_all_questions_from_a_three_question_payload(monkeypatch):
    """A parser-session response carrying three questions in `questions[]`
    must pass through drain intact -- issue #207's core scenario: grilling
    output with multiple questions must NOT be collapsed into one entry."""
    three_q_payload = {
        "header": "A few things to settle before we start.",
        "footer": "",
        "questions": [
            {
                "id": "q1",
                "text": "Should this be Python or Node?",
                "kind": "single",
                "options": ["Python", "Node"],
                "recommended": [1],
                "recommended_text": None,
            },
            {
                "id": "q2",
                "text": "Which environments need support?",
                "kind": "multi",
                "options": ["Dev", "Staging", "Prod"],
                "recommended": [1, 3],
                "recommended_text": None,
            },
            {
                "id": "q3",
                "text": "Where should this run?",
                "kind": "open",
                "options": None,
                "recommended": None,
                "recommended_text": "On the existing droplet.",
            },
        ],
    }
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", three_q_payload)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="three questions here")

    results = run(_drain_all(1))

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["source_session_id"] == 5
    assert len(results[0]["questions"]) == 3
    assert results[0]["questions"] == three_q_payload["questions"]
    assert results[0]["header"] == three_q_payload["header"]


def test_drain_preserves_mixed_kind_payload_fields(monkeypatch):
    """A mixed-kind payload (single / multi / open) must arrive at the
    caller with every per-question field intact -- kind, options,
    recommended, recommended_text -- so the left main card can render
    each question in its correct UI shape."""
    mixed_payload = {
        "header": "",
        "footer": "Thanks!",
        "questions": [
            {
                "id": "q1",
                "text": "Pick one language.",
                "kind": "single",
                "options": ["Python", "Go"],
                "recommended": [2],
                "recommended_text": None,
            },
            {
                "id": "q2",
                "text": "Which features are required?",
                "kind": "multi",
                "options": ["Auth", "Logging", "Metrics"],
                "recommended": [1, 2],
                "recommended_text": None,
            },
            {
                "id": "q3",
                "text": "Any other constraints?",
                "kind": "open",
                "options": None,
                "recommended": None,
                "recommended_text": "Keep it simple for now.",
            },
        ],
    }
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", mixed_payload)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=7, phase="grilling", text="mixed question types")

    results = run(_drain_all(1))

    q = results[0]["questions"]
    assert q[0]["kind"] == "single" and q[0]["options"] == ["Python", "Go"] and q[0]["recommended"] == [2]
    assert q[1]["kind"] == "multi" and q[1]["recommended"] == [1, 2] and q[1]["recommended_text"] is None
    assert q[2]["kind"] == "open" and q[2]["options"] is None and q[2]["recommended_text"] == "Keep it simple for now."


def test_extraction_prompt_invokes_the_parse_interview_skill_with_phase_and_text(monkeypatch):
    """Issue #228: the inline `_EXTRACTION_PROMPT_TEMPLATE` prompt string is
    gone -- the multi-question splitting guidance and the `Recommended:`
    mapping rule now live in the `rhubarb` plugin's own
    `/rhubarb:parse-interview` skill file
    (`rhubarb/claude_plugin/skills/parse-interview/SKILL.md`), discoverable
    by every parser-session subprocess via `--plugin-dir` regardless of
    phase. What this module must still get right is invoking that skill,
    uniformly, with this item's own `phase` and raw `text` -- so this
    asserts the prompt actually written to the subprocess references the
    skill and carries both inputs through, rather than re-asserting prose
    that no longer lives in this file."""
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", _VALID_PAYLOAD_1)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="Does this matter?")

    run(_drain_all(1))

    # written_lines[0] is the init/resume handshake; written_lines[1] is the
    # first user-turn JSON written to the subprocess stdin.
    assert len(backend.written_lines) >= 1
    prompt_json = backend.written_lines[-1]
    assert "/rhubarb:parse-interview" in prompt_json
    assert "phase: grilling" in prompt_json
    assert "Does this matter?" in prompt_json


def test_build_extraction_prompt_invokes_the_skill_uniformly_across_phases():
    """No phase-specific branching (issue #228's acceptance criterion): the
    same `/rhubarb:parse-interview` invocation shape is produced for every
    phase that routes through this extraction path, differing only in the
    `phase:` value itself."""
    for phase in ("grilling", "qa", "qa_grilling", "implement"):
        prompt = parser_session._build_extraction_prompt(phase=phase, text="raw turn text")

        assert prompt.startswith("/rhubarb:parse-interview ")
        assert f"phase: {phase}" in prompt
        assert "raw turn text" in prompt


def test_reproduces_original_bug_scenario_prompt_carries_the_full_raw_text_through(monkeypatch):
    """Issue #228's regression scenario (root cause of PRD #227): a turn
    whose raw text has a paragraph of prose, then a transition phrase, then
    a `❓ **Q5**` block with options and a `Recommended:` line, where the
    old inline prompt had no explicit rule for mapping that line onto
    `recommended`/`recommended_text` and silently dropped it. This module
    can't verify the skill's actual parsing accuracy (that's the skill's own
    prose, exercised via worked examples, not this Python code) -- what it
    verifies is (a) the prompt actually sent to the parser session invokes
    the new skill with this exact raw text intact, and (b) a scripted
    response that correctly preserves `options`/`recommended` threads all
    the way through the existing JSON-response handling to the tagged
    result, end to end."""
    raw_text = (
        "We've settled the schema and the API shape already. The last open "
        "branch is how retries should behave under load, since that changes "
        "how aggressively the client backs off.\n\n"
        "One more branch to close:\n\n"
        "❓ **Q5** - **Retry backoff**: How should the client back off "
        "between retries?\n"
        "- Fixed 1s delay\n"
        "- Exponential backoff\n"
        "- No retry, fail fast\n"
        "Recommended: Exponential backoff\n"
    )
    scripted_response = {
        "header": (
            "We've settled the schema and the API shape already. The last open "
            "branch is how retries should behave under load, since that changes "
            "how aggressively the client backs off.\n\nOne more branch to close:"
        ),
        "footer": "",
        "questions": [
            {
                "id": "q5",
                "text": "How should the client back off between retries?",
                "kind": "single",
                "options": ["Fixed 1s delay", "Exponential backoff", "No retry, fail fast"],
                "recommended": [2],
                "recommended_text": None,
            }
        ],
    }
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", scripted_response)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text=raw_text)

    results = run(_drain_all(1))

    # (a) the constructed prompt invokes the new skill and carries the raw
    # text (including the transition phrase) through untouched.
    prompt_json = backend.written_lines[-1]
    assert "/rhubarb:parse-interview" in prompt_json
    assert "phase: grilling" in prompt_json
    assert "One more branch to close" in prompt_json
    assert "Recommended: Exponential backoff" in prompt_json

    # (b) the scripted response's populated options/recommended thread
    # through the existing response handling untouched.
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["source"] == "parser_session"
    question = results[0]["questions"][0]
    assert question["kind"] == "single"
    assert question["options"] == ["Fixed 1s delay", "Exponential backoff", "No retry, fail fast"]
    assert question["recommended"] == [2]
    assert question["recommended_text"] is None


def test_drain_never_raises_when_no_parser_session_is_registered_for_the_project():
    """No live parser session at all is the LookupError branch of
    `_process_one_queued_item`'s `except Exception` -- the same
    subprocess-failure shape issue #194's fallback treats generically.
    "turn text" doesn't parse/rescue into anything, so the fallback also
    comes up empty, but the result is still tagged `"source": "fallback"`
    to record that it was attempted."""
    parser_session.enqueue_needs_input_turn(999, card_id=5, phase="grilling", text="turn text")

    results = run(_drain_all(999))

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert results[0]["source_session_id"] == 5
    assert results[0]["source"] == "fallback"
    assert parser_session.get_needs_input_queue(999) == []


def test_drain_on_an_empty_queue_yields_nothing():
    results = run(_drain_all(12345))
    assert results == []


def test_get_parsed_results_accumulates_across_multiple_drains(monkeypatch):
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="first")
    run(_drain_all(1))

    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="second")
    run(_drain_all(1))

    all_results = parser_session.get_parsed_results(1)
    assert [r["source_session_id"] for r in all_results] == [5, 6]


def test_get_parsed_results_returns_empty_list_for_unknown_project():
    assert parser_session.get_parsed_results(54321) == []


# ---------------------------------------------------------------------------
# `GET /api/projects/{project_id}/parsed-results` (issue #193, "Route parsed
# results to the focused card or a toast" -- `gh issue view 193` for full
# context): the poll endpoint `prompt.html`'s `pollParsedResultsTick` uses to
# learn about new parser-session results and route each one (client-side) to
# the focused left card or a toast. This module has no way to drive real
# browser JS, so per this repo's own testing convention (see #188's own
# verification, and this issue's task description) the backend routing/
# storage half -- the endpoint's own draining and `since`-offset paging --
# gets real automated tests here; the client-side routing decision itself
# (focused card vs. toast, held-result-on-focus) is verified statically
# instead (`node --check` on the extracted <script> body, plus a manual
# reading of `routeParsedResult`/`showHeldParsedResultIfAny`/`focusCard` in
# `rhubarb/web/templates/prompt.html`).
#
# Deliberately exercises the endpoint function directly (`run(app_module.
# get_parsed_results_endpoint(...))`), not via `TestClient.get(...)` --
# mirrors this file's own existing convention of calling `app_module.
# open_project(...)` directly rather than through the HTTP layer (see the
# lifecycle-endpoint tests above), so parser-session setup
# (`ensure_parser_session`, whose `asyncio.Lock` is bound to whichever event
# loop creates it) and the endpoint call both run on the exact same
# `asyncio.run(...)` loop, never crossing into `TestClient`'s own separate
# internal loop.
# ---------------------------------------------------------------------------


def test_parsed_results_endpoint_returns_empty_when_nothing_queued_or_produced():
    data = run(app_module.get_parsed_results_endpoint(777, since=0))
    assert data == {"results": [], "total": 0}


def test_parsed_results_endpoint_drains_the_queue_and_returns_the_tagged_result(monkeypatch):
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", _VALID_PAYLOAD_1)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="Python or Node?")

    data = run(app_module.get_parsed_results_endpoint(1, since=0))

    assert data["total"] == 1
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["ok"] is True
    assert result["source_session_id"] == 5
    assert result["questions"] == _VALID_PAYLOAD_1["questions"]
    # The endpoint itself drove the drain -- nothing is left queued behind it.
    assert parser_session.get_needs_input_queue(1) == []


def test_parsed_results_endpoint_since_offset_never_returns_an_already_delivered_result(monkeypatch):
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="first")
    first_poll = run(app_module.get_parsed_results_endpoint(1, since=0))
    assert first_poll["total"] == 1
    assert [r["source_session_id"] for r in first_poll["results"]] == [5]

    # A poll repeated with the SAME `since` (e.g. a retried/duplicate
    # request) must not re-deliver what was already returned.
    repeated_poll = run(app_module.get_parsed_results_endpoint(1, since=0))
    assert repeated_poll["total"] == 1
    assert len(repeated_poll["results"]) == 1

    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="second")
    next_poll = run(app_module.get_parsed_results_endpoint(1, since=first_poll["total"]))
    assert next_poll["total"] == 2
    assert [r["source_session_id"] for r in next_poll["results"]] == [6]


def test_parsed_results_endpoint_drains_multiple_source_sessions_in_fifo_order_none_dropped(monkeypatch):
    """Backend half of #193's "multiple non-focused results queued up this
    way each raise their own toast ... none dropped" acceptance criterion --
    the frontend can only raise a toast per result if the storage/draining
    layer actually hands back every one of them, in order."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2),
            _extraction_result_line("session-abc", _VALID_PAYLOAD_3),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo"))

    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="from card 5")
    parser_session.enqueue_needs_input_turn(1, card_id=6, phase="implementing", text="from card 6")
    parser_session.enqueue_needs_input_turn(1, card_id=7, phase="qa_grilling", text="from card 7")

    data = run(app_module.get_parsed_results_endpoint(1, since=0))

    assert data["total"] == 3
    assert [r["source_session_id"] for r in data["results"]] == [5, 6, 7]
    assert all(r["ok"] is True for r in data["results"])


def test_parsed_results_endpoint_keeps_different_projects_independent(monkeypatch):
    backend_1 = FakeStreamJsonBackend([_extraction_result_line("session-p1", _VALID_PAYLOAD_1)], eof_after=False)
    backend_2 = FakeStreamJsonBackend([_extraction_result_line("session-p2", _VALID_PAYLOAD_2)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend_1, backend_2])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(1, cwd="/repo-1"))
    run(parser_session.ensure_parser_session(2, cwd="/repo-2"))
    parser_session.enqueue_needs_input_turn(1, card_id=5, phase="grilling", text="project one's turn")
    parser_session.enqueue_needs_input_turn(2, card_id=9, phase="creating_prd", text="project two's turn")

    data_1 = run(app_module.get_parsed_results_endpoint(1, since=0))
    data_2 = run(app_module.get_parsed_results_endpoint(2, since=0))

    assert data_1["total"] == 1 and [r["source_session_id"] for r in data_1["results"]] == [5]
    assert data_2["total"] == 1 and [r["source_session_id"] for r in data_2["results"]] == [9]


# ---------------------------------------------------------------------------
# Legacy fallback on parse failure (issue #194 originally, `gh issue view
# 194` for context; retired by issue #230, `gh issue view 230`): when a
# queued item's primary parser-session extraction fails -- subprocess
# error/timeout, or a response failing schema validation -- that ONE item's
# result is tagged `"source": "fallback"`. This used to also attempt a
# regex-parser + Ollama-rescue extraction pipeline (`qa_parser.
# parse_grilling_response` / `ollama_rescue.rescue_grilling_response`) as a
# second try, which could still recover a question (`ok: True`); issue #230
# retired that chain -- the parser-session pipeline is now the ONLY
# extraction mechanism, so a primary failure is always `ok: False` now. A
# clean primary-path success is still tagged `"source": "parser_session"`,
# so a test (or the frontend) can positively distinguish "parsed normally"
# from "failed" rather than only ever checking for the fallback marker's
# absence.
#
# Nothing here persists any per-project or per-session state -- each item's
# own outcome is decided completely independently (retry-per-turn, never a
# permanent downgrade); see `test_retry_per_turn_...`/
# `test_repeated_failures_...` below.
# ---------------------------------------------------------------------------

_GRILLING_FORMAT_TEXT = (
    'Question 1: "Should this be Python or Node?"\n'
    "Options:\n"
    'Option 1: "Python"\n'
    'Option 2: "Node"\n'
    "Recommended: [1]\n"
)


def test_primary_success_is_tagged_with_the_parser_session_source(monkeypatch):
    """Baseline for the `"source"` marker itself: an ordinary successful
    primary-path parse is tagged `"source": "parser_session"`, distinct
    from a fallback-derived result."""
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", _VALID_PAYLOAD_1)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(2001, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(2001, card_id=5, phase="grilling", text="Python or Node?")

    results = run(_drain_all(2001))

    assert results[0]["ok"] is True
    assert results[0]["source"] == "parser_session"


def test_primary_subprocess_error_produces_a_tagged_fallback_failure():
    """Acceptance criterion (issue #230): a parser-session turn that errors
    (subprocess error, timeout) is tagged `"source": "fallback"` with
    `ok: False` -- issue #230 retired the regex/Ollama-rescue extraction that
    used to be attempted as a second try here, so there is nothing left that
    could recover a question; a primary failure is now always just a tagged
    failure. No parser session is registered for this project at all, so
    `stream_turn` raises `LookupError` -- the same subprocess-failure shape
    `_process_one_queued_item`'s `except Exception` branch treats
    generically."""
    parser_session.enqueue_needs_input_turn(2002, card_id=6, phase="grilling", text=_GRILLING_FORMAT_TEXT)

    results = run(_drain_all(2002))

    assert len(results) == 1
    result = results[0]
    assert result["ok"] is False
    assert result["source_session_id"] == 6
    assert result["source"] == "fallback"
    assert "error" in result
    assert parser_session.get_needs_input_queue(2002) == []


def test_primary_timeout_produces_a_tagged_fallback_failure_the_same_way(monkeypatch):
    """Acceptance criterion: a timeout is another "parser-session turn
    errors" shape -- simulated here by making `stream_turn` itself raise
    `TimeoutError`, exercised the same way `_process_one_queued_item`'s own
    `except Exception` branch would see a real one."""

    async def _raise_timeout(project_id, prompt):
        raise TimeoutError("parser session timed out")
        yield  # pragma: no cover -- makes this an async generator function

    monkeypatch.setattr(parser_session, "stream_turn", _raise_timeout)
    parser_session.enqueue_needs_input_turn(2003, card_id=7, phase="grilling", text=_GRILLING_FORMAT_TEXT)

    results = run(_drain_all(2003))

    assert results[0]["ok"] is False
    assert results[0]["source"] == "fallback"
    assert results[0]["source_session_id"] == 7


def test_primary_schema_invalid_response_produces_a_tagged_fallback_failure_the_same_way(monkeypatch):
    """Acceptance criterion: "A parser-session turn that returns a response
    failing schema validation" is tagged the same way as any other primary
    failure -- `ok: False`, `"source": "fallback"`."""
    invalid_payload = {"header": "", "questions": []}  # missing required "footer"
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", invalid_payload)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(2004, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(2004, card_id=8, phase="grilling", text=_GRILLING_FORMAT_TEXT)

    results = run(_drain_all(2004))

    assert results[0]["ok"] is False
    assert results[0]["source"] == "fallback"
    assert results[0]["source_session_id"] == 8
    assert "error" in results[0]


def test_retry_per_turn_next_item_after_a_failure_goes_through_parser_session_normally(monkeypatch):
    """Acceptance criteria: "The next turn processed for that project ...
    is attempted via the parser session normally, not automatically routed
    to the fallback" and "repeated failures ... do not accumulate into a
    permanent downgrade." Two items queued: the FIRST fails primary
    extraction and falls back; the SECOND must still be attempted via the
    same, still-live parser session first -- and succeed there -- proving
    nothing sticky was set by the first item's failure."""
    invalid_payload = {"header": "", "questions": []}  # missing required "footer" -- fails validation
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", invalid_payload),  # item 1: primary fails
            _extraction_result_line("session-abc", _VALID_PAYLOAD_2),  # item 2: primary succeeds normally
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(2007, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(2007, card_id=11, phase="grilling", text=_GRILLING_FORMAT_TEXT)
    parser_session.enqueue_needs_input_turn(2007, card_id=12, phase="grilling", text="an unrelated completed turn")

    results = run(_drain_all(2007))

    assert len(results) == 2
    assert results[0]["source"] == "fallback"  # item 1 fell back
    assert results[0]["source_session_id"] == 11
    assert results[1]["ok"] is True
    assert results[1]["source"] == "parser_session"  # item 2 went through the parser session normally
    assert results[1]["source_session_id"] == 12
    assert results[1]["questions"] == _VALID_PAYLOAD_2["questions"]


def test_retry_per_turn_applies_even_when_the_next_item_is_the_same_source_session(monkeypatch):
    """Same acceptance criterion, but explicitly for "including the same
    source session" -- both queued items share the same `card_id`."""
    invalid_payload = {"header": "", "questions": []}
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", invalid_payload),  # item 1: primary fails
            _extraction_result_line("session-abc", _VALID_PAYLOAD_1),  # item 2: primary succeeds normally
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(2008, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(2008, card_id=13, phase="grilling", text=_GRILLING_FORMAT_TEXT)
    parser_session.enqueue_needs_input_turn(2008, card_id=13, phase="grilling", text="a second turn from card 13")

    results = run(_drain_all(2008))

    assert results[0]["source_session_id"] == 13 and results[0]["source"] == "fallback"
    assert results[1]["source_session_id"] == 13 and results[1]["source"] == "parser_session"
    assert results[1]["ok"] is True


def test_repeated_failures_never_accumulate_into_a_permanent_downgrade(monkeypatch):
    """Three items queued, ALL of which fail primary extraction (schema-
    invalid every time) -- every single one must independently attempt the
    parser session and independently fall back; nothing accumulates that
    would skip the primary attempt for a later item just because an earlier
    one failed."""
    invalid_payload = {"header": "", "questions": []}
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", invalid_payload),
            _extraction_result_line("session-abc", invalid_payload),
            _extraction_result_line("session-abc", invalid_payload),
        ],
        eof_after=False,
    )
    factory, calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(2009, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(2009, card_id=14, phase="grilling", text=_GRILLING_FORMAT_TEXT)
    parser_session.enqueue_needs_input_turn(2009, card_id=15, phase="grilling", text=_GRILLING_FORMAT_TEXT)
    parser_session.enqueue_needs_input_turn(2009, card_id=16, phase="grilling", text=_GRILLING_FORMAT_TEXT)

    results = run(_drain_all(2009))

    assert len(results) == 3
    assert all(r["source"] == "fallback" for r in results)
    assert all(r["ok"] is False for r in results)  # no fallback recovers any of them (issue #230)
    assert [r["source_session_id"] for r in results] == [14, 15, 16]
    # Exactly one subprocess spawn throughout, and one turn written per
    # queued item -- no extra "downgrade" bookkeeping turns snuck in.
    assert len(calls) == 1
    assert len(_written_prompts(backend)) == 3


# ---------------------------------------------------------------------------
# Post-extraction validation + single corrective retry (issue #229, `gh
# issue view 229` for full context; parent PRD #227). After a schema-valid
# primary extraction, `_detect_extraction_mismatches` compares it against
# the turn's raw text (Recommended:/Recommended text: line counts, nearby
# bulleted-option lines vs. the returned `options` array) and, on a
# mismatch, `_process_one_queued_item` sends exactly one corrective retry
# to the same parser session before returning. Driven the same way as the
# #192/#194 tests above (a fake `process_factory`, no real subprocess).
# ---------------------------------------------------------------------------

_CLEAN_SINGLE_QUESTION_TEXT = (
    "❓ **Q1** - **Language**: Which language should we use?\n"
    "- Python\n"
    "- Node\n"
    "Recommended: Python\n"
)

_CLEAN_SINGLE_QUESTION_PAYLOAD = {
    "header": "",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which language should we use?",
            "kind": "single",
            "options": ["Python", "Node"],
            "recommended": [1],
            "recommended_text": None,
        }
    ],
}


def test_clean_first_response_never_triggers_a_retry(monkeypatch):
    """Acceptance criterion: "A first-response success ... must never
    trigger a retry at all." The raw text's single `Recommended:` line
    matches the one question that ended up with a populated `recommended`
    field, and its two bulleted option lines match a two-entry `options`
    array -- nothing here should look mismatched, so exactly one write
    (the primary extraction turn) must reach the subprocess."""
    backend = FakeStreamJsonBackend(
        [_extraction_result_line("session-abc", _CLEAN_SINGLE_QUESTION_PAYLOAD)], eof_after=False
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3001, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(3001, card_id=20, phase="grilling", text=_CLEAN_SINGLE_QUESTION_TEXT)

    results = run(_drain_all(3001))

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["source"] == "parser_session"
    assert "extraction_incomplete" not in results[0]["questions"][0]
    # Exactly one write reached the subprocess -- no retry turn at all.
    assert len(_written_prompts(backend)) == 1


_MISMATCH_SINGLE_QUESTION_TEXT = (
    "We've settled the schema already.\n\n"
    "❓ **Q1** - **Language**: Which language should we use?\n"
    "- Python\n"
    "- Node\n"
    "Recommended: Python\n"
)

# First attempt silently drops both the options and the recommendation --
# exactly the PRD #227 bug scenario (a `kind: "open"` question with nothing
# populated, despite the raw text clearly carrying both signals).
_MISMATCHED_FIRST_PAYLOAD = {
    "header": "We've settled the schema already.",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which language should we use?",
            "kind": "open",
            "options": None,
            "recommended": None,
            "recommended_text": None,
        }
    ],
}

_CORRECTED_SECOND_PAYLOAD = {
    "header": "We've settled the schema already.",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which language should we use?",
            "kind": "single",
            "options": ["Python", "Node"],
            "recommended": [1],
            "recommended_text": None,
        }
    ],
}


def test_mismatched_first_response_retries_once_and_succeeds(monkeypatch):
    """Acceptance criterion: on a detected mismatch, exactly one retry is
    sent naming the specific mismatch, and a corrected second response
    leaves the final result with no `extraction_incomplete` anywhere."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _MISMATCHED_FIRST_PAYLOAD),
            _extraction_result_line("session-abc", _CORRECTED_SECOND_PAYLOAD),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3002, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(3002, card_id=21, phase="grilling", text=_MISMATCH_SINGLE_QUESTION_TEXT)

    results = run(_drain_all(3002))

    written = _written_prompts(backend)
    assert len(written) == 2  # exactly one retry, not zero and not more
    retry_prompt = written[1]
    # The retry names the specific mismatch: which question, and what was
    # found wrong with it.
    assert "question index 0" in retry_prompt
    assert "q1" in retry_prompt
    assert "recommended" in retry_prompt.lower()
    assert "/rhubarb:parse-interview" in retry_prompt

    assert len(results) == 1
    assert results[0]["ok"] is True
    question = results[0]["questions"][0]
    assert question["kind"] == "single"
    assert question["options"] == ["Python", "Node"]
    assert question["recommended"] == [1]
    assert "extraction_incomplete" not in question  # retry succeeded -- no flag needed


_TWO_QUESTION_TEXT = (
    "❓ **Q1** - **Language**: Which language should we use?\n"
    "- Python\n"
    "- Node\n"
    "Recommended: Python\n\n"
    "❓ **Q2** - **Deploy target**: Where should this run?\n"
    "Recommended: On the existing droplet.\n"
)

# Q1 drops its options/recommendation (the mismatch); Q2 is extracted
# correctly from the very first attempt.
_TWO_QUESTION_FIRST_PAYLOAD = {
    "header": "",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which language should we use?",
            "kind": "open",
            "options": None,
            "recommended": None,
            "recommended_text": None,
        },
        {
            "id": "q2",
            "text": "Where should this run?",
            "kind": "open",
            "options": None,
            "recommended": None,
            "recommended_text": "On the existing droplet.",
        },
    ],
}

# The retry recovers Q1's options but STILL drops the recommendation itself
# (still fails the same check) -- Q2 comes back with a slightly different
# (but still populated) recommended_text, to prove the final result is
# "whatever the last attempt produced," not a merge with the first.
_TWO_QUESTION_RETRY_PAYLOAD = {
    "header": "",
    "footer": "",
    "questions": [
        {
            "id": "q1",
            "text": "Which language should we use?",
            "kind": "single",
            "options": ["Python", "Node"],
            "recommended": None,
            "recommended_text": None,
        },
        {
            "id": "q2",
            "text": "Where should this run?",
            "kind": "open",
            "options": None,
            "recommended": None,
            "recommended_text": "On the existing droplet, near the API.",
        },
    ],
}


def test_still_mismatched_retry_tags_only_the_affected_question_and_does_not_retry_again(monkeypatch):
    """Acceptance criteria: "If the retry still fails the same check, tag
    the specific affected question(s) ... never retry again" and "all other
    questions ... in the same turn unaffected." Only Q1 is broken in both
    attempts; Q2 is fine throughout and must never be tagged."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _TWO_QUESTION_FIRST_PAYLOAD),
            _extraction_result_line("session-abc", _TWO_QUESTION_RETRY_PAYLOAD),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3003, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(3003, card_id=22, phase="grilling", text=_TWO_QUESTION_TEXT)

    results = run(_drain_all(3003))

    # Exactly one retry -- not a second one even though it's still broken.
    assert len(_written_prompts(backend)) == 2

    assert len(results) == 1
    questions = results[0]["questions"]
    assert len(questions) == 2

    q1, q2 = questions
    assert q1["id"] == "q1"
    assert q1["extraction_incomplete"] is True
    # Q1's other fields are preserved from the retry (the last attempt),
    # not the first attempt or some merge of the two.
    assert q1["kind"] == "single"
    assert q1["options"] == ["Python", "Node"]
    assert q1["recommended"] is None
    assert q1["recommended_text"] is None

    assert q2["id"] == "q2"
    assert "extraction_incomplete" not in q2  # completely unaffected
    assert q2["recommended_text"] == "On the existing droplet, near the API."


def test_retry_turn_failure_falls_back_to_the_first_attempt_tagged_incomplete(monkeypatch):
    """The retry itself is never guaranteed to succeed -- if the retry turn
    produces no usable JSON at all, the function must still return
    something usable (issue #229: "never raise/error ... always return
    something usable"), built from the FIRST attempt's own data, tagged."""
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _MISMATCHED_FIRST_PAYLOAD),
            _result_line_with_usage("session-abc", text="Sorry, I can't produce that JSON."),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3004, cwd="/repo"))
    parser_session.enqueue_needs_input_turn(3004, card_id=23, phase="grilling", text=_MISMATCH_SINGLE_QUESTION_TEXT)

    results = run(_drain_all(3004))

    assert len(_written_prompts(backend)) == 2  # still exactly one retry attempt, no more
    assert len(results) == 1
    assert results[0]["ok"] is True
    question = results[0]["questions"][0]
    assert question["extraction_incomplete"] is True
    # Falls back to the first attempt's own (still-broken) fields -- there
    # was nothing better to prefer.
    assert question["kind"] == "open"
    assert question["options"] is None
    assert question["recommended_text"] is None


def test_detect_extraction_mismatches_flags_missing_recommendation():
    mismatches = parser_session._detect_extraction_mismatches(
        _MISMATCH_SINGLE_QUESTION_TEXT, _MISMATCHED_FIRST_PAYLOAD
    )
    assert len(mismatches) == 1
    assert mismatches[0]["index"] == 0
    assert mismatches[0]["id"] == "q1"


def test_detect_extraction_mismatches_returns_empty_for_a_clean_payload():
    mismatches = parser_session._detect_extraction_mismatches(
        _CLEAN_SINGLE_QUESTION_TEXT, _CLEAN_SINGLE_QUESTION_PAYLOAD
    )
    assert mismatches == []


def test_tag_extraction_incomplete_only_touches_named_indices():
    data = {
        "header": "",
        "footer": "",
        "questions": [{"id": "q1", "text": "a"}, {"id": "q2", "text": "b"}],
    }

    tagged = parser_session._tag_extraction_incomplete(data, [{"index": 1, "id": "q2", "reasons": ["x"]}])

    assert "extraction_incomplete" not in tagged["questions"][0]
    assert tagged["questions"][1]["extraction_incomplete"] is True
    # Original data is untouched (no in-place mutation).
    assert "extraction_incomplete" not in data["questions"][1]


def test_tag_extraction_incomplete_with_no_mismatches_returns_data_unchanged():
    data = {"header": "", "footer": "", "questions": [{"id": "q1", "text": "a"}]}

    assert parser_session._tag_extraction_incomplete(data, []) is data


# ---------------------------------------------------------------------------
# `extract_with_validation` (PRD #227 follow-up, gap 1, `gh issue view 227`):
# the one shared implementation of "call the skill, validate, retry once on
# mismatch, tag extraction_incomplete" that both `_process_one_queued_item`
# above (already exercised by every test above this point) and
# `session_runner._extract_questions_via_parser_session` (the live-turn path
# -- see `tests/test_sessions.py` for its own end-to-end coverage) now call.
# These tests exercise the function directly, at the module level, the same
# fake-`process_factory` way the rest of this file already does.
# ---------------------------------------------------------------------------


def test_extract_with_validation_clean_response_returns_data_with_no_retry(monkeypatch):
    backend = FakeStreamJsonBackend(
        [_extraction_result_line("session-abc", _CLEAN_SINGLE_QUESTION_PAYLOAD)], eof_after=False
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3010, cwd="/repo"))

    result = run(
        parser_session.extract_with_validation(3010, _CLEAN_SINGLE_QUESTION_TEXT, phase="grilling")
    )

    assert len(_written_prompts(backend)) == 1  # no retry -- nothing looked mismatched
    assert result["source"] == "parser_session"
    assert result["questions"][0]["options"] == ["Python", "Node"]
    assert "extraction_incomplete" not in result["questions"][0]
    # A response with real open questions never carries a completion verdict
    # -- the skill only attaches one for a grilling turn with zero questions
    # (issue #242, child of PRD #241).
    assert "completion" not in result


def test_extract_with_validation_passes_through_a_grilling_completion_verdict(monkeypatch):
    """Issue #242 (child of PRD #241): a `phase: grilling` response with
    `"questions": []` may carry an additional top-level `"completion"`
    field (`{"done": bool, "reason": str}`) -- `extract_with_validation`
    must pass it through verbatim to its caller (`session_runner._run_
    grilling_turn_stream_json`), not silently drop it the way the flat
    `{header, questions, footer, source}` return used to before this
    field existed."""
    payload = {
        "header": "Sounds like we've covered everything.",
        "questions": [],
        "footer": "",
        "completion": {"done": True, "reason": "Every open branch was resolved."},
    }
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", payload)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3015, cwd="/repo"))

    result = run(
        parser_session.extract_with_validation(3015, "Sounds like we've covered everything.", phase="grilling")
    )

    assert result["questions"] == []
    assert result["completion"] == {"done": True, "reason": "Every open branch was resolved."}


def test_extract_with_validation_retries_once_on_mismatch_and_returns_corrected_data(monkeypatch):
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _MISMATCHED_FIRST_PAYLOAD),
            _extraction_result_line("session-abc", _CORRECTED_SECOND_PAYLOAD),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3011, cwd="/repo"))

    result = run(
        parser_session.extract_with_validation(3011, _MISMATCH_SINGLE_QUESTION_TEXT, phase="implementing")
    )

    written = _written_prompts(backend)
    assert len(written) == 2  # exactly one corrective retry
    assert "/rhubarb:parse-interview" in written[1]
    assert "phase: implementing" in written[0]  # phase passed straight through, no branching in this module

    question = result["questions"][0]
    assert question["options"] == ["Python", "Node"]
    assert question["recommended"] == [1]
    assert "extraction_incomplete" not in question  # the retry recovered everything


def test_extract_with_validation_still_mismatched_after_retry_tags_extraction_incomplete(monkeypatch):
    backend = FakeStreamJsonBackend(
        [
            _extraction_result_line("session-abc", _MISMATCHED_FIRST_PAYLOAD),
            _result_line_with_usage("session-abc", text="Sorry, I can't produce that JSON."),
        ],
        eof_after=False,
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3012, cwd="/repo"))

    result = run(
        parser_session.extract_with_validation(3012, _MISMATCH_SINGLE_QUESTION_TEXT, phase="grilling")
    )

    assert len(_written_prompts(backend)) == 2
    assert result["questions"][0]["extraction_incomplete"] is True


def test_extract_with_validation_with_detect_mismatches_false_skips_the_retry_safety_net(monkeypatch):
    """Gap 2 of PRD #227's follow-up: the nested QA-grilling shape opts out
    of issue #229's mismatch-detection/retry entirely (`detect_mismatches=
    False`), a deliberate narrower-scope judgment call rather than
    generalizing `_detect_extraction_mismatches` to the nested `issues[].
    questions` structure. Scripted here with a QA-shaped response that,
    if it went through the flat-shape mismatch check at all, would look
    identical to `_MISMATCHED_FIRST_PAYLOAD` in spirit (a `Recommended
    text:` line present in the raw text with nothing populated in the
    JSON) -- proving no retry is attempted and the response is returned
    exactly as received, once `detect_mismatches=False` is passed."""
    from rhubarb.ollama_rescue import _is_valid_qa_shape

    qa_payload = {
        "prd": None,
        "issues": [
            {
                "number": 1,
                "title": "Some issue",
                "questions": [{"id": "issue1-q1", "text": "Does it work?", "recommended_text": None}],
            }
        ],
    }
    raw_text = 'Issue 1: "Some issue"\nQuestion 1: "Does it work?"\nRecommended text: "Yes."\n'
    backend = FakeStreamJsonBackend([_extraction_result_line("session-abc", qa_payload)], eof_after=False)
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3013, cwd="/repo"))

    result = run(
        parser_session.extract_with_validation(
            3013, raw_text, phase="qa_grilling_issues", validator=_is_valid_qa_shape, detect_mismatches=False
        )
    )

    assert len(_written_prompts(backend)) == 1  # no retry attempted at all
    assert result == {**qa_payload, "source": "parser_session"}


def test_extract_with_validation_returns_none_on_schema_invalid_response(monkeypatch):
    backend = FakeStreamJsonBackend(
        [_result_line_with_usage("session-abc", text=json.dumps({"not": "the right shape"}))], eof_after=False
    )
    factory, _calls = _sequenced_process_factory([backend])
    _patch_stream_json_engine_process_factory(monkeypatch, factory)
    run(parser_session.ensure_parser_session(3014, cwd="/repo"))

    result = run(parser_session.extract_with_validation(3014, "some text", phase="grilling"))

    assert result is None


def test_extract_with_validation_returns_none_when_no_live_session():
    result = run(parser_session.extract_with_validation(999999, "some text", phase="grilling"))

    assert result is None
