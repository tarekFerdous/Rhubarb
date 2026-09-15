import pytest
from fastapi.testclient import TestClient

from rhubarb import afk_loop, db, live_stream, ollama_rescue, parser_session, session_runner, stream_json_engine
from rhubarb.web import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_path / "rhubarb.db")
    # Also isolate the legacy #93 migration source path -- otherwise
    # `get_connection()` (called with no explicit path all over the app)
    # would check the real `~/.baton/baton.db` on whatever machine runs the
    # tests and, if one happens to exist there, migrate its real data into
    # this test's fresh tmp db.
    monkeypatch.setattr(db, "OLD_DB_PATH", tmp_path / "not-a-real-legacy-db" / "baton.db")
    monkeypatch.setattr(app_module, "_active_project_id", None)
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def _isolated_live_stream(monkeypatch):
    """Session row ids restart at 1 in every test's fresh tmp db, but
    `live_stream`'s buffers are process-global -- reset them per test so
    unrelated tests never share a card_id's event history."""
    monkeypatch.setattr(live_stream, "_buffers", {})
    monkeypatch.setattr(live_stream, "_subscribers", {})
    monkeypatch.setattr(live_stream, "_last_usage", None)


@pytest.fixture(autouse=True)
def _isolated_implement_queues(monkeypatch):
    """Same story as `_isolated_live_stream` above, but for the serial-mode
    per-project implement queue -- also process-lifetime-only state, keyed
    by project ids that restart at 1 in every test's fresh tmp db."""
    monkeypatch.setattr(session_runner, "_implement_queues", {})


@pytest.fixture(autouse=True)
def _isolated_pty_engines(monkeypatch):
    """Same story again, but for the resident PtyEngine-per-card_id registry
    (issue #87) -- card_id values restart at 1 in every test's fresh tmp db
    too, so a leftover (fake) engine from one test must never be handed to
    an unrelated test's identically-numbered card_id."""
    monkeypatch.setattr(session_runner, "_pty_engines", {})


@pytest.fixture(autouse=True)
def _isolated_standby_engines(monkeypatch):
    """Same story again, but for the pre-warmed standby-PtyEngine-per-project
    registry (issue #136) -- project ids also restart at 1 in every test's
    fresh tmp db, so a leftover (fake) standby from one test must never be
    handed to an unrelated test's identically-numbered project."""
    monkeypatch.setattr(session_runner, "_standby_engines", {})


@pytest.fixture(autouse=True)
def _isolated_stream_json_engines(monkeypatch):
    """Same story again, but for the resident StreamJsonEngine-per-card_id
    registry a grilling card now uses (issue #184) -- mirrors
    `_isolated_pty_engines` above exactly, for the same reason."""
    monkeypatch.setattr(session_runner, "_stream_json_engines", {})


@pytest.fixture(autouse=True)
def _isolated_standby_stream_json_engines(monkeypatch):
    """Same story again, but for the pre-warmed standby-StreamJsonEngine-
    per-project registry a brand-new (grilling-phase) session now claims
    from (issue #184) -- mirrors `_isolated_standby_engines` above exactly,
    for the same reason."""
    monkeypatch.setattr(session_runner, "_standby_stream_json_engines", {})


@pytest.fixture(autouse=True)
def _isolated_parser_sessions(monkeypatch):
    """Same story again, but for the persistent-per-project parser-session
    registry (issue #189) -- project ids also restart at 1 in every test's
    fresh tmp db, so a leftover (fake) parser session from one test must
    never be handed to an unrelated test's identically-numbered project."""
    monkeypatch.setattr(parser_session, "_parser_sessions", {})
    monkeypatch.setattr(parser_session, "_locks", {})
    # Also reset issue #190's per-project tracked context-usage fraction --
    # same "process-lifetime, project-id-keyed" state as the registry above,
    # so a leftover reading from one test can never be seen by an unrelated
    # test's identically-numbered project.
    monkeypatch.setattr(parser_session, "_context_pct", {})


@pytest.fixture(autouse=True)
def _isolated_needs_input_queues(monkeypatch):
    """Same story again, but for the per-project needs-input FIFO queue
    (issue #191) -- project ids also restart at 1 in every test's fresh tmp
    db, so a leftover queued item from one test must never be handed to an
    unrelated test's identically-numbered project."""
    monkeypatch.setattr(parser_session, "_needs_input_queues", {})
    # Also reset issue #192's per-project accumulated tagged-parse-result
    # list -- same reasoning as the queue itself above.
    monkeypatch.setattr(parser_session, "_parsed_results", {})


@pytest.fixture(autouse=True)
def _isolated_afk_loop(monkeypatch):
    """Same story again, but for the AFK loop's per-project idle clock and
    its per-project undismissed-notification queue."""
    monkeypatch.setattr(afk_loop, "_last_activity", {})
    monkeypatch.setattr(afk_loop, "_notifications", {})


@pytest.fixture(autouse=True)
def _isolated_error_notifications(monkeypatch):
    """Same story again, but for session_runner's per-project undismissed
    background-session-error notification queue."""
    monkeypatch.setattr(session_runner, "_error_notifications", {})


@pytest.fixture(autouse=True)
def _parser_session_engine_is_a_fake_by_default(monkeypatch):
    """`rhubarb.web.app.open_project` fire-and-forgets a per-project
    parser-session warm on every project open
    (`parser_session.ensure_parser_session`, issue #189). Unlike
    `session_runner.py`'s own `PtyEngine`/`StreamJsonEngine` names (which
    plenty of existing tests already monkeypatch via `_mock_engine`),
    `parser_session.py` imports the real `StreamJsonEngine` directly under
    its own name -- so left unpatched, EVERY existing test that opens a
    project (the overwhelming majority of tests/test_sessions.py and
    tests/test_app.py, via `_open_project`) would attempt to spawn a REAL
    `claude` subprocess in the background on every single run: slow, flaky,
    and a genuinely unwanted side effect on whatever machine runs the suite.

    Defaults every test to a fake `StreamJsonEngine` that never spawns a
    real process; `tests/test_parser_session.py` (the tests that actually
    want to exercise this path) overrides this stub with its own fake/
    injected-backend engine, using the same `monkeypatch` fixture instance
    this autouse fixture already used, so it simply wins for the rest of
    that one test -- same pattern as `_ollama_unreachable_by_default` below
    and test_sessions.py's own `_classify_needs_input_is_a_no_op_by_default`."""

    class _NoopParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.cwd = cwd
            self.model = model
            self.effort = effort
            self.session_id = resume_session_id
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            raise AssertionError(
                "stream_turn must not be called on the default no-op parser-session "
                "fake -- this test needs its own fake/injected StreamJsonEngine if it "
                "actually drives a parser-session turn"
            )
            yield  # pragma: no cover -- makes this an async generator function

    monkeypatch.setattr(parser_session, "StreamJsonEngine", _NoopParserEngine)


@pytest.fixture(autouse=True)
def _no_real_stream_json_subprocess_spawns_by_default(monkeypatch):
    """`rhubarb.web.app.open_project` ALSO fire-and-forgets a pre-warmed
    STANDBY `StreamJsonEngine` on every open (`ensure_standby_stream_json_
    engine`, pre-existing since issue #184, unrelated to issue #189's own
    parser-session addition above) -- and, unlike `PtyEngine`/
    `StreamJsonEngine`-level mocking via `_mock_engine`, a handful of
    existing tests never mock that class at all (e.g.
    `test_pty_tab_count_lists_resident_and_standby_engines_with_model_
    effort`, and the argv-capturing tests that intentionally leave the real
    class in place -- `_capture_real_stream_json_spawns`). On a machine
    where `claude` is genuinely installed on PATH (true for a real dev
    machine, not just a minimal CI image), those tests were relying on an
    unguarded real OS subprocess spawn LOSING a race against their own
    synchronous assertions to pass -- not a guarantee, and adding any other
    concurrent background task (such as issue #189's own parser-session
    warm, scheduled from the very same `open_project` call) can easily tip
    that race the other way.

    Defaults every test's real spawn seam (`stream_json_engine.
    _spawn_subprocess`, the one place every non-test `StreamJsonEngine`
    resolves its `process_factory` through when none is injected) to an
    inert fake that never touches the OS. `_capture_real_stream_json_
    spawns` (`tests/test_sessions.py`) already explicitly overrides this
    exact same seam via its own `monkeypatch.setattr` call, which simply
    wins for the rest of that one test -- same override pattern as
    `_ollama_unreachable_by_default` below."""

    class _InertStreamJsonBackend:
        def write_line(self, line):
            pass

        def read_line(self):
            raise EOFError

        def is_alive(self):
            return False

        def terminate(self, force=False):
            pass

    def _inert_spawn_subprocess(argv, *, cwd, env):
        return _InertStreamJsonBackend()

    monkeypatch.setattr(stream_json_engine, "_spawn_subprocess", _inert_spawn_subprocess)


@pytest.fixture(autouse=True)
def _ollama_unreachable_by_default(monkeypatch):
    """Every Ollama-backed call (`ollama_rescue._call_ollama`) that doesn't
    receive its own `http_post` override falls back to
    `_default_http_post`, which makes a real HTTP call to a real local
    Ollama install. Every existing test that cares about that call's outcome
    already passes its own fake `http_post` (see tests/test_ollama_rescue.py
    and the classify_needs_input tests in tests/test_sessions.py) -- this
    fixture only ever affects a caller that DIDN'T, so it can never change
    what any of those already-passing tests exercise.

    Issue #179 wires `classify_needs_input` into implementing's own turn
    handling unconditionally (once per completed turn, not gated behind a
    rare parse-failure trigger the way the pre-existing rescue calls are),
    so an untouched pre-existing implement test now reaches this fallback on
    every run. Left unpatched, on any machine that happens to have a real
    Ollama installed and running (as opposed to one where the call fails
    fast with a connection error), that test's outcome would depend on a
    live, non-deterministic local model's actual judgment -- exactly the
    flakiness a unit test must never have. Patching `_default_http_post`
    itself (rather than, say, `classify_needs_input`) makes every such
    fallback call fail the same fast, deterministic way a clean CI machine
    without Ollama installed already gets for free, regardless of what
    happens to be running on whichever machine actually runs the suite."""

    def _unreachable(url, body, *, timeout):
        raise ConnectionError("Ollama is not reachable in tests unless a test supplies its own http_post")

    monkeypatch.setattr(ollama_rescue, "_default_http_post", _unreachable)
