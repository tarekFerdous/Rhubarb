import json

import pytest
from starlette.websockets import WebSocketDisconnect

from rhubarb import live_stream, session_runner
from rhubarb.pty_engine import PtyEngine
from rhubarb.web import app as app_module
from tests.test_pty_engine import FakePtyBackend


async def _noop_job(*args, **kwargs):
    return None


def test_start_session_returns_immediately_with_only_card_id(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_active_project_id", 1)
    monkeypatch.setattr(app_module, "_active_project_cwd", lambda: None)
    monkeypatch.setattr(app_module.db, "claim_available_session", lambda conn, project_id: None)
    monkeypatch.setattr(app_module.session_runner, "start_session_job", _noop_job)

    resp = client.post("/api/session/start", json={"prompt": "a feature"})
    data = resp.json()

    assert set(data.keys()) == {"card_id"}


def test_continue_session_returns_immediately_with_only_card_id(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_active_project_id", 1)
    monkeypatch.setattr(app_module, "_active_project_cwd", lambda: None)
    monkeypatch.setattr(app_module.db, "claim_available_session", lambda conn, project_id: None)
    monkeypatch.setattr(app_module.session_runner, "start_session_job", _noop_job)
    card_id = client.post("/api/session/start", json={"prompt": "a feature"}).json()["card_id"]

    monkeypatch.setattr(app_module.session_runner, "continue_session_job", _noop_job)
    resp = client.post("/api/session/continue", json={"card_id": card_id, "reply": "ok"})
    data = resp.json()

    assert data == {"card_id": card_id}


def test_stream_session_replays_history_then_closes_after_done(client):
    card_id = 987654321
    live_stream.publish(card_id, {"type": "phase", "phase": "grilling"})
    live_stream.publish(card_id, {"type": "text", "text": "hi"})
    live_stream.publish(card_id, {"type": "done"})

    with client.stream("GET", f"/api/sessions/{card_id}/stream") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    events = [json.loads(line[len("data:"):].strip()) for line in lines]
    assert events == [
        {"type": "phase", "phase": "grilling"},
        {"type": "text", "text": "hi"},
        {"type": "done"},
    ]


def test_stream_session_reconnect_replays_full_history_again(client):
    card_id = 987654322
    live_stream.publish(card_id, {"type": "action", "summary": "Read foo.py"})
    live_stream.publish(card_id, {"type": "done"})

    def _connect_and_collect():
        with client.stream("GET", f"/api/sessions/{card_id}/stream") as response:
            lines = [line for line in response.iter_lines() if line.startswith("data:")]
        return [json.loads(line[len("data:"):].strip()) for line in lines]

    expected = [{"type": "action", "summary": "Read foo.py"}, {"type": "done"}]
    assert _connect_and_collect() == expected
    # Reconnecting later (e.g. a page reload) replays the same history again.
    assert _connect_and_collect() == expected


def test_stream_session_carries_terminal_output_events_verbatim(client):
    """Issue #88: raw/ANSI PTY output rides the same per-card SSE channel as
    the existing turn events, as a `{"type": "terminal_output", "data":
    ...}` event -- and must survive the JSON round trip byte-for-byte
    (escape sequences, control characters, all of it) since a terminal
    emulator on the frontend depends on getting the exact original bytes."""
    card_id = 987654324
    raw_chunk = "\x1b[1;32mRunning tests...\x1b[0m\r\n"
    live_stream.publish(card_id, {"type": "terminal_output", "data": raw_chunk})
    live_stream.publish(card_id, {"type": "done"})

    with client.stream("GET", f"/api/sessions/{card_id}/stream") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    events = [json.loads(line[len("data:"):].strip()) for line in lines]
    assert events == [
        {"type": "terminal_output", "data": raw_chunk},
        {"type": "done"},
    ]


def test_pty_tab_count_endpoint_reflects_resident_engines(client, monkeypatch):
    """Backs the web UI's tab-count indicator (issue #88) -- the endpoint
    just surfaces `session_runner.open_pty_tab_count()`, plus (issue #140)
    the additive `"engines"` listing from `session_runner.list_live_engines`."""

    class _FakeEngine:
        model = "claude-sonnet-5"
        effort = "auto"

    assert client.get("/api/pty-tabs/count").json() == {"count": 0, "engines": []}

    monkeypatch.setitem(session_runner._pty_engines, 1, _FakeEngine())
    data = client.get("/api/pty-tabs/count").json()
    assert data["count"] == 1
    assert data["engines"] == [{"card_id": 1, "model": "claude-sonnet-5", "effort": "auto"}]

    monkeypatch.setitem(session_runner._pty_engines, 2, _FakeEngine())
    data = client.get("/api/pty-tabs/count").json()
    assert data["count"] == 2
    assert len(data["engines"]) == 2


def test_usage_endpoint_returns_unknown_before_any_session_has_run(client):
    resp = client.get("/api/usage")
    assert resp.json() == {"five_hour_pct": None, "seven_day_pct": None}


def test_usage_endpoint_reflects_the_latest_published_usage_event(client):
    live_stream.publish(111, {"type": "usage", "five_hour_pct": 12.5, "seven_day_pct": 3.1})

    resp = client.get("/api/usage")

    assert resp.json() == {"five_hour_pct": 12.5, "seven_day_pct": 3.1}
    assert "$" not in resp.text


def test_retry_session_returns_immediately_with_only_card_id(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_active_project_id", 1)
    monkeypatch.setattr(app_module, "_active_project_cwd", lambda: None)
    monkeypatch.setattr(app_module.db, "claim_available_session", lambda conn, project_id: None)
    monkeypatch.setattr(app_module.session_runner, "start_session_job", _noop_job)
    card_id = client.post("/api/session/start", json={"prompt": "a feature"}).json()["card_id"]

    monkeypatch.setattr(app_module.session_runner, "retry_session_job", _noop_job)
    resp = client.post(f"/api/sessions/{card_id}/retry")

    assert resp.json() == {"card_id": card_id}


def test_stream_session_replay_continues_past_an_earlier_done_from_a_retry(client):
    """A retry can append a fresh run onto a session that already reached
    `done` once. Reconnecting must replay everything, not stop at the first
    (now stale) `done` partway through history."""
    card_id = 987654323
    live_stream.publish(card_id, {"type": "turn", "phase": "creating_prd", "error": "not logged in"})
    live_stream.publish(card_id, {"type": "done"})
    # Retry runs and appends a fresh, successful attempt onto the same buffer.
    live_stream.publish(card_id, {"type": "phase", "phase": "creating_prd"})
    live_stream.publish(card_id, {"type": "turn", "phase": "details", "details": {"prd": None, "issues": []}})
    live_stream.publish(card_id, {"type": "done"})

    with client.stream("GET", f"/api/sessions/{card_id}/stream") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    events = [json.loads(line[len("data:"):].strip()) for line in lines]
    assert events == live_stream._buffers[card_id]
    assert events[-1] == {"type": "done"}


# ---------------------------------------------------------------------------
# Raw passthrough WebSocket channel (issue #165, child of PRD #162) -- purely
# additive alongside the SSE stream above; none of the tests in this section
# touch `/api/sessions/{card_id}/stream` or its behavior.
# ---------------------------------------------------------------------------


def test_pty_passthrough_websocket_forwards_bytes_to_the_correct_cards_engine(client):
    """Bytes sent over `/ws/sessions/{card_id}/pty` must reach that card's
    resident PtyEngine's write path -- exercised here through a real
    `PtyEngine` wired to a `FakePtyBackend` (the same fake used throughout
    test_pty_engine.py), so this test goes through `PtyEngine.write` itself,
    not a stand-in double."""
    backend = FakePtyBackend([], eof_after=False)
    engine = PtyEngine(pty_factory=lambda argv, *, cwd, env: backend)
    engine.start()
    session_runner.register_engine(555, engine)

    with client.websocket_connect("/ws/sessions/555/pty") as ws:
        ws.send_text("echo hello")

    assert backend.writes == ["echo hello"]


def test_pty_passthrough_websocket_forwards_multiple_frames_in_order(client):
    backend = FakePtyBackend([], eof_after=False)
    engine = PtyEngine(pty_factory=lambda argv, *, cwd, env: backend)
    engine.start()
    session_runner.register_engine(556, engine)

    with client.websocket_connect("/ws/sessions/556/pty") as ws:
        ws.send_text("a")
        ws.send_text("b")
        ws.send_text("c")

    assert backend.writes == ["a", "b", "c"]


def test_pty_passthrough_websocket_never_forwards_to_a_different_cards_engine(client):
    """A frame sent to one card's socket must never reach another card's
    engine -- the endpoint is scoped by the `card_id` path parameter."""
    backend_1 = FakePtyBackend([], eof_after=False)
    backend_2 = FakePtyBackend([], eof_after=False)
    engine_1 = PtyEngine(pty_factory=lambda argv, *, cwd, env: backend_1)
    engine_2 = PtyEngine(pty_factory=lambda argv, *, cwd, env: backend_2)
    engine_1.start()
    engine_2.start()
    session_runner.register_engine(1, engine_1)
    session_runner.register_engine(2, engine_2)

    with client.websocket_connect("/ws/sessions/1/pty") as ws:
        ws.send_text("only for card 1")

    assert backend_1.writes == ["only for card 1"]
    assert backend_2.writes == []


def test_pty_passthrough_websocket_closes_immediately_for_a_card_with_no_live_engine(client):
    """No resident engine for this card_id -- nothing to forward to, so the
    connection is refused (closed with policy-violation code 1008) rather
    than accepted and silently dropping input."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/sessions/999999/pty"):
            pass


def test_pty_passthrough_websocket_does_not_affect_the_sse_stream_for_the_same_card(client):
    """The passthrough channel is purely additive -- using it alongside the
    existing SSE stream for the same card_id must not change what that
    stream replays."""
    card_id = 777
    backend = FakePtyBackend([], eof_after=False)
    engine = PtyEngine(pty_factory=lambda argv, *, cwd, env: backend)
    engine.start()
    session_runner.register_engine(card_id, engine)

    live_stream.publish(card_id, {"type": "phase", "phase": "grilling"})
    live_stream.publish(card_id, {"type": "done"})

    with client.websocket_connect(f"/ws/sessions/{card_id}/pty") as ws:
        ws.send_text("keystrokes")

    with client.stream("GET", f"/api/sessions/{card_id}/stream") as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    events = [json.loads(line[len("data:"):].strip()) for line in lines]
    assert events == [
        {"type": "phase", "phase": "grilling"},
        {"type": "done"},
    ]
    assert backend.writes == ["keystrokes"]
