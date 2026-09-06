import json

from rhubarb import live_stream, session_runner
from rhubarb.web import app as app_module


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
    just surfaces `session_runner.open_pty_tab_count()`."""
    assert client.get("/api/pty-tabs/count").json() == {"count": 0}

    monkeypatch.setitem(session_runner._pty_engines, 1, object())
    assert client.get("/api/pty-tabs/count").json() == {"count": 1}

    monkeypatch.setitem(session_runner._pty_engines, 2, object())
    assert client.get("/api/pty-tabs/count").json() == {"count": 2}


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
