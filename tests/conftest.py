import pytest
from fastapi.testclient import TestClient

from baton import afk_loop, db, live_stream, session_runner
from baton.web import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_path / "baton.db")
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
