import json
import subprocess

from rhubarb import afk_loop, db, session_runner
from rhubarb.web import app as app_module


def _init_repo(path, remote_url):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=path, check=True)


def test_app_state_defaults_afk_hours_to_six(client):
    state = client.get("/api/app-state").json()
    assert state["afk_hours"] == 6


def test_set_afk_hours_persists_and_reflects_in_app_state(client):
    resp = client.post("/api/settings/afk-hours", json={"afk_hours": 12})
    assert resp.json() == {"afk_hours": 12}

    state = client.get("/api/app-state").json()
    assert state["afk_hours"] == 12


def test_app_state_defaults_parallel_implementation_to_false(client):
    state = client.get("/api/app-state").json()
    assert state["parallel_implementation"] is False


def test_set_parallel_implementation_persists_and_reflects_in_app_state(client):
    resp = client.post("/api/settings/parallel-implementation", json={"parallel_implementation": False})
    assert resp.json() == {"parallel_implementation": False}

    state = client.get("/api/app-state").json()
    assert state["parallel_implementation"] is False

    resp = client.post("/api/settings/parallel-implementation", json={"parallel_implementation": True})
    assert resp.json() == {"parallel_implementation": True}
    assert client.get("/api/app-state").json()["parallel_implementation"] is True


def test_app_state_defaults_terminal_view_hidden_to_false(client):
    state = client.get("/api/app-state").json()
    assert state["terminal_view_hidden"] is False


def test_set_terminal_view_hidden_persists_and_reflects_in_app_state(client):
    resp = client.post("/api/settings/terminal-view-hidden", json={"terminal_view_hidden": True})
    assert resp.json() == {"terminal_view_hidden": True}

    state = client.get("/api/app-state").json()
    assert state["terminal_view_hidden"] is True

    resp = client.post("/api/settings/terminal-view-hidden", json={"terminal_view_hidden": False})
    assert resp.json() == {"terminal_view_hidden": False}
    assert client.get("/api/app-state").json()["terminal_view_hidden"] is False


def test_app_state_defaults_model_to_claude_sonnet(client):
    state = client.get("/api/app-state").json()
    assert state["model"] == "claude-sonnet-4-6"


def test_set_model_persists_and_reflects_in_app_state(client):
    resp = client.post("/api/settings/model", json={"model": "claude-opus-4-8"})
    assert resp.json() == {"model": "claude-opus-4-8"}

    state = client.get("/api/app-state").json()
    assert state["model"] == "claude-opus-4-8"


def test_app_state_defaults_effort_to_auto(client):
    state = client.get("/api/app-state").json()
    assert state["effort"] == "auto"


def test_set_effort_persists_and_reflects_in_app_state(client):
    resp = client.post("/api/settings/effort", json={"effort": "high"})
    assert resp.json() == {"effort": "high"}

    state = client.get("/api/app-state").json()
    assert state["effort"] == "high"


def _open_project(client, tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    _init_repo(root / "repo", f"https://github.com/x/{name}.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")
    return project_id


def test_session_start_accepts_effort_and_seeds_the_session_row(client, tmp_path, monkeypatch):
    _open_project(client, tmp_path, "proj")

    async def _noop(card_id, prompt, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_session_job", _noop)

    resp = client.post("/api/session/start", json={"prompt": "a feature", "effort": "low"})
    card_id = resp.json()["card_id"]

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    assert row["effort"] == "low"


def test_patch_session_effort_updates_the_row_immediately(client, tmp_path, monkeypatch):
    _open_project(client, tmp_path, "proj")

    async def _noop(card_id, prompt, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_session_job", _noop)

    resp = client.post("/api/session/start", json={"prompt": "a feature", "effort": "auto"})
    card_id = resp.json()["card_id"]

    patch_resp = client.post(f"/api/sessions/{card_id}/effort", json={"effort": "medium"})
    assert patch_resp.json() == {"effort": "medium"}

    conn = db.get_connection()
    row = db.get_session(conn, card_id)
    assert row["effort"] == "medium"


def test_patch_session_effort_returns_error_for_unknown_session(client):
    resp = client.post("/api/sessions/999999/effort", json={"effort": "high"})
    assert resp.json() == {"error": "Session not found"}


def test_root_dir_change_without_confirmation_leaves_projects_untouched(client, tmp_path):
    root_a = tmp_path / "root_a"
    root_a.mkdir()
    _init_repo(root_a / "repo1", "https://github.com/x/repo1.git")

    resp = client.post("/api/settings/root-dir", json={"root_dir": str(root_a)})
    assert resp.json()["projects"], "expected repo1 to be discovered"

    root_b = tmp_path / "root_b"
    root_b.mkdir()

    resp = client.post("/api/settings/root-dir", json={"root_dir": str(root_b)})
    assert resp.json() == {"needs_confirmation": True}

    projects = client.get("/api/app-state").json()["projects"]
    assert len(projects) == 1
    assert projects[0]["name"] == "repo1"


def test_root_dir_change_with_confirmation_clears_projects(client, tmp_path):
    root_a = tmp_path / "root_a"
    root_a.mkdir()
    _init_repo(root_a / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root_a)})

    root_b = tmp_path / "root_b"
    root_b.mkdir()
    _init_repo(root_b / "repo2", "https://github.com/x/repo2.git")

    resp = client.post("/api/settings/root-dir", json={"root_dir": str(root_b), "confirm": True})
    names = {p["name"] for p in resp.json()["projects"]}
    assert names == {"repo2"}


def test_project_open_close_reopen_round_trip(client, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})

    project_id = client.get("/api/app-state").json()["projects"][0]["id"]

    opened = client.post(f"/api/projects/{project_id}/open").json()
    assert opened["project"]["name"] == "repo1"
    assert opened["session_state"] == {}

    state = client.get("/api/app-state").json()
    assert state["active_project"]["id"] == project_id

    client.post(
        f"/api/projects/{project_id}/close",
        json={"session_state": {"session_id": "abc", "console_text": "hi"}},
    )

    state = client.get("/api/app-state").json()
    assert state["active_project"] is None

    reopened = client.post(f"/api/projects/{project_id}/open").json()
    assert reopened["session_state"] == {"session_id": "abc", "console_text": "hi"}


def test_prds_endpoint_returns_sorted_blockage_annotated_list(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    prds = [
        {"number": 34, "title": "AFK feature", "body": "", "labels": []},
        {"number": 30, "title": "Grilling fix", "body": "", "labels": []},
    ]
    all_open_issues = [
        {"number": 31, "title": "child", "body": "## Parent\n\n#30\n\n## Blocked by\n\nNone - can start immediately.\n"},
        {
            "number": 35,
            "title": "child",
            "body": "## Parent\n\n#34\n\n## Blocked by\n\n- #99\n",
        },
        {"number": 99, "title": "blocker", "body": "## Blocked by\n\nNone - can start immediately.\n"},
    ]
    monkeypatch.setattr(app_module, "_fetch_ready_prds", lambda cwd: prds)
    monkeypatch.setattr(app_module, "_fetch_all_open_issues", lambda cwd: all_open_issues)

    resp = client.get(f"/api/projects/{project_id}/prds")
    assert resp.json() == {
        "prds": [
            {"number": 30, "title": "Grilling fix", "blocked": False},
            {"number": 34, "title": "AFK feature", "blocked": True},
        ]
    }


def test_prds_endpoint_returns_empty_for_non_active_project(client):
    resp = client.get("/api/projects/999/prds")
    assert resp.json() == {"prds": []}


def test_fetch_ready_prds_filters_by_prd_label_not_ready_for_agent(monkeypatch):
    captured = {}

    class _FakeResult:
        returncode = 0
        stdout = "[]"

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeResult()

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)

    app_module._fetch_ready_prds("/some/cwd")

    assert "--label" in captured["cmd"]
    label_index = captured["cmd"].index("--label")
    assert captured["cmd"][label_index + 1] == "prd"
    assert "ready-for-agent" not in captured["cmd"]


def test_start_implement_endpoint_rejects_duplicate_session_for_same_prd(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")
    client.post("/api/settings/parallel-implementation", json={"parallel_implementation": True})

    async def _noop_job(card_id, prd_number, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_implement_job", _noop_job)

    first = client.post("/api/session/start-implement", json={"number": 5, "title": "My PRD"})
    assert "card_id" in first.json()

    # A second click for the same PRD number while the first is still live
    # must be rejected without creating a second session row.
    second = client.post("/api/session/start-implement", json={"number": 5, "title": "My PRD"})
    assert second.json() == {"error": "Already implementing"}

    conn = db.get_connection()
    sessions = db.list_sessions_for_project(conn, project_id)
    implement_sessions = [s for s in sessions if s["session_type"] == "implement"]
    assert len(implement_sessions) == 1

    # A different PRD number is unaffected by the first one's in-flight session.
    third = client.post("/api/session/start-implement", json={"number": 6, "title": "Other PRD"})
    assert "card_id" in third.json()


def test_open_project_records_afk_activity(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]

    calls = []
    monkeypatch.setattr(app_module.afk_loop, "record_activity", lambda pid: calls.append(pid))

    client.post(f"/api/projects/{project_id}/open")

    assert calls == [project_id]


def test_start_implement_records_afk_activity(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    async def _noop_job(card_id, prd_number, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_implement_job", _noop_job)

    calls = []
    monkeypatch.setattr(app_module.afk_loop, "record_activity", lambda pid: calls.append(pid))

    client.post("/api/session/start-implement", json={"number": 5, "title": "My PRD"})

    assert calls == [project_id]


def test_afk_notifications_endpoint_reflects_the_server_side_queue(client, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    assert client.get(f"/api/projects/{project_id}/afk-notifications").json() == {"notifications": []}

    afk_loop.add_notification(project_id, 5, "Top PRD")
    afk_loop.add_notification(project_id, 6, "Second PRD")

    resp = client.get(f"/api/projects/{project_id}/afk-notifications")
    assert resp.json() == {
        "notifications": [
            {"number": 5, "title": "Top PRD"},
            {"number": 6, "title": "Second PRD"},
        ]
    }


def test_dismiss_afk_notifications_clears_the_queue(client, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    afk_loop.add_notification(project_id, 5, "Top PRD")

    resp = client.post(f"/api/projects/{project_id}/afk-notifications/dismiss")
    assert resp.json() == {"dismissed": True}

    assert client.get(f"/api/projects/{project_id}/afk-notifications").json() == {"notifications": []}


def test_dismiss_afk_notifications_does_not_touch_session_rows(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    async def _noop_job(card_id, prd_number, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_implement_job", _noop_job)

    started = client.post("/api/session/start-implement", json={"number": 5, "title": "My PRD"})
    card_id = started.json()["card_id"]

    conn = db.get_connection()
    before = dict(db.get_session(conn, card_id))

    afk_loop.add_notification(project_id, 5, "My PRD")
    client.post(f"/api/projects/{project_id}/afk-notifications/dismiss")

    after = dict(db.get_session(conn, card_id))
    assert after == before


def test_qa_complete_endpoint_accepts_notes_and_returns_ok(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    conn = db.get_connection()
    qa_card_id = db.create_session(
        conn, project_id,
        session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-abc",
        details={"prd": {"number": 7, "title": "Test PRD"}},
    )

    received = {}

    async def _fake_continue(card_id, notes, *, cwd):
        received["card_id"] = card_id
        received["notes"] = notes

    monkeypatch.setattr(session_runner, "continue_qa_job", _fake_continue)

    resp = client.post("/api/session/qa-complete", json={"card_id": qa_card_id, "notes": "All good"})
    assert resp.json() == {"ok": True}
    assert received["card_id"] == qa_card_id
    assert received["notes"] == "All good"


def test_qa_complete_endpoint_returns_error_for_unknown_session(client):
    resp = client.post("/api/session/qa-complete", json={"card_id": 9999, "notes": ""})
    assert "error" in resp.json()


def test_session_error_notifications_endpoint_round_trip(client, tmp_path):
    _open_project(client, tmp_path, "proj")
    project_id = client.get("/api/app-state").json()["active_project"]["id"]

    assert client.get(f"/api/projects/{project_id}/session-error-notifications").json() == {"notifications": []}

    session_runner.add_error_notification(project_id, 42, "implementing", "boom")

    resp = client.get(f"/api/projects/{project_id}/session-error-notifications")
    assert resp.json() == {"notifications": [{"card_id": 42, "phase": "implementing", "message": "boom"}]}

    dismiss_resp = client.post(f"/api/projects/{project_id}/session-error-notifications/dismiss")
    assert dismiss_resp.json() == {"dismissed": True}
    assert client.get(f"/api/projects/{project_id}/session-error-notifications").json() == {"notifications": []}


def test_implement_reply_endpoint_reaches_continue_implement_job(client, tmp_path, monkeypatch):
    _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(
        conn, 1, session_type="implement", phase="blocked",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )

    received = {}

    async def _fake_continue(card_id_arg, reply, *, cwd):
        received["card_id"] = card_id_arg
        received["reply"] = reply

    monkeypatch.setattr(session_runner, "continue_implement_job", _fake_continue)

    resp = client.post(f"/api/sessions/{card_id}/implement-reply", json={"reply": "Use GitHub OAuth"})
    assert resp.json() == {"ok": True}
    assert received["card_id"] == card_id
    assert received["reply"] == "Use GitHub OAuth"


def test_implement_reply_endpoint_returns_error_for_unknown_session(client):
    resp = client.post("/api/sessions/9999/implement-reply", json={"reply": "anything"})
    assert resp.json() == {"error": "Session not found"}


def test_sessions_list_exposes_blocked_field(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(
        conn, project_id, session_type="implement", phase="blocked",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    db.update_session(
        conn, card_id, blocked_json=json.dumps({"phase": "implement_blocked", "question": "Which one?"})
    )

    sessions = client.get(f"/api/projects/{project_id}/sessions").json()["sessions"]
    [session] = [s for s in sessions if s["card_id"] == card_id]
    assert session["blocked"] == {"phase": "implement_blocked", "question": "Which one?"}


def test_sessions_list_blocked_is_none_when_not_blocked(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    sessions = client.get(f"/api/projects/{project_id}/sessions").json()["sessions"]
    [session] = [s for s in sessions if s["card_id"] == card_id]
    assert session["blocked"] is None
