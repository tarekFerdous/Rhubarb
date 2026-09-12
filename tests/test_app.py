import json
import subprocess

from rhubarb import afk_loop, db, error_log, ollama_installer, session_runner
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


# ---------------------------------------------------------------------------
# Ollama consent/install gate (issue #115)
# ---------------------------------------------------------------------------


def test_ollama_status_reports_presence_and_declined_state(client, monkeypatch):
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_NOT_PRESENT)

    data = client.get("/api/ollama-status").json()

    assert data == {"presence": "not_present", "declined": False}


def test_set_ollama_declined_persists_and_reflects_in_status(client, monkeypatch):
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_WITH_MODEL)

    resp = client.post("/api/settings/ollama-declined", json={"ollama_declined": True})
    assert resp.json() == {"ollama_declined": True}
    assert client.get("/api/ollama-status").json()["declined"] is True

    resp = client.post("/api/settings/ollama-declined", json={"ollama_declined": False})
    assert resp.json() == {"ollama_declined": False}
    assert client.get("/api/ollama-status").json()["declined"] is False


def test_ollama_install_is_a_noop_and_ok_when_already_present_with_model(client, monkeypatch):
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_WITH_MODEL)
    monkeypatch.setattr(
        ollama_installer, "install_and_pull_model", lambda: (_ for _ in ()).throw(AssertionError("should not install"))
    )
    monkeypatch.setattr(ollama_installer, "pull_model", lambda: (_ for _ in ()).throw(AssertionError("should not pull")))

    resp = client.post("/api/ollama-install")

    assert resp.json() == {"ok": True}


def test_ollama_install_runs_full_install_when_not_present(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_NOT_PRESENT)
    monkeypatch.setattr(ollama_installer, "install_and_pull_model", lambda: calls.append("install_and_pull"))

    resp = client.post("/api/ollama-install")

    assert resp.json() == {"ok": True}
    assert calls == ["install_and_pull"]


def test_ollama_install_only_pulls_when_present_without_model(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_WITHOUT_MODEL)
    monkeypatch.setattr(ollama_installer, "pull_model", lambda: calls.append("pull"))

    resp = client.post("/api/ollama-install")

    assert resp.json() == {"ok": True}
    assert calls == ["pull"]


def test_ollama_install_reports_the_error_on_failure_instead_of_raising(client, monkeypatch):
    monkeypatch.setattr(ollama_installer, "check_ollama_presence", lambda: ollama_installer.PRESENCE_NOT_PRESENT)

    def failing_install():
        raise RuntimeError("winget not found")

    monkeypatch.setattr(ollama_installer, "install_and_pull_model", failing_install)

    resp = client.post("/api/ollama-install")

    assert resp.json() == {"ok": False, "error": "winget not found"}


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


# ---------------------------------------------------------------------------
# Model/Effort card ground truth for the open Live Terminal (issue #139)
# ---------------------------------------------------------------------------


def test_app_state_returns_engine_ground_truth_for_a_card_with_a_live_engine(client, tmp_path):
    client.post("/api/settings/model", json={"model": "claude-opus-4-8"})
    client.post("/api/settings/effort", json={"effort": "high"})

    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()
    card_id = db.create_session(conn, project_id, model="claude-opus-4-8", effort="high")

    class _FakeEngine:
        model = "claude-sonnet-5"
        effort = "low"

    session_runner._pty_engines[card_id] = _FakeEngine()
    try:
        state = client.get(f"/api/app-state?card_id={card_id}").json()
        assert state["model"] == "claude-sonnet-5"
        assert state["effort"] == "low"
        assert state["session_model_effort_live"] is True
    finally:
        session_runner._pty_engines.pop(card_id, None)


def test_app_state_falls_back_to_global_settings_when_card_has_no_live_engine(client, tmp_path):
    client.post("/api/settings/model", json={"model": "claude-sonnet-4-6"})
    client.post("/api/settings/effort", json={"effort": "auto"})

    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()
    # A pooled/finished session row with its own (different) recorded
    # model/effort -- still must fall back to the *global* setting, not this
    # row's own field, since there's no live engine to be ground truth for.
    card_id = db.create_session(conn, project_id, model="claude-opus-4-8", effort="high")

    assert card_id not in session_runner._pty_engines
    state = client.get(f"/api/app-state?card_id={card_id}").json()
    assert state["model"] == "claude-sonnet-4-6"
    assert state["effort"] == "auto"
    assert state["session_model_effort_live"] is False


def test_app_state_without_card_id_is_unaffected_by_a_live_engine_elsewhere(client, tmp_path):
    """No `card_id` passed (today's exact call shape) must behave exactly as
    before this issue -- global settings only, `session_model_effort_live`
    False -- even while some other card has a live resident engine."""
    client.post("/api/settings/model", json={"model": "claude-sonnet-4-6"})
    client.post("/api/settings/effort", json={"effort": "auto"})

    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    class _FakeEngine:
        model = "claude-sonnet-5"
        effort = "low"

    session_runner._pty_engines[card_id] = _FakeEngine()
    try:
        state = client.get("/api/app-state").json()
        assert state["model"] == "claude-sonnet-4-6"
        assert state["effort"] == "auto"
        assert state["session_model_effort_live"] is False
    finally:
        session_runner._pty_engines.pop(card_id, None)


def _open_project(client, tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    _init_repo(root / "repo", f"https://github.com/x/{name}.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")
    return project_id


# ---------------------------------------------------------------------------
# Background/other-tab model+effort visibility for resident and standby
# engines (issue #140)
# ---------------------------------------------------------------------------


def test_pty_tab_count_lists_resident_and_standby_engines_with_model_effort(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()
    card_id = db.create_session(conn, project_id, model="claude-opus-4-8", effort="high")

    class _FakeEngine:
        model = "claude-sonnet-5"
        effort = "low"

    class _FakeStandbyEngine:
        model = "claude-sonnet-4-6"
        effort = "auto"

    session_runner._pty_engines[card_id] = _FakeEngine()
    session_runner._standby_engines[project_id] = (_FakeStandbyEngine(), "claude-sonnet-4-6", "auto")

    data = client.get("/api/pty-tabs/count").json()

    assert data["count"] == 2
    engines = data["engines"]
    assert len(engines) == 2
    assert {"card_id": card_id, "model": "claude-sonnet-5", "effort": "low"} in engines
    assert {"card_id": "standby", "model": "claude-sonnet-4-6", "effort": "auto"} in engines


def test_pty_tab_count_contract_is_unaffected_when_no_engines_are_live(client, tmp_path):
    """The pre-#140 `{"count": N}` shape must remain valid/unaffected: with
    no live engines, `"count"` is 0 and the new `"engines"` field is an empty
    list rather than missing or breaking anything an existing consumer that
    only reads `"count"` relies on."""
    _open_project(client, tmp_path, "proj")

    data = client.get("/api/pty-tabs/count").json()

    assert data["count"] == 0
    assert data["engines"] == []


def test_list_live_engines_combines_resident_and_standby_across_projects(client, tmp_path):
    """Exercise `session_runner.list_live_engines` directly (not just through
    the endpoint) with more than one resident engine plus a standby, to
    confirm every live entry across both registries is represented."""
    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()
    card_a = db.create_session(conn, project_id, model="claude-opus-4-8", effort="high")
    card_b = db.create_session(conn, project_id, model="claude-sonnet-5", effort="auto")

    class _FakeEngine:
        def __init__(self, model, effort):
            self.model = model
            self.effort = effort

    session_runner._pty_engines[card_a] = _FakeEngine("claude-opus-4-8", "high")
    session_runner._pty_engines[card_b] = _FakeEngine("claude-sonnet-5", "auto")
    session_runner._standby_engines[project_id] = (_FakeEngine("claude-sonnet-4-6", "low"), "claude-sonnet-4-6", "low")

    engines = session_runner.list_live_engines()

    assert len(engines) == 3
    assert {"card_id": card_a, "model": "claude-opus-4-8", "effort": "high"} in engines
    assert {"card_id": card_b, "model": "claude-sonnet-5", "effort": "auto"} in engines
    assert {"card_id": "standby", "model": "claude-sonnet-4-6", "effort": "low"} in engines


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


def test_open_project_schedules_a_standby_prewarm(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]

    calls = []

    async def _fake_ensure(pid, *, cwd, model, effort):
        calls.append({"project_id": pid, "cwd": cwd, "model": model, "effort": effort})

    monkeypatch.setattr(app_module.session_runner, "ensure_standby_engine", _fake_ensure)

    client.post(f"/api/projects/{project_id}/open")

    assert len(calls) == 1
    assert calls[0]["project_id"] == project_id
    assert calls[0]["effort"] == db.DEFAULT_EFFORT


def test_close_project_closes_its_standby(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]

    calls = []
    monkeypatch.setattr(app_module.session_runner, "close_standby_engine", lambda pid: calls.append(pid))

    client.post(f"/api/projects/{project_id}/close", json={})

    assert calls == [project_id]


def test_session_start_claims_a_matching_standby_engine(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    class FakeStandby:
        claude_session_id = "standby-session-id"

    fake_standby = FakeStandby()
    claim_calls = []
    register_calls = []

    def fake_claim(pid, *, model, effort):
        claim_calls.append({"project_id": pid, "model": model, "effort": effort})
        return fake_standby

    monkeypatch.setattr(app_module.session_runner, "claim_standby_engine", fake_claim)
    monkeypatch.setattr(app_module.session_runner, "register_engine", lambda cid, eng: register_calls.append((cid, eng)))

    async def _noop_job(card_id, prompt, *, cwd):
        return None

    monkeypatch.setattr(app_module.session_runner, "start_session_job", _noop_job)

    called_claim_available = []
    monkeypatch.setattr(app_module.db, "claim_available_session", lambda conn, pid: called_claim_available.append(pid) or None)

    resp = client.post("/api/session/start", json={"prompt": "a feature"})
    card_id = resp.json()["card_id"]

    assert len(claim_calls) == 1
    assert register_calls == [(card_id, fake_standby)]
    assert called_claim_available == []  # DB-pool fallback never consulted when a standby matched

    row = db.get_session(db.get_connection(), card_id)
    assert row["claude_session_id"] == "standby-session-id"


def test_session_start_falls_back_to_db_pool_when_no_standby_matches(client, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "repo1", "https://github.com/x/repo1.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")

    monkeypatch.setattr(app_module.session_runner, "claim_standby_engine", lambda pid, *, model, effort: None)

    register_calls = []
    monkeypatch.setattr(app_module.session_runner, "register_engine", lambda cid, eng: register_calls.append((cid, eng)))

    called_claim_available = []

    def fake_claim_available(conn, pid):
        called_claim_available.append(pid)
        return None

    monkeypatch.setattr(app_module.db, "claim_available_session", fake_claim_available)

    async def _noop_job(card_id, prompt, *, cwd):
        return None

    monkeypatch.setattr(app_module.session_runner, "start_session_job", _noop_job)

    client.post("/api/session/start", json={"prompt": "a feature"})

    assert called_claim_available == [project_id]
    assert register_calls == []


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


def test_qa_complete_endpoint_accepts_answers_and_extra_notes_and_returns_ok(client, tmp_path, monkeypatch):
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

    async def _fake_continue(card_id, answers, extra_notes, *, cwd):
        received["card_id"] = card_id
        received["answers"] = answers
        received["extra_notes"] = extra_notes

    monkeypatch.setattr(session_runner, "continue_qa_job", _fake_continue)

    resp = client.post(
        "/api/session/qa-complete",
        json={"card_id": qa_card_id, "answers": {"issue7-q1": "Works great"}, "extra_notes": "All good"},
    )
    assert resp.json() == {"ok": True}
    assert received["card_id"] == qa_card_id
    assert received["answers"] == {"issue7-q1": "Works great"}
    assert received["extra_notes"] == "All good"


def test_qa_complete_endpoint_returns_error_for_unknown_session(client):
    resp = client.post("/api/session/qa-complete", json={"card_id": 9999, "answers": {}, "extra_notes": ""})
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


def test_project_errors_endpoint_returns_empty_list_when_log_file_missing(client, tmp_path, monkeypatch):
    """Issue #155: before any error has ever been logged, the log file
    doesn't exist yet -- the endpoint must return an empty list, not error."""
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "no-such-home" / "logs" / "errors.log")
    project_id = _open_project(client, tmp_path, "proj")

    resp = client.get(f"/api/projects/{project_id}/errors")
    assert resp.status_code == 200
    assert resp.json() == {"errors": []}


def test_project_errors_endpoint_scopes_to_project(client, tmp_path, monkeypatch):
    """Entries logged for a different project_id must never leak into this
    project's response."""
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "err-home" / "logs" / "errors.log")
    root = tmp_path / "root"
    root.mkdir()
    _init_repo(root / "proj-a", "https://github.com/x/proj-a.git")
    _init_repo(root / "proj-b", "https://github.com/x/proj-b.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root)})
    projects = {p["name"]: p["id"] for p in client.get("/api/app-state").json()["projects"]}
    project_a, project_b = projects["proj-a"], projects["proj-b"]

    error_log.log_error(project_id=project_a, card_id=1, phase="implementing", message="a broke")
    error_log.log_error(project_id=project_b, card_id=2, phase="implementing", message="b broke")

    resp_a = client.get(f"/api/projects/{project_a}/errors").json()
    assert [e["message"] for e in resp_a["errors"]] == ["a broke"]

    resp_b = client.get(f"/api/projects/{project_b}/errors").json()
    assert [e["message"] for e in resp_b["errors"]] == ["b broke"]


def test_project_errors_endpoint_filters_by_phase(client, tmp_path, monkeypatch):
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "err-home" / "logs" / "errors.log")
    project_id = _open_project(client, tmp_path, "proj")

    error_log.log_error(project_id=project_id, card_id=1, phase="grilling", message="grill boom")
    error_log.log_error(project_id=project_id, card_id=2, phase="implementing", message="impl boom")

    resp = client.get(f"/api/projects/{project_id}/errors", params={"phase": "implementing"}).json()
    assert [e["message"] for e in resp["errors"]] == ["impl boom"]


def test_project_errors_endpoint_filters_by_card_id(client, tmp_path, monkeypatch):
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "err-home" / "logs" / "errors.log")
    project_id = _open_project(client, tmp_path, "proj")

    error_log.log_error(project_id=project_id, card_id=1, phase="implementing", message="card one")
    error_log.log_error(project_id=project_id, card_id=2, phase="implementing", message="card two")

    resp = client.get(f"/api/projects/{project_id}/errors", params={"card_id": 2}).json()
    assert [e["message"] for e in resp["errors"]] == ["card two"]


def test_project_errors_endpoint_filters_by_free_text_search(client, tmp_path, monkeypatch):
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "err-home" / "logs" / "errors.log")
    project_id = _open_project(client, tmp_path, "proj")

    error_log.log_error(project_id=project_id, card_id=1, phase="implementing", message="Disk was full")
    error_log.log_error(project_id=project_id, card_id=2, phase="implementing", message="network timeout")

    resp = client.get(f"/api/projects/{project_id}/errors", params={"q": "disk"}).json()
    assert [e["message"] for e in resp["errors"]] == ["Disk was full"]


def test_project_errors_endpoint_filters_by_date_range(client, tmp_path, monkeypatch):
    """Uses direct JSONL writes (rather than `log_error`, whose timestamp is
    always `now`) so the test can control each entry's timestamp precisely."""
    log_path = tmp_path / "err-home" / "logs" / "errors.log"
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", log_path)
    project_id = _open_project(client, tmp_path, "proj")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "project_id": project_id, "card_id": 1, "phase": "implementing", "message": "too early"},
        {"timestamp": "2026-02-01T00:00:00+00:00", "project_id": project_id, "card_id": 2, "phase": "implementing", "message": "in range"},
        {"timestamp": "2026-03-01T00:00:00+00:00", "project_id": project_id, "card_id": 3, "phase": "implementing", "message": "too late"},
    ]
    with log_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    resp = client.get(
        f"/api/projects/{project_id}/errors",
        params={"since": "2026-01-15T00:00:00+00:00", "until": "2026-02-15T00:00:00+00:00"},
    ).json()
    assert [e["message"] for e in resp["errors"]] == ["in range"]


def test_project_errors_endpoint_combines_filters_with_and_semantics(client, tmp_path, monkeypatch):
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", tmp_path / "err-home" / "logs" / "errors.log")
    project_id = _open_project(client, tmp_path, "proj")

    error_log.log_error(project_id=project_id, card_id=1, phase="implementing", message="disk full")
    error_log.log_error(project_id=project_id, card_id=2, phase="implementing", message="disk full")
    error_log.log_error(project_id=project_id, card_id=2, phase="publishing", message="disk full")

    resp = client.get(
        f"/api/projects/{project_id}/errors",
        params={"phase": "implementing", "card_id": 2, "q": "disk"},
    ).json()
    assert len(resp["errors"]) == 1
    assert resp["errors"][0]["card_id"] == 2
    assert resp["errors"][0]["phase"] == "implementing"


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


def test_sessions_list_exposes_stalled_field(client, tmp_path):
    """Issue #169: `stalled_json` (persisted by `session_runner._run_turn`
    the same way `blocked_json` already is -- see `test_sessions_list_
    exposes_blocked_field` above) must round-trip through the session list
    endpoint as a `stalled` field, so a reconnect/page-refresh can recover
    and re-show it."""
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)
    db.update_session(
        conn, card_id, stalled_json=json.dumps({"phase": "grilling", "context": "still working..."})
    )

    sessions = client.get(f"/api/projects/{project_id}/sessions").json()["sessions"]
    [session] = [s for s in sessions if s["card_id"] == card_id]
    assert session["stalled"] == {"phase": "grilling", "context": "still working..."}


def test_sessions_list_stalled_is_none_when_not_stalled(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    sessions = client.get(f"/api/projects/{project_id}/sessions").json()["sessions"]
    [session] = [s for s in sessions if s["card_id"] == card_id]
    assert session["stalled"] is None


def test_close_session_endpoint_marks_the_row_closed(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    resp = client.post(f"/api/sessions/{card_id}/close")
    assert resp.json() == {"card_id": card_id}

    row = db.get_session(conn, card_id)
    assert row["phase"] == "closed"


def test_close_session_endpoint_returns_error_for_unknown_session(client):
    resp = client.post("/api/sessions/999999/close")
    assert resp.json() == {"error": "Session not found"}


def test_sessions_list_excludes_a_closed_session(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    client.post(f"/api/sessions/{card_id}/close")

    sessions = client.get(f"/api/projects/{project_id}/sessions").json()["sessions"]
    assert card_id not in [s["card_id"] for s in sessions]


# ---------------------------------------------------------------------------
# Stall-reply endpoint (issue #169, child of PRD #168 "Recover from a
# stalled turn instead of hanging the turn lock forever"): mirrors
# `resize_session_pty` in shape -- a small, separate endpoint that looks up
# a card's resident engine and forwards straight into its lock-protected
# `PtyEngine.write()`, without going through `_run_turn`'s own prompt-write
# machinery, starting a new turn, or touching the turn lock a second time.
# ---------------------------------------------------------------------------


class _FakeWriteEngine:
    """Minimal stand-in for a resident `PtyEngine` -- only `write()` is
    exercised by the stall-reply endpoint, so that's all this fake needs to
    implement. Records every call so a test can assert exactly what reached
    it, and never touches any turn lock -- there is none here, since this
    fake is registered directly into `session_runner._pty_engines` rather
    than driven through `_run_turn`."""

    def __init__(self):
        self.writes = []

    async def write(self, data):
        self.writes.append(data)


def test_stall_reply_endpoint_forwards_straight_into_the_cards_engine_write(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    engine = _FakeWriteEngine()
    monkeypatch.setattr(session_runner, "_pty_engines", {card_id: engine})

    resp = client.post(f"/api/sessions/{card_id}/stall-reply", json={"text": "please continue"})

    assert resp.json() == {"replied": True}
    assert engine.writes == ["please continue"]


def test_stall_reply_endpoint_is_a_noop_for_a_card_with_no_live_engine(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)
    # No engine registered for this card_id at all.

    resp = client.post(f"/api/sessions/{card_id}/stall-reply", json={"text": "anything"})

    assert resp.json() == {"replied": False}


def test_stall_reply_endpoint_never_touches_the_turn_lock_or_starts_a_new_turn(client, tmp_path, monkeypatch):
    """The whole point of this endpoint: it must reach the engine's `write()`
    directly, never `_run_turn` (which would try to acquire the per-card
    turn lock a second time and start a brand-new turn on top of whatever's
    already in flight)."""
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id = db.create_session(conn, project_id)

    engine = _FakeWriteEngine()
    monkeypatch.setattr(session_runner, "_pty_engines", {card_id: engine})

    def _run_turn_must_not_be_called(*args, **kwargs):
        raise AssertionError("stall-reply must not go through _run_turn")

    monkeypatch.setattr(session_runner, "_run_turn", _run_turn_must_not_be_called)

    # Hold the turn lock ourselves, exactly as a genuinely in-flight turn
    # would -- the endpoint must still succeed, proving it never tries to
    # acquire this same lock.
    lock = session_runner._get_turn_lock(card_id)
    assert not lock.locked()

    resp = client.post(f"/api/sessions/{card_id}/stall-reply", json={"text": "nudge"})

    assert resp.json() == {"replied": True}
    assert engine.writes == ["nudge"]
    # Still untouched -- the endpoint never acquired or released it.
    assert not lock.locked()


def test_stall_reply_endpoint_forwards_to_the_correct_cards_engine_only(client, tmp_path, monkeypatch):
    """With two cards each carrying their own resident engine, a reply for
    one card must never reach the other's."""
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    card_id_a = db.create_session(conn, project_id)
    card_id_b = db.create_session(conn, project_id)

    engine_a = _FakeWriteEngine()
    engine_b = _FakeWriteEngine()
    monkeypatch.setattr(session_runner, "_pty_engines", {card_id_a: engine_a, card_id_b: engine_b})

    resp = client.post(f"/api/sessions/{card_id_a}/stall-reply", json={"text": "for A"})

    assert resp.json() == {"replied": True}
    assert engine_a.writes == ["for A"]
    assert engine_b.writes == []
