import json

import pytest

from baton import cli_client


def test_run_prompt_skips_permission_checks(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)

    cli_client.run_prompt("hello")

    assert "--dangerously-skip-permissions" in captured["args"]


def test_run_prompt_loads_baton_own_plugin(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)

    cli_client.run_prompt("hello")

    assert "--plugin-dir" in captured["args"]
    plugin_dir = captured["args"][captured["args"].index("--plugin-dir") + 1]
    assert plugin_dir == cli_client._PLUGIN_DIR


@pytest.mark.parametrize("effort", ["high", "medium", "low"])
def test_run_prompt_passes_effort_flag_for_high_medium_low(monkeypatch, effort):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)

    cli_client.run_prompt("hello", effort=effort)

    assert "--effort" in captured["args"]
    assert captured["args"][captured["args"].index("--effort") + 1] == effort


@pytest.mark.parametrize("effort", ["auto", None])
def test_run_prompt_omits_effort_flag_for_auto_or_unset(monkeypatch, effort):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)

    cli_client.run_prompt("hello", effort=effort)

    assert "--effort" not in captured["args"]


def test_get_auth_status_parses_json_output(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = json.dumps({"loggedIn": True})
        stderr = ""

    def fake_run(args, **kwargs):
        assert args == ["claude", "auth", "status", "--json"]
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)

    assert cli_client.get_auth_status() == {"loggedIn": True}


def test_get_auth_status_raises_on_nonzero_exit(monkeypatch):
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "not logged in"

    monkeypatch.setattr(cli_client.subprocess, "run", lambda *a, **kw: FakeResult())

    with pytest.raises(cli_client.ClaudeCLIError):
        cli_client.get_auth_status()
