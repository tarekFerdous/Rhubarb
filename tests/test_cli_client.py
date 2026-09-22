import json

import pytest

from rhubarb import cli_client
from rhubarb.headroom_installer import HEADROOM_BASE_URL, HEADROOM_PROXY_PORT
from rhubarb.lean_ctx_installer import LEAN_CTX_HOOKS_SETTINGS_PATH, LEAN_CTX_MCP_CONFIG_PATH


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


def test_run_prompt_loads_rhubarb_own_plugin(monkeypatch):
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


# ---------------------------------------------------------------------------
# Headroom proxy port (issue #203)
# ---------------------------------------------------------------------------


def test_headroom_base_url_derives_from_the_shared_port_constant():
    """`cli_client._HEADROOM_BASE_URL` must not be an independent hardcoded
    copy -- it derives from `headroom_installer.HEADROOM_BASE_URL`/
    `HEADROOM_PROXY_PORT`, the single shared source of truth."""
    assert cli_client._HEADROOM_BASE_URL == HEADROOM_BASE_URL
    assert cli_client._HEADROOM_BASE_URL == f"http://localhost:{HEADROOM_PROXY_PORT}"
    assert cli_client._HEADROOM_BASE_URL == "http://localhost:8787"


def test_run_prompt_sets_anthropic_base_url_to_the_shared_port_when_headroom_active(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)
    cli_client.set_headroom_proxy_active(True)
    try:
        cli_client.run_prompt("hello")
    finally:
        cli_client.set_headroom_proxy_active(False)

    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8787"


# ---------------------------------------------------------------------------
# lean-ctx args (issue #233, child of PRD #232)
# ---------------------------------------------------------------------------


def test_lean_ctx_args_is_empty_when_disabled():
    cli_client.set_lean_ctx_enabled(False)

    assert cli_client._lean_ctx_args() == []


def test_lean_ctx_args_points_at_the_generated_config_files_when_enabled():
    cli_client.set_lean_ctx_enabled(True)
    try:
        args = cli_client._lean_ctx_args()
    finally:
        cli_client.set_lean_ctx_enabled(False)

    assert args == [
        "--mcp-config",
        str(LEAN_CTX_MCP_CONFIG_PATH),
        "--settings",
        str(LEAN_CTX_HOOKS_SETTINGS_PATH),
    ]


def test_run_prompt_passes_lean_ctx_args_when_enabled(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)
    cli_client.set_lean_ctx_enabled(True)
    try:
        cli_client.run_prompt("hello")
    finally:
        cli_client.set_lean_ctx_enabled(False)

    assert "--mcp-config" in captured["args"]
    assert captured["args"][captured["args"].index("--mcp-config") + 1] == str(LEAN_CTX_MCP_CONFIG_PATH)
    assert "--settings" in captured["args"]
    assert captured["args"][captured["args"].index("--settings") + 1] == str(LEAN_CTX_HOOKS_SETTINGS_PATH)


def test_run_prompt_omits_lean_ctx_args_when_disabled(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = json.dumps({"session_id": "abc"})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_client.subprocess, "run", fake_run)
    cli_client.set_lean_ctx_enabled(False)

    cli_client.run_prompt("hello")

    assert "--mcp-config" not in captured["args"]
    assert "--settings" not in captured["args"]
