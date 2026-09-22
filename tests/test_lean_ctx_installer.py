import json
import subprocess

import pytest

from rhubarb import lean_ctx_installer
from rhubarb.lean_ctx_installer import (
    PRESENCE_NOT_PRESENT,
    PRESENCE_PRESENT,
    check_lean_ctx_presence,
    install_for_platform,
    install_linux,
    install_macos,
    install_windows,
)

_NPM_INSTALL_LEAN_CTX = ["npm", "install", "-g", "lean-ctx-bin"]


@pytest.fixture(autouse=True)
def _fixed_commands(monkeypatch):
    """Every test below (other than `resolve_npm_command`/`resolve_lean_ctx_
    command`'s own dedicated tests, which override this themselves) asserts
    on the exact argv passed to `run` -- pin `shutil.which` so both resolvers
    resolve deterministically regardless of where `npm`/`lean-ctx` actually
    happen to live on whatever machine runs this test."""
    monkeypatch.setattr(lean_ctx_installer.shutil, "which", lambda name: name)


# ---------------------------------------------------------------------------
# `resolve_npm_command`/`resolve_lean_ctx_command` -- same `.cmd`-shim
# `WinError 2` fix as `caveman_installer.resolve_npx_command`.
# ---------------------------------------------------------------------------


def test_resolve_npm_command_uses_shutil_which(monkeypatch):
    monkeypatch.setattr(lean_ctx_installer.shutil, "which", lambda name: r"C:\Program Files\nodejs\npm.CMD")

    assert lean_ctx_installer.resolve_npm_command() == r"C:\Program Files\nodejs\npm.CMD"


def test_resolve_npm_command_falls_back_to_bare_name_when_which_finds_nothing(monkeypatch):
    monkeypatch.setattr(lean_ctx_installer.shutil, "which", lambda name: None)

    assert lean_ctx_installer.resolve_npm_command() == "npm"


def test_resolve_lean_ctx_command_uses_shutil_which(monkeypatch):
    monkeypatch.setattr(
        lean_ctx_installer.shutil, "which", lambda name: r"C:\Users\someone\AppData\Roaming\npm\lean-ctx.CMD"
    )

    assert lean_ctx_installer.resolve_lean_ctx_command() == r"C:\Users\someone\AppData\Roaming\npm\lean-ctx.CMD"


def test_resolve_lean_ctx_command_falls_back_to_bare_name_when_which_finds_nothing(monkeypatch):
    monkeypatch.setattr(lean_ctx_installer.shutil, "which", lambda name: None)

    assert lean_ctx_installer.resolve_lean_ctx_command() == "lean-ctx"


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


def test_presence_reports_not_present_when_command_raises():
    def failing_run(argv):
        raise FileNotFoundError("lean-ctx not found")

    assert check_lean_ctx_presence(run=failing_run) == PRESENCE_NOT_PRESENT


def test_presence_reports_present_when_command_succeeds():
    calls = []

    def succeeding_run(argv):
        calls.append(argv)

    result = check_lean_ctx_presence(run=succeeding_run)
    assert result == PRESENCE_PRESENT
    assert calls == [["lean-ctx", "--version"]]


def test_presence_check_uses_the_resolved_command_not_a_bare_name(monkeypatch):
    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: r"C:\npm\lean-ctx.CMD")
    calls = []

    check_lean_ctx_presence(run=calls.append)

    assert calls == [[r"C:\npm\lean-ctx.CMD", "--version"]]


def test_presence_reports_not_present_when_command_exits_nonzero():
    def failing_run(argv):
        raise subprocess.CalledProcessError(1, argv)

    assert check_lean_ctx_presence(run=failing_run) == PRESENCE_NOT_PRESENT


# ---------------------------------------------------------------------------
# Per-OS install -- all three run the identical `npm install -g lean-ctx-bin`
# ---------------------------------------------------------------------------


def test_install_windows_invokes_npm_install_lean_ctx_bin():
    calls = []
    install_windows(run=calls.append)

    assert calls == [_NPM_INSTALL_LEAN_CTX]


def test_install_macos_invokes_npm_install_lean_ctx_bin():
    calls = []
    install_macos(run=calls.append)

    assert calls == [_NPM_INSTALL_LEAN_CTX]


def test_install_linux_invokes_npm_install_lean_ctx_bin():
    calls = []
    install_linux(run=calls.append)

    assert calls == [_NPM_INSTALL_LEAN_CTX]


def test_install_uses_the_resolved_npm_command_not_a_bare_name(monkeypatch):
    """A bare `"npm"` fails on Windows even when it's genuinely on `PATH` --
    `CreateProcess` can't launch a `.cmd` shim without the full path
    including its extension (see `resolve_npm_command`'s docstring)."""
    monkeypatch.setattr(lean_ctx_installer, "resolve_npm_command", lambda: r"C:\Program Files\nodejs\npm.CMD")
    calls = []
    install_windows(run=calls.append)

    assert calls[0][0] == r"C:\Program Files\nodejs\npm.CMD"


def test_install_for_platform_dispatches_to_windows():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Windows")

    assert calls == [_NPM_INSTALL_LEAN_CTX]


def test_install_for_platform_dispatches_to_macos():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Darwin")

    assert calls == [_NPM_INSTALL_LEAN_CTX]


def test_install_for_platform_dispatches_to_linux_for_anything_else():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Linux")

    assert calls == [_NPM_INSTALL_LEAN_CTX]


# ---------------------------------------------------------------------------
# `run_streaming` -- same `check=True` contract as `_default_run`, but calls
# `on_line` for each line of output as it arrives.
# ---------------------------------------------------------------------------


class _FakeStreamingProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.waited = False

    def wait(self):
        self.waited = True


def test_run_streaming_calls_on_line_for_every_line_of_output():
    process = _FakeStreamingProcess(["npm warn ...\n", "added 3 packages\n"])
    received = []

    lean_ctx_installer.run_streaming(["fake"], on_line=received.append, popen_factory=lambda *a, **kw: process)

    assert received == ["npm warn ...", "added 3 packages"]
    assert process.waited is True


def test_run_streaming_raises_called_process_error_on_nonzero_exit():
    process = _FakeStreamingProcess(["some output\n"], returncode=1)

    with pytest.raises(subprocess.CalledProcessError):
        lean_ctx_installer.run_streaming(["fake"], popen_factory=lambda *a, **kw: process)


def test_run_streaming_passes_argv_and_pipe_kwargs_to_popen_factory():
    process = _FakeStreamingProcess([])
    captured = {}

    def fake_popen_factory(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    lean_ctx_installer.run_streaming(["fake", "arg"], popen_factory=fake_popen_factory)

    assert captured["argv"] == ["fake", "arg"]
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert captured["kwargs"]["text"] is True


# ---------------------------------------------------------------------------
# `generate_scoped_config` -- the two Rhubarb-owned scoped config files
# ---------------------------------------------------------------------------


def test_generate_scoped_config_writes_mcp_config_pointing_at_resolved_command(tmp_path, monkeypatch):
    mcp_path = tmp_path / "lean-ctx-mcp-config.json"
    settings_path = tmp_path / "lean-ctx-hooks-settings.json"
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_MCP_CONFIG_PATH", mcp_path)
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_HOOKS_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: r"C:\npm\lean-ctx.CMD")

    lean_ctx_installer.generate_scoped_config()

    mcp_config = json.loads(mcp_path.read_text())
    assert mcp_config["mcpServers"]["lean-ctx"]["command"] == r"C:\npm\lean-ctx.CMD"


def test_generate_scoped_config_writes_hooks_settings_denying_grep_and_glob(tmp_path, monkeypatch):
    mcp_path = tmp_path / "lean-ctx-mcp-config.json"
    settings_path = tmp_path / "lean-ctx-hooks-settings.json"
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_MCP_CONFIG_PATH", mcp_path)
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_HOOKS_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: "lean-ctx")

    lean_ctx_installer.generate_scoped_config()

    settings = json.loads(settings_path.read_text())
    assert settings["permissions"]["deny"] == ["Grep", "Glob"]
    assert "PreToolUse" in settings["hooks"]
    assert "PostToolUse" in settings["hooks"]


def test_generate_scoped_config_creates_parent_directory(tmp_path, monkeypatch):
    nested_dir = tmp_path / "does" / "not" / "exist" / "yet"
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_MCP_CONFIG_PATH", nested_dir / "lean-ctx-mcp-config.json")
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_HOOKS_SETTINGS_PATH", nested_dir / "lean-ctx-hooks-settings.json")
    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: "lean-ctx")

    lean_ctx_installer.generate_scoped_config()  # must not raise

    assert (nested_dir / "lean-ctx-mcp-config.json").exists()
    assert (nested_dir / "lean-ctx-hooks-settings.json").exists()


def test_generate_scoped_config_regenerates_on_every_call_not_just_when_missing(tmp_path, monkeypatch):
    """Issue #233's acceptance criteria: regenerated every time lean-ctx
    transitions to enabled, never merely checked for existence -- a
    relocated/reinstalled `lean-ctx` command must be reflected."""
    mcp_path = tmp_path / "lean-ctx-mcp-config.json"
    settings_path = tmp_path / "lean-ctx-hooks-settings.json"
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_MCP_CONFIG_PATH", mcp_path)
    monkeypatch.setattr(lean_ctx_installer, "LEAN_CTX_HOOKS_SETTINGS_PATH", settings_path)

    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: "old-path/lean-ctx")
    lean_ctx_installer.generate_scoped_config()
    assert json.loads(mcp_path.read_text())["mcpServers"]["lean-ctx"]["command"] == "old-path/lean-ctx"

    monkeypatch.setattr(lean_ctx_installer, "resolve_lean_ctx_command", lambda: "new-path/lean-ctx")
    lean_ctx_installer.generate_scoped_config()
    assert json.loads(mcp_path.read_text())["mcpServers"]["lean-ctx"]["command"] == "new-path/lean-ctx"


# ---------------------------------------------------------------------------
# Injectable defaults sanity check
# ---------------------------------------------------------------------------


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Every test above injects `run`, so `_default_run` is never exercised --
    matching the injectable-only testing contract."""
    assert callable(lean_ctx_installer._default_run)
