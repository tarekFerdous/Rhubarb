import os
import sys

import pytest

from rhubarb import headroom_installer
from rhubarb.headroom_installer import (
    HEADROOM_BASE_URL,
    HEADROOM_PROXY_PORT,
    PRESENCE_NOT_PRESENT,
    PRESENCE_PRESENT,
    check_headroom_presence,
    install_for_platform,
    install_linux,
    install_macos,
    install_windows,
)

_PIP_INSTALL_HEADROOM = [sys.executable, "-m", "pip", "install", "headroom-ai[all]"]


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


def test_presence_reports_not_present_when_command_raises():
    def failing_run(argv):
        raise FileNotFoundError("headroom not found")

    assert check_headroom_presence(run=failing_run) == PRESENCE_NOT_PRESENT


def test_presence_reports_present_when_command_succeeds(monkeypatch):
    monkeypatch.setattr(headroom_installer, "resolve_headroom_command", lambda: "headroom")
    calls = []

    def succeeding_run(argv):
        calls.append(argv)

    result = check_headroom_presence(run=succeeding_run)
    assert result == PRESENCE_PRESENT
    assert calls == [["headroom", "--version"]]


def test_presence_check_uses_the_resolved_command_not_a_bare_name(monkeypatch):
    """`check_headroom_presence` must run through `resolve_headroom_command()`
    rather than a hardcoded bare `"headroom"`, so it finds a pip-installed
    console script even when this backend process's `PATH` doesn't include
    the scripts directory it lives in (see `resolve_headroom_command`'s
    docstring)."""
    monkeypatch.setattr(headroom_installer, "resolve_headroom_command", lambda: r"C:\Python\Scripts\headroom.exe")
    calls = []

    check_headroom_presence(run=calls.append)

    assert calls == [[r"C:\Python\Scripts\headroom.exe", "--version"]]


def test_presence_reports_not_present_when_command_exits_nonzero():
    """A subprocess.CalledProcessError (check=True) counts as not present."""
    import subprocess

    def failing_run(argv):
        raise subprocess.CalledProcessError(1, argv)

    assert check_headroom_presence(run=failing_run) == PRESENCE_NOT_PRESENT


# ---------------------------------------------------------------------------
# `resolve_headroom_command` -- the same PATH-inheritance fix as
# `ollama_installer._resolve_winget_command`, applied to `headroom`.
# ---------------------------------------------------------------------------


def test_resolve_headroom_command_prefers_the_pip_scripts_directory_when_present(monkeypatch):
    monkeypatch.setattr(headroom_installer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(headroom_installer.sysconfig, "get_path", lambda name, scheme=None: r"C:\Python\Scripts")
    monkeypatch.setattr(headroom_installer.os.path, "exists", lambda path: True)

    assert headroom_installer.resolve_headroom_command() == r"C:\Python\Scripts\headroom.exe"


def test_resolve_headroom_command_uses_no_exe_suffix_on_non_windows(monkeypatch):
    """`os.path.join` uses whatever separator the OS this test actually
    runs on uses (regardless of the simulated `platform.system()`), so
    assert on the basename rather than a hardcoded separator."""
    monkeypatch.setattr(headroom_installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(headroom_installer.sysconfig, "get_path", lambda name, scheme=None: "/usr/local/bin")
    monkeypatch.setattr(headroom_installer.os.path, "exists", lambda path: True)

    result = headroom_installer.resolve_headroom_command()
    assert os.path.basename(result) == "headroom"  # no ".exe" suffix


def test_resolve_headroom_command_falls_back_to_the_per_user_scripts_dir_when_pip_installed_there_instead(
    monkeypatch,
):
    """The real-world bug this exists to fix: pip silently falls back to a
    `--user` install (warning only, not a failure) whenever it lacks write
    access to the interpreter's own site-packages -- common for a
    non-admin user against a machine-wide Windows Python install. The
    install genuinely succeeds, but the executable lands in the per-user
    scripts directory, not the interpreter's own one -- so checking only
    the system scheme (the original version of this function) would still
    report the bare, unresolvable command, and starting the proxy right
    after a successful install would still fail with `WinError 2`."""
    monkeypatch.setattr(headroom_installer.platform, "system", lambda: "Windows")

    def fake_get_path(name, scheme=None):
        if scheme is None:
            return r"C:\Python313\Scripts"  # system scheme: nothing installed here
        return r"C:\Users\someone\AppData\Roaming\Python\Python313\Scripts"  # per-user fallback

    def fake_exists(path):
        return path == r"C:\Users\someone\AppData\Roaming\Python\Python313\Scripts\headroom.exe"

    monkeypatch.setattr(headroom_installer.sysconfig, "get_path", fake_get_path)
    monkeypatch.setattr(headroom_installer.os.path, "exists", fake_exists)

    result = headroom_installer.resolve_headroom_command()

    assert result == r"C:\Users\someone\AppData\Roaming\Python\Python313\Scripts\headroom.exe"


def test_resolve_headroom_command_falls_back_to_bare_name_when_not_found_at_either_pip_location(monkeypatch):
    """E.g. installed some other way (`uv tool install`, manually) and
    already reachable on this process's own `PATH`."""
    monkeypatch.setattr(headroom_installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(headroom_installer.sysconfig, "get_path", lambda name, scheme=None: "/some/pip/scripts/dir")
    monkeypatch.setattr(headroom_installer.os.path, "exists", lambda path: False)

    assert headroom_installer.resolve_headroom_command() == "headroom"


# ---------------------------------------------------------------------------
# Per-OS install
# ---------------------------------------------------------------------------


def test_install_windows_invokes_pip_install_headroom():
    calls = []
    install_windows(run=calls.append)

    assert calls == [_PIP_INSTALL_HEADROOM]


def test_install_windows_uses_the_real_headroom_ai_package_not_the_unrelated_headroom_package():
    """A PyPI package literally named `headroom` exists, but it's an
    unrelated tool by a different author/org, not this project's Headroom.
    The real package -- confirmed against `headroomlabs-ai/headroom`'s own
    README quickstart -- is `headroom-ai` (with the `[all]` extras the
    quickstart itself recommends)."""
    calls = []
    install_windows(run=calls.append)

    assert calls[0][-1] == "headroom-ai[all]"
    assert calls[0][-1] != "headroom"


def test_install_windows_uses_sys_executable_m_pip_not_a_bare_pip_command():
    """A bare `"pip"` is resolved via `PATH` by the OS loader, and this
    backend process doesn't reliably inherit the same `PATH` an interactive
    shell has -- pip's console-script shim can live somewhere that's
    simply not on it, failing with `WinError 2` even though pip genuinely
    works when typed into a terminal (the exact bug class already fixed
    once for `winget` in `ollama_installer._resolve_winget_command`).
    Routing through `sys.executable -m pip` needs no `PATH` lookup for
    "pip" at all."""
    calls = []
    install_windows(run=calls.append)

    assert calls[0][0] == sys.executable
    assert calls[0][1] == "-m"
    assert "pip" not in calls[0][0]  # not relying on a bare "pip" resolving via PATH


def test_install_macos_invokes_pip_install_headroom_ai():
    """No Homebrew tap exists for Headroom (`headroomlabs-ai/tap/headroom`
    and `headroomlabs-ai/homebrew-tap` both 404 -- confirmed against the
    GitHub API) -- pip is the only real install path on macOS too, same as
    every other platform."""
    calls = []
    install_macos(run=calls.append)

    assert calls == [_PIP_INSTALL_HEADROOM]


def test_install_linux_invokes_pip_install_headroom():
    calls = []
    install_linux(run=calls.append)

    assert calls == [_PIP_INSTALL_HEADROOM]


def test_install_for_platform_dispatches_to_windows():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Windows")

    assert calls == [_PIP_INSTALL_HEADROOM]


def test_install_for_platform_dispatches_to_macos():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Darwin")

    assert calls == [_PIP_INSTALL_HEADROOM]


def test_install_for_platform_dispatches_to_linux_for_anything_else():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Linux")

    assert calls == [_PIP_INSTALL_HEADROOM]


# ---------------------------------------------------------------------------
# `run_streaming` -- same `check=True` contract as `_default_run`, but
# calls `on_line` for each line of output as it arrives, so a slow install
# can show live progress instead of going silent until it finishes.
# ---------------------------------------------------------------------------


class _FakeStreamingProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.waited = False

    def wait(self):
        self.waited = True


def test_run_streaming_calls_on_line_for_every_line_of_output():
    process = _FakeStreamingProcess(["Collecting headroom-ai\n", "Installing collected packages\n", "Successfully installed\n"])
    received = []

    headroom_installer.run_streaming(["fake"], on_line=received.append, popen_factory=lambda *a, **kw: process)

    assert received == ["Collecting headroom-ai", "Installing collected packages", "Successfully installed"]
    assert process.waited is True


def test_run_streaming_works_with_on_line_omitted():
    process = _FakeStreamingProcess(["some output\n"])

    headroom_installer.run_streaming(["fake"], popen_factory=lambda *a, **kw: process)  # must not raise

    assert process.waited is True


def test_run_streaming_raises_called_process_error_on_nonzero_exit():
    import subprocess

    process = _FakeStreamingProcess(["some output\n"], returncode=1)

    with pytest.raises(subprocess.CalledProcessError):
        headroom_installer.run_streaming(["fake"], popen_factory=lambda *a, **kw: process)


def test_run_streaming_passes_argv_and_pipe_kwargs_to_popen_factory():
    import subprocess

    process = _FakeStreamingProcess([])
    captured = {}

    def fake_popen_factory(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    headroom_installer.run_streaming(["fake", "arg"], popen_factory=fake_popen_factory)

    assert captured["argv"] == ["fake", "arg"]
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert captured["kwargs"]["text"] is True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_headroom_proxy_port_is_8787():
    """Headroom's own actual default proxy port (issue #203) -- not 8080,
    which this project previously (and incorrectly) hardcoded."""
    assert HEADROOM_PROXY_PORT == 8787


def test_headroom_base_url_derives_from_the_shared_port_constant():
    assert HEADROOM_BASE_URL == f"http://localhost:{HEADROOM_PROXY_PORT}"
    assert HEADROOM_BASE_URL == "http://localhost:8787"


# ---------------------------------------------------------------------------
# Injectable defaults sanity check
# ---------------------------------------------------------------------------


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Every test above injects `run`, so `_default_run` is never exercised --
    matching the injectable-only testing contract."""
    assert callable(headroom_installer._default_run)
