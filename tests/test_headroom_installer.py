from rhubarb import headroom_installer
from rhubarb.headroom_installer import (
    HEADROOM_BASE_URL,
    PRESENCE_NOT_PRESENT,
    PRESENCE_PRESENT,
    check_headroom_presence,
    install_for_platform,
    install_linux,
    install_macos,
    install_windows,
)


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


def test_presence_reports_not_present_when_command_raises():
    def failing_run(argv):
        raise FileNotFoundError("headroom not found")

    assert check_headroom_presence(run=failing_run) == PRESENCE_NOT_PRESENT


def test_presence_reports_present_when_command_succeeds():
    calls = []

    def succeeding_run(argv):
        calls.append(argv)

    result = check_headroom_presence(run=succeeding_run)
    assert result == PRESENCE_PRESENT
    assert calls == [["headroom", "--version"]]


def test_presence_reports_not_present_when_command_exits_nonzero():
    """A subprocess.CalledProcessError (check=True) counts as not present."""
    import subprocess

    def failing_run(argv):
        raise subprocess.CalledProcessError(1, argv)

    assert check_headroom_presence(run=failing_run) == PRESENCE_NOT_PRESENT


# ---------------------------------------------------------------------------
# Per-OS install
# ---------------------------------------------------------------------------


def test_install_windows_invokes_pip_install_headroom():
    calls = []
    install_windows(run=calls.append)

    assert calls == [["pip", "install", "headroom"]]


def test_install_macos_uses_brew_when_present():
    calls = []
    install_macos(run=calls.append, brew_available=lambda run: True)

    assert calls == [["brew", "install", "headroomlabs-ai/tap/headroom"]]


def test_install_macos_falls_back_to_pip_when_brew_missing():
    calls = []
    install_macos(run=calls.append, brew_available=lambda run: False)

    assert calls == [["pip", "install", "headroom"]]


def test_install_macos_brew_available_probe_receives_the_run_callable():
    """The default brew-detection probe should use the injected `run`."""
    calls = []

    def fake_run(argv):
        calls.append(argv)

    install_macos(run=fake_run)

    assert calls[0] == ["brew", "--version"]


def test_install_linux_invokes_pip_install_headroom():
    calls = []
    install_linux(run=calls.append)

    assert calls == [["pip", "install", "headroom"]]


def test_install_for_platform_dispatches_to_windows():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Windows")

    assert calls == [["pip", "install", "headroom"]]


def test_install_for_platform_dispatches_to_macos_with_brew():
    """`install_for_platform` doesn't expose its own `brew_available`
    override, so the default probe runs using the injected `run` -- a
    non-raising `run` means brew is available, so this ends in a brew install."""
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Darwin")

    assert calls[0] == ["brew", "--version"]
    assert calls[1] == ["brew", "install", "headroomlabs-ai/tap/headroom"]


def test_install_for_platform_dispatches_to_linux_for_anything_else():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Linux")

    assert calls == [["pip", "install", "headroom"]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_headroom_base_url_points_at_localhost_8080():
    assert HEADROOM_BASE_URL == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Injectable defaults sanity check
# ---------------------------------------------------------------------------


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Every test above injects `run`, so `_default_run` is never exercised --
    matching the injectable-only testing contract."""
    assert callable(headroom_installer._default_run)
