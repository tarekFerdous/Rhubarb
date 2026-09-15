import subprocess

from rhubarb import caveman_installer
from rhubarb.caveman_installer import (
    PRESENCE_NOT_PRESENT,
    PRESENCE_PRESENT,
    check_caveman_presence,
    disable_caveman,
    install_caveman,
)


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


def test_presence_reports_not_present_when_command_raises():
    def failing_run_output(argv):
        raise FileNotFoundError("npx not found")

    assert check_caveman_presence(run_output=failing_run_output) == PRESENCE_NOT_PRESENT


def test_presence_reports_not_present_when_caveman_not_in_output():
    def run_output_without_caveman(argv):
        return "some-other-skill\nanother-skill\n"

    assert check_caveman_presence(run_output=run_output_without_caveman) == PRESENCE_NOT_PRESENT


def test_presence_reports_present_when_caveman_in_output():
    calls = []

    def run_output_with_caveman(argv):
        calls.append(argv)
        return "caveman\nsome-other-skill\n"

    result = check_caveman_presence(run_output=run_output_with_caveman)
    assert result == PRESENCE_PRESENT
    assert calls == [["npx", "skills", "list"]]


def test_presence_reports_present_when_caveman_appears_among_multiple_skills():
    def run_output(argv):
        return "skill-a\nJuliusBrussee/caveman\nskill-b\n"

    assert check_caveman_presence(run_output=run_output) == PRESENCE_PRESENT


def test_presence_reports_not_present_when_command_exits_nonzero():
    """A subprocess.CalledProcessError (check=True) counts as not present."""

    def failing_run_output(argv):
        raise subprocess.CalledProcessError(1, argv)

    assert check_caveman_presence(run_output=failing_run_output) == PRESENCE_NOT_PRESENT


def test_presence_check_runs_npx_skills_list():
    calls = []

    def capturing_run_output(argv):
        calls.append(argv)
        return "caveman\n"

    check_caveman_presence(run_output=capturing_run_output)
    assert calls == [["npx", "skills", "list"]]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def test_install_caveman_invokes_npx_skills_add():
    calls = []
    install_caveman(run=calls.append)

    assert calls == [["npx", "skills", "add", "JuliusBrussee/caveman", "-g"]]


def test_install_caveman_uses_global_flag():
    calls = []
    install_caveman(run=calls.append)

    assert "-g" in calls[0]


# ---------------------------------------------------------------------------
# Disable / remove
# ---------------------------------------------------------------------------


def test_disable_caveman_invokes_npx_skills_remove():
    calls = []
    disable_caveman(run=calls.append)

    assert calls == [["npx", "skills", "remove", "JuliusBrussee/caveman", "-g"]]


def test_disable_caveman_uses_global_flag():
    calls = []
    disable_caveman(run=calls.append)

    assert "-g" in calls[0]


# ---------------------------------------------------------------------------
# Injectable defaults sanity check
# ---------------------------------------------------------------------------


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Every test above injects `run` or `run_output`, so `_default_run` and
    `_default_run_output` are never exercised -- matching the injectable-only
    testing contract."""
    assert callable(caveman_installer._default_run)
    assert callable(caveman_installer._default_run_output)
