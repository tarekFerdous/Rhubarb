import subprocess

import pytest

from rhubarb import caveman_installer
from rhubarb.caveman_installer import (
    PRESENCE_NOT_PRESENT,
    PRESENCE_PRESENT,
    check_caveman_presence,
    disable_caveman,
    install_caveman,
)


@pytest.fixture(autouse=True)
def _fixed_npx_command(monkeypatch):
    """Every test below (other than `resolve_npx_command`'s own dedicated
    tests further down, which override this themselves) asserts on the
    exact argv passed to `run`/`run_output` -- pin `shutil.which` so
    `resolve_npx_command()` resolves deterministically to a bare "npx"
    regardless of where `npx` actually happens to live on whatever machine
    runs this test."""
    monkeypatch.setattr(caveman_installer.shutil, "which", lambda name: "npx")


# ---------------------------------------------------------------------------
# `resolve_npx_command` -- Windows can't launch a bare `.cmd` shim like
# `npx` via a raw `CreateProcess` call, even when it's genuinely on `PATH`;
# it needs the full path including the `.cmd` extension. Different
# mechanism than the `PATH`-inheritance bug fixed for `winget`/`pip`/
# `headroom`, so it gets its own resolver rather than reusing theirs.
# ---------------------------------------------------------------------------


def test_resolve_npx_command_uses_shutil_which(monkeypatch):
    monkeypatch.setattr(caveman_installer.shutil, "which", lambda name: r"C:\Program Files\nodejs\npx.CMD")

    assert caveman_installer.resolve_npx_command() == r"C:\Program Files\nodejs\npx.CMD"


def test_resolve_npx_command_falls_back_to_bare_name_when_which_finds_nothing(monkeypatch):
    monkeypatch.setattr(caveman_installer.shutil, "which", lambda name: None)

    assert caveman_installer.resolve_npx_command() == "npx"


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


def test_presence_check_uses_the_resolved_npx_command_not_a_bare_name(monkeypatch):
    monkeypatch.setattr(caveman_installer, "resolve_npx_command", lambda: r"C:\Program Files\nodejs\npx.CMD")
    calls = []

    check_caveman_presence(run_output=lambda argv: (calls.append(argv), "caveman\n")[1])

    assert calls == [[r"C:\Program Files\nodejs\npx.CMD", "skills", "list"]]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def test_install_caveman_invokes_npx_skills_add():
    calls = []
    install_caveman(run=calls.append)

    assert calls == [
        ["npx", "skills", "add", "JuliusBrussee/caveman", "--skill", "caveman", "-a", "claude-code", "-g", "-y"]
    ]


def test_install_caveman_uses_global_flag():
    calls = []
    install_caveman(run=calls.append)

    assert "-g" in calls[0]


def test_install_caveman_is_fully_non_interactive():
    """Without `--skill`/`-y`/`-a`, `skills add` clones the repo and then
    drops into an interactive checkbox-picker wizard (confirmed in QA of
    issue #201: piping it produces garbled, repeated spinner-frame text
    and never actually installs anything) -- these three flags are what
    make this command able to run unattended at all, not optional tuning."""
    calls = []
    install_caveman(run=calls.append)

    argv = calls[0]
    assert "--skill" in argv and argv[argv.index("--skill") + 1] == "caveman"
    assert "-a" in argv and argv[argv.index("-a") + 1] == "claude-code"
    assert "-y" in argv


def test_install_caveman_uses_the_resolved_npx_command_not_a_bare_name(monkeypatch):
    """A bare `"npx"` fails on Windows even when it's genuinely on `PATH`
    -- `CreateProcess` can't launch a `.cmd` shim without the full path
    including its extension (see `resolve_npx_command`'s docstring)."""
    monkeypatch.setattr(caveman_installer, "resolve_npx_command", lambda: r"C:\Program Files\nodejs\npx.CMD")
    calls = []
    install_caveman(run=calls.append)

    assert calls[0][0] == r"C:\Program Files\nodejs\npx.CMD"


# ---------------------------------------------------------------------------
# `run_streaming` -- mirrors `headroom_installer.run_streaming` exactly:
# same `check=True` contract, but calls `on_line` for each line of output
# as it arrives, so a slow `npx` install can show live progress instead
# of going silent until it finishes.
# ---------------------------------------------------------------------------


class _FakeStreamingProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.waited = False

    def wait(self):
        self.waited = True


def test_run_streaming_calls_on_line_for_every_line_of_output():
    process = _FakeStreamingProcess(["npm warn ...\n", "added 12 packages\n"])
    received = []

    caveman_installer.run_streaming(["fake"], on_line=received.append, popen_factory=lambda *a, **kw: process)

    assert received == ["npm warn ...", "added 12 packages"]
    assert process.waited is True


def test_run_streaming_raises_called_process_error_on_nonzero_exit():
    process = _FakeStreamingProcess(["some output\n"], returncode=1)

    with pytest.raises(subprocess.CalledProcessError):
        caveman_installer.run_streaming(["fake"], popen_factory=lambda *a, **kw: process)


def test_run_streaming_passes_argv_and_pipe_kwargs_to_popen_factory():
    process = _FakeStreamingProcess([])
    captured = {}

    def fake_popen_factory(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return process

    caveman_installer.run_streaming(["fake", "arg"], popen_factory=fake_popen_factory)

    assert captured["argv"] == ["fake", "arg"]
    assert captured["kwargs"]["stdout"] == subprocess.PIPE
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert captured["kwargs"]["text"] is True


# ---------------------------------------------------------------------------
# Disable / remove
# ---------------------------------------------------------------------------


def test_disable_caveman_invokes_npx_skills_remove():
    calls = []
    disable_caveman(run=calls.append)

    assert calls == [["npx", "skills", "remove", "caveman", "-a", "claude-code", "-g", "-y"]]


def test_disable_caveman_targets_the_skill_name_not_the_repo_path():
    """`skills remove` looks up an already-installed skill by its own
    name ("caveman"), not the source repo path ("JuliusBrussee/caveman")
    `skills add` takes -- passing the repo path here (the old behavior)
    never matched anything installed, so it was a no-op at best."""
    calls = []
    disable_caveman(run=calls.append)

    assert calls[0][3] == "caveman"
    assert "JuliusBrussee/caveman" not in calls[0]


def test_disable_caveman_uses_global_flag():
    calls = []
    disable_caveman(run=calls.append)

    assert "-g" in calls[0]


def test_disable_caveman_skips_confirmation_prompt():
    calls = []
    disable_caveman(run=calls.append)

    assert "-y" in calls[0]


# ---------------------------------------------------------------------------
# Injectable defaults sanity check
# ---------------------------------------------------------------------------


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Every test above injects `run` or `run_output`, so `_default_run` and
    `_default_run_output` are never exercised -- matching the injectable-only
    testing contract."""
    assert callable(caveman_installer._default_run)
    assert callable(caveman_installer._default_run_output)
