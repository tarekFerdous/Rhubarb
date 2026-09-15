"""Caveman skill install and presence-detection automation (issue #201).

Pure functions with every external command call injectable, mirroring
`headroom_installer.py`'s injection pattern -- tests never require a real
Caveman install, network access, or a package manager.

Caveman is the `JuliusBrussee/caveman` Claude Code skill. Install:
  npx skills add JuliusBrussee/caveman -g
Presence check:
  npx skills list  (look for "caveman" in output)
Disable/remove:
  npx skills remove JuliusBrussee/caveman -g

Only the skill-only install path is ever used -- the proxy tier is never
installed by this feature. No proxy process to manage: once the skill is
installed globally via `npx`, it is available to any `claude` invocation
Rhubarb spawns without any additional per-spawn wiring.

This module has no UI or session-runner wiring of its own; the app-level
consent/install gate and the Settings toggle in `rhubarb/web/app.py` are
the callers.
"""

import subprocess

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_PRESENT = "present"


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def _default_run_output(argv: list[str]) -> str:
    """Run `argv` and return its stdout as a string, raising on non-zero exit."""
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    return result.stdout


def check_caveman_presence(*, run_output=None) -> str:
    """Check whether the Caveman skill is installed by running
    `npx skills list` and looking for "caveman" in the output.
    Returns `PRESENCE_PRESENT` if found, `PRESENCE_NOT_PRESENT` otherwise.

    `run_output` is injectable (a callable taking an argv list and returning
    the command's stdout as a string, or raising on failure) so tests never
    need a real Caveman install or network access."""
    run_output = run_output or _default_run_output
    try:
        output = run_output(["npx", "skills", "list"])
        return PRESENCE_PRESENT if "caveman" in output else PRESENCE_NOT_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_caveman(*, run=None) -> None:
    """Install the Caveman skill globally via
    `npx skills add JuliusBrussee/caveman -g`.
    `npx` is cross-platform, so the same command is used on every OS.

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real install or network access."""
    run = run or _default_run
    run(["npx", "skills", "add", "JuliusBrussee/caveman", "-g"])


def disable_caveman(*, run=None) -> None:
    """Remove the Caveman skill globally via
    `npx skills remove JuliusBrussee/caveman -g`. Called when the user
    turns off the Settings toggle.

    `run` is injectable so tests never need a real install or network access."""
    run = run or _default_run
    run(["npx", "skills", "remove", "JuliusBrussee/caveman", "-g"])
