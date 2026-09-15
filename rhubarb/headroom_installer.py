"""Per-OS Headroom install and presence-detection automation (issue #200).

Pure functions with every external command call injectable, mirroring
`ollama_installer.py`'s injection pattern -- tests never require a real
Headroom install, network access, or a package manager.

This module has no UI or session-runner wiring of its own; the app-level
consent/install gate and the `headroom proxy` process management in
`rhubarb/web/app.py` are the callers.

Headroom is from `headroomlabs-ai/headroom`. It shells out to Headroom's
own official install paths and CLI -- it does not reimplement Headroom's
compression logic.
"""

import platform
import subprocess

HEADROOM_BASE_URL = "http://localhost:8080"

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_PRESENT = "present"


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def _default_brew_available(run) -> bool:
    try:
        run(["brew", "--version"])
        return True
    except Exception:
        return False


def check_headroom_presence(*, run=None) -> str:
    """Check whether the `headroom` CLI is present by running
    `headroom --version` -- returns `PRESENCE_PRESENT` if the command
    succeeds (exit 0), `PRESENCE_NOT_PRESENT` otherwise.

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real Headroom install."""
    run = run or _default_run
    try:
        run(["headroom", "--version"])
        return PRESENCE_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_windows(*, run=None) -> None:
    """Install Headroom via `pip install headroom` on Windows."""
    run = run or _default_run
    run(["pip", "install", "headroom"])


def install_macos(*, run=None, brew_available=None) -> None:
    """Install Headroom via Homebrew if it's present on the machine,
    falling back to `pip install headroom` if it isn't."""
    run = run or _default_run
    brew_available = brew_available if brew_available is not None else _default_brew_available
    if brew_available(run):
        run(["brew", "install", "headroomlabs-ai/tap/headroom"])
    else:
        run(["pip", "install", "headroom"])


def install_linux(*, run=None) -> None:
    """Install Headroom via `pip install headroom` on Linux."""
    run = run or _default_run
    run(["pip", "install", "headroom"])


def install_for_platform(*, run=None, system=None) -> None:
    """Dispatch to the right per-OS install function for the current
    platform. `system` is injectable (a zero-arg callable returning a
    `platform.system()`-shaped string) so tests can force a specific OS
    without actually running on it."""
    resolved_system = system() if system else platform.system()
    if resolved_system == "Windows":
        install_windows(run=run)
    elif resolved_system == "Darwin":
        install_macos(run=run)
    else:
        install_linux(run=run)
