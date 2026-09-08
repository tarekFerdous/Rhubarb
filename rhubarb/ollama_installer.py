"""Per-OS Ollama install and presence-detection automation (issue #113).

Pure functions with every external command/HTTP call injectable, mirroring
`pty_engine.PtyEngine`'s `pty_factory` injection pattern -- tests never
require a real Ollama install, network access, or a package manager.

This module has no UI or `session_runner.py` wiring of its own; issue #114
(the rescue-call module) and issue #115 (the app-level consent/install gate)
are the callers.
"""

import json
import os
import platform
import subprocess
import urllib.request

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_WITHOUT_MODEL = "present_without_model"
PRESENCE_WITH_MODEL = "present_with_model"


def _default_http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama_presence(*, http_get=None) -> str:
    """Check whether Ollama is running locally and whether `OLLAMA_MODEL`
    has already been pulled, via Ollama's own local HTTP API
    (`GET /api/tags`) -- not by shelling out and parsing CLI output.
    Returns one of `PRESENCE_NOT_PRESENT`/`PRESENCE_WITHOUT_MODEL`/
    `PRESENCE_WITH_MODEL`.

    `http_get` is injectable (a callable taking a URL and returning the
    parsed JSON body as a dict, or raising on failure) so tests never need
    a real Ollama install or network access."""
    http_get = http_get or _default_http_get
    try:
        data = http_get(f"{OLLAMA_BASE_URL}/api/tags")
    except Exception:
        return PRESENCE_NOT_PRESENT

    models = [m.get("name") for m in data.get("models", [])]
    return PRESENCE_WITH_MODEL if OLLAMA_MODEL in models else PRESENCE_WITHOUT_MODEL


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _default_brew_available(run) -> bool:
    try:
        run(["brew", "--version"])
        return True
    except Exception:
        return False


def _resolve_winget_command() -> str:
    """`winget` lives in a Windows "app execution alias" (a stub under
    `%LOCALAPPDATA%\\Microsoft\\WindowsApps`) that's only on `PATH` when the
    *parent* process inherited a normal interactive user session's
    environment. A backend process started some other way (a different
    launcher, a service) can have a `PATH` missing that directory, making a
    bare `"winget"` invocation fail with `WinError 2` even though winget is
    genuinely installed (found in QA of issue #113/#115). Resolving the
    well-known full path directly sidesteps that PATH-inheritance quirk
    entirely; falls back to the bare command name if that path doesn't
    exist (e.g. a portable/manual install elsewhere on PATH)."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = os.path.join(local_app_data, "Microsoft", "WindowsApps", "winget.exe")
        if os.path.exists(candidate):
            return candidate
    return "winget"


def install_windows(*, run=None) -> None:
    """Silently install Ollama via `winget`."""
    run = run or _default_run
    run([_resolve_winget_command(), "install", "--id", "Ollama.Ollama", "-e", "--silent"])


def _install_macos_pkg(*, run) -> None:
    """Download and silently run Ollama's official `.pkg` installer --
    the fallback when Homebrew isn't present."""
    run(["curl", "-fsSL", "-o", "/tmp/Ollama.pkg", "https://ollama.com/download/Ollama.pkg"])
    run(["installer", "-pkg", "/tmp/Ollama.pkg", "-target", "/"])


def install_macos(*, run=None, brew_available=None) -> None:
    """Install Ollama via Homebrew if it's present on the machine, falling
    back to the official `.pkg` installer if it isn't."""
    run = run or _default_run
    brew_available = brew_available if brew_available is not None else _default_brew_available
    if brew_available(run):
        run(["brew", "install", "ollama"])
    else:
        _install_macos_pkg(run=run)


def install_linux(*, run=None) -> None:
    """Install Ollama via the official, cross-distro install script."""
    run = run or _default_run
    run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])


def install_for_platform(*, run=None, system=None) -> None:
    """Dispatch to the right per-OS install function for the current
    platform -- mirrors `pty_engine._default_pty_factory`'s OS-dispatch
    pattern. `system` is injectable (a zero-arg callable returning a
    `platform.system()`-shaped string) so tests can force a specific OS
    without actually running on it."""
    resolved_system = system() if system else platform.system()
    if resolved_system == "Windows":
        install_windows(run=run)
    elif resolved_system == "Darwin":
        install_macos(run=run)
    else:
        install_linux(run=run)


def pull_model(*, run=None) -> None:
    """Pull `OLLAMA_MODEL` via the `ollama` CLI. Safe to call on its own
    when Ollama is already installed but the model isn't (`PRESENCE_WITHOUT_MODEL`)."""
    run = run or _default_run
    run(["ollama", "pull", OLLAMA_MODEL])


def install_and_pull_model(*, run=None, system=None) -> None:
    """Full install flow for a completely missing Ollama
    (`PRESENCE_NOT_PRESENT`): install for the current platform, then pull
    `OLLAMA_MODEL`. Callers that already know Ollama is present without the
    model (`PRESENCE_WITHOUT_MODEL`) should call `pull_model()` directly
    instead, to skip the OS-level install step."""
    install_for_platform(run=run, system=system)
    pull_model(run=run)
