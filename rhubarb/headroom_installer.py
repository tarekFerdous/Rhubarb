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

import os
import platform
import subprocess
import sys
import sysconfig

# Headroom's own actual default proxy port (confirmed from Headroom's own
# docs, `docs/content/docs/proxy.mdx` in `headroomlabs-ai/headroom`, which
# states the `--port` CLI option's default is 8787) -- NOT 8080, which this
# constant used to be (issue #203). This is the single shared source of
# truth for the port: `cli_client.py`'s `_HEADROOM_BASE_URL` (used to set
# `ANTHROPIC_BASE_URL` for spawned `claude` processes) imports it from here
# rather than hardcoding its own copy, and `rhubarb/web/app.py`'s
# `_start_headroom_proxy()` passes it explicitly via `--port` rather than
# relying on Headroom's own implicit default -- so the port Rhubarb assumes
# and the port the proxy actually binds to can never silently drift apart
# again.
HEADROOM_PROXY_PORT = 8787
HEADROOM_BASE_URL = f"http://localhost:{HEADROOM_PROXY_PORT}"

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_PRESENT = "present"

# The real PyPI package is `headroom-ai` (which ships the `headroom` CLI
# itself, per the project's own README quickstart: `pip install
# "headroom-ai[all]"`) -- NOT a package literally named `headroom`. A
# PyPI package named `headroom` does exist, but it's a completely
# unrelated tool by a different author/org -- installing it either fails
# to provide a `headroom` command at all, or silently installs the wrong
# CLI under that name, neither of which is this project's Headroom. `[all]`
# is the extras group the quickstart itself recommends, pulling in the
# full compression stack (not just the bare library).
#
# `[sys.executable, "-m", "pip", ...]` rather than a bare `"pip"`: a bare
# command name is resolved via `PATH` by the OS loader, and Rhubarb's own
# backend process doesn't reliably inherit the same `PATH` an interactive
# shell has (see `resolve_headroom_command`/`ollama_installer._resolve_
# winget_command`'s docstrings for the same class of bug with `headroom`/
# `winget`, found in QA of issue #113/#115) -- `pip`'s console-script
# `.exe`/shim can live in a Scripts directory that's simply not on that
# reduced `PATH`, so a bare `["pip", ...]` fails with `WinError 2` even
# though pip is genuinely installed and works fine when typed into a
# terminal. `sys.executable` is always an absolute path to the exact
# interpreter already running this code, so routing through `-m pip` needs
# no `PATH` lookup for "pip" at all.
#
# There is no Homebrew tap for Headroom (`headroomlabs-ai/tap/headroom`/
# `headroomlabs-ai/homebrew-tap` don't exist -- confirmed 404 against the
# GitHub API); the project's README only lists `pip install`, `uv tool
# install`, and `npm install` (the npm package is a TypeScript-only SDK
# with no CLI) as install methods. So every platform installs the same
# way -- no brew branch, nothing OS-specific beyond the executable's file
# extension (handled in `resolve_headroom_command`).
_PIP_INSTALL_HEADROOM = [sys.executable, "-m", "pip", "install", "headroom-ai[all]"]


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def run_streaming(argv: list[str], *, on_line=None, popen_factory=None) -> None:
    """Same `subprocess.run(argv, check=True)` contract as `_default_run`
    (raises `subprocess.CalledProcessError` on a non-zero exit), but calls
    `on_line(line)` for each line of merged stdout/stderr AS IT ARRIVES,
    not only after the process exits -- so a caller (the app-level install
    endpoint) can surface live output while a slow install (real network
    downloads, wheel builds) is still running, instead of the UI going
    silent/"stuck" until the whole thing finishes.

    `popen_factory` is injectable (a callable with `subprocess.Popen`'s
    signature) so tests never need a real subprocess.

    `encoding="utf-8", errors="replace"` explicitly: `text=True` alone
    leaves Python to decode the subprocess's output bytes using
    `locale.getpreferredencoding()`, which on Windows is the legacy ANSI
    codepage ("charmap"/cp1252) -- not UTF-8. Modern `pip`/`npm` output
    routinely contains UTF-8 characters (progress-bar glyphs, checkmarks,
    smart quotes) that aren't valid in that codepage, so decoding raises
    mid-stream (`'charmap' codec can't decode byte ...`) and the whole
    install is reported as failed even though the underlying command may
    have been working fine. `errors="replace"` additionally means one
    genuinely undecodable byte swaps in a placeholder character instead
    of crashing the entire streamed read."""
    popen_factory = popen_factory or subprocess.Popen
    process = popen_factory(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1
    )
    for line in process.stdout:
        if on_line:
            on_line(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, argv)


_USER_SCHEME = "nt_user" if os.name == "nt" else "posix_user"


def resolve_headroom_command() -> str:
    """`headroom`'s console-script entry point is installed by
    `python -m pip install "headroom-ai[all]"` into a scripts directory
    next to wherever pip decided to install the package -- which isn't
    reliably on this backend process's `PATH` (the exact same class of bug
    already found, and fixed the same way, for `winget` in
    `ollama_installer._resolve_winget_command`): a bare `"headroom"`
    invocation can fail with `WinError 2` even though pip installed it
    successfully seconds earlier.

    There are two candidate scripts directories, checked in order, because
    pip itself picks between them silently:
    1. `sysconfig.get_path("scripts")` -- the interpreter's own scheme
       (e.g. `...\\Scripts\\headroom.exe` on Windows, `.../bin/headroom`
       on macOS/Linux). This is where pip installs when it has write
       access to the interpreter's own site-packages.
    2. `sysconfig.get_path("scripts", _USER_SCHEME)` -- the per-user
       scheme (e.g. `%APPDATA%\\Python\\PythonXY\\Scripts` on Windows).
       pip silently falls back to a `--user` install here (with only a
       warning, not a failure) whenever it lacks write access to the
       interpreter's own site-packages -- common for a non-admin user
       against a machine-wide Python install. Checking only candidate 1
       misses exactly this case: the install genuinely succeeds (so it's
       not a `CalledProcessError`), but the executable ends up here
       instead, and a bare `"headroom"` -- or a lookup that only checks
       candidate 1 -- still can't find it.

    Falls back to the bare command name when it's not found at either --
    e.g. installed some other way (`uv tool install`, manually) and
    already on `PATH`."""
    exe_name = "headroom.exe" if platform.system() == "Windows" else "headroom"
    candidates = [sysconfig.get_path("scripts")]
    try:
        candidates.append(sysconfig.get_path("scripts", _USER_SCHEME))
    except KeyError:
        pass  # scheme unavailable on this platform/build -- system scheme only
    for scripts_dir in candidates:
        candidate = os.path.join(scripts_dir, exe_name)
        if os.path.exists(candidate):
            return candidate
    return "headroom"


def check_headroom_presence(*, run=None) -> str:
    """Check whether the `headroom` CLI is present by running
    `headroom --version` -- returns `PRESENCE_PRESENT` if the command
    succeeds (exit 0), `PRESENCE_NOT_PRESENT` otherwise.

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real Headroom install."""
    run = run or _default_run
    try:
        run([resolve_headroom_command(), "--version"])
        return PRESENCE_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_windows(*, run=None) -> None:
    """Install Headroom via `python -m pip install "headroom-ai[all]"` on
    Windows."""
    run = run or _default_run
    run(_PIP_INSTALL_HEADROOM)


def install_macos(*, run=None) -> None:
    """Install Headroom via `python -m pip install "headroom-ai[all]"` on
    macOS. No Homebrew tap exists for Headroom -- pip is the only real
    install path here, same as every other platform."""
    run = run or _default_run
    run(_PIP_INSTALL_HEADROOM)


def install_linux(*, run=None) -> None:
    """Install Headroom via `python -m pip install "headroom-ai[all]"` on
    Linux."""
    run = run or _default_run
    run(_PIP_INSTALL_HEADROOM)


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
