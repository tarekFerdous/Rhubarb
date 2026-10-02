"""Per-OS lean-ctx install/presence automation and scoped-config generation
(issue #233, child of PRD #232).

Pure functions with every external command call injectable, mirroring
`headroom_installer.py`/`caveman_installer.py`'s injection pattern -- tests
never require a real lean-ctx install, network access, or npm.

lean-ctx is the `lean-ctx-bin` npm package. Rhubarb installs its own local
copy under `~/.rhubarb/` (issue #252, child of PRD #251) rather than relying
on a global `npm install -g` one, and points every lean-ctx process it causes
at its own data/config dirs -- so Rhubarb's savings ledger never mixes with
the user's interactive lean-ctx usage and the user's personal lean-ctx config
never changes how unattended Rhubarb sessions behave.
This module has no UI or session-runner wiring of its own; the app-level
consent/install gate and the Settings toggle in `rhubarb/web/app.py` are
the callers.
"""

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_PRESENT = "present"

# The two Rhubarb-owned, self-contained config files passed to every `claude`
# subprocess Rhubarb spawns (via `--mcp-config`/`--settings`, see
# `cli_client._lean_ctx_args`) once lean-ctx is enabled. Written flat under
# `~/.rhubarb/` -- never under the user's own `~/.claude.json`/
# `~/.claude/CLAUDE.md`, since Rhubarb deliberately never runs lean-ctx's own
# `init --agent claude` (that mutates the user's global Claude Code config;
# only Rhubarb's own subprocess invocations are meant to be affected, same
# precedent Headroom already set for `ANTHROPIC_BASE_URL`).
LEAN_CTX_MCP_CONFIG_PATH = Path.home() / ".rhubarb" / "lean-ctx-mcp-config.json"
LEAN_CTX_HOOKS_SETTINGS_PATH = Path.home() / ".rhubarb" / "lean-ctx-hooks-settings.json"

# Rhubarb's own npm prefix for its local lean-ctx install, plus the data dir
# (ledger, cache, knowledge) and config dir lean-ctx is pointed at via its own
# `LEAN_CTX_DATA_DIR`/`LEAN_CTX_CONFIG_DIR` env vars. The config dir starts
# empty so lean-ctx's own defaults apply. A plain (non-`-g`) `npm install
# --prefix` gives the same `node_modules/.bin/` layout on every OS.
LEAN_CTX_PREFIX = Path.home() / ".rhubarb" / "lean-ctx"
LEAN_CTX_DATA_DIR = Path.home() / ".rhubarb" / "lean-ctx-data"
LEAN_CTX_CONFIG_DIR = Path.home() / ".rhubarb" / "lean-ctx-config"

_NPM_INSTALL_LEAN_CTX_ARGS = ["install", "--prefix", str(LEAN_CTX_PREFIX), "lean-ctx-bin"]

# The tool-name matcher lean-ctx's own hooks policy uses to redirect a
# native read/search/list call to its own `ctx_*` tools instead -- every
# spelling variant a client might name that family of tool (Read/Grep/Glob
# and their lowercase/aliased forms), confirmed against a real installed
# lean-ctx's own generated hooks policy.
_LEAN_CTX_READ_SEARCH_MATCHER = (
    "Read|read|ReadFile|read_file|View|view|Grep|grep|Search|search|"
    "ListFiles|list_files|ListDirectory|list_directory|Glob|glob"
)


def lean_ctx_env() -> dict:
    """The two env vars that point a lean-ctx process at Rhubarb's own data
    and config dirs -- applied to every spawned `claude` subprocess while
    lean-ctx is enabled (so its hooks and MCP server inherit them), to the
    generated MCP config's server entry, and to every lean-ctx command the
    backend runs itself."""
    return {"LEAN_CTX_DATA_DIR": str(LEAN_CTX_DATA_DIR), "LEAN_CTX_CONFIG_DIR": str(LEAN_CTX_CONFIG_DIR)}


def _install_env() -> dict:
    """`lean-ctx-bin`'s npm postinstall runs `lean-ctx onboard` unless
    `LEAN_CTX_NO_ONBOARD=1` -- which rewrites the user's *global*
    `~/.claude/settings.json` hooks, `~/.claude.json` MCP entry and shell
    hook to point at whichever binary was just installed (confirmed live
    against lean-ctx 3.10.5). Rhubarb must never touch the user's global
    setup, so every install it runs suppresses onboarding."""
    return {**os.environ, **lean_ctx_env(), "LEAN_CTX_NO_ONBOARD": "1"}


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, env=_install_env())


def run_streaming(argv: list[str], *, on_line=None, popen_factory=None) -> None:
    """Same `subprocess.run(argv, check=True)` contract as `_default_run`,
    but calls `on_line(line)` for each line of merged stdout/stderr AS IT
    ARRIVES -- mirrors `headroom_installer.run_streaming`/`caveman_
    installer.run_streaming` exactly, including the explicit
    `encoding="utf-8", errors="replace"` (an `npm install` can emit UTF-8
    progress glyphs that Windows' default ANSI codepage can't decode), plus
    `_install_env()` so the postinstall never onboards the user's global
    Claude Code config."""
    popen_factory = popen_factory or subprocess.Popen
    process = popen_factory(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_install_env(),
    )
    for line in process.stdout:
        if on_line:
            on_line(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, argv)


def resolve_npm_command() -> str:
    """`npm` (like `npx`) is a `.cmd` batch-file shim on Windows -- a bare
    `"npm"` invocation fails with `WinError 2` under `CreateProcess`
    (`subprocess.run(..., shell=False)`) even when it's genuinely on `PATH`,
    the same mechanism already fixed for `npx` in
    `caveman_installer.resolve_npx_command`. `shutil.which` does its own
    real `PATH` search and returns the full path including the `.CMD`
    extension, so this needs no per-platform branching -- on macOS/Linux
    `npm` is a plain executable and this just returns its resolved path
    unchanged. Falls back to the bare name if `npm` isn't found on `PATH`
    at all."""
    return shutil.which("npm") or "npm"


def resolve_lean_ctx_command(*, system=None) -> str:
    """The `lean-ctx` binary inside Rhubarb's own local prefix -- never a
    `PATH` lookup, so a global-only lean-ctx is never picked up. npm puts it
    in `node_modules/.bin/`, as a `.cmd` shim on Windows (a bare name would
    hit the same `WinError 2` `CreateProcess` limitation `resolve_npm_
    command` documents). The path is returned whether or not the install
    has happened yet; `check_lean_ctx_presence` is what tells the two apart.
    `system` is injectable like `install_for_platform`'s."""
    resolved_system = system() if system else platform.system()
    name = "lean-ctx.cmd" if resolved_system == "Windows" else "lean-ctx"
    return str(LEAN_CTX_PREFIX / "node_modules" / ".bin" / name)


def check_lean_ctx_presence(*, run=None) -> str:
    """Check whether Rhubarb's local `lean-ctx` is present by running
    its `--version` (a global-only install therefore reports not present)
    -- returns `PRESENCE_PRESENT` if the command succeeds (exit 0),
    `PRESENCE_NOT_PRESENT` otherwise (including a not-yet-installed local
    binary, which raises `FileNotFoundError`).

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real lean-ctx install."""
    run = run or _default_run
    try:
        run([resolve_lean_ctx_command(), "--version"])
        return PRESENCE_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_windows(*, run=None) -> None:
    """Install lean-ctx into Rhubarb's local prefix on Windows."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_macos(*, run=None) -> None:
    """Install lean-ctx into Rhubarb's local prefix on macOS."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_linux(*, run=None) -> None:
    """Install lean-ctx into Rhubarb's local prefix on Linux."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_for_platform(*, run=None, system=None) -> None:
    """Dispatch to the right per-OS install function for the current
    platform -- all three run the identical `npm install --prefix ...`,
    so this only exists to mirror `headroom_installer.install_for_platform`/
    `ollama_installer.install_for_platform`'s shape. `system` is injectable
    (a zero-arg callable returning a `platform.system()`-shaped string) so
    tests can force a specific OS without actually running on it."""
    resolved_system = system() if system else platform.system()
    if resolved_system == "Windows":
        install_windows(run=run)
    elif resolved_system == "Darwin":
        install_macos(run=run)
    else:
        install_linux(run=run)


def _build_hooks_settings(command: str) -> dict:
    """The actual Claude Code hooks policy that makes lean-ctx's compression
    real rather than merely available (PRD #232's user story #14): every
    native read/search/list-shaped tool call is redirected to lean-ctx's own
    `ctx_*` tools (`PreToolUse` "redirect"), every Bash/PowerShell call is
    rewritten for lean-ctx's shell-compression path (`PreToolUse` "rewrite"),
    a `Read` result is deduplicated against lean-ctx's own cache
    (`PostToolUse` "read-dedup"), and every lifecycle event is observed so
    lean-ctx can compute the savings the (future) widget/modal reads back
    via `lean-ctx gain --json`. `Grep`/`Glob` are additionally denied
    outright via `permissions.deny` -- confirmed against a real installed
    lean-ctx's own generated policy shape."""
    observe_hook = {"matcher": ".*", "hooks": [{"type": "command", "command": f"{command} hook observe"}]}
    return {
        "permissions": {"deny": ["Grep", "Glob"]},
        "hooks": {
            "SessionStart": [observe_hook],
            "SessionEnd": [observe_hook],
            "Stop": [observe_hook],
            "UserPromptSubmit": [observe_hook],
            "PreCompact": [observe_hook],
            "PreToolUse": [
                {
                    "matcher": "Bash|bash|PowerShell|powershell",
                    "hooks": [{"type": "command", "command": f"{command} hook rewrite"}],
                },
                {
                    "matcher": _LEAN_CTX_READ_SEARCH_MATCHER,
                    "hooks": [{"type": "command", "command": f"{command} hook redirect"}],
                },
            ],
            "PostToolUse": [
                observe_hook,
                {"matcher": "Read", "hooks": [{"type": "command", "command": f"{command} hook read-dedup"}]},
            ],
        },
    }


def generate_scoped_config() -> None:
    """Write `LEAN_CTX_MCP_CONFIG_PATH` (an MCP-server registration pointing
    at Rhubarb's local `lean-ctx` command, with an explicit `env` block for
    its data/config dirs so isolation never depends on env inheritance
    alone) and `LEAN_CTX_HOOKS_SETTINGS_PATH` (the hooks policy above) under
    `~/.rhubarb/`, regenerating both every time lean-ctx transitions to
    enabled -- never merely checked for existence, so a reinstalled/
    relocated `lean-ctx` command is always reflected in what gets passed to
    the next spawned `claude` subprocess.

    The hooks get the forward-slash form of the path: Claude Code runs hook
    commands through a shell (Git Bash on Windows), which would eat an
    unquoted backslash path -- the same form lean-ctx's own onboarding
    writes."""
    command = resolve_lean_ctx_command()
    LEAN_CTX_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEAN_CTX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEAN_CTX_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    mcp_config = {"mcpServers": {"lean-ctx": {"command": command, "env": lean_ctx_env()}}}
    LEAN_CTX_MCP_CONFIG_PATH.write_text(json.dumps(mcp_config, indent=2))
    hooks_settings = _build_hooks_settings(Path(command).as_posix())
    LEAN_CTX_HOOKS_SETTINGS_PATH.write_text(json.dumps(hooks_settings, indent=2))
