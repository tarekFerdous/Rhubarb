"""Per-OS lean-ctx install/presence automation and scoped-config generation
(issue #233, child of PRD #232).

Pure functions with every external command call injectable, mirroring
`headroom_installer.py`/`caveman_installer.py`'s injection pattern -- tests
never require a real lean-ctx install, network access, or npm.

lean-ctx is the `lean-ctx-bin` npm package (`npm install -g lean-ctx-bin`).
This module has no UI or session-runner wiring of its own; the app-level
consent/install gate and the Settings toggle in `rhubarb/web/app.py` are
the callers.
"""

import json
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

_NPM_INSTALL_LEAN_CTX_ARGS = ["install", "-g", "lean-ctx-bin"]

# The tool-name matcher lean-ctx's own hooks policy uses to redirect a
# native read/search/list call to its own `ctx_*` tools instead -- every
# spelling variant a client might name that family of tool (Read/Grep/Glob
# and their lowercase/aliased forms), confirmed against a real installed
# lean-ctx's own generated hooks policy.
_LEAN_CTX_READ_SEARCH_MATCHER = (
    "Read|read|ReadFile|read_file|View|view|Grep|grep|Search|search|"
    "ListFiles|list_files|ListDirectory|list_directory|Glob|glob"
)


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def run_streaming(argv: list[str], *, on_line=None, popen_factory=None) -> None:
    """Same `subprocess.run(argv, check=True)` contract as `_default_run`,
    but calls `on_line(line)` for each line of merged stdout/stderr AS IT
    ARRIVES -- mirrors `headroom_installer.run_streaming`/`caveman_
    installer.run_streaming` exactly, including the explicit
    `encoding="utf-8", errors="replace"` (an `npm install` can emit UTF-8
    progress glyphs that Windows' default ANSI codepage can't decode)."""
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


def resolve_lean_ctx_command() -> str:
    """`lean-ctx` is installed by `npm install -g lean-ctx-bin` as a
    `.cmd` shim on Windows (npm's standard global-install layout, same
    shape as `npx` itself) -- a bare `"lean-ctx"` invocation hits the exact
    same `WinError 2` `CreateProcess` limitation `resolve_npm_command`/
    `caveman_installer.resolve_npx_command` already document. `shutil.which`
    resolves it via a real `PATH` search, extension included, with no
    per-platform branching needed. Falls back to the bare command name when
    it's not found on `PATH` at all (e.g. this check runs before the very
    first install completes)."""
    return shutil.which("lean-ctx") or "lean-ctx"


def check_lean_ctx_presence(*, run=None) -> str:
    """Check whether the `lean-ctx` CLI is present by running
    `lean-ctx --version` -- returns `PRESENCE_PRESENT` if the command
    succeeds (exit 0), `PRESENCE_NOT_PRESENT` otherwise (including a
    missing command, since a nonexistent bare fallback name raises
    `FileNotFoundError`).

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real lean-ctx install."""
    run = run or _default_run
    try:
        run([resolve_lean_ctx_command(), "--version"])
        return PRESENCE_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_windows(*, run=None) -> None:
    """Install lean-ctx via `npm install -g lean-ctx-bin` on Windows."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_macos(*, run=None) -> None:
    """Install lean-ctx via `npm install -g lean-ctx-bin` on macOS."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_linux(*, run=None) -> None:
    """Install lean-ctx via `npm install -g lean-ctx-bin` on Linux."""
    run = run or _default_run
    run([resolve_npm_command(), *_NPM_INSTALL_LEAN_CTX_ARGS])


def install_for_platform(*, run=None, system=None) -> None:
    """Dispatch to the right per-OS install function for the current
    platform -- all three run the identical `npm install -g lean-ctx-bin`,
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
    at the resolved `lean-ctx` command) and `LEAN_CTX_HOOKS_SETTINGS_PATH`
    (the hooks policy above) under `~/.rhubarb/`, regenerating both every
    time lean-ctx transitions to enabled -- never merely checked for
    existence, so a reinstalled/relocated `lean-ctx` command is always
    reflected in what gets passed to the next spawned `claude` subprocess."""
    command = resolve_lean_ctx_command()
    LEAN_CTX_MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    mcp_config = {"mcpServers": {"lean-ctx": {"command": command}}}
    LEAN_CTX_MCP_CONFIG_PATH.write_text(json.dumps(mcp_config, indent=2))
    LEAN_CTX_HOOKS_SETTINGS_PATH.write_text(json.dumps(_build_hooks_settings(command), indent=2))
