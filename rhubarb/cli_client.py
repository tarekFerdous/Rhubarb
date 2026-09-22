"""Thin wrapper around the `claude` CLI, run as a subprocess.

Deliberately does not use the Claude Agent SDK: the SDK requires an API key
and always bills pay-per-token, while spawning the CLI directly can run
under the user's Claude subscription login instead. See CLAUDE.md.
"""

import json
import os
import platform
import subprocess
from pathlib import Path

from rhubarb.headroom_installer import HEADROOM_BASE_URL as _HEADROOM_BASE_URL
from rhubarb.lean_ctx_installer import LEAN_CTX_HOOKS_SETTINGS_PATH, LEAN_CTX_MCP_CONFIG_PATH


class ClaudeCLIError(RuntimeError):
    pass


# Rhubarb's own private, plugin-scoped skill set (do/grilling/to-prd/to-issues/
# implement/qa/etc, namespaced as /rhubarb:*) -- shipped inside the `rhubarb`
# package itself, never inside a target project's repo. Passed via
# `--plugin-dir` on every subprocess call so Rhubarb-driven sessions never see
# (and can never accidentally invoke) the user's global ~/.claude/skills.
_PLUGIN_DIR = str(Path(__file__).parent / "claude_plugin")


def _plugin_args() -> list[str]:
    return ["--plugin-dir", _PLUGIN_DIR]


def _effort_args(effort: str | None) -> list[str]:
    """"auto" (Rhubarb's default) and `None` both omit `--effort` entirely,
    letting the model's own built-in default apply -- "auto" is not a valid
    `--effort` flag value (only low/medium/high/xhigh/max are), so there is
    no flag that means "auto" here, only the absence of one."""
    if not effort or effort == "auto":
        return []
    return ["--effort", effort]


# Set to True by app.py when lean-ctx is enabled and present (issue #233,
# child of PRD #232). No proxy process to start/stop, unlike Headroom --
# this flag alone gates whether `_lean_ctx_args()` points a spawned
# `claude` subprocess at the Rhubarb-owned scoped config files
# `lean_ctx_installer.generate_scoped_config()` writes.
_lean_ctx_enabled: bool = False


def set_lean_ctx_enabled(enabled: bool) -> None:
    """Called by app.py when lean-ctx is enabled/disabled (install success,
    Settings toggle, or a declined/undeclined transition). Only affects
    subsequently spawned subprocesses -- an already-running `claude`
    process never sees this change."""
    global _lean_ctx_enabled
    _lean_ctx_enabled = enabled


def _lean_ctx_args() -> list[str]:
    """Mirrors `_plugin_args()`'s shape: a fixed pair of flags pointing at
    Rhubarb's own generated config files when lean-ctx is enabled, else no
    flags at all. Both `run_prompt()` below and `stream_json_engine.py`'s
    `_build_args()` call this -- the two places Rhubarb ever spawns
    `claude` -- so enabling lean-ctx actually reaches every subprocess
    Rhubarb drives, not just one of the two spawn paths."""
    if not _lean_ctx_enabled:
        return []
    return ["--mcp-config", str(LEAN_CTX_MCP_CONFIG_PATH), "--settings", str(LEAN_CTX_HOOKS_SETTINGS_PATH)]


# Set to True by app.py when a Headroom proxy is running and claude
# subprocess traffic should be routed through it (issue #200).
# `_clean_env` reads this flag to add ANTHROPIC_BASE_URL when needed.
# Always False until app.py explicitly activates it -- no Headroom by default.
_headroom_proxy_active: bool = False
# `_HEADROOM_BASE_URL` itself is imported from `headroom_installer` above
# (issue #203) rather than hardcoded here -- see that module's
# `HEADROOM_PROXY_PORT`/`HEADROOM_BASE_URL` docstring for why.


def set_headroom_proxy_active(active: bool) -> None:
    """Called by app.py when the Headroom proxy process starts or stops."""
    global _headroom_proxy_active
    _headroom_proxy_active = active


def _clean_env() -> dict:
    env = os.environ.copy()
    # Always strip these so the subprocess uses the subscription login, not
    # an API key -- this must happen regardless of Headroom's state.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    # Route traffic through the local Headroom proxy when it's active.
    # When not active, ensure any ambient ANTHROPIC_BASE_URL in the parent
    # environment is not passed through either.
    if _headroom_proxy_active:
        env["ANTHROPIC_BASE_URL"] = _HEADROOM_BASE_URL
    else:
        env.pop("ANTHROPIC_BASE_URL", None)
    return env


def _isolated_process_group() -> dict:
    """On Windows, spawn the child in its own process group so it's immune to
    CTRL_C_EVENT.

    `claude` runs under `cmd.exe` here (shell=True). uvicorn's `--reload`
    restarts its server process on a file change by sending it CTRL_C_EVENT,
    which Windows broadcasts to every process sharing the console -- including
    an in-flight `cmd.exe`/`claude` child, whose default reaction to that is
    to print "Terminate batch job (Y/N)?" to stdout and hang waiting for an
    answer nobody gives it. A totally unrelated dev-server reload (e.g. from
    an unrelated file edit while a `/implement` turn is mid-flight) would
    otherwise tear down that turn. CREATE_NEW_PROCESS_GROUP makes Windows
    exempt the child from CTRL_C_EVENT entirely; `Popen.kill()`/`terminate()`
    still work regardless, since those target the process by handle."""
    if platform.system() == "Windows":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def run_prompt(
    prompt: str,
    *,
    session_id: str | None = None,
    cwd: str | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> dict:
    """Run a single non-interactive `claude -p` call and return the parsed JSON result.

    Pass `session_id` (from a previous call's result) to continue that
    conversation instead of starting a fresh one.

    The prompt is piped via stdin rather than passed as a CLI argument: on
    Windows, `claude` is invoked through `cmd.exe` (it's an npm .cmd shim),
    and a multi-line prompt passed as an argument gets its embedded newlines
    treated as command separators, silently truncating the call.
    """
    env = _clean_env()

    args = ["claude", "-p", "--output-format", "json", "--dangerously-skip-permissions"]
    args += _plugin_args()
    args += _lean_ctx_args()
    if session_id:
        args += ["--resume", session_id]
    if model:
        args += ["--model", model]
    args += _effort_args(effort)

    result = subprocess.run(
        args,
        input=prompt,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=platform.system() == "Windows",
        **_isolated_process_group(),
    )

    if result.returncode != 0:
        raise ClaudeCLIError(f"claude exited with {result.returncode}: {result.stderr.strip()}")

    return json.loads(result.stdout)


def get_auth_status() -> dict:
    """Return the parsed output of `claude auth status`."""
    env = _clean_env()

    result = subprocess.run(
        ["claude", "auth", "status", "--json"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=platform.system() == "Windows",
        **_isolated_process_group(),
    )

    if result.returncode != 0:
        raise ClaudeCLIError(f"claude auth status exited with {result.returncode}: {result.stderr.strip()}")

    return json.loads(result.stdout)
