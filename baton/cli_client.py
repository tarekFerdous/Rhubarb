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


class ClaudeCLIError(RuntimeError):
    pass


# Baton's own private, plugin-scoped skill set (do/grilling/to-prd/to-issues/
# implement/qa/etc, namespaced as /baton:*) -- shipped inside the `baton`
# package itself, never inside a target project's repo. Passed via
# `--plugin-dir` on every subprocess call so Baton-driven sessions never see
# (and can never accidentally invoke) the user's global ~/.claude/skills.
_PLUGIN_DIR = str(Path(__file__).parent / "claude_plugin")


def _plugin_args() -> list[str]:
    return ["--plugin-dir", _PLUGIN_DIR]


def _effort_args(effort: str | None) -> list[str]:
    """"auto" (Baton's default) and `None` both omit `--effort` entirely,
    letting the model's own built-in default apply -- "auto" is not a valid
    `--effort` flag value (only low/medium/high/xhigh/max are), so there is
    no flag that means "auto" here, only the absence of one."""
    if not effort or effort == "auto":
        return []
    return ["--effort", effort]


def _clean_env() -> dict:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
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
