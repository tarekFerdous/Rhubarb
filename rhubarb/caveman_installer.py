"""Caveman skill install and presence-detection automation (issue #201).

Pure functions with every external command call injectable, mirroring
`headroom_installer.py`'s injection pattern -- tests never require a real
Caveman install, network access, or a package manager.

Caveman is the `JuliusBrussee/caveman` Claude Code skill, from a repo
containing 20 skills (`caveman`, `caveman-commit`, `caveman-compress`, ...)
-- `skills add <repo>` alone (with no `--skill`/`-y`) clones the repo and
then drops into an interactive checkbox-picker wizard to choose which of
those 20 to install, which is fundamentally incompatible with running it
as a piped, non-interactive subprocess (confirmed in QA of issue #201:
piping it produces garbled, repeated spinner-frame text, never completes,
and never actually installs anything). The `skills` CLI (`vercel-labs/
skills` on npm/GitHub) documents proper non-interactive flags for exactly
this ("Non-interactive installation (CI/CD friendly)" in its own README) --
`--skill <name>` picks a specific skill instead of prompting, `-y`/`--yes`
skips all confirmation prompts, and `-a/--agent` targets a specific agent
instead of prompting for one. Passing all three below is what actually
makes this a genuinely non-interactive, scriptable command:
  npx skills add JuliusBrussee/caveman --skill caveman -a claude-code -g -y
Presence check:
  npx skills list  (look for "caveman" in output; already non-interactive
  -- `list` is a plain read-only listing, no picker, and covers both
  project and global scope by default, so no extra flags needed here)
Disable/remove:
  npx skills remove caveman -a claude-code -g -y
  (the target here is the SKILL NAME "caveman", not the repo path
  "JuliusBrussee/caveman" `skills remove` took previously -- `remove`'s
  positional argument names an already-installed skill by its own name,
  not a source to resolve; the repo path never matched anything installed,
  so this call was accomplishing nothing before this fix, and would fall
  into its own interactive picker if no skills happened to match)

Only the skill-only install path is ever used -- the proxy tier is never
installed by this feature. No proxy process to manage: once the skill is
installed globally via `npx`, it is available to any `claude` invocation
Rhubarb spawns without any additional per-spawn wiring.

This module has no UI or session-runner wiring of its own; the app-level
consent/install gate and the Settings toggle in `rhubarb/web/app.py` are
the callers.
"""

import shutil
import subprocess

PRESENCE_NOT_PRESENT = "not_present"
PRESENCE_PRESENT = "present"


def resolve_npx_command() -> str:
    """`npx` (like `npm`) is a `.cmd` batch-file shim on Windows, not a
    native `.exe` -- and Windows' `CreateProcess` (what `subprocess.run`
    uses under `shell=False`) cannot launch a `.cmd` file directly from a
    *bare* name, even when it's genuinely on `PATH`: it needs the full
    path INCLUDING the `.cmd` extension before it'll run it without a
    shell (empirically confirmed in QA of issue #201 -- `subprocess.run(
    ["npx", "--version"])` raises `WinError 2` on a machine where `npx` is
    demonstrably installed and callable from an interactive terminal, while
    `subprocess.run([shutil.which("npx"), "--version"])` succeeds). This
    is a different mechanism than the `PATH`-inheritance bug already fixed
    for `winget`/`pip`/`headroom` (`ollama_installer._resolve_winget_
    command`, `headroom_installer.resolve_headroom_command`): those are
    genuinely missing from this process's `PATH`; `npx` is actually found
    by `shutil.which` (which does check `PATHEXT` and returns the full
    `...\\npx.CMD` path) -- it's specifically the bare, extension-less
    invocation that Windows refuses to run without a shell.

    `shutil.which` does its own real `PATH` search (unlike a hardcoded
    directory guess), so this needs no per-platform branching: on
    macOS/Linux `npx` is a plain executable and this just returns its
    resolved path unchanged. Falls back to the bare name if `npx` isn't
    found on `PATH` at all (the resulting `FileNotFoundError` then
    correctly reports "not found" rather than masking it)."""
    return shutil.which("npx") or "npx"


def _default_run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def run_streaming(argv: list[str], *, on_line=None, popen_factory=None) -> None:
    """Same `subprocess.run(argv, check=True)` contract as `_default_run`
    (raises `subprocess.CalledProcessError` on a non-zero exit), but calls
    `on_line(line)` for each line of merged stdout/stderr AS IT ARRIVES,
    not only after the process exits -- mirrors
    `headroom_installer.run_streaming` exactly, so a caller (the app-level
    install endpoint) can surface live `npx` output while it's still
    downloading/resolving packages, instead of the UI going silent/"stuck"
    until the whole thing finishes.

    `popen_factory` is injectable (a callable with `subprocess.Popen`'s
    signature) so tests never need a real subprocess.

    `encoding="utf-8", errors="replace"` explicitly, for the same reason
    as `headroom_installer.run_streaming`: `text=True` alone decodes with
    `locale.getpreferredencoding()`, which on Windows is the legacy ANSI
    codepage, not UTF-8 -- and `npm`/`npx` output routinely contains UTF-8
    characters that codepage can't decode, crashing the whole streamed
    read (`'charmap' codec can't decode byte ...`) over what's often just
    cosmetic progress output."""
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
        output = run_output([resolve_npx_command(), "skills", "list"])
        return PRESENCE_PRESENT if "caveman" in output else PRESENCE_NOT_PRESENT
    except Exception:
        return PRESENCE_NOT_PRESENT


def install_caveman(*, run=None) -> None:
    """Install the Caveman skill globally, fully non-interactively, via
    `npx skills add JuliusBrussee/caveman --skill caveman -a claude-code -g -y`.
    `npx` is cross-platform, so the same command is used on every OS.

    `--skill caveman` picks the one skill this project cares about out of
    the repo's 20; `-a claude-code` targets Claude Code specifically
    instead of prompting for an agent; `-y` skips every remaining
    confirmation prompt. Without these three flags this drops into an
    interactive checkbox-picker wizard instead of installing anything
    (see the module docstring) -- they are not optional tuning, they are
    what makes this command able to run unattended at all.

    `run` is injectable (a callable taking an argv list that raises on
    failure) so tests never need a real install or network access."""
    run = run or _default_run
    run(
        [
            resolve_npx_command(),
            "skills",
            "add",
            "JuliusBrussee/caveman",
            "--skill",
            "caveman",
            "-a",
            "claude-code",
            "-g",
            "-y",
        ]
    )


def disable_caveman(*, run=None) -> None:
    """Remove the Caveman skill globally, fully non-interactively, via
    `npx skills remove caveman -a claude-code -g -y`. Called when the user
    turns off the Settings toggle.

    The target is the skill's own name ("caveman"), not the source repo
    path -- `skills remove` looks up an already-installed skill by name,
    unlike `skills add`'s source-then-`--skill` shape (see the module
    docstring). `-y` skips the confirmation prompt the same way `-y` does
    for `install_caveman`.

    `run` is injectable so tests never need a real install or network access."""
    run = run or _default_run
    run([resolve_npx_command(), "skills", "remove", "caveman", "-a", "claude-code", "-g", "-y"])
