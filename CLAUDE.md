# Rhubarb

Python controller that drives Claude Code through phases of software
development, gating progression on scenario-based decisions rather than on
artifact presence.

## Architecture decisions

- **Control mechanism: CLI subprocess.** The controller spawns the `claude`
  binary as a subprocess per phase/task and drives it via stdin/args, parsing
  stdout/exit codes. Not the Claude Agent SDK.
- **Billing: subscription, not API key.** Reasoning must run under the
  user's Claude subscription (Pro/Max/Team/Enterprise) login, not pay-per-
  token API billing. This means:
  - Do **not** set `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the
    subprocess environment — an API key present in env always overrides the
    subscription login, regardless of how many seats exist.
  - The Agent SDK was ruled out for this reason: it requires an API key and
    always bills pay-per-token, with no subscription path.
  - Watch for subscription usage limits (rolling time-window allowances on
    Pro/Max/Team/Enterprise) since heavy automation across many phases can
    hit these caps faster than interactive use.
- **Unattended sessions skip permission checks.** Every session spawned via
  `run_prompt()` runs headless (`-p`, stdin-piped, no TTY), so there is no
  terminal for Claude to prompt into and no human present to answer an
  interactive tool-permission request. `run_prompt()` therefore passes
  `--dangerously-skip-permissions`. This is a deliberately minimal, unblocking
  change — sandboxing/isolation per session, an audit trail of what a session
  did, and rollback/recovery from an unattended destructive action are known
  open gaps, not solved here and deferred to future work.

## Open questions

- Phase/gate design not yet decided: fixed SDLC phase sequence with
  hardcoded gates vs. config/DSL-driven phases with customizable gate
  conditions per project.
