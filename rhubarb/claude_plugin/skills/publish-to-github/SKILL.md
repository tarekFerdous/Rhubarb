---
name: publish-to-github
description: Publish a PRD and its child issues (drafted earlier in this same conversation by /rhubarb:to-prd and /rhubarb:to-issues) to GitHub via gh issue create.
---

# /rhubarb:publish-to-github

This skill runs as a resumed turn of the same conversation `/rhubarb:to-prd` and `/rhubarb:to-issues` just ran in — both drafts are already in your context. Do NOT re-derive or re-draft anything; publish exactly what was already drafted.

**Never spawn a subagent.** Publish every issue directly, one at a time, in this same session — never dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism), and never run anything in parallel.

## Process

### 1. Publish the PRD

Find the PRD draft earlier in this conversation (the `PRD Draft: <title>` line and the rendered PRD markdown that followed it). Publish it:

```
gh issue create --title "PRD: <title>" --body "<the full rendered PRD markdown>" --label "ready-for-agent" --label "prd"
```

The title is the drafted title prefixed with `PRD: ` literally — this is how the PRD is identified at a glance in GitHub's own issue list, separate from Rhubarb's own UI.

Apply both the `ready-for-agent` label (so the AFK session runner picks it up) and the `prd` label (so it appears in the "To be implemented" panel, distinguishing it from the child issues below, which carry `ready-for-agent` only).

After `gh issue create` succeeds, parse the real issue number from the URL it prints. Output exactly one line in this format (where N is the real issue number, and `<title>` is the undecorated drafted title, without the `PRD: ` prefix):

```
PRD #N: <title>
```

This line is required — the backend parses the issue number from it.

### 2. Publish each child issue

Find every `Issue Draft S<n>: <title>` block drafted earlier in this conversation by `/rhubarb:to-issues`, in the same dependency order they were drafted in (blockers first — do not reorder them).

For each one, before publishing:

- Replace its `## Parent` placeholder with the real PRD issue number published in step 1, formatted as `#N`.
- Replace every `S<n>` label in its `## Blocked by` section with the real issue number `#M` of the sibling already published earlier in THIS publishing pass (a slice can only be blocked by a slice that comes before it in dependency order, so that sibling's real number is always already known by the time you get here).

Then publish it:

```
gh issue create --title "<title>" --body "<body with placeholders resolved>" --label "ready-for-agent"
```

After each `gh issue create` succeeds, parse the issue number from the URL it prints and output exactly one line in this format (where N is the real issue number):

```
Issue #N: <title>
```

These lines are required — the backend parses issue numbers from them.

### 3. Summarize

After every issue is published, give a short closing summary of what was created (the PRD and its child issues) — this is shown to the user as the do-finished summary.

Do NOT close or modify any parent issue. Do NOT start implementing anything.
