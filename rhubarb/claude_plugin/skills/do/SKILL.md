---
name: do
description: Requirements → PRD → issues. Grills the idea, writes a PRD as a GitHub issue, breaks it into child issues, then clears context. Hand off to /rhubarb:implement to execute.
---

# /rhubarb:do — Requirements to Issues

Orchestrate the first half of the feature lifecycle: grilling → PRD → issue breakdown → context reset. Follow each phase in order without skipping.

**Never spawn a subagent.** No part of this workflow — including `/rhubarb:grilling`/`/rhubarb:grill-with-docs`, `/rhubarb:to-prd`, and `/rhubarb:to-issues` as invoked from here — may dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism) for any purpose, including exploration and fact-finding. This explicitly overrides `/rhubarb:grilling`'s own instruction to "dispatch a sub-agent" for fact-finding: when grilling is invoked from here, find any needed facts directly, in this same session, instead.

## Arguments

`$ARGUMENTS` may contain `@file` references. If any `@` references are present, use `/rhubarb:grill-with-docs`; otherwise use `/rhubarb:grilling`.

---

## Phase 1 — Requirements Grilling

- If `$ARGUMENTS` contains any `@` file references → invoke `/rhubarb:grill-with-docs $ARGUMENTS`
- Otherwise → invoke `/rhubarb:grilling`

Run the full grilling session until you have a clear, stable picture of what is being built.

---

## Phase 2 — PRD (automatic, no confirmation)

Invoke `/rhubarb:to-prd`.

**Override the default `/rhubarb:to-prd` behaviour**: do NOT pause to ask the user whether the seams look correct. Proceed directly through all steps and publish the PRD as a GitHub issue using `gh issue create`. Apply both the `ready-for-agent` and `prd` triage labels.

---

## Phase 3 — Issue Breakdown (automatic, no confirmation)

Invoke `/rhubarb:to-issues`.

**Override the default `/rhubarb:to-issues` behaviour**: do NOT run the "Quiz the user" step. Do not ask whether granularity or dependencies look right. Proceed directly to publishing all slices as individual GitHub issues via `gh issue create`, in dependency order (blockers first).

---

## Phase 4 — Context Reset

Run `/clear`.

---

**Done.** Use `/rhubarb:implement` to pick a PRD and execute the implementation, testing, QA, and close phases.
