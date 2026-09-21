---
name: to-issues
description: Break a plan, spec, or PRD into independently-grabbable draft issues using tracer-bullet vertical slices, for /rhubarb:publish-to-github to publish next.
---

# /rhubarb:to-issues

Break a plan into independently-grabbable issues using vertical slices (tracer bullets).

This is a pure drafting step, only ever reached through the automated `/do` chain — it is never used interactively on its own. Do NOT run `gh issue create` and do NOT quiz the user about the breakdown: draft the slices and output them; the next phase (`/rhubarb:publish-to-github`, a resumed turn of this same conversation) is the one place that publishes them.

**Never spawn a subagent.** All exploration and drafting below happens directly, in this same session — never dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism), and never run anything in parallel.

## Process

### 1. Gather context

Work from whatever is already in the conversation context. If the user passes an issue reference (issue number, URL, or path) as an argument, fetch it from the issue tracker and read its full body and comments.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so to understand the current state of the code. Issue titles and descriptions should use the project's domain glossary vocabulary, and respect ADRs in the area you're touching.

Look for opportunities to prefactor the code to make the implementation easier. "Make the change easy, then make the easy change."

### 3. Draft vertical slices

Break the plan into **tracer bullet** issues. Each issue is a thin vertical slice that cuts through ALL integration layers end-to-end, NOT a horizontal slice of one layer.

<vertical-slice-rules>

- Each slice delivers a narrow but COMPLETE path through every layer (schema, API, UI, tests)
- A completed slice is demoable or verifiable on its own
- Any prefactoring should be done first

</vertical-slice-rules>

### 4. Output the draft slices

Assign each slice a stable per-run label in dependency order: `S1`, `S2`, `S3`, ... These labels exist only to let `/rhubarb:publish-to-github` resolve sibling cross-references later in this same conversation — they are never shown to the user and never published anywhere.

For each slice, output exactly one label line in this format, followed by the issue body using the template below:

```
Issue Draft S1: <title>
```

Use the issue body template below for each drafted issue's body. The real parent PRD issue number doesn't exist yet at draft time, so leave `## Parent` as a placeholder. Likewise, no sibling has been published yet, so in "Blocked by" reference the blocking slice's own label (`S1`, `S2`, ...) instead of a real `#N` — `/rhubarb:publish-to-github` resolves every label into the real issue number it just created for that slice, in the same dependency order established here (blockers first).

<issue-template>
## Parent

(assigned by /rhubarb:publish-to-github once the PRD is published)

## What to build

A concise description of this vertical slice. Describe the end-to-end behavior, not layer-by-layer implementation.

Avoid specific file paths or code snippets — they go stale fast. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it here and note briefly that it came from a prototype. Trim to the decision-rich parts — not a working demo, just the important bits.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Blocked by

- S1 (per-run label of the blocking slice, assigned above)

Or "None - can start immediately" if no blockers.

</issue-template>

Do NOT close or modify any parent issue.
