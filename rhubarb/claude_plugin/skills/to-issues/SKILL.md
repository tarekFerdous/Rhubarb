---
name: to-issues
description: Break a plan, spec, or PRD into independently-grabbable issues on the project issue tracker using tracer-bullet vertical slices.
---

# /rhubarb:to-issues

Break a plan into independently-grabbable issues using vertical slices (tracer bullets).

Rhubarb-managed sessions publish to GitHub as a separate, external step that runs after this skill finishes — it reads a local draft file rather than the CLI turn calling GitHub directly. Write to that draft file as instructed below. Do NOT run `gh issue create` or any other GitHub CLI command from this skill.

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

### 4. Quiz the user

Present the proposed breakdown as a numbered list. For each slice, show:

- **Title**: short descriptive name
- **Blocked by**: which other slices (if any) must complete first
- **User stories covered**: which user stories this addresses (if the source material has them)

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the dependency relationships correct?
- Should any slices be merged or split further?

Iterate until the user approves the breakdown.

### 5. Write the issues to `.claude/prd_draft.json`

For each approved slice, do NOT publish it to GitHub yourself — instead read the existing `.claude/prd_draft.json` at the project root (already written by `/rhubarb:to-prd` with the `prd` key) and add an `issues` array to it, preserving the existing `prd` key:

```json
{
  "issues": [
    { "title": "...", "body": "...", "labels": ["..."] }
  ]
}
```

**Ordering matters and stands in for real issue numbers.** No real GitHub issue numbers exist yet at this point — they're only assigned once `rhubarb/github_publisher.py` actually creates the issues. So:

- List `issues` in the array in dependency order (blockers first). The external publisher creates them in that same order and resolves each entry's real issue number as it goes.
- In each issue's `body`, write the "Blocked by" section using the *title* of the blocking slice (not a `#N` reference, since no number exists yet) — e.g. "Blocked by: the slice titled '<title>'" or "None - can start immediately". Do not fabricate placeholder issue numbers.

Use the issue body template below for each entry's `body`. This file follows a fixed schema shared with `/rhubarb:to-prd` (which writes the `prd` key) and with `rhubarb/github_publisher.py`, the Python module that later reads this file and actually creates the GitHub issues in order, resolving real `#N` references as it creates each one. Keep the shape exactly as shown above so that reader can parse it without any special-casing.

<issue-template>
## Parent

A reference to the parent issue on the issue tracker (if the source was an existing issue, otherwise omit this section).

## What to build

A concise description of this vertical slice. Describe the end-to-end behavior, not layer-by-layer implementation.

Avoid specific file paths or code snippets — they go stale fast. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it here and note briefly that it came from a prototype. Trim to the decision-rich parts — not a working demo, just the important bits.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Blocked by

- The title of the blocking slice (real issue numbers don't exist yet — the external publisher resolves these when it creates the issues in order)

Or "None - can start immediately" if no blockers.

</issue-template>

Do NOT close or modify any parent issue.
