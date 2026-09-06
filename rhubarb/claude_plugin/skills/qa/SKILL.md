---
name: qa
description: Reads the implement tracker file, runs the QA grilling loop, closes child issues and the PRD, clears the tracker, commits, pushes, and does a final /clear.
---

# /rhubarb:qa — QA, Close, and Commit

Pick up where `/rhubarb:implement` left off. Reads the tracker file, runs QA, closes everything, commits, and cleans up. Follow each phase in order without skipping.

**Never spawn a subagent.** No part of this workflow — applying QA-requested fixes, closing issues, committing, or anything invoked from here including `/rhubarb:qa-grilling` — may dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism). Everything happens directly, in this same continuous session.

---

## Phase 1 — Load Tracker

Read `.claude/implement-tracker.json` from the project root.

- If the file does not exist or `status` is not `"implemented"`, stop and tell the user: "No completed implementation found. Run `/rhubarb:implement` first."
- Extract the PRD number/title, all child issues, their summaries, and acceptance criteria.

---

## Phase 2 — QA Grilling Loop (repeat until user says "perfect!")

Invoke `/rhubarb:qa-grilling` to verify everything is working as intended.

- If the user requests any fixes or adjustments during this session, apply them.
- After applying fixes, append a record to `qa_changes` in the tracker file:
  ```json
  { "issue": <number-or-null>, "change": "<brief description of what was changed>" }
  ```
- After applying fixes, invoke `/rhubarb:qa-grilling` again.
- **Keep looping** until the user explicitly says the word **"perfect!"**

---

## Phase 3 — Close Child Issues

For each issue in the tracker:

1. Check off every acceptance criterion that was satisfied (change `- [ ]` to `- [x]` in the GitHub issue body via `gh issue edit`).
2. Build a closing comment that includes:
   - The `summary` from the tracker
   - Any `qa_changes` entries that reference this issue's number (or general changes with `issue: null`)
3. Close the issue:
   ```
   gh issue close <number> --comment "<message>"
   ```

---

## Phase 4 — Close the PRD

Close the parent PRD issue with a comment that:
- Links every child issue that was implemented (`#<n>`)
- Notes any implementation decisions or deviations (from issue summaries)
- Summarises all `qa_changes` entries

Run:
```
gh issue close <prd-number> --comment "<message>"
```

---

## Phase 5 — Clear Tracker

Overwrite `.claude/implement-tracker.json` with an empty reset state so it is ready for the next run:

```json
{
  "prd": null,
  "issues": [],
  "qa_changes": [],
  "status": "idle"
}
```

---

## Phase 6 — Commit

Stage all changes (including the cleared tracker) and create a single commit:
- Subject line: concise feature summary (derive from PRD title)
- Body: what was built, referencing the PRD issue and all child issues (`Closes #<n>`)
- Include a "QA changes" section listing everything in `qa_changes`
- End with: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

---

## Phase 7 — Push

Run `git push` to push the commit to the remote.

---

## Phase 8 — Final Context Reset

Run `/clear`.
