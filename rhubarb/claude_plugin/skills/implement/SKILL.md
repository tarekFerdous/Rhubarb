---
name: implement
description: Pick a PRD from open GitHub issues, implement its child issues one at a time in this session, run tests, write a tracker file, then clear. Hand off to /rhubarb:qa to finish.
---

# /rhubarb:implement — Implementation Workflow

Execute implementation and testing for a chosen PRD. Follow each phase in order without skipping.

**Never spawn a subagent.** No part of this skill — including issue implementation, Phase 3's test-fixing, and any exploration needed to diagnose a failure — may dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism). Everything happens directly, in this same continuous session. (Phase 2 below restates this for issue implementation specifically, along with the rate-limit-budget rationale for that case.)

**Never pause to ask the user a question during this skill.** Make the best autonomous call and proceed — do not seek confirmation ("are we ready?", "should I proceed?"), and do not ask which of several reasonable options to pick when a reasonable default exists. This applies everywhere in this skill except Phase 1's fallback PRD-selection prompt below, which only fires for a fresh, argument-less invocation with no PRD in context — a different kind of question (which PRD to work on at all, before any work has started) than the "can't proceed" case this directive is about.

If you genuinely cannot proceed without information only the user has — not "which of two reasonable choices," but a real blocker — stop and emit this marker instead of asking inline:

```json
{"phase": "implement_blocked", "issue": <number-or-null>, "question": "<the question>", "context": "<why it's blocked>"}
```

Rhubarb detects this marker and hands your question to the user through the app UI; your next turn will carry their reply as a normal message, resuming from here. Do not use this for anything a reasonable default would resolve — it exists for the rare case, not as a routine checkpoint.

Before printing that marker to chat, also write the exact same JSON object to a file at `.claude/rhubarb_blocked.json` in the project directory (create it if it doesn't exist, overwrite it if it does) -- this is what Rhubarb actually reads to route this to the blocked-card UI, more reliably than scraping it back out of your printed chat output. Still print the marker to chat exactly as specified above; the file is additive, not a replacement.

---

## Phase 1 — PRD Selection

If the invocation already specifies `prd: <N>` (e.g. `/rhubarb:implement prd: 34`), skip this entire phase — no `gh issue list` fetch, no table, no pause — and go straight to Phase 2 using PRD issue N as the selected PRD.

Otherwise: if a PRD was already created earlier in this same conversation (e.g. via `/rhubarb:do`), use that PRD's issue number directly and go straight to Phase 2 — it's already in your context, no need to ask or re-fetch anything.

Otherwise (a genuinely fresh invocation, nothing in context, no `prd:` argument), continue with steps 1-6 below:

1. Fetch all open issues:
   ```
   gh issue list --state open --json number,title,body,labels
   ```
2. Identify **PRD issues** — those with the `ready-for-agent` label or whose title/body marks them as a PRD.
3. For each PRD, collect its related child issues (issues whose body references the PRD number, e.g. "Part of #N" or "Blocked by #N").
4. Display a table like the following (one row per PRD; child issue titles in the last column):

   | # | PRD Title | Child Issues |
   |---|-----------|--------------|
   | 12 | User auth flow | #13 Sign-up page, #14 Login page, #15 JWT middleware |
   | 20 | Menu management | #21 Category CRUD, #22 Item CRUD, #23 Image upload |

5. **Pause here.** Ask the user: "Which PRD would you like to implement? (enter the issue number)"
6. Wait for the user's selection before continuing.

---

## Phase 2 — Implementation

Using only the child issues that belong to the selected PRD:

1. Re-fetch those issues to get their current bodies and labels.
2. Identify **unblocked** issues (those whose body says "None - can start immediately" or whose blockers are already closed).
3. Implement all unblocked issues **one at a time, in this same continuous session** — do not dispatch sub-agents and do not implement issues in parallel, regardless of how many are unblocked at once. Rhubarb-sized PRDs are small enough that sequential is cheap; parallel sub-agent fan-out is what was burning the account's rate-limit budget.
4. Once unblocked issues are complete, resolve the next wave of now-unblocked issues — also one at a time.
5. **Triage decisions are made autonomously by default.** Pick the most reasonable interpretation and proceed; note the call you made in that issue's tracker summary (Phase 4) rather than pausing over it. Only emit the blocked marker (above) if the issue is genuinely ambiguous in a way no reasonable default resolves.

---

## Phase 3 — Testing

Run the project's full test suite. All tests must pass before continuing. Fix any failures autonomously; if a failure genuinely cannot be diagnosed or fixed without user-only information, emit the blocked marker (above) rather than leaving it broken or guessing destructively.

---

## Phase 4 — Write Tracker File

Write `.claude/implement-tracker.json` at the project root with the following structure:

```json
{
  "prd": {
    "number": <prd-issue-number>,
    "title": "<prd-issue-title>"
  },
  "issues": [
    {
      "number": <issue-number>,
      "title": "<issue-title>",
      "summary": "<one or two sentences describing what was built for this issue>",
      "acceptance_criteria": ["<criterion 1>", "<criterion 2>"]
    }
  ],
  "qa_changes": [],
  "status": "implemented"
}
```

- `qa_changes` starts as an empty array — `/rhubarb:qa` will populate it during the QA loop.
- `acceptance_criteria` lists every criterion from the issue body so `/rhubarb:qa` can check them off.

---

## Phase 5 — Context Reset

Run `/clear`.

---

**Done.** Use `/rhubarb:qa` to run the QA loop, close issues, commit, and clean up.
