---
name: grilling
description: Grill the user relentlessly about a plan, decision, or idea. Use when the user wants to stress-test their thinking, or uses any 'grill' trigger phrases.
---

Interview the user relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **frontier** is every decision whose prerequisites are already settled: the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole frontier in one round: number each question and give your recommended answer. Then wait for the user's answers before the next round.

Format a round in this structure — free-text framing before the first question and after the last one is fine (e.g. "Here is round 1 of questions." / "Based on your answers, a second wave might be needed."), but every question itself should follow this shape:

```
❓ **Q1** - **Scope**: <question text>
- Yes
- No
Recommended: Yes

❓ **Q2** (select multiple) - **Environments**: <question text>
- Dev
- Staging
- Prod
Recommended: Dev, Prod

❓ **Q3** - **Deployment target**: <question text, no discrete options>
Recommended: <your recommended free-text answer>
```

- Each question starts with `❓ **Qn** - **<short title>**: <question text>`.
- A question with a bulleted options list is single-select by default (mutually exclusive); mark it `❓ **Qn** (select multiple) - **<title>**: ...` to make it multi-select instead (independently toggleable).
- `Recommended:` lists the recommended option(s) by their exact text (comma-separated for multi-select).
- A question with no options list is open-ended — give a `Recommended:` line with your free-text recommendation, or omit it if you have no recommendation.
- Ask the whole frontier in one round, one `❓ **Qn**` block per question, numbered sequentially within the round.

Before printing a round to chat, also write the exact same `❓ **Qn**` blocks (no free-text framing, just the structured blocks themselves, one after another) to a file at `.claude/rhubarb_question.md` in the project directory (create it if it doesn't exist, overwrite it if it does) -- this is what Rhubarb actually reads to render the round on screen, more reliably than scraping it back out of your printed chat output. Still print the round to chat exactly as specified above; the file is additive, not a replacement.

Each round the user answers reshapes the tree: settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the user's. When a frontier question needs a fact from the environment (filesystem, tools, etc.), look it up yourself directly, in this same session — never dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism), and never run explorations in parallel; don't ask the user for anything you could look up yourself. Ask the rest of the frontier now regardless of what you're still looking up. The _decisions_ are the user's: put each to them and wait.

The session is done when the frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the user confirms you have reached a shared understanding.
