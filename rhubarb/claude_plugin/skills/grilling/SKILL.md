---
name: grilling
description: Grill the user relentlessly about a plan, decision, or idea. Use when the user wants to stress-test their thinking, or uses any 'grill' trigger phrases.
---

Interview the user relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **frontier** is every decision whose prerequisites are already settled: the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole frontier in one round: number each question and give your recommended answer. Then wait for the user's answers before the next round.

Format a round in this exact structure, so it can be parsed deterministically — free-text framing before the first question and after the last one is fine (e.g. "Here is round 1 of questions." / "Based on your answers, a second wave might be needed."), but every question itself must follow this shape precisely:

```
Question 1: "<question text>"
Options:
Option 1: "<option text>"
Option 2: "<option text>"
Recommended: [1]

Question 2 (select multiple): "<question text>"
Options:
Option 1: "<option text>"
Option 2: "<option text>"
Option 3: "<option text>"
Recommended: [1, 3]

Question 3: "<question text, no discrete options>"
Recommended text: "<your recommended free-text answer>"
```

- A question with an `Options:` block is single-select by default (mutually exclusive); mark it `Question N (select multiple):` to make it multi-select instead (independently toggleable).
- `Recommended: [n]` (single-select) or `Recommended: [n, m, ...]` (multi-select) references option numbers, 1-based.
- A question with no `Options:` block is open-ended — give a `Recommended text: "<...>"` line instead of `Recommended:`, or omit it if you have no recommendation.
- Ask the whole frontier in one round, one `Question N:` block per question, numbered sequentially within the round.

Each round the user answers reshapes the tree: settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the user's. When a frontier question needs a fact from the environment (filesystem, tools, etc.), dispatch a sub-agent to find it; don't ask the user for anything you could look up yourself. Don't block on it: a running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the sub-agent to report; ask the rest of the frontier now. The _decisions_ are the user's: put each to them and wait.

The session is done when the frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the user confirms you have reached a shared understanding.
