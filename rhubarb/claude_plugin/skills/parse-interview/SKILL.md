---
name: parse-interview
description: Extract the question(s)/options a Claude Code turn's raw reply text is asking into structured JSON. Used by Rhubarb's resident per-project parser session for grilling, QA, QA-grilling, and implement turns alike.
---

# /rhubarb:parse-interview

You are given the complete, final reply text of one turn a Claude Code assistant just gave during one of Rhubarb's phases. That turn has already been flagged as needing a human's input before its session can usefully continue. Your job is to extract the question(s) it is actually asking (or the choice(s) it is waiting on) as JSON, and reply with **ONLY that JSON object** — no other prose, no code fences, no markdown.

**Never spawn a subagent.** Do this extraction directly, in this same session — never dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism), and never run anything in parallel.

## Invocation

The first line of your invocation is `phase: <phase>`, naming which Rhubarb phase this turn came from (e.g. `grilling`, `qa`, `qa_grilling`, `implement`, `qa_grilling_issues`). Everything from the next `Text:` marker onward, verbatim, is the raw turn text to extract from.

For every phase EXCEPT `qa_grilling_issues`, the phase name only tells you which session produced this text — it never changes how you extract; the exact same rules below (the flat `{header, questions, footer}` shape) apply uniformly regardless of phase. `qa_grilling_issues` is the one exception: it asks for a different, nested output shape instead — see "Phase-specific output shape: `qa_grilling_issues`" near the end of this file. Read that section only when `phase: qa_grilling_issues` is what you were given; every other phase should ignore it entirely.

## Output shape

```
{"header": <free text before the first question, or the whole text if there is no clear question in it>,
 "questions": [
   {"id": <a short stable string id, e.g. "q1">,
    "text": <the question itself, verbatim or lightly cleaned up>,
    "kind": "single" | "multi" | "open",
    "options": <array of option strings verbatim, or null if this question is open-ended>,
    "recommended": <array of 1-based indexes into "options" that were recommended, or null>,
    "recommended_text": <free-text recommendation for an open-ended question, or null>},
   ...
 ],
 "footer": <free text after the last question, or "" if none>}
```

Use `"single"` for a pick-one choice, `"multi"` for a pick-several choice, and `"open"` (with `"options"` and `"recommended"` both `null`) for anything without discrete options — free-text questions still go in `"questions"` with `"kind": "open"`.

If the text genuinely contains no question at all (this should be rare, since it was already flagged as needing input), return `"questions": []` with the whole text as `"header"` and `""` as `"footer"`.

## Grilling completion verdict (`phase: grilling` only)

This section applies ONLY when `phase: grilling` AND your extraction above came back with `"questions": []` (the turn looks like a wrap-up, with nothing left to ask). In that case, add one more top-level field to your JSON object, alongside `header`/`questions`/`footer`:

```
"completion": {"done": <true or false>, "reason": <short string>}
```

A turn with zero questions is not automatically "done" — read the FULL grilling transcript text you were given (not just this turn's own closing remark) and judge for yourself: has this interview actually reached a stable, complete shared understanding that a PRD could be drafted from right now? Every open branch resolved, nothing the transcript itself still flags as unresolved?

- `"done": true` — the understanding is genuinely stable and complete. This should be the common case whenever a wrap-up turn truly has nothing left to ask.
- `"done": false` — something is still unresolved even though this turn's own text didn't pose an explicit follow-up question (e.g. the model glossed over an earlier open branch, or its closing remark doesn't actually match what the rest of the transcript still leaves undecided). Set `"reason"` to a short, specific, human-readable explanation of what's still missing — this is shown directly to the person waiting on this session.

For every phase OTHER than `grilling`, and for `grilling` whenever `"questions"` is non-empty, omit `"completion"` entirely — do not include it at all.

## CRITICAL SPLITTING RULE

If the text contains multiple distinct questions, each question MUST be a separate object in the `"questions"` array — never combine two or more questions into a single entry. A question is distinct if it asks about a different topic, offers a different set of options, or calls for a separate recommendation. Use these signals to find question boundaries:

1. Explicit `❓ **Qn**` / `Question N:` markers (e.g. `❓ **Q1**`, `❓ **Q2**`, `Question 1:`, `Question 2:`) — each `n`/`N` is a separate entry.
2. Numbered question lists (e.g. lines starting with `1.`, `2.`, `3.`) — each number starts a new separate entry.
3. Structural separation such as blank lines or horizontal rules between self-contained question blocks.

When in doubt, split rather than merge — a Rhubarb UI card is rendered per entry, so merging collapses distinct questions into one unreadable block.

## THE RECOMMENDED-LINE MAPPING RULE

Every question that carries a `Recommended: <text>` (or `Recommended text: "<text>"`) line must have that recommendation reflected in the JSON — never drop it, and never let it silently disappear just because the question follows a long paragraph of prior context or comes last in a batch.

- If the question has a bulleted options list (`kind: "single"` or `"multi"`): match the `Recommended:` line's text **exactly** (case- and wording-sensitive) against the option list, and put the **1-based index** of the matching option(s) into `"recommended"`. `"recommended_text"` stays `null`. For `"multi"`, `Recommended:` may name more than one option (comma-separated) — every matched option's index goes into the `"recommended"` array, in the order the options themselves appear (not the order named in the line).
- If the question has no options list (`kind: "open"`): the `Recommended:` line's text goes verbatim into `"recommended_text"`. `"options"` and `"recommended"` both stay `null`.
- A question with no `Recommended:` line at all: leave `"recommended"`/`"recommended_text"` as `null`, whichever applies to its `kind`. This is the only case where both are legitimately empty — never treat a present `Recommended:` line as optional to encode.

### Worked example 1 — options-bearing question with a recommendation

Given this raw text (note the paragraph and transition phrase before the question — recommendations must survive this just as reliably as a short, isolated question):

```
We've settled the schema and the API shape already. The last open branch is how
retries should behave under load, since that changes how aggressively the
client backs off.

One more branch to close:

❓ **Q5** - **Retry backoff**: How should the client back off between retries?
- Fixed 1s delay
- Exponential backoff
- No retry, fail fast
Recommended: Exponential backoff
```

The correct extraction:

```json
{
  "header": "We've settled the schema and the API shape already. The last open branch is how retries should behave under load, since that changes how aggressively the client backs off.\n\nOne more branch to close:",
  "questions": [
    {
      "id": "q5",
      "text": "How should the client back off between retries?",
      "kind": "single",
      "options": ["Fixed 1s delay", "Exponential backoff", "No retry, fail fast"],
      "recommended": [2],
      "recommended_text": null
    }
  ],
  "footer": ""
}
```

`"recommended": [2]` because `"Exponential backoff"` is the second entry in `"options"` — matched by its exact text, not guessed from position or paraphrase. `"options"` is fully populated; it is never dropped or replaced with an `"open"` kind just because a recommendation is also present.

### Worked example 2 — open/free-text question with a recommendation

```
❓ **Q6** - **Deployment target**: Where should this service run in production?
Recommended: On the existing droplet, behind the current nginx proxy.
```

The correct extraction:

```json
{
  "header": "",
  "questions": [
    {
      "id": "q6",
      "text": "Where should this service run in production?",
      "kind": "open",
      "options": null,
      "recommended": null,
      "recommended_text": "On the existing droplet, behind the current nginx proxy."
    }
  ],
  "footer": ""
}
```

No bulleted options exist here, so the recommendation goes verbatim into `"recommended_text"`, and `"options"`/`"recommended"` both stay `null` — this is what distinguishes a genuinely open question from an options-bearing one that lost its options.

## Phase-specific output shape: `qa_grilling_issues`

Everything above this section is for every phase EXCEPT `qa_grilling_issues`. When `phase: qa_grilling_issues` is what you were given, ignore the flat `{header, questions, footer}` shape entirely and use this section instead.

The input text is one round of a QA-grilling verification pass (produced by the `qa-grilling` skill), checking a completed PRD's issues one at a time. It is meant to contain a PRD header followed by one or more issues, each with its own verification question(s), in this format:

```
QA session for PRD N: "<prd title>"

Issue N: "<issue title>"
Question N: "<question text>"
Recommended text: "<a likely answer>" (optional)
Question N: "<question text>"

Issue N: "<issue title>"
...
```

For this phase, reply with this nested shape instead of the flat one:

```
{"prd": {"number": <int>, "title": <string>} or null,
 "issues": [
   {"number": <int>,
    "title": <string>,
    "questions": [
      {"id": <a short stable string id, shaped "issue<issue_number>-q<n>">,
       "text": <the question itself, verbatim or lightly cleaned up>,
       "kind": "single" | "multi" | "open",
       "options": <array of option strings verbatim, or null if this question is open-ended>,
       "recommended": <array of 1-based indexes into "options" that were recommended, or null>,
       "recommended_text": <free-text recommendation for an open-ended question, or null>},
      ...
    ]},
   ...
 ]}
```

Grouping rule: each `Issue N: "<issue title>"` line starts a new entry in `"issues"`, with `"number"` taken from `N` and `"title"` from the quoted text; every `Question N: "..."` line up to the next `Issue N:` header (or end of text) belongs to that issue's own `"questions"` array. The `QA session for PRD N: "<prd title>"` header (if present) becomes `"prd"`; if the text has no such header, `"prd"` is `null`.

Every field on each nested question follows the exact same rules as the flat shape's own `"questions"` entries above — this is the SAME underlying per-question schema (`id`, `text`, `kind`, `options`, `recommended`, `recommended_text`), just grouped under each issue instead of collected into one flat list. In particular:

- The CRITICAL SPLITTING RULE above applies per issue: multiple `Question N:` lines under the same issue are separate entries in that issue's `"questions"` array, never merged.
- THE RECOMMENDED-LINE MAPPING RULE above applies identically: a `Recommended text: "<text>"` (or `Recommended: <text>`) line must never be dropped, exactly as for the flat shape. A real QA verification question is almost always open-ended (no bulleted options list) — expect `kind: "open"` with the recommendation landing in `"recommended_text"` far more often than not — but if a QA question genuinely does present a bulleted choice, extract it exactly as the flat shape would (`kind: "single"`/`"multi"`, `"options"` populated verbatim, `"recommended"` as 1-based indexes into `"options"`). Never force a question into `"open"` just because that is the common case for this phase — check for actual bulleted option lines the same way you would for any other phase.

There is no `"header"`/`"footer"` in this shape — omit both fields entirely; they do not apply here.

If the text contains no recognizable QA session content at all, reply with `{"prd": null, "issues": []}`.

### Worked example — two issues, each with their own question

Given this raw text:

```
QA session for PRD 12: "Add retry backoff"

Issue 13: "Client retry loop"
Question 1: "Does the client actually stop retrying after 3 attempts?"
Recommended text: "Yes, confirmed in the logs."

Issue 14: "Backoff timing"
Question 1: "Is the backoff delay actually exponential, not fixed?"
```

The correct extraction:

```json
{
  "prd": {"number": 12, "title": "Add retry backoff"},
  "issues": [
    {
      "number": 13,
      "title": "Client retry loop",
      "questions": [
        {
          "id": "issue13-q1",
          "text": "Does the client actually stop retrying after 3 attempts?",
          "kind": "open",
          "options": null,
          "recommended": null,
          "recommended_text": "Yes, confirmed in the logs."
        }
      ]
    },
    {
      "number": 14,
      "title": "Backoff timing",
      "questions": [
        {
          "id": "issue14-q1",
          "text": "Is the backoff delay actually exponential, not fixed?",
          "kind": "open",
          "options": null,
          "recommended": null,
          "recommended_text": null
        }
      ]
    }
  ]
}
```

Note the second question has no `Recommended text:` line at all, so both `"recommended"`/`"recommended_text"` stay `null` — same "only legitimately empty when there was truly nothing there" rule as the flat shape.

## Reminder

Respond with the JSON object only. No prose before or after it, no code fences, no markdown formatting around it.
