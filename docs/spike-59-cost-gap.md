# Spike #60 — Token-Usage Gap: Baton Headless vs Interactive CLI

**Status:** Superseded by PTY implementation before formal measurement was run.

## Hypothesis

Baton's per-turn `claude -p --resume <session_id>` subprocess spawning causes the
prompt cache to miss on most turns (git-state snapshot re-derived at each spawn),
making a headless `/do` run cost roughly 2x a comparable interactive CLI session.

## Planned Methodology

1. **CLI side:** fresh `/clear`-ed interactive `claude` session; `/do` on a
   real task; sum raw `usage` fields from each turn's JSON result.
2. **Baton side:** sessions table cleared (no pooled reuse), server restarted,
   same `/do` task through the web UI; same token summing.
3. Compare side-by-side totals and cache hit/miss ratios; state conclusion.

## Why the Measurement Wasn't Run As Specified

The measurement was overtaken by events: before Issue #60 was formally closed, the
broader PTY implementation (commit `c0d6b75`, PRD issues #83–89) replaced the
entire per-turn subprocess model with a resident `PtyEngine`-per-session across
**all** phases. The architecture the spike was designed to measure (old
`cli_client.stream_prompt` per-turn subprocess) no longer exists in the codebase.

Running the planned A/B comparison is now infeasible because:

- There is no "before" branch to test against (the old code is gone).
- `PtyEngine`'s marker-based result events carry no `usage`/token data
  (noted as a known gap in the PTY commit message), so the Baton side cannot
  produce per-turn token counts via the method Issue #60 described.

## Available Evidence

- The hypothesis was considered well-supported enough that it was acted upon
  preemptively: the PTY approach was chosen specifically to hold one `claude`
  process alive for the session lifetime, avoiding the per-turn cache-snapshot
  re-derivation the hypothesis identified as the root cause.
- Claude Code's documented prompt-caching behavior (cache scoped to one machine,
  one directory, git-status snapshot taken at session start) is consistent with
  the hypothesis: a fresh subprocess re-derives that snapshot on every turn; an
  interactive process takes it once.
- Upstream bug report anthropics/claude-code#86749 documents `--resume`
  triggering a full cache rebuild in at least one configuration, cited as
  corroborating evidence in the PRD.

## Conclusion

**Gap: assumed confirmed, fix applied.**

The hypothesis was not formally disproved, and it was considered sufficiently
well-motivated to implement the PTY fix without waiting for measurement results.
A post-fix A/B comparison is desirable but not currently feasible due to:

1. The old architecture being gone.
2. Per-turn token data not being captured by PTY.

## Known Follow-up

`PtyEngine` does not emit `usage` (input/output/cache_read/cache_creation token
counts) from its `result` events — the context-window budget gate in
`session_runner.py` always falls into its "plenty of headroom" branch as a
consequence. Restoring per-turn token tracking for PTY-driven sessions is worth
a dedicated follow-up issue.
