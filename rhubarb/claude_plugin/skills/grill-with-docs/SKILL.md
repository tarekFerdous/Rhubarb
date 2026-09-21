---
name: grill-with-docs
description: A relentless interview to sharpen a plan or design, invoked when the prompt contains @file references.
disable-model-invocation: true
---

Run a `/rhubarb:grilling` session.

**Never spawn a subagent.** This applies to the `/rhubarb:grilling` session it runs too — never dispatch a subagent (via the Agent tool, a fork, or any other delegation mechanism), and never run anything in parallel.

Note: the interactive `/grill-with-docs` skill also generates ADR/glossary docs via a `/domain-modeling` skill; that skill is not part of Rhubarb's private plugin (Rhubarb's automated prompts don't originate from a workflow that expects those docs), so this falls back to plain `/rhubarb:grilling`. If Rhubarb's usage ever needs ADR/glossary generation, bring `/domain-modeling` into this plugin at that point.
