---
name: grill-with-docs
description: A relentless interview to sharpen a plan or design, invoked when the prompt contains @file references.
disable-model-invocation: true
---

Run a `/rhubarb:grilling` session.

Note: the interactive `/grill-with-docs` skill also generates ADR/glossary docs via a `/domain-modeling` skill; that skill is not part of Rhubarb's private plugin (Rhubarb's automated prompts don't originate from a workflow that expects those docs), so this falls back to plain `/rhubarb:grilling`. If Rhubarb's usage ever needs ADR/glossary generation, bring `/domain-modeling` into this plugin at that point.
