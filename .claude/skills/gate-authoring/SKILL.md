---
name: gate-authoring
description: Add a new refusal condition to the pipeline. Use when a data or identification problem needs to become a named gate rather than an ad-hoc check.
---

# Adding a gate

A gate turns a silent failure into a named, explainable refusal. Every gate needs five things, in this order.

## 1. Reason code

Add a constant to `promo/gates.py`, screaming snake case, describing the condition and not the fix: `NO_OVERLAP`, not `NEEDS_MORE_DATA`.

## 2. Detection

A pure function in the relevant stage module returning `(bool, dict)` — whether the condition fired, plus the evidence that decided it. The evidence dict must contain the numbers a sceptic would ask for, not just a boolean.

## 3. Severity

Choose one:
- `refuse` — the estimate is not identified. The pipeline stops and downstream stages do not run.
- `bounded` — a weaker quantity is available. State which one, for example ordinal ranking where cardinal depth is unavailable.
- `pass` — recorded for the audit trail even when nothing is wrong.

## 4. Message

One or two sentences a category manager could act on. Name what is missing and what it would take to fix it. Never blame the user, never use the word error, and never imply the effect does not exist when what you mean is that this comparison cannot see it.

Write the message as a template in `promo/gates.py` with the evidence interpolated. The LLM layer rewrites it for display but must have a correct deterministic fallback.

## 5. Test

A test that constructs data guaranteed to trigger the gate, and a second that constructs data guaranteed not to. A gate that has never fired in a test does not work.

## Finally

Add the row to the reason-code table in `docs/runbook.md` and to the audit screen in `app/app.py`. A gate the user cannot see in the interface does not exist as far as the demo is concerned.
