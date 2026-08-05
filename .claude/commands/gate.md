---
description: Run the feasibility audit and report whether a campaign is measurable
argument-hint: <campaign or treatment spec>
---

Run the Gate 2 feasibility audit for `$1`.

1. Load `data/interim/panel.parquet`.
2. Run every check in `promo/audit.py`: variation axes, overlap via the propensity classifier, treatment collisions, horizon versus repurchase cycle, and the break-even share.
3. Emit a `GateResult` for each.

Report in this order: the overall verdict (measurable, bounded, or not identified), the single binding constraint if it is not measurable, and what would have to change about the data or the campaign design to make it measurable.

Do not proceed to estimation even if the audit passes. This command is the audit only.
