---
description: Locate the estimator's zero by running the placebo harness
argument-hint: [n_windows]
---

Run the placebo harness over ${1:-300} windows where no promotion occurred.

Report the distribution's centre, spread, and 5th/95th percentiles, and state where zero actually sits for this estimator. Then compare the estimates in `data/out/campaigns.parquet` against that band and list which fall inside it.

For anything inside the band, phrase the conclusion as a diagnosis about the comparison — it carries no signal — not as a verdict that the effect is absent. Preserve that distinction in the wording.
