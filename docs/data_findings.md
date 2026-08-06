# Data findings

What is actually in the Dunnhumby files, established in Phase 1. Sections are
added as tasks complete; a heading absent here means the question has not been
asked yet, not that it has no answer.

Every figure below was computed on `data/interim/transactions.parquet` (the
mirror of `transaction_data.csv`, which CLAUDE.md forbids reading directly) and
`data/raw/product.csv`, by full scan in DuckDB. Each section names the script
that reproduces it.

---

## Phase 1 summary — the settled decisions

Five decisions, settled in Phase 1, in one place. Each is argued from evidence in
the task section named beside it; this list is the reference, those sections are
the record of why. Do not re-litigate these — reopen one only if the validity
check named against it fails.

| # | Decision | From |
|---|---|---|
| 1 | **The panel grain is `PRODUCT_ID` × `STORE_ID` × `WEEK_NO`.** Household-level work is a separate, optional axis. | Task 1.4 |
| 2 | **Volume-measured rows are excluded**, identified by `DEPARTMENT IN ('KIOSK-GAS', 'MISC SALES TRAN')`. | Task 1.2 |
| 3 | **Regular price uses reconstruction A**: `(SALES_VALUE + \|RETAIL_DISC\| + \|COUPON_MATCH_DISC\|) / QUANTITY`. `COUPON_DISC` is excluded. | Task 1.3 |
| 4 | **The treatment is `display`**, with `mailer` as a covariate and never as the treatment. Campaign membership and coupon redemption are excluded as panel treatments. | Task 1.4 |
| 5 | **The estimation window is weeks 18–101.** | Tasks 1.4, 1.5 |

**On (2):** these rows are 1.07% of transactions but 98.71% of all raw units, at
a fraction of a cent each. `DEPARTMENT` is the discriminator, not unit value and
not `COMMODITY_DESC` — the latter labels petrol as `COUPON/MISC ITEMS`. The
exclusion is executed in Phase 2 and must record its effect on row count, unit
total, and sales value, before and after.

**On (5):** the window is the intersection of two independent constraints.

- **Panel entry.** The household panel is still recruiting through the early
  weeks: 3.52% of households shop in week 1, and 99% have not entered until week
  18. Weeks 1–17 are excluded because **panel size is still growing there, so any
  level comparison against those weeks compares panel size rather than demand**.
  Weeks 1 and 102 are additionally 5- and 6-day calendar stubs.
- **Treatment coverage.** `causal_data` records no treatment outside weeks 9–101.

18–101 is where both hold. The window is fixed here, before any estimate is
produced, rather than chosen after seeing results.

**Nothing from Phase 1 is still open.** The two items Task 1.4 left undecided —
the `causal_data` duplicate-key resolution rule and the collapse-before-join
requirement — were both resolved in Task 2.5 and are recorded as settled
decision 7 below.

**Closed since Phase 1:** the price level test owed by Task 1.3 was run in Task
2.3 and **confirms decision 3** — reconstruction A is closer to the observed
regular price on 6.75% of test groups and B on none. See *The level test* at the
end of Task 1.3.

**Added in Task 2.3:** a sixth settled decision, on price-status support.

| # | Decision | From |
|---|---|---|
| 6 | **A product needs at least 8 distinct priced weeks before anything is claimed about its depth.** Below that it is `insufficient_support`, tested before the bounded rule. | Task 2.3 |

**On (6):** the two-way identified/bounded split conflated two different
failures. 18,217 products looked `bounded`, but their median support was **one**
priced product-store-week — they were not perpetually discounted, they were
barely observed. Separating them leaves **3,445 genuinely bounded** products and
moves **14,772** to `insufficient_support`.

The threshold is 8 for three independent reasons that agree:

1. **Precedent.** Task 1.3 required at least 8 weeks of a product before scoring
   its price stability. Same question, same bar.
2. **Inference.** At the observed on-deal base rate of 50.25%, eight consecutive
   on-deal weeks arise by chance with probability 0.5⁸ ≈ **0.4%**, so "always on
   deal" is a real inference at eight weeks. At three weeks it is 12.5% — a
   coin-flip dressed as a finding.
3. **Modelling.** Task 2.6's middle rolling window is 8 weeks. A product that
   cannot fill it has no rolling feature to be modelled on.

What the threshold costs, and why it is affordable:

| status | products | share | units | sales value |
|---|---:|---:|---:|---:|
| `identified` | 24,537 | 26.7% | 79.54% | **75.45%** |
| `bounded` | 3,445 | 3.7% | 15.06% | 14.99% |
| `insufficient_support` | 63,929 | 69.6% | 5.40% | **9.56%** |

**Seven products in ten, under a tenth of the money.** That asymmetry is the
justification: the refusal is wide in the catalogue and narrow in revenue, which
is the correct shape for a long-tail exclusion. 19 of the thin products have no
reconstructable price at all and are a subset of this status, not a fourth one.

**Added in Task 2.5:** a seventh settled decision, closing both of Task 1.4's
open items.

| # | Decision | From |
|---|---|---|
| 7 | **Duplicate `causal_data` keys resolve by "any treated wins",** after collapsing to one row per key and before the join. | Task 2.5 |

**On (7), the rationale is structural rather than conservative.** `causal_data`
contains **zero** untreated rows — measured, not assumed, and asserted on every
run. It is a treatment log, not a panel, so a row exists only because something
was promoted. A `display = '0'` row is therefore present on account of its
mailer, and its zero display field records *absence of relevance*, not absence
of display. Reading it as evidence of no-display misreads the file.

**The evidence, checked before the rule was adopted:**

| | |
|---|---:|
| conflicting keys | 15,245 |
| of which `display` disagrees | 15,208 |
| zero-display conflict rows carrying a **real mailer** | **15,208 / 15,208 = 100.0000%** |
| zero-display conflict rows with `mailer = '0'` too | **0** |
| `display` conflicts that are real-code against real-code | **0** |
| `mailer` conflicts that are real-code against real-code | 9 |

Not one exception. Had a material share of those rows carried `mailer = '0'`
they would have no reason to exist under this reading and the decision would
have been reopened; none does. The nine real-vs-real `mailer` conflicts take the
lexicographic max, which moves the recorded *code* only and never the boolean.

**"First wins" was considered and rejected.** The `'0'` record appears first in
the file **99.76%** of the time — a real code is first on only 37 of 15,245
keys — so file order is not arbitrary. "First wins" would function as
"untreated always wins" in disguise, stripping the treatment from 15,208 keys.

**What the rule costs.** Only **614** of the 15,245 conflicting keys appear in
the panel at all, so the three candidate rules differ by 614 rows — 0.026
percentage points of the panel. The rule is recorded because it must be, not
because it moves the number.

| rule | treated product-store-weeks | of panel |
|---|---:|---:|
| **any treated wins** (chosen) | **236,689** | **10.07%** |
| all must agree | 236,075 | 10.05% |
| drop conflicts | 236,075 | 10.05% |

**Collapse before join is executed and asserted.** `promo/treatment.py` collapses
`causal_data` to one row per key in DuckDB — restricted by semi-join to keys the
panel holds, so only ~483k of 36.8M rows are ever materialised — then joins with
`validate="one_to_one"` and raises `CoverageError` if the row count moves. The
615 silently duplicated keys Task 1.4 found in a naive join cannot recur.

**Added in Task 3.3:** an eighth settled decision, on the baseline's covariates.

| # | Decision | From |
|---|---|---|
| 8 | **`in_mailer` joins the baseline feature set as a control covariate.** Not as a treatment. | Task 3.3 |

**On (8): the decision was already recorded, it was never implemented.** Settled
decision 4 states that "`mailer` is retained as a control covariate, never as
the treatment". The Phase 2.6 feature set does not contain it. Decision 4 was
documented and not carried out, and Task 3.3 is what surfaced the gap.

**What it costs to leave it out.** The Phase 4 baseline trains on `treated == 0`
rows. **13.06% of those control rows carry a mailer** — they were promoted, and
the model is told nothing about it. It learns an inflated normal, the
counterfactual is fitted too high, and **every measured display effect is biased
towards zero**. That is 339,613 rows and 22.11% of panel units quietly lifting
the baseline.

**This makes `in_mailer` a control, not a treatment.** The treatment is still
`display` alone. The counterfactual for a treated row holds the mailer at its
observed value, which is what isolates the display effect: *what would this row
have done with the mailer that actually ran and without the display*.

**Reconciling with the causal-inference rule.** `.claude/rules/causal-inference.md`
says features "may not include anything derived from the promotion: no discount
depth, no promo flag, no post-treatment aggregates". `in_mailer` is a promo flag,
so the reconciliation must be explicit rather than assumed:

- The rule exists to stop the counterfactual seeing **the promotion being
  measured**. `in_mailer` is a *different* mechanic, and conditioning on other
  treatments is what prevents omitted-variable bias rather than causing leakage.
- `in_mailer` is not derived from `display`, is known before the display
  decision, and is not a post-treatment aggregate.
- Depth, `on_deal` and the discount columns remain excluded. This decision
  admits exactly one flag and nothing else.

**Alternatives considered and rejected:**

1. **Restrict controls to mailer-free rows.** Discards 339,613 rows of control
   mass — 22.11% of panel units — to solve by deletion what a covariate solves
   by conditioning.
2. **Report a joint display-and-mailer effect.** Answers a different question
   than decision 4 asks. Decision 4 chose `display` precisely because it varies
   across stores within a week and `mailer` does not; merging them discards that.

**What the covariate does not fix.** **39.80% of treated rows carry both
mechanics.** For those, the estimate remains a joint effect — the display effect
*conditional on a mailer also running* — because holding `in_mailer` fixed at
True is holding an interaction fixed, not removing it. Those rows must be
**reported separately from the clean 60.20%**, not pooled into a single "display
effect". Adding the covariate fixes the control group; it does not purify the
treated group.

**Added in Task 4.1:** two more settled decisions, both about what the baseline
is allowed to see.

| # | Decision | From |
|---|---|---|
| 9 | **`price_rel_category` may never be a model feature, because its missingness *is* the outcome.** The baseline uses `price_rel_category_lag` instead, and `missingness_coupling()` checks every feature for the same shape. | Task 4.1 |
| 10 | **Identity — `PRODUCT_ID`, `STORE_ID`, `COMMODITY_DESC` — is off by default in the baseline.** A default, not a prohibition: `include_identity=True` turns it on, and Task 4.4 must settle it by evidence. | Task 4.1 |

**On (9): the column is observed if and only if the product sold.** Measured on
the real panel over the control rows in weeks 18–101:

| | |
|---|---:|
| training rows | 2,351,749 |
| `price_rel_category` null | 2,046,518 |
| `units == 0` | 2,046,518 |
| null **and** zero | **2,046,518** |
| null and `units > 0` | **0** |
| priced and `units == 0` | **0** |

`P(units = 0 | price null) = 1.0000` and `P(price null | units = 0) = 1.0000`,
with no exception in either direction. The ratio needs an observed price and a
week with no sale has none, so the column's *absence* reports the outcome. The
first fit of `promo/baseline.py` gave it **94.3% of the gain**: the model was
not predicting demand, it was reading zeros off a null.

**What it cost to stop reading the answer.** Held-out error on weeks 94–101,
225,038 rows, same split both times:

| | MAE (units) | RMSE (units) |
|---|---:|---:|
| with `price_rel_category` (leaking) | **0.100** | 0.494 |
| with `price_rel_category_lag` | **0.273** | 0.721 |

**The better number was the leaking one**, and it was better precisely because
it was leaking. Anyone comparing baseline error across this commit will see a
2.7× regression; it is the price of an estimate that is about demand rather
than about which rows happened to sell. Recorded here so the improvement is
never "restored".

**Why the Phase 2 leakage certificate could not catch it.**
`test_no_feature_uses_data_from_a_later_week` rebuilds the panel with every
value after week W perturbed and requires that no column differ at weeks ≤ W.
That is a strong test of the right thing — **temporal** leakage, in feature
*values*. This leak is neither: it is contemporaneous, within week *w*, and it
lives in the *missingness pattern* rather than in any value. Perturbing later
weeks leaves the null pattern of earlier weeks untouched, so the certificate
passes on a panel carrying a perfect outcome readout. The two checks are
complements, not substitutes, and neither replaces the other.

**The replacement.** `add_price_history()` derives `price_rel_category_lag` —
the last ratio observed **strictly before** week *w*, within product-store,
computed over the whole panel *before* any row filter, so that "the previous
week" cannot quietly become "the previous unpromoted week". It is still a
*regular* price ratio, never a paid price, so it does not encode the deal. The
coupling drops to `P(units = 0 | null) = 0.9586` and `P(null | units = 0) =
0.1572` — it is null where the pair has no priced history yet, which is past
information of the same class as `units_lag_1`.

**`missingness_coupling()` is the general check, and it replaces
blocklisting.** A named list of forbidden columns only ever catches the leak
someone already found. The check measures, for every feature with nulls, both
`P(units = 0 | feature null)` and `P(feature null | units = 0)`, and raises
`OutcomeLeakError` when both exceed 0.99. Both directions are required, so a
merely rare null cannot trip it. A test smuggles the leaking column in under a
name the blocklist does not know, which is how we know the *measurement* is
what fires. `price_rel_category` stays on the forbidden list as well, but the
list is now the second line of defence rather than the only one.

**Recorded and not repaired:** `n_stores_carrying` counts stores with a sale of
the product that week **including the focal store**, so a selling row adds one
to its own feature. It is not deterministic, so the check does not fire, but it
is a partial use of the current week's outcome and it is now the second-largest
gain in the fit at 30.7%. The fix belongs in Phase 2.6, which builds the column.

**On (10): identity is a default, not a prohibition.** Task 3.2 already excluded
identifiers from the overlap classifier, for a reason that transfers directly:
*a tree given `PRODUCT_ID` memorises which products get promoted rather than
learning why*. The baseline has the same exposure in a different shape —
identity lets the model fit each product's level directly instead of learning
demand structure, and a counterfactual that has memorised a product's mean is
not a model of its demand. The lags and rolling means already carry
product-level persistence, so the level is available without the identifier;
`units_roll_mean_13` takes 46.3% of the gain in the shipped fit.

**It is a default because the argument is not decisive.** Identity may well fit
better on a 300-product scope, and "better fit" is not the same as "better
counterfactual" — which is exactly the kind of claim that should be settled by
measurement. `fit_baseline(include_identity=True)` turns it on and the
diagnostics record which set was used, so the comparison is a parameter change
rather than an edit.

**Task 4.4 owes the evidence, on both open axes.** The synthetic-truth harness
must report bias and interval coverage **with and without identity**, and **with
and without the contemporaneous block** — the second axis being the one Task 3.2
already asked for, where `n_stores_carrying`, `category_units_ex_focal` and
`store_traffic` carry 77.5% of the classifier's gain and are all measurable
consequences of the promotion. Two modelling choices, four cells, settled by
recovery against a known truth rather than by argument. The obligation is
written into Task 4.4's **Done when** in `docs/plan.md`.

**Added in Task 4.3:** an eleventh settled decision, on which cells a campaign
is made of.

| # | Decision | From |
|---|---|---|
| 11 | **A product-store belongs to a campaign if it was treated in *any* of the campaign weeks, and thereafter every week of the measurement window counts for that cell** — including campaign weeks its own display was down. | Task 4.3 |

**On (11): the alternative was to count only the weeks each cell was actually
treated.** That rule is tighter and it is wrong for this estimand. A display
that ran two of four weeks still pulls demand forward, and the dip that follows
is attributable to it — but it lands in weeks that cell was untreated, so the
tighter rule drops exactly the observations the horizon rule exists to capture.
It would bank each cell's peak and discard its trough, which is settled decision
5's error committed one cell at a time instead of one campaign at a time.
Carryover inside the window has the same shape: a store whose display came down
in week 3 has not returned to baseline by week 3.

**What it costs, and the cost is not small.** Exposure inside a campaign window
is very uneven. Across every product-store treated in weeks 80–83:

| weeks treated | cells | share |
|---:|---:|---:|
| 1 | 2,750 | **43.9%** |
| 2 | 1,337 | 21.3% |
| 3 | 999 | 15.9% |
| 4 | 1,185 | 18.9% |

Nearly half of all treated cells ran for a single week of the four, and fewer
than one in five ran throughout. Under this rule **a cell treated once carries
the same window weight as a cell treated four times.** The reported figure is
therefore a **campaign-level average over uneven exposure, not a per-week dose
response**, and it must never be read as "what a week of display does". A
campaign whose cells mostly ran once will report a smaller per-cell effect than
one whose cells all ran throughout, with no difference in the underlying
response.

`estimate_lift` records `treated_share_of_campaign_weeks` for exactly this
reason — the share of campaign-window pair-weeks that were actually treated.
On the carbonated-water campaign above it is **0.74**, so a quarter of the
window's pair-weeks are counted-but-untreated. A reader who sees that number
knows how much averaging sits behind the estimate; a reader who does not, does
not.

**Task 4.4 owes a check on this.** The synthetic harness can generate uneven
exposure directly — cells treated for one, two, three, and four weeks of the
same campaign, against a known per-week effect — and report whether recovery of
the campaign-level truth is biased by the rule and in which direction. If it is,
the fix is a stated weighting, not a quiet change to which cells count. Recorded
in Task 4.4's section in `docs/plan.md`.

---

## Task 1.1 — Shape and coverage

Reproduce: `.venv/bin/python scratch/explore_01.py`.

### The spine

`transaction_data` (read from the parquet mirror): **2,595,732 rows**, DAY
**1–711**, WEEK_NO **1–102**, 2,500 households, 92,339 products, 582 stores,
276,484 baskets.

### Demographics cover a third of households but most of the money

| | |
|---|---:|
| households in `transaction_data` | 2,500 |
| households in `hh_demographic` | 801 |
| in both | 801 |
| **demographics as a share of shoppers** | **32.0%** |
| **share of transaction rows they carry** | **55.0%** |
| share of sales value they carry | 55.8% |

Every household in `hh_demographic` also shops, so the coverage gap is one-way.
The two percentages are the finding: demographics reach **under a third of
households but over half the rows**, because the covered households are the
heavier shoppers. So a demographic segmentation is not a small sample of the
data, but it *is* a biased sample of the households — it over-represents
frequent shoppers on both counts. Anything using `hh_demographic` runs on that
subset and must say so.

### Treatment coverage is the binding constraint

`causal_data` reaches **115 of 582 stores** (19.8%) and **weeks 9–101** (93 of
102). Those two limits, not the size of the transaction file, are what bound the
usable panel. They are the reason for settled decision (5); see Task 1.4 for the
per-axis coverage table and Task 1.5 for the panel-entry half of the window.

### Baskets exist, so cannibalisation is identifiable

2,595,732 rows over 276,484 baskets = **9.39 items per basket**. The data is
basket-level, not store-total: what else was in the trolley when the promoted
item was bought is observable. Projects with store totals only cannot ask
whether a lift came from the category next to it. This is what makes Phase 6
possible at all.

---

## Task 1.2 — The quantity column

Reproduce: `.venv/bin/python scratch/explore_02.py`. Full scan, no sampling, so
the shares are exact for the mirror.

**Finding: `QUANTITY` is not one unit of measure. 1.07% of rows carry 98.71% of
all raw units, and those rows are petrol and miscellaneous kiosk transactions
priced at roughly a quarter of a cent per unit. Summing `QUANTITY` across the
file measures fuel volume, not grocery demand.**

### The column as a whole

| | |
|---|---|
| rows | 2,595,732 |
| total `QUANTITY` | 260,685,622 |
| total `SALES_VALUE` | 8,057,463.08 |
| mean `QUANTITY` per row | 100.43 |
| min / max `QUANTITY` | 0 / 89,638 |
| median `QUANTITY` | 1 |
| p99 `QUANTITY` | 10 |
| p99.9 `QUANTITY` | 16,963 |

The median row buys one of something and the mean row buys a hundred. That gap
between p99 (10) and p99.9 (16,963) is the whole finding in one line.

Degenerate rows, which matter because they make unit value undefined or zero:

| case | rows | share |
|---|---:|---:|
| `QUANTITY = 0` | 14,466 | 0.56% |
| `QUANTITY < 0` | 0 | 0.00% |
| `SALES_VALUE = 0` | 18,850 | 0.73% |
| `SALES_VALUE < 0` | 0 | 0.00% |

There are **no returns** in this file — neither quantity nor sales value is ever
negative. The refunds live in the discount columns, not here.

### Where the units are

Ranking rows by `QUANTITY` descending:

| slice | share of all units |
|---|---:|
| top 0.01% of rows | 2.91% |
| top 0.1% | 20.39% |
| top 1% | 98.73% |
| top 10% | 99.00% |

### Unit value, `SALES_VALUE / QUANTITY`

Defined on the 2,581,266 rows with `QUANTITY > 0`. Range 0.000000 to 499.99,
mean 2.44, median 1.92.

| decade band | rows | row % | units | unit % |
|---|---:|---:|---:|---:|
| `SALES_VALUE <= 0` | 4,451 | 0.17% | 4,544 | 0.00% |
| 1e-3 .. 1e-2 | 23,153 | 0.90% | 257,330,170 | **98.71%** |
| 1e-2 .. 1e-1 | 281 | 0.01% | 538 | 0.00% |
| 1e-1 .. 1e+0 | 492,147 | 19.07% | 841,748 | 0.32% |
| 1e+0 .. 1e+1 | 2,022,688 | 78.36% | 2,466,522 | 0.95% |
| 1e+1 .. 1e+2 | 38,464 | 1.49% | 42,007 | 0.02% |
| 1e+2 .. 1e+3 | 82 | 0.00% | 93 | 0.00% |

Two populations, not a tail. Counted groceries sit in the 1e-1 to 1e+1 decades —
97% of rows, under 1.3% of units. A separate population sits two decades below
at a fraction of a cent and carries almost everything.

### Below the 0.05 line

| | below line | all rows | share |
|---|---:|---:|---:|
| rows | 27,714 | 2,595,732 | **1.07%** |
| `QUANTITY` | 257,334,912 | 260,685,622 | **98.71%** |
| `SALES_VALUE` | 604,322.40 | 8,057,463.08 | 7.50% |

Mean `QUANTITY` per row below the line: 9,285. Above it: about 1.3.

### What those rows are

By `DEPARTMENT`, ranked by units carried below the line:

| DEPARTMENT | rows | row % | units | unit % | sales | mean unit value |
|---|---:|---:|---:|---:|---:|---:|
| KIOSK-GAS | 20,277 | 73.17% | 221,253,105 | 85.98% | 514,974.01 | 0.0023 |
| MISC SALES TRAN | 2,884 | 10.41% | 36,077,082 | 14.02% | 89,342.26 | 0.0025 |
| GROCERY | 1,246 | 4.50% | 1,329 | 0.00% | 4.06 | 0.0009 |
| MEAT-PCKGD | 939 | 3.39% | 955 | 0.00% | 0.02 | 0.0000 |
| DRUG GM | 681 | 2.46% | 719 | 0.00% | 1.01 | 0.0007 |
| COUP/STR & MFG | 448 | 1.62% | 460 | 0.00% | 0.07 | 0.0002 |

Those two departments carry **99.998%** of the units below the line between them.
Everything after them is a long list of ordinary grocery rows that happen to be
priced at zero — giveaways and coupon lines — contributing rows but no units.

Measured the other way, against each department's own totals:

| DEPARTMENT | all rows | all units | % of its units below line | % of its rows below line |
|---|---:|---:|---:|---:|
| KIOSK-GAS | 22,059 | 221,254,887 | 100.00% | 91.92% |
| MISC SALES TRAN | 6,050 | 36,080,860 | 99.99% | 47.67% |
| COUP/STR & MFG | 817 | 1,011 | 45.50% | 54.83% |
| GROCERY | 1,646,076 | 2,194,762 | 0.06% | 0.08% |
| DRUG GM | 277,232 | 353,844 | 0.20% | 0.25% |

KIOSK-GAS and MISC SALES TRAN are wholly volume-measured. No real merchandise
department loses more than about 1% of its units to the line.

**`COMMODITY_DESC` does not identify these rows.** 99.95% of below-line units are
labelled `COUPON/MISC ITEMS`; the commodity literally named `FUEL` accounts for
6 rows and 0.05% of them. The single largest row in the file is `PRODUCT_ID`
6534178, `KIOSK-GAS` / `COUPON/MISC ITEMS`, 89,638 units for 250.00 — a tank of
petrol recorded in hundredths of a gallon under a catalogue label that says
nothing. `DEPARTMENT` is the usable discriminator here, not `COMMODITY_DESC`.

### The 0.05 threshold is not load-bearing

| threshold | rows | row % | units | unit % | sales % |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 4,451 | 0.17% | 4,544 | 0.00% | 0.00% |
| 0.005 | 27,604 | 1.06% | 257,334,714 | 98.71% | 7.50% |
| 0.010 | 27,604 | 1.06% | 257,334,714 | 98.71% | 7.50% |
| 0.050 | 27,714 | 1.07% | 257,334,912 | 98.71% | 7.50% |
| 0.100 | 28,301 | 1.09% | 257,337,464 | 98.72% | 7.50% |
| 0.250 | 48,630 | 1.87% | 257,391,890 | 98.74% | 7.62% |
| 0.500 | 154,734 | 5.96% | 257,612,825 | 98.82% | 8.59% |

Anywhere from 0.005 to 0.10 returns the same 98.71% of units. The two
populations are separated by a genuinely empty band, so the answer is a property
of the data rather than of the cutoff. Below 0.005 the filter stops catching
volume rows and catches only the 4,451 zero-priced rows, which are a different
phenomenon (free goods) carrying 4,544 units.

### Consequences for the pipeline

1. **Units are not comparable across departments.** Any `sum(QUANTITY)` at the
   PRODUCT_ID × STORE_ID × WEEK_NO grain that includes KIOSK-GAS or
   MISC SALES TRAN is a mixture of gallons and tins. A baseline trained on raw
   units in a panel that includes them fits petrol volume.
2. **The identifying column is `DEPARTMENT`, not unit value and not
   `COMMODITY_DESC`.** A unit-value threshold also sweeps in several thousand
   zero-priced grocery rows, which are a separate issue; `COMMODITY_DESC`
   labels petrol as `COUPON/MISC ITEMS`.
3. **Sales value is much less distorted than units.** These rows are 7.50% of
   `SALES_VALUE` against 98.71% of `QUANTITY`, so a revenue-denominated
   quantity is far more robust to this problem than a unit-denominated one.
4. **Nothing was dropped in Phase 1.** The flag proposed here —
   `DEPARTMENT IN ('KIOSK-GAS', 'MISC SALES TRAN')`, with the zero-price rows
   flagged separately — is now **settled decision (2)** in the Phase 1 summary.
   The exclusion itself happens in Phase 2 and must record its before/after
   effect on rows, units, and sales value; per CLAUDE.md these rows are flagged,
   never silently removed.

---

## Task 1.3 — The discount columns

Reproduce: `.venv/bin/python scratch/explore_03.py`. Part 1 is a full scan; part 2 scans
the 200 sampled products only.

**The decision is recorded at the end of this section.** Everything before it is
the evidence the decision was made on.

### The three columns

| column | nonzero rows | nonzero % | sum | as % of `SALES_VALUE` |
|---|---:|---:|---:|---:|
| `RETAIL_DISC` | 1,303,062 | 50.20% | -1,398,334.84 | 17.35% |
| `COUPON_DISC` | 36,422 | 1.40% | -42,611.54 | 0.53% |
| `COUPON_MATCH_DISC` | 17,449 | 0.67% | -7,575.81 | 0.09% |

Magnitude over the rows where each fires:

| column | min | p1 | p25 | median | p75 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RETAIL_DISC` | -180.00 | -6.98 | -1.22 | -0.69 | -0.32 | -0.04 | **+3.99** |
| `COUPON_DISC` | -55.93 | -6.73 | -1.00 | -1.00 | -0.50 | -0.25 | -0.08 |
| `COUPON_MATCH_DISC` | -7.70 | -1.00 | -0.50 | -0.45 | -0.30 | -0.25 | -0.01 |

**The retailer's own markdown is the dominant mechanic by two orders of
magnitude.** It touches half of all rows and 17.35% of revenue; the two coupon
mechanics together touch under 2% of rows and 0.62% of revenue.

**`RETAIL_DISC` is not uniformly negative.** 36 rows are positive, summing
+8.53, max +3.99; 17 of them have `SALES_VALUE = 0`. Both candidate
reconstructions use `abs()`, which silently converts those surcharges into
discounts. Sign-flip count is small, but the rule "all three arrive negative"
has exceptions and code that assumes otherwise will not notice. `COUPON_DISC`
and `COUPON_MATCH_DISC` have no positive rows.

### How the mechanics co-occur

| `RETAIL_DISC` | `COUPON_DISC` | `COUPON_MATCH_DISC` | rows | row % | sales % |
|---|---|---|---:|---:|---:|
| yes | – | – | 1,283,013 | 49.43% | 49.41% |
| – | – | – | 1,276,297 | 49.17% | 49.00% |
| yes | yes | yes | 10,846 | 0.42% | 0.36% |
| – | yes | – | 9,770 | 0.38% | 0.42% |
| yes | yes | – | 9,203 | 0.35% | 0.57% |
| – | yes | yes | 6,603 | 0.25% | 0.24% |

`COUPON_MATCH_DISC` fires on **zero** rows where `COUPON_DISC` is zero. The
match is strictly nested inside the manufacturer coupon, which is what the
mechanic's name claims, and means it can never be reconstructed independently.

Discount depth where each mechanic fires, `|disc| / (SALES_VALUE + |disc|)`:

| column | rows | p25 | median | p75 | p95 | at 100% |
|---|---:|---:|---:|---:|---:|---:|
| `RETAIL_DISC` | 1,303,026 | 15.67% | 24.62% | 37.11% | 50.13% | 0.29% |
| `COUPON_DISC` | 36,422 | 15.81% | 24.01% | 37.50% | 100.00% | **13.72%** |
| `COUPON_MATCH_DISC` | 17,449 | 10.03% | 15.08% | 22.50% | 40.00% | 0.07% |

13.72% of coupon rows are 100% discounts — the shopper paid nothing. Under
reconstruction B those rows price at the coupon face value; under A they price
at `|RETAIL_DISC|` alone, which is often zero, giving a reconstructed regular
price of 0.00. Neither is obviously right and both are wrong for some rows.

### The two reconstructions

```
A = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
B = (SALES_VALUE + |RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
```

Sample: the 200 products with the most transactions that appear in at least 8
weeks — 501,918 rows, 249.7M units, 1,661,358.26 of sales, spanning all 102
weeks. Departments: GROCERY (106), PRODUCE (55), DELI (10), MEAT (9), DRUG GM
(8), PASTRY (3), MEAT-PCKGD (3), MISC SALES TRAN (2), SALAD BAR (2), KIOSK-GAS
(2). The four volume-measured products flagged in Task 1.2 are scored, then
scored again with them removed.

**A and B differ on 1,417 rows — 0.28% of the sample.** They are algebraically
identical everywhere `COUPON_DISC = 0`. That share is the ceiling on any
difference between them, and it is why most comparisons below are ties rather
than close calls.

Stability of the modal price, scored per product and compared pairwise on the
same products, weeks and rows. Medians across the 200 products:

| metric | A | B | better |
|---|---:|---:|:--|
| within-week support of the modal price | 0.9507 | 0.9495 | **A** |
| distinct modal prices across weeks | 4.50 | 4.50 | tie |
| share of weeks at the dominant price | 0.5514 | 0.5514 | tie |
| coefficient of variation of weekly modal price | 0.1048 | 0.1048 | tie |
| median week-over-week move of modal price | 0.0000 | 0.0000 | tie |

Paired per-product wins (tolerance 1e-9; see *Determinism* below):

| metric | A wins | B wins | tie |
|---|---:|---:|---:|
| within-week support of the modal price | **77** | **0** | 123 |
| distinct modal prices across weeks | 2 | 1 | 197 |
| share of weeks at the dominant price | 2 | 0 | 198 |
| coefficient of variation of modal price | 2 | 2 | 196 |
| median week-over-week move of modal price | 1 | 0 | 199 |

Excluding the four volume-measured products (196 products) the picture is
unchanged: within-week support 77–0 to A, everything else 2–1, 2–0, 2–2, 1–0.

**The one metric that separates them separates them unanimously.** On 77 of 200
products, adding `COUPON_DISC` back scatters the within-week price distribution;
on none of them does it tighten it. The other four metrics see essentially
nothing, because they read the modal price itself rather than its support, and
0.28% of rows almost never move a mode.

Where the two disagree most on the count of distinct modal prices:

| PRODUCT_ID | COMMODITY | weeks | modes A | modes B | dominant A | dominant B |
|---:|---|---:|---:|---:|---:|---:|
| 971922 | CHEESES | 101 | 38 | 37 | 3.05 | 3.00 |
| 1132771 | CHEESES | 102 | 27 | 28 | 3.11 | 3.05 |
| 870547 | MARGARINES | 102 | 7 | 8 | 1.99 | 1.99 |
| 820165 | CITRUS | 91 | 9 | 9 | 0.50 | 0.50 |

Even the worst disagreement is one mode in thirty-eight, and the two dominant
prices differ by five cents. For PRODUCT_ID 870547 the two weekly modal series
are identical over its first 26 weeks (3.19 for nine weeks, then 1.99).

### Reading of the evidence

The direction is consistent with the mechanics: a manufacturer coupon is
deducted at the till and does not move the shelf price, so adding it back pushes
those rows above the price everyone else in that store-week paid, which is
exactly the scatter the support metric picks up. Reconstruction A never loses on
that metric. But the effect is small, confined to 0.28% of rows, and measured on
a metric already saturated at 95%.

### What this evidence cannot settle

- **Stability is not correctness.** The criterion rewards any transformation
  that varies less; a constant would score perfectly. It says A's prices are
  more concentrated, not that A's level is the true shelf price. A level test —
  comparing each reconstruction against the modal price on rows where all three
  discounts are zero — would test that, and has not been run.
- **The 100%-coupon rows** (13.72% of coupon rows) are mispriced by both
  candidates and need their own handling either way.
- **Neither candidate addresses the 36 positive `RETAIL_DISC` rows.**
- **Cost-bearer, not just price level.** A and B answer "what would the shopper
  have paid"; the accounting in Phase 5 needs "what did the retailer forgo",
  which is a different combination of the same three columns. Choosing a shelf
  price does not choose that.

### Determinism

The paired win counts are computed at a relative tolerance of 1e-9, not by exact
equality. DuckDB sums doubles in thread-arrival order, so scores that are
mathematically identical differ in the last bits between runs. With exact `=`,
the within-week-support count moved between 104–28–68 and 97–36–67 on two
consecutive runs of the same script — noise, entirely from ties being
reclassified as wins. At tolerance it is a stable 77–0–123. Any downstream
comparison of two float aggregates needs the same treatment.

---

### DECISION — reconstruction A

**The regular price is reconstructed as A:**

```
regular_price_per_unit = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
```

`COUPON_DISC` is **excluded** from the shelf price.

**Rationale.** `COUPON_DISC` is manufacturer-funded and reimbursed to the
retailer, so the retailer's revenue is unaffected by it. Adding it back would
double-count that reimbursement and invent a price that was never charged to
anyone. `RETAIL_DISC` and `COUPON_MATCH_DISC` are both borne by the retailer and
both move the price the shopper faces, so both belong in the reconstruction.

**Empirical support.** Within-week support of the modal price favours A **77–0**
with 123 ties across the 200 sampled products, and 77–0 again with the
volume-measured products removed. No metric favours B.

**Scope of this decision.** It fixes the *shelf price* only. What the retailer
forgave — the Phase 5 accounting quantity — is a different combination of the
same three columns and is not decided here. All three columns stay separate
through the pipeline.

### Implementation requirements for Phase 2

These are conditions on the code that implements A, not suggestions.

1. **Do not call `abs()` on `RETAIL_DISC`.** 36 rows are positive (max +3.99);
   `abs()` would silently convert surcharges into discounts. Flag those rows as
   surcharges and exclude them, recording the exclusion's effect on row count,
   units, and sales value like any other exclusion.
2. **Guard the divide.** 14,466 rows have `QUANTITY = 0`, so the per-unit price
   is undefined for 0.56% of rows. The reconstruction must produce a null there,
   never an infinity and never a silent zero.
3. **Compare prices at a tolerance of 1e-9, never with `==`.** Exact float
   equality was not reproducible across runs of `explore_03.py` — DuckDB sums
   doubles in thread-arrival order — and the same hazard applies to any price
   equality test, modal-price grouping, or "did the price change" check.

### The level test — run in Task 2.3, decision 3 stands

Reproduce: `promo.prices.level_test()`; the result is stored under `level_test`
in `data/interim/prices_diagnostics.json`.

The Phase 1 evidence established that A's prices are more *concentrated* than
B's. Concentration is not correctness — a constant would score perfectly. The
test owed here was of the *level*: within a product-store-week holding both
undiscounted and discounted rows, the undiscounted rows show the regular price
directly, so the reconstruction computed from the discounted rows can be scored
against it. B is scored on the identical groups, so the test discriminates
rather than merely describes.

**5,350 product-store-weeks support the test** — those holding both kinds of
row. A second pass at product-store grain, which relaxes the week requirement,
carries 161,316 groups.

| | A | B |
|---|---:|---:|
| median absolute error, product-store-week | **0.0000** | 0.0000 |
| within 1 cent | **70.84%** | 64.11% |
| within 5 cents | **72.36%** | 65.64% |
| strictly closer to observed | **6.75%** | **0.00%** |
| median absolute error, product-store | **0.0020** | 0.0133 |
| strictly closer, product-store | **5.07%** | 0.34% |

**A is closer on 6.75% of groups; B is closer on none of them.** The shape
repeats Phase 1's 77–0 for the same reason: the two reconstructions are
algebraically identical wherever `COUPON_DISC = 0`, so most groups tie, and
every group that breaks the tie breaks it for A. At product-store grain, where
more groups contain a coupon, B's median error is 6.7× A's.

**DECISION 3 STANDS on level as well as on stability.** It is not reopened.

**What the 29% that miss by more than a cent are, and why they do not bear on
this question.** Their median signed error is **+0.19**, about **+10%**
relative, and 62% overshoot. But **only 0.02% of them involve a coupon at all**,
so they are not evidence about `COUPON_DISC` and cannot separate A from B. They
are a different phenomenon: the observed regular price is not constant within a
product-store-week, and the undiscounted row is not always at the shelf price —
one `PRODUCT_ID` can span pack sizes, and a temporary price cut recorded through
no discount column looks undiscounted. That the error is mostly positive says
the reconstruction lands *above* those rows, which is what a mid-week markdown
outside the three columns would produce. Task 2.4's price index is built from
unpromoted rows and inherits this; it is a known limit, not a defect in the
reconstruction.

---

## Task 1.4 — The treatment candidates

Reproduce: `.venv/bin/python scratch/explore_04.py`. **The decision is recorded at the end
of this section**; everything before it is the evidence it was made on.

**Finding: `display` and `mailer` are the only candidates that vary on the panel
axis, which confirms the settled decision. But they do not vary the same way —
`mailer` is a chain-wide decision with almost no cross-store variation, so a
store-level comparison identifies `display` and not `mailer`. And the treatment
file reaches only 20% of stores and 24% of products.**

### The week calendar, since campaign dates depend on it

`WEEK_NO = floor((DAY + 8) / 7)`, verified at **0** disagreeing rows across all
2,595,732 transactions. The obvious `floor((DAY - 1) / 7) + 1` is wrong on
694,450 rows. The calendar does not start on a week boundary: week 1 is a 5-day
stub (DAY 1–5) and week 102 a 6-day stub (DAY 706–711). Weekly totals in those
two weeks are understated, and any day-to-week mapping — campaign windows,
redemption dates — must use this formula.

### `causal_data` — the codes

36,786,524 rows over 36,771,279 distinct product-store-weeks, `WEEK_NO` **9–101**
only; no treatment is recorded for weeks 1–8 or 102.

| `display` | rows | share | | `mailer` | rows | share |
|---|---:|---:|---|---|---:|---:|
| `'0'` | 21,038,745 | 57.19% | | `'A'` | 17,106,789 | 46.50% |
| `'9'` | 2,699,467 | 7.34% | | `'0'` | 11,534,183 | 31.35% |
| `'5'` | 2,575,289 | 7.00% | | `'D'` | 4,467,453 | 12.14% |
| `'7'` | 2,362,118 | 6.42% | | `'H'` | 1,560,395 | 4.24% |
| `'3'` | 2,073,738 | 5.64% | | `'F'` | 1,077,549 | 2.93% |
| `'6'` | 1,816,021 | 4.94% | | `'J'` | 306,924 | 0.83% |
| `'2'` | 1,812,840 | 4.93% | | `'L'` | 301,327 | 0.82% |
| `'1'` | 1,102,141 | 3.00% | | `'C'` | 291,059 | 0.79% |
| `'A'` | 713,180 | 1.94% | | `'X'` | 120,823 | 0.33% |
| `'4'` | 592,985 | 1.61% | | `'Z'` | 19,453 | 0.05% |
| | | | | `'P'` | 569 | 0.00% |

Both are categorical, not ordinal — `display` `'9'` is not more display than
`'2'`. `'A'` is valid in **both** columns and means a different thing in each.
`display` has no `'8'`.

**`causal_data` contains no untreated rows.** Every row has `display <> '0'` or
`mailer <> '0'`; the crosstab has three cells, not four (57.19% mailer-only,
31.35% display-only, 11.45% both). The file is a treatment log, not a panel.
Presence in it *is* treatment, which is exactly why "absent means untreated" is
the right reading rather than a convenient one.

### Duplicate keys — 15,245 product-store-weeks disagree with themselves

15,245 keys appear twice (30,490 rows), and **100% of them conflict**:

- `display` disagrees on 15,208, and in every single case it is `'0'` against a
  real code;
- `mailer` disagrees on 105, of which 96 are `'0'` against a real code;
- 68 disagree on both.

For those keys "was this product on display in this store that week" has two
answers. **Phase 2 must state a rule** (any / first / drop) and record it — the
rule moves the treated share. The figures below use *any treated wins*, which is
a reporting choice made to get one row per key, not the decision.

This also inflates a naive `LEFT JOIN`: joining the panel to raw `causal_data`
produced 2,371,399 rows for 2,370,784 distinct product-store-weeks. 615 panel
keys were silently duplicated before the collapse was added.

### Coverage — the treatment reaches a slice, not the panel

| axis | in panel | in `causal_data` | ever treated in panel | covered |
|---|---:|---:|---:|---:|
| products | 92,339 | 68,377 | 22,153 | **23.99%** |
| stores | 582 | 115 | 115 | **19.76%** |
| weeks | 102 | 93 | 93 | 91.18% |

Of 2,370,784 product-store-weeks that recorded a sale, 483,320 (20.39%) appear
in `causal_data`. 36,287,959 causal rows (98.69%) have no matching sale — but
that is a **sampling artifact, not demand**: `causal_data` is chain-wide while
`transaction_data` is a 2,500-household sample of it. Those are mostly "no
sampled household bought it", not "advertised and nothing sold". They must not
be read as demand zeros.

Treated share, panel denominator (absent = untreated):

| treatment | panel rows | % of panel | % of causal-covered |
|---|---:|---:|---:|
| `display <> '0'` | 237,052 | 10.00% | 49.05% |
| `mailer <> '0'` | 350,188 | 14.77% | 72.45% |
| either | 483,320 | 20.39% | 100.00% |

### Variation — and where it lives

Share of entities never / always / sometimes treated, panel denominator:

| treatment | axis | entities | never | always | switches |
|---|---|---:|---:|---:|---:|
| display | products | 92,339 | 82.79% | 0.58% | 16.63% |
| display | stores | 582 | 80.41% | 0.00% | 19.59% |
| display | weeks | 102 | 8.82% | 0.00% | 91.18% |
| mailer | products | 92,339 | 83.60% | 0.77% | 15.63% |
| mailer | stores | 582 | 80.24% | 0.00% | 19.76% |
| mailer | weeks | 102 | 8.82% | 0.00% | 91.18% |
| either | products | 92,339 | 76.01% | 1.33% | 22.66% |

The store figures restate coverage: the 80% of stores that never appear treated
are the 467 stores missing from `causal_data` entirely.

**The table that matters**, restricted to products that are ever treated —
because these are the two comparisons a product-store-week estimator can make:

| treatment | ever-treated products | varies within a store, across weeks | varies within a week, across stores | varies at neither |
|---|---:|---:|---:|---:|
| display | 15,888 | 61.74% | **65.34%** | 24.30% |
| mailer | 15,146 | 69.83% | **2.28%** | 30.03% |
| either | 22,153 | 62.13% | 40.59% | 31.07% |

**`display` and `mailer` are not interchangeable treatments.** `display` varies
both ways: for 65% of treated products there is a week in which some stores had
it on display and others did not. `mailer` does not: for only **2.28%** of
treated products does the mailer flag differ across stores within a week. The
mailer is decided for the chain, so within a week it is collinear with
product-week and the cross-store comparison is unavailable for it. Timing
variation across weeks remains for both (62% and 70%).

Roughly 25–30% of ever-treated products vary at neither — they are treated in
every product-store-week they appear in, so for them treated and untreated never
sit side by side.

`mailer` `'A'` alone is 55.66% of treated panel rows and reaches 13,262
products; `'D'` another 27.25%. Any single "on the mailer" boolean is mostly
those two codes. Some codes are near-degenerate on the time axis — `'J'` appears
in 14 weeks, `'X'` in 29, `'Z'` in 8 weeks and 50 products.

### Campaign membership — household-week

7,208 memberships, 1,584 households (**63.36%** of the 2,500 shoppers), 30
campaigns in three types (TypeA 5, TypeB 19, TypeC 6). Median 4 campaigns per
member household, max 17. Campaign windows run 33–162 days (median 38) and span
weeks **33–103** — nothing is live before week 33, and the last window runs past
the end of the 102-week panel. Both ends are censored.

Of 255,000 household-weeks, **15.74%** are in at least one live campaign. 36.64%
of households are never in one; no household is always in one; the median
household is in a campaign 12.75% of its weeks.

**It does not vary across products or stores — not sparsely, but structurally.**
A household in a campaign is in it for every product and every store it shops
in. At `PRODUCT_ID × STORE_ID × WEEK_NO` the flag is constant within a week and
therefore collinear with a week fixed effect. That is what household targeting
means; no amount of data fixes it. It is a treatment on a different panel
(household × week), not a weak treatment on this one.

### Coupon redemption — household-day, and very sparse

2,318 redemption events, 434 households (**17.36%** of shoppers), 556 coupons,
30 campaigns, DAY 225–704. Median 3 redemptions per redeeming household (p90 12,
max 35). **0.47%** of household-weeks contain any redemption — 1,193 of 255,000.

A redemption names a coupon, not a product. `coupon.csv` maps 1,135 coupons to
products with a median of 12 and a maximum of **14,477** products per coupon; of
the 556 coupons actually redeemed (all present in `coupon.csv`), the median fans
out to 20 products. A redemption cannot be attributed to a product without a
stated splitting rule, and it is post-treatment behaviour besides — the shopper
chose to redeem.

### Summary

| treatment | exists at | varies across product | across store | across week | usable at panel grain |
|---|---|---|---|---|---|
| `display` | product-store-week | yes | **yes** | yes | yes |
| `mailer` | product-store-week | yes | **barely (2.28%)** | yes | timing only |
| campaign | household-week | no | no | yes | no |
| redemption | household-day | no (coupon fans out) | no | yes | no |

Read the switch shares, not the treated shares. A treatment can be on 20% of the
panel and identify nothing if the same entities are always the treated ones.

### Consequences for Phase 2 and Phase 3

1. **The usable panel is much smaller than 2.37M product-store-weeks.** 115
   stores, 22,153 ever-treated products, weeks 9–101. The feasibility gate
   should compute its overlap on that slice, not on the full panel.
2. **The duplicate-key rule must be stated and recorded** before any treated
   share is quoted.
3. **`display` and `mailer` need separate treatment variables.** Collapsing them
   into one "promoted" boolean merges a treatment with cross-store variation
   into one without it.
4. **Weeks 1–8 and 102 have no treatment data**, and weeks 1 and 102 are partial
   calendar weeks. Both are reasons to trim the estimation window, recorded
   before rather than after seeing results.
5. **Campaign and redemption are out of scope as panel treatments** — recorded
   here as measured, not assumed.

---

### DECISION — the treatment is `display`

**The primary treatment is `display`.** Not `mailer`, and not a merged
"promoted" flag.

**Rationale.** `display` varies within a week across stores for **65.34%** of
treated products, which gives a contemporaneous cross-store comparison: in the
same week, for the same product, some stores had it on display and others did
not. `mailer` varies that way for only **2.28%** of treated products. Within a
week the mailer flag is collinear with product-week — it is a chain-wide
decision — so it offers no same-week untreated comparison. A merged flag would
inherit `mailer`'s collinearity while appearing to have `display`'s variation,
which is worse than either column alone.

**`mailer` is retained as a control covariate, never as the treatment.** It is
real promotional activity and omitting it would load its effect onto `display`;
including it as a treatment would claim an identification that does not exist.
Both columns keep their raw categorical codes — `display` `'9'` is not more
display than `'2'`, and `'A'` means different things in the two columns.

**Campaign membership and coupon redemption are excluded as treatments**, for
structural reasons rather than sparsity:

- **Campaign membership** is constant across products and stores for a household,
  so at `PRODUCT_ID × STORE_ID × WEEK_NO` it is collinear with a week fixed
  effect. More data would not change this; it is what household targeting means.
- **Coupon redemption** names a coupon, and a redeemed coupon fans out to a
  median of 20 products (max 14,477). There is no principled rule for attributing
  a redemption to one product. It is also post-treatment behaviour — the shopper
  chose to redeem.

Both remain available as a separate household × week analysis, which is a
different panel and an optional axis, not this one.

### Open Phase 2 decisions arising from this task — both closed in Task 2.5

1. ~~The 15,245 conflicting `causal_data` keys need an explicit resolution
   rule.~~ **Resolved: any treated wins**, on the structural grounds recorded as
   settled decision 7. The *any treated wins* figures used for reporting in this
   section are therefore now the decision as well, which was not knowable when
   they were written.
2. ~~The join must collapse `causal_data` to one row per key before merging.~~
   **Done and asserted** in `promo/treatment.py`; see settled decision 7.

---

## Task 1.5 — Structural zeros and the repurchase cycle

Reproduce: `.venv/bin/python scratch/explore_05.py`.

**Finding: half of all household-weeks contain no shopping trip at all
(51.38%), so most of the zero mass in household-level work is "not in the shop"
rather than "declined to buy". And the panel does not start full — 99% of
households have not entered until week 18, so weeks 1–17 are a recruitment ramp,
not weak demand. The median repurchase cycle across the top 50 commodities is
3 weeks, which is the floor for any measurement horizon.**

### The household × week grid

2,500 households × 102 weeks = 255,000 household-weeks.

| | household-weeks | share |
|---|---:|---:|
| with at least one transaction | 123,976 | 48.62% |
| **with none at all** | **131,024** | **51.38%** |

That raw figure mixes two different zeros. A household-week before a household's
first-ever trip is not a refused purchase — it is a household that is not in the
panel yet. Restricting to each household's observed span (first shopping week to
last):

| | household-weeks | share |
|---|---:|---:|
| inside some household's span | 223,664 | 87.71% of the grid |
| **of those, no trip** | **99,688** | **44.57%** |

**44.57% is the honest no-trip share**; 51.38% is that plus panel entry and exit.
2,412 households are first seen after week 1 and 1,295 are last seen before week
102. Median first week 11, median last week 101, median span 90 weeks.

### The panel is a recruitment ramp for its first ~17 weeks

| week | households shopping | inside their span |
|---:|---:|---:|
| 1 | 3.52% | 3.52% |
| 2 | 7.00% | 8.68% |
| 4 | 10.80% | 17.92% |
| 6 | 17.32% | 28.20% |
| 8 | 21.20% | 38.72% |
| 10 | 28.32% | 49.44% |
| 12 | 36.20% | 63.28% |
| 14 | 44.76% | 75.96% |

90% of households have entered by week 16, 95% by week 17, 99% by week 18. (The
last straggler enters at week 97.) In steady state about 52% of households shop
in a given week — min 3.52% at week 1, max 57.12% at week 92, median 52.52%.

**Any level comparison involving weeks 1–17 is comparing panel size, not
demand.** Weeks 1 and 102 are additionally 5- and 6-day calendar stubs (Task
1.4). Combined with `causal_data` covering only weeks 9–101, the window where
both the panel is full and treatment is observed is roughly **weeks 18–101**.

### How the trips are spread

Weeks with a trip, per household: min 1, p10 15, p25 28, **median 50**, p75 71,
max 102. Exactly **one** household shopped in all 102 weeks; 134 households
shopped in fewer than 10 weeks. The no-trip mass is not a few absent households
— it is spread across nearly everyone.

### Inter-purchase gaps, top 50 commodities

"Frequently purchased" = most household-week purchase events, where one event is
a household buying anything in that commodity in one week. `median` is the
pooled median gap in weeks; `median of hh medians` computes each household's own
median gap first and then takes the median across households, which is the less
purchase-weighted view.

| COMMODITY_DESC | events | median gap (wk) | p75 | p90 | median of hh medians |
|---|---:|---:|---:|---:|---:|
| FLUID MILK PRODUCTS | 57,370 | 2.0 | 3 | 6 | 3.0 |
| SOFT DRINKS | 51,594 | 2.0 | 3 | 8 | 3.0 |
| BAKED BREAD/BUNS/ROLLS | 51,126 | 2.0 | 3 | 7 | 3.0 |
| CHEESE | 40,772 | 2.0 | 4 | 9 | 4.0 |
| BAG SNACKS | 35,811 | 2.0 | 5 | 10 | 4.0 |
| BEEF | 32,432 | 2.0 | 5 | 10 | 4.0 |
| TROPICAL FRUIT | 28,905 | 2.0 | 4 | 10 | 4.0 |
| EGGS | 26,177 | 3.0 | 6 | 12 | 5.0 |
| COUPON/MISC ITEMS * | 23,607 | 2.0 | 4 | 10 | 4.0 |
| COLD CEREAL | 23,315 | 3.0 | 6 | 13 | 5.0 |
| REFRGRATD JUICES/DRNKS | 23,135 | 2.0 | 5 | 13 | 5.0 |
| SOUP | 21,782 | 3.0 | 7 | 15 | 5.5 |
| LUNCHMEAT | 21,582 | 3.0 | 7 | 14 | 5.5 |
| CRACKERS/MISC BKD FD | 21,255 | 3.0 | 7 | 15 | 6.0 |
| VEGETABLES - SHELF STABLE | 20,973 | 3.0 | 7 | 14 | 5.5 |
| FROZEN PIZZA | 20,784 | 3.0 | 6 | 14 | 5.0 |
| FRZN MEAT/MEAT DINNERS | 20,289 | 3.0 | 6 | 14 | 5.0 |
| CANDY - PACKAGED | 18,982 | 3.0 | 7 | 16 | 6.0 |
| CANDY - CHECKLANE | 18,840 | 3.0 | 7 | 16 | 6.5 |
| ICE CREAM/MILK/SHERBTS | 18,440 | 3.0 | 8 | 17 | 6.5 |
| MILK BY-PRODUCTS | 18,367 | 3.0 | 8 | 15 | 6.0 |
| CANNED JUICES | 18,250 | 3.0 | 7 | 16 | 6.0 |
| CONDIMENTS/SAUCES | 16,818 | 4.0 | 9 | 19 | 7.5 |
| BAKED SWEET GOODS | 16,804 | 3.0 | 7 | 15 | 6.0 |
| YOGURT | 16,251 | 2.0 | 6 | 14 | 5.0 |
| SALD DRSNG/SNDWCH SPRD | 15,978 | 5.0 | 10 | 20 | 7.5 |
| COOKIES/CONES | 15,887 | 4.0 | 8 | 18 | 7.0 |
| DELI MEATS | 15,723 | 2.0 | 6 | 14 | 5.0 |
| WATER - CARBONATED/FLVRD DRI | 15,494 | 3.0 | 6 | 16 | 6.0 |
| VEGETABLES - ALL OTHERS | 15,293 | 3.0 | 7 | 16 | 7.0 |
| ONIONS | 14,756 | 4.0 | 9 | 17 | 7.5 |
| POTATOES | 14,650 | 4.0 | 9 | 18 | 7.0 |
| SALAD MIX | 14,452 | 3.0 | 7 | 16 | 6.0 |
| TOMATOES | 14,259 | 3.0 | 7 | 17 | 6.5 |
| CHICKEN | 14,192 | 4.0 | 8 | 18 | 6.5 |
| DRY NOODLES/PASTA | 13,482 | 4.0 | 10 | 20 | 8.0 |
| HISPANIC | 13,450 | 4.0 | 9 | 19 | 7.0 |
| REFRGRATD DOUGH PRODUCTS | 13,313 | 4.0 | 9 | 18 | 7.0 |
| CONVENIENT BRKFST/WHLSM SNAC | 13,148 | 3.0 | 7 | 17 | 6.0 |
| MEAT - SHELF STABLE | 13,071 | 4.0 | 9 | 19 | 7.0 |
| MARGARINES | 13,064 | 5.0 | 10 | 20 | 8.5 |
| BATH TISSUES | 13,013 | 4.0 | 9 | 19 | 8.0 |
| BEERS/ALES | 12,870 | 2.0 | 6 | 16 | 6.5 |
| APPLES | 12,740 | 3.0 | 8 | 18 | 6.0 |
| DINNER MXS:DRY | 12,684 | 4.0 | 9 | 19 | 7.0 |
| FRUIT - SHELF STABLE | 12,636 | 4.0 | 9 | 20 | 7.0 |
| DRY BN/VEG/POTATO/RICE | 12,621 | 4.0 | 10 | 21 | 8.0 |
| FRZN VEGETABLE/VEG DSH | 11,945 | 4.0 | 9 | 19 | 8.0 |
| PORK | 11,778 | 4.0 | 9 | 19 | 7.5 |
| PASTA SAUCE | 11,384 | 5.0 | 10 | 21 | 8.0 |

`COUPON/MISC ITEMS` is in this list at rank 9 and sits in the volume-measured
departments flagged in Task 1.2 — its "repurchase cycle" is a refuelling
interval, not a grocery restock interval. Listed, not dropped.

**Across the 50 commodities the median gap ranges 2.0 to 5.0 weeks; the
quartiles of the per-commodity median are p25 3.0, median 3.0, p75 4.0.** The
pooled median understates what a horizon needs: the p75 gap is 3–10 weeks and
the p90 is 6–21 weeks depending on category. Staples (milk, bread, soft drinks)
cycle in 2 weeks; store-cupboard goods (pasta sauce, margarine, salad dressing)
in 5.

The `median of hh medians` column is consistently **higher** than the pooled
median — 3.0 against 2.0 for milk, 8.0 against 5.0 for pasta sauce. The pooled
figure is dominated by frequent buyers, who contribute more gaps each. For a
horizon, the per-household view is the safer one.

### Both figures are biased short, and knowably so

- Of 100,757 household-commodity pairs, **15,645 (15.53%) bought in exactly one
  week** and contribute no gap at all. Those are the slowest buyers, so every
  median above is biased **short**.
- Gaps are right-censored by the 102-week window: no gap longer than 101 weeks
  is observable, and a household's last purchase has no successor.

A horizon set from these numbers is therefore a **lower bound** on the true
cycle. Rounding up is the conservative direction.

### The zeros decomposed

For each commodity, of its zero household-weeks, what share are weeks the
household made no trip at all:

| COMMODITY_DESC | bought in | zero | of the zeros, no trip |
|---|---:|---:|---:|
| FLUID MILK PRODUCTS | 57,370 | 197,630 | 66.30% |
| SOFT DRINKS | 51,594 | 203,406 | 64.42% |
| BAKED BREAD/BUNS/ROLLS | 51,126 | 203,874 | 64.27% |
| CHEESE | 40,772 | 214,228 | 61.16% |
| BAG SNACKS | 35,811 | 219,189 | 59.78% |
| EGGS | 26,177 | 228,823 | 57.26% |
| SOUP | 21,782 | 233,218 | 56.18% |

Even for the most-bought category in the file, two thirds of its zeros are weeks
the household never entered a shop. That is not demand information, and a model
that treats it as a decision not to buy is fitting shopping-trip frequency.

### Consequences for the pipeline

1. **The no-trip share applies to household-level work, not to the
   PRODUCT × STORE × WEEK panel.** A store is open every week. Household-level
   analysis needs an availability flag rather than a zero; the panel needs the
   distinction between "product not stocked" and "stocked and unsold", which is
   a different question and is not answered here.
2. **Trim the estimation window to weeks 18–101.** Below 18 the panel is still
   recruiting, above 101 there is no treatment record. This is a decision to
   record before seeing results, not after.
3. **The measurement horizon must extend past campaign end by at least the
   commodity's median gap** — 3 weeks for a typical category, 5 for slow ones,
   and more if the p75 is used. Because of the censoring above, these are floors.
4. **The horizon is per-commodity, not global.** A single 2-week window banks the
   peak for pasta sauce and misses its entire trough.

---

## Task 2.4 — The price index does not come back flat

Reproduce: `promo.prices.build_price_index()`; the full result is in
`data/interim/price_index_diagnostics.json`, the index itself in
`data/interim/price_index.parquet`.

**`docs/plan.md` predicted near-flat drift on this US dataset. It is not
near-flat.** The chained matched-pair index rises **+16.68%** across weeks 1–102
and **+16.31%** across the estimation window, weeks 18–101 — an annualised
**+8.3%**. The prediction was wrong and the number is reported as measured.

### What was checked before reporting it

| check | result | rules out |
|---|---:|---|
| trimmed 5% of extreme relatives | +14.48% | a handful of outliers carrying the chain |
| direct fixed-base index, week *w* vs week 1 | ≈ +17.8% by week 101 | chain drift from bouncing prices |
| weeks with an imputed link | 0 of 102 | thin links propped up by 1.0 |
| minimum matched pairs in any week | 40 | a link resting on nothing |

The chained and the fixed-base constructions agree, so this is not the classic
scanner-data chain drift. Trimming moves it by two points, so it is not
outliers.

### The composition finding, which is the actually interesting one

Three measurements of the same 102 weeks, answering three different questions:

| measure | drift | what it asks |
|---|---:|---|
| **matched** (the index) | **+16.68%** | what happened to the price of a *fixed* item |
| balanced panel, 684 product-stores present in ≥40 weeks | +11.81% | same, by brute force rather than by matching |
| **pooled** — geometric mean of every undiscounted price that week | **−6.83%** | what the average price paid did |

**The matched index and the naive average move in opposite directions.** The
pool of undiscounted observations grows from about 3,700 to 13,106 per week as
the household panel fills, and the entrants are cheaper items. A fixed item got
more expensive; the basket got cheaper, because the basket changed.

This is exactly what a price index is for, and it is the sentence to say on
stage: *the average price paid fell 7% while prices rose 17%, and only one of
those is inflation.* A pipeline that deflated by the pooled figure would push
real prices the wrong way by a quarter over two years.

### What this does not establish

- **Whether +16.7% is real inflation or a residual artefact.** The matched and
  balanced figures differ by five points, which is more than sampling noise
  should give. The dataset carries no calendar dates — only `DAY` 1–711 — so it
  cannot be checked against a published CPI series for the period, and there is
  no external anchor available.
- The 29% of level-test groups whose reconstruction misses by more than a cent
  (Task 1.3, *The level test*) indicate within-week price heterogeneity under a
  single `PRODUCT_ID`. Pack-size mix inside one identifier moves a unit value
  without any price moving, and the index inherits that.

**Consequence for the pipeline.** Deflation is load-bearing here, not
decorative. `deflate_prices()` adds `real_paid_price` and `real_regular_price`
alongside the nominal columns and overwrites neither: Phase 5 accounting is in
the currency the shopper actually paid, and only cross-week price *comparisons*
use real terms. Depth is a within-week ratio, so the index cancels and depth
needs no real counterpart — asserted in the tests rather than assumed.

---

## Task 2.5 — The treatment join, and what "untreated" is allowed to mean

Reproduce: `promo.treatment.build_treatment_panel()`; the full result is in
`data/interim/treatment_diagnostics.json`, the panel in
`data/interim/panel_treated.parquet`.

The duplicate-key rule and the collapse requirement are recorded above as
settled decision 7. This section records what the join produced.

### The treated share

Treatment is `display` per settled decision 4, taken as a parameter rather than
hardcoded so Phase 3 can re-run the audit under `mailer`, `display_or_mailer`,
and `display_and_mailer`.

| | rows | share of panel | units | sales value |
|---|---:|---:|---:|---:|
| in `causal_data` | 482,672 | 20.54% | 794,587 | 1,557,644 |
| **treated (`display`)** | **236,689** | **10.07%** | 382,440 | 724,374 |
| in mailer | 349,814 | 14.89% | | |
| both mechanics | 103,831 | 4.42% | | |

Treated is **49.04%** of the rows the log covers and **10.54%** of the rows
where absence is informative. Both denominators are reported because the second
is the one a treated share should be quoted against, and they differ by a factor
of five.

### The finding: absence means two different things, and only one is a control

The plan's phrasing — "weeks and product-stores absent from `causal_data` are
untreated, not missing" — is right inside the log's coverage and wrong outside
it. `causal_data` spans weeks 9–101 and 115 stores. Inside that envelope a
missing key is a real negative, because the log records every promotion. Outside
it the log is simply silent, and calling silence "untreated" manufactures
controls out of nothing.

The panel therefore carries `treatment_observed` beside `treated`:

| | rows | share | what it is |
|---|---:|---:|---|
| in the log | 482,672 | 20.54% | treatment read directly |
| untreated by inference | 1,763,222 | **75.04%** | inside the envelope, absent from the log — a real negative |
| **unobserved** | **103,666** | **4.41%** | outside the envelope — absence carries no information |

**Three quarters of the panel is untreated by inference.** That is the
assumption's true weight, and it is exactly why the 4.41% must be separable
rather than folded in with it.

**Consequence for Phase 3.** Controls must be drawn from `treatment_observed`
rows only. An unobserved row used as an untreated control is a fabricated
comparison, and at 103,666 rows it is large enough to matter and small enough to
go unnoticed.

### Store coverage is far better than the store count suggests

Task 1.4 recorded that `causal_data` reaches 115 of 582 stores — **19.76%** —
which reads like a severe limitation. At panel grain it is not:

| | rows | sales value | share of sales |
|---|---:|---:|---:|
| the 115 logged stores | 2,313,740 | 7,285,564 | **98.54%** |
| the other 441 stores | 35,820 | 107,681 | 1.46% |

**The logged stores are the large ones: a fifth of the estate carrying 98.5% of
the money.** The 4.41% unobserved share decomposes into 35,820 rows from
unlogged stores and 67,846 rows in logged stores but outside weeks 9–101 —
summing exactly. Quote the sales share, not the store count; the store count
understates coverage by a factor of five and invites a needless refusal.

---

## Task 2.6 — The modelling panel, and the holiday flag that cannot be built

Reproduce: `promo.features.build_feature_panel()`; full result in
`data/interim/feature_diagnostics.json`, panel in `data/interim/panel.parquet`.

### The scope, and what it buys

`CLAUDE.md` forbids the full 92,339 × 582 × 102 grid — 5.5 billion cells. The
scope applied here is the top **300 products by transaction count, drawn from
ever-treated products only**, the **115** logged stores, and weeks **9–101**.

| | |
|---|---:|
| carried product-store pairs | 31,896 |
| rows (pairs × 93 weeks) | **2,966,328** |
| of which observed | 404,465 |
| of which zero-filled | 2,561,863 (86.4%) |
| share of all transactions | **20.83%** |
| share of all sales value | **16.32%** |

**Zeros are filled only for product-store pairs observed at least once.** The
full cross product would be 3,208,500 rows; the extra 242,172 are shelves a
store does not stock. Task 1.5 recorded that this dataset cannot distinguish
"not stocked" from "stocked and unsold", so those rows would be invented demand
observations, not zeros.

### Finding: 79% of treated rows are weeks with no sale

Task 2.5 joined treatment onto rows that recorded a sale. Once explicit zeros
exist, the treatment has to be looked up again — and the reason is stark:

| | rows |
|---|---:|
| treated rows in the scoped panel | 366,346 |
| **of which sold nothing that week** | **290,895 (79.4%)** |

A product can be on display and sell nothing, and `causal_data` is chain-wide so
it records exactly that. Had the treatment been carried over from Task 2.5
instead of re-derived, **four in five treated rows would have been silently
labelled untreated** — and they would have entered the Phase 4 baseline as
training data, since the baseline trains on `treated == 0`. That is the precise
failure the "raise if treated rows reach the fit" invariant exists to catch, and
it would have been introduced upstream of the check.

### The holiday flag cannot be built from this dataset, and was not

The plan asks for a holiday flag. **This dataset carries no calendar dates** —
only `DAY` 1–711 and `WEEK_NO` 1–102, with no anchor to a real year. There is
therefore no way to know which week contains which holiday.

Two ways to fake it were rejected:

- **Anchoring to a guessed year.** Any anchor is an assumption presented as a
  fact, and every seasonal coefficient would inherit it.
- **Inferring holiday weeks from demand spikes.** This builds a feature out of
  the outcome it is meant to predict. It would improve fit and destroy the
  estimate.

`is_holiday_week` is therefore emitted as **False everywhere**, with
`holiday_flag.populated = false` and the reason recorded in the diagnostics. It
takes a `holiday_weeks` parameter, so an external anchor can populate it without
a code change. **`week_of_year` carries the seasonality instead** — computed as
`((WEEK_NO - 1) % 52) + 1`, a position in a 52-week cycle, which needs no
calendar. A tree learns the December bump wherever it falls; it just cannot name
it.

### Feature missingness, all of it structural

| feature | missing | why |
|---|---:|---|
| `price_rel_category` | 86.36% | a week with no sale has no observed price |
| `units_lag_52` | 55.91% | the scope is 93 weeks, so the first 52 cannot have one |
| `units_lag_4` | 4.30% | grid edge |
| `units_lag_1`, all rolling means | 1.08% | grid edge |
| everything else | 0.00% | |

`price_rel_category` is left null rather than carried forward from the last
priced week. Filling it would be a modelling choice, and an imputed price is
indistinguishable from an observed one once it is in the column. LightGBM splits
on null natively, so the feature still contributes where it exists.

### Two leakage rules, both tested

1. **No feature uses the current week's outcome.** Rolling means span *w−k .. w−1*
   and never include *w*. A test recomputes every lag and window by hand against
   a shifted series and fails on an off-by-one.
2. **No feature is derived from the promotion.** `price_rel_category` uses the
   reconstructed **regular** price, never the paid price — a paid price is low
   precisely *because* the product is on deal, so a control built on it encodes
   the treatment.

**A caveat the tests cannot settle.** `store_traffic` and
`category_units_ex_focal` are contemporaneous, and both can be *affected by* the
promotion. Conditioning on a mediator absorbs part of the effect. They are not
future information and the plan asks for them, so they are built — but the
diagnostics tag every feature `lagged` or `contemporaneous`, and Phase 3 should
report the estimate with and without the contemporaneous block rather than
assume the controls are innocent.

---

## Task 2.7 — Availability, zero states, and the horizon that is longer than we said

Reproduce: `promo.quality`; results in `data/interim/quality.json` and
`data/interim/repurchase_cycles.parquet`.

Task 1.5's figures reproduce exactly: **51.38%** raw no-trip share, **44.57%**
within-span, median first week 11, median last week 101, and the top-8
commodity gap table to the decimal.

### Finding: the "3 week" horizon is the fastest categories, not the typical one

Task 1.5's summary reads "the median repurchase cycle across the top 50
commodities is 3 weeks, which is the floor for any measurement horizon", and
consequence 3 gives "3 weeks for a typical category, 5 for slow ones". Computed
on the basis Task 1.5 itself argued for — the **median of household medians**,
because the pooled median is dominated by frequent buyers — that understates
the typical category by half:

| basis | median horizon |
|---|---:|
| top 50 commodities | **6 weeks** |
| all 306 commodities | 9 weeks |
| p90 across commodities | 16 weeks |
| slowest with adequate support (`SEASONAL`) | 49 weeks |

3 weeks is right for `FLUID MILK PRODUCTS`, `SOFT DRINKS` and
`BAKED BREAD/BUNS/ROLLS` — the three fastest. It is wrong for the median
top-50 category, which needs **6**. A single global 3-week window would bank the
peak and miss the trough for most of the catalogue, which is precisely the
failure the horizon rule exists to prevent.

**The per-commodity horizon in `repurchase_cycles.parquet` is the number to
use.** 298 of 306 commodities have one; 36 are flagged `low_support` (fewer than
30 gap observations) and 8 have no gap at all and get no invented horizon.

### The single-week share was also quoted on a partial basis

Task 1.5's 15.53% of household-commodity pairs buying in exactly one week is the
**top-50** figure (reproduced here as 15.60%). Across all 306 commodities it is
**31.49%** — the long tail is mostly one-off purchases. Both are now in the
diagnostics under their own names. The bias direction is unchanged and, on the
full basis, twice as strong: every horizon is a floor.

### Finding: a fuel-only visit is still a trip

The shopping flag counts **any** transaction, not any *usable* one. Task 2.2's
`usable` filter removes fuel and zero-quantity rows because they are not
comparable demand — but a household that bought only petrol was in the shop and
able to buy groceries. Applying that filter here turns **2,827 real trips into
no-trips** and pushes the raw no-trip share from 51.38% to 52.49%.

The distinction matters because it changes what a zero means: for those 2,827
household-weeks, not buying the commodity is a **decision**, not an absence.
`usable_only=True` exists to measure the gap, not as an option worth taking.

### Zero states: four, not two

| state | zero kind | carries demand information |
|---|---|---|
| `bought` | — | — |
| `no_buy_on_trip` | **sampling** | **yes** |
| `no_trip` | structural | no |
| `out_of_panel` | structural | no — not even an observation |

`out_of_panel` is kept separate from `no_trip` because a household not yet
recruited is a different object from one who shopped elsewhere, and merging them
hides the recruitment ramp. For `FLUID MILK PRODUCTS` the structural share of
zeros is over 60%, matching Task 1.5.

### The Phase 2 honesty report

`data/interim/quality.json` consolidates all six stage diagnostics. It records
**6 exclusions** with before-and-after row, unit, and sales-value effects:

| stage | exclusion | action | rows |
|---|---|---|---:|
| clean | `nonpositive_quantity` | exclude | 14,466 |
| clean | `unmatched_product` | exclude | 0 |
| clean | `volume_measured` | exclude | 28,052 |
| clean | `free_good` | flag | 4,451 |
| clean | `blank_department` | flag | 7,839 |
| prices | `retail_disc_surcharge` | exclude | 6 |

It also carries the scope coverage, the `NO_MARGIN` finding, the treatment
absence assumption, and four **unresolved** limits stated as limits: stockouts
are unobservable; "not stocked" is indistinguishable from "stocked and unsold";
the holiday flag has no calendar anchor; and 103,666 panel rows are unobserved
rather than untreated and must never serve as controls. A stage whose
diagnostics file is missing is named in `sources`, so the report can never be
short without saying so.

---

## Task 3.1 — Variation on the scoped panel

Reproduce: `promo.audit.variation_axes()`; result in
`data/interim/variation_diagnostics.json`.

Task 1.4 measured variation at entity level on the full panel, weighting every
product equally. This is the same shape weighted by **units**, on the scoped
modelling panel — the slice `data_findings.md` already directed the gate to use.

| axis | fully treated | fully untreated | **mixed** |
|---|---:|---:|---:|
| product | 0.00% | 25.67% | **74.33%** |
| store | 0.00% | 0.01% | **99.99%** |
| week | 0.00% | 0.00% | **100.00%** |

Shares are of units. **No level on any axis is fully treated** — every treated
product has untreated weeks, and every treated store has untreated
product-weeks. There is comparison mass everywhere the estimator might look.

### Finding: a quarter of the scoped panel can never inform a display effect

**72 of the 300 scoped products are never on display — and all 72 are
mailer-only.** Not one is untreated on both mechanics.

The cause is a mismatch inside Task 2.6's scope rule. It selects the top N
products from those *ever treated*, where ever-treated is
`BOOL_OR(on_display) OR BOOL_OR(in_mailer)` — either mechanic. But the treatment
is `display` (settled decision 4). A product carried only in the mailer
therefore passes the scope filter and then contributes nothing to estimating the
thing being measured.

It is not free: those 72 products carry **709,590 rows and 25.67% of the scoped
panel's units**. `CLAUDE.md`'s own justification for the ever-treated rule —
"an untreated-only product cannot contribute to estimating a treatment effect,
so it consumes the row budget and returns nothing" — applies to them exactly,
and the rule as implemented does not catch them.

This is a scope-efficiency problem, not a correctness one. Those rows are valid
untreated observations and can serve as controls; they simply cannot be treated
observations.

### DECISION — the 72 are kept as controls

**They are retained deliberately as untreated controls, not by oversight.** The
panel is not rebuilt.

**The effective treated product count is therefore 228, not 300.** Quote 228
whenever the estimator's base is described; 300 is the scoped product count and
overstates what can carry a treated observation by a third.

The rationale is that a control is not waste. The scope rule was written to
avoid spending the row budget on products that "return nothing", and a
mailer-only product does return something — 709,590 rows of untreated
observation in the same stores and weeks as the treated ones, from products
selected for being high-frequency. Discarding them to buy more treated products
would trade control mass for treatment mass at a point where neither is scarce.

**If the row budget becomes binding in Phase 4**, narrowing the scope rule from
`BOOL_OR(on_display) OR BOOL_OR(in_mailer)` to `BOOL_OR(on_display)` frees
**709,590 rows** for products that can actually switch. That is the lever to
reach for, and it requires rebuilding Task 2.6's panel.

### The scope-rule mismatch, recorded rather than inferred

Task 2.6 scopes on **either** mechanic while the treatment is **display alone**.
That mismatch is now stated in three places rather than left to be reconstructed
from two diagnostics files: `scope_rule_mismatch` in
`variation_diagnostics.json`, the `unresolved` list in `quality.json`, and here.
`treated_base` in the variation diagnostics carries the 228 against the 300 so
the effective count travels with the audit that established it.

### The audit is marginal, and cannot re-derive decision 4

A level counts as mixed if it holds both kinds of observation *anywhere*. On the
store axis both `display` (99.99%) and `mailer` (100.00%) come back essentially
fully mixed, because a store need only have had something treated while
something else was not.

That is much weaker than the joint condition decision 4 rests on — variation
across stores *within a week, for the same product*, where display scores 65.34%
and mailer 2.28%. **This module cannot separate the two treatments and must not
be cited as if it could.** It reports the necessary condition; Task 3.2's overlap
check is the one with teeth. The limitation is carried in the diagnostics under
`limitation`, not only in the source, and is asserted by a test.

---

## Task 3.2 — Overlap

Reproduce: `promo.audit.overlap()`; results in
`data/interim/overlap_diagnostics.json` and `overlap_importances.parquet`.

**Cross-validated AUC 0.7036**, folds 0.7025–0.7049 — a spread of 0.0024, so the
split is not driving it. Treated and untreated are distinguishable but far from
separable. **Diagnosis: `SEPARABLE_BUT_PLAUSIBLE`.**

### Not leakage, and the importance table is how we know

| feature | gain share |
|---|---:|
| `n_stores_carrying` | 45.06% |
| `category_units_ex_focal` | 20.19% |
| `store_traffic` | 12.25% |
| `price_rel_category` | 8.81% |
| `week_of_year` | 4.23% |
| everything else (9 features) | 9.46% |

No single feature carries a majority, and the AUC is nowhere near the 0.95 bar.
A leaked covariate looks different: AUC above 0.99 with one feature taking most
of the gain. The synthetic version of that failure is asserted in the tests, so
the diagnosis is calibrated against a known-leaking panel rather than asserted.

**The top three are all contemporaneous, and Task 2.6 flagged exactly them.**
`n_stores_carrying`, `category_units_ex_focal` and `store_traffic` together
carry 77.5% of the gain, and all three are measured in week *w*. Task 2.6's
mediator warning applies directly: they are not future information, but they can
be *affected by* the promotion, so part of what the classifier is learning may
be the treatment's own consequences rather than its causes. The lagged features
contribute almost nothing by comparison — 6.1% between the seven of them. Phase 4
should fit with and without the contemporaneous block and report both.

### Overlap is one-sided, and the good side is the one that matters

| | rows | share |
|---|---:|---:|
| propensity < 0.02 | 177,977 | **6.00%** |
| propensity > 0.98 | **0** | **0.00%** |
| treated rows below 0.02 | 560 | |
| untreated rows above 0.98 | 0 | |

**No region of the panel is treatment-saturated.** Nothing sits above 0.98, so
there is no treated row whose covariates make an untreated comparison
impossible — the direction that would force a refusal. The 6% below 0.02 are
overwhelmingly untreated rows in regions where display is rare; they are
unused controls, not a broken comparison. The 560 treated rows below 0.02 are
the genuinely awkward ones: treated units in territory where treatment almost
never happens.

### The extreme share is a property of the model as well as the data

30 boosting rounds at learning rate 0.05 shrink every probability towards the
base rate. On a synthetic panel with a covariate that *is* the treatment — AUC
above 0.99, perfect separation — a short fit reports **zero** rows outside the
bounds and a 300-round fit reports over 90%. Same data, opposite readings.

The reported figures use 200 rounds at 0.05, recorded in the diagnostics
alongside the count. **Read 6.00% as a lower bound**, and compare it across runs
only at fixed hyperparameters. This is asserted by a test rather than left as a
footnote.

### Two choices that shape the number, both recorded

- **Folds are grouped by product-store.** Adjacent weeks of one product-store
  share their lagged units almost exactly; a random split puts near-copies on
  both sides and inflates the AUC. `cv="random"` is available for comparison.
- **Identifiers are not covariates.** A tree given `PRODUCT_ID` memorises which
  products are displayed and returns a near-perfect AUC that says nothing about
  confounding. They can be passed explicitly when that is the question.

The `seed` parameter is inert under `cv="group"`: `GroupKFold` does not shuffle
and the learner has no stochastic component at these settings. It bites only
under `cv="random"`. Pinned by a test so a seed sweep is not mistaken for a
robustness check.

---

## Task 3.3 — Collisions and horizon

Reproduce: `promo.audit.collisions()` and `promo.audit.horizon_check()`;
collision result in `data/interim/collisions_diagnostics.json`.

### Display and mailer collide on two fifths of treated rows

**Status: `OVERLAPPING_TREATMENTS`.**

| cell | rows | row share | units | unit share |
|---|---:|---:|---:|---:|
| treated **with** mailer | 145,812 | 4.92% | 79,233 | 10.79% |
| treated, clean | 220,534 | 7.43% | 68,920 | 9.38% |
| control **with** mailer | 339,613 | 11.45% | 162,371 | 22.11% |
| control, clean | 2,260,369 | 76.20% | 423,933 | 57.72% |

- **39.80% of treated rows also carry a mailer.** On those rows the estimand is
  the joint effect of display *and* mailer, not of display alone.
- **13.06% of control rows carry a mailer.** These are promoted rows sitting in
  the control group.

### The second figure is the one the plan's check would have missed

The plan asks for product-store-weeks "where display and mailer both fire and
are being treated as one effect" — the first row of that table. The second is
worse and easier to miss.

A control row carrying a mailer is untreated by the `display` definition but was
promoted. It enters the Phase 4 baseline's training set, where its units are
lifted by a promotion the model is told nothing about. The counterfactual is
fitted **too high**, and the measured display effect is biased **towards zero**.

**The two biases push opposite ways and do not cancel**, because they act on
different rows: collisions inflate the effect by crediting display with the
mailer's work, contamination deflates it by raising the baseline. In unit terms
the contamination is the larger of the two — 22.11% of panel units against
10.79%.

Three responses are recorded in the diagnostics, none taken here because the
choice belongs to Phase 4's estimator design:

1. Restrict to rows where `in_mailer` is false — a clean display effect on a
   much smaller panel: 76.20% of rows survive but only 57.72% of units.
2. Keep `in_mailer` as a **covariate** so the model can separate them. This is
   what settled decision 4 assumes ("`mailer` is retained as a control
   covariate") — **and the Phase 2.6 feature set does not include it.** That is
   a gap between the recorded decision and the built panel, surfaced here.
3. Report the joint effect and label it as joint.

### The horizon check takes campaigns, it does not invent them

Phase 3's objective is a verdict "for any **proposed** campaign", so
`horizon_check()` compares a supplied campaign table against the Task 2.7
cycles rather than deriving campaigns from the panel. What constitutes a
campaign is left to the caller, as the plan intends.

Statuses are three, not two:

| status | meaning |
|---|---|
| `OK` | horizon ≥ the commodity's recorded cycle (equality clears it) |
| `HORIZON_TOO_SHORT` | with the shortfall in weeks |
| `UNKNOWN_CYCLE` | no cycle recorded, or the campaign has no horizon |

**An unknown cycle is never a pass.** A commodity absent from the cycles table,
or one whose cycle is null because it had no gap to measure, returns
`UNKNOWN_CYCLE` so it cannot be read as "long enough". Cycles flagged
`low_support` are checked but carry the flag through: a median over a handful of
gaps is a weak requirement, not a safe one.

**Clearing the check is necessary and not sufficient.** Task 2.7 established
every recorded cycle is a floor — one-week-only pairs contribute no gap and are
the slowest buyers, and gaps are right-censored at 102 weeks. The diagnostics
say so in the `floors_not_estimates` field rather than only here.

---

## Task 3.4 — Break-even

Reproduce: `promo.audit.kappa_star()` and `promo.audit.margin_sweep()`.

### The required incremental share is depth over margin, exactly

`kappa_star = depth / margin`. The derivation is worth recording because the
cancellation is not obvious. For promoted units `Q` at regular price `p`, depth
`d`, gross margin rate `m`:

- a promoted unit earns `p(1-d) - p(1-m) = p(m-d)`;
- the incremental share `k` earns that on `kQ` units;
- the `(1-k)Q` units that would have sold anyway each lose `p·d`;
- break-even is `k·p(m-d) = (1-k)·p·d`.

Both the price and the `kd` terms cancel, leaving `k·m = d`. **The required
share depends on neither price nor volume** — only on the ratio of depth to
margin. A test checks the algebra numerically on a worked example rather than
trusting the cancellation.

Since `k ≤ 1` iff `m ≥ d`, **the depth is itself the minimum margin at which a
promotion can break even at all.** That is Task 5.2's `m_star` in the case where
promotional cost is the discount subsidy only.

### What the panel's own depths require

75,451 treated, priced product-store-weeks. Median depth **24.2%**, p75 37.3%,
p90 46.3%; **4.46% run deeper than 50%**.

At the median depth of 24.2%:

| assumed margin | 10% | 15% | 20% | 25% | 30% | 35% | 40% | 45% | 50% |
|---|---|---|---|---|---|---|---|---|---|
| required incremental share | — | — | — | 97% | 81% | 69% | 60% | 54% | 48% |

A dash is arithmetically impossible: it asks for more incremental units than
were sold. **The typical promotion in this panel needs a gross margin above
24.2% before it can break even at any volume**, and at a 25% margin it needs
97% of promoted units to be incremental — which no display promotion achieves.

At the p90 depth of 46.3% only the 50% margin column is feasible at all, and it
requires 93% incrementality.

**This is the MVP 03 answer and it is an honest one.** The dataset has no COGS —
`promo/io.py` establishes that by searching all 46 column names — so the sweep
varies the assumption instead of inventing it. A merchant reads their own row.

### `NO_MARGIN` is a returned reason code, not a missing value

`kappa_star(depth, None)` returns `kappa=None` with `reason_code="NO_MARGIN"`
and a detail line explaining that no margin can be derived. `kappa_star > 1`
returns `reason_code="KAPPA_IMPOSSIBLE"`. Neither is an exception, so the gate
can carry them into a `GateResult` in Task 3.5 without catching anything.

### The grid is defined once

`MARGIN_GRID` — 10% to 50% in 5-point steps, nine points — is written as
literals rather than built by arithmetic, so no float drift can put 0.35 at
0.35000000000000003 and make two tables disagree on a key. A test asserts the
steps are exactly 0.05 and that `0.35 in MARGIN_GRID` holds exactly.

Task 5.2 is the authority on this grid; the constant lives in `promo/audit.py`
only because that module exists and `promo/accounting.py` does not yet.
**Phase 5 must import it, never restate it.**

### The sweep is optimistic wherever free goods were given away

`kappa_star = d/m` counts the discount subsidy alone. Task 5.1 established that
promotional cost is subsidy **plus free goods**, and the second component raises
`m_star` above the depth. So every figure above is a floor on what the promotion
needed to clear. Recorded in the sweep's `identity` field, not only here.

---

## Task 3.5 — The refusal engine

Reproduce: `promo.gates.run_audit()`; a worked verdict in
`data/interim/audit_soup.json`. The reason-code table lives in
`docs/runbook.md`.

**Eleven reason codes, each with a severity, a trigger and a deterministic
message.** Severity belongs to the code, not to the caller: `refuse` stops the
pipeline, `bounded` means a weaker quantity survives and the run continues,
`pass` is recorded for the audit trail.

Seven refuse — `NO_VARIATION`, `NO_OVERLAP`, `LEAKED_FEATURE`,
`INSUFFICIENT_SUPPORT`, `KAPPA_IMPOSSIBLE`, `PLACEBO_OVERLAP`,
`HORIZON_TOO_SHORT`. Four are bounded — `DEPTH_BOUNDED`, `NO_MARGIN`,
`OVERLAPPING_TREATMENTS`, `ROI_UNBOUNDED` — and each names the weaker quantity
that survives, which is what makes it bounded rather than a refusal.

### The verdict on a real campaign

A display promotion on `SOUP` at the panel's median depth of 24.2%, measured
over a 9-week post-campaign window:

| gate | status | code |
|---|---|---|
| break_even | bounded | `NO_MARGIN` |
| horizon | pass | — |
| variation | pass | — |
| collisions | bounded | `OVERLAPPING_TREATMENTS` |

**Verdict: bounded.** Two weaker quantities survive and both are stated —
the break-even margin of 24.2% with its sweep, and a joint display-and-mailer
effect rather than a clean display effect. Nothing is refused, and nothing is
reported as if it were the clean quantity.

### Short-circuiting is cheapest-first, and that is load-bearing

`GATE_ORDER` is `break_even`, `horizon`, `variation`, `collisions`, `overlap`.
The overlap check costs about two and a half minutes on the full panel;
everything before it costs seconds. A campaign at 60% depth against a 30% margin
stops at `break_even` having run one gate, so the arithmetic refusal never pays
for a model fit. Asserted by a test.

The short-circuit hides any problem after the first, which is a real cost:
`stop_on_refuse=False` collects every verdict, and the diagnostics record
`gates_run` and `gates_skipped` so a partial audit cannot be mistaken for a
clean one.

### Two codes are defined but cannot yet fire

`PLACEBO_OVERLAP` and `ROI_UNBOUNDED` have templates, severities and rendering
tests, but their detection belongs to Phases 4 and 5. `run_audit()` cannot emit
them today. Recorded here and in the runbook so their absence from a verdict is
not read as a pass.

### What the message rules cost, and why they are tested

The gate-authoring skill requires that a message name what is missing and what
would fix it, never blame the caller, never use the word *error*, and never
imply the effect does not exist when the truth is that this comparison cannot
see it. All four are asserted for all eleven codes rather than left to review.

Two failed on first write and were fixed rather than waived:

- `OVERLAPPING_TREATMENTS` opened with an interpolated percentage, so the
  sentence began with a digit.
- `PLACEBO_OVERLAP` said what the result *was not* without saying what would
  change it. It now names a larger promotion, more weeks, or closer comparison
  weeks.

The placebo message carries the project's sharpest distinction and has its own
test: it must contain "not evidence that the promotion did nothing" and must
never contain "no effect".

---

## Task 3.5 addendum — the price-status gate

Reason codes that fire through `run_audit()` went from **7 of 11 to 9 of 11**.

`DEPTH_BOUNDED` and `INSUFFICIENT_SUPPORT` had templates and severities from
Task 3.5 but no detection wired: the codes existed and no gate could emit them.
The Phase 3 verification caught it by instrumenting `gate_result` and recording
which codes actually reached a `GateResult` during the test run, rather than by
reading the tests — rendering a message is not firing a gate.

### The gate reads `prices.parquet`, not `panel.parquet`

`panel.parquet` carries `price_status` but Task 2.6's projection dropped
`deal_share` and `n_weeks_priced` — the two numbers the messages have to quote.
The gate therefore takes a `statuses` source defaulting to `prices.parquet`,
mirroring how `horizon_check` takes `cycles`. Worth knowing before anyone
assumes the modelling panel carries everything the gate needs.

### What it can and cannot say without a product

`CampaignSpec.product` is optional, so a commodity-level campaign still audits.
When no product is named the gate returns `pass` and reports the portfolio
distribution — 24,537 identified, 3,445 bounded, 63,929 insufficient support —
while stating plainly that no verdict was reached, because a commodity spans
products with different statuses and these two codes are properties of a
product. An unknown product likewise passes with "it has not been shown to be
usable" rather than silently clearing.

### The two codes stay separate, and the severities prove it

| status | code | severity | what survives |
|---|---|---|---|
| `bounded` | `DEPTH_BOUNDED` | bounded | ordinal depth — its deals rank against each other |
| `insufficient_support` | `INSUFFICIENT_SUPPORT` | refuse | nothing — a coverage limit implying no action |
| `identified` | — | pass | the depth is a usable number |

A test asserts the severities differ, which is the whole point of settled
decision 6: a bounded product is a pricing problem a manager can act on, a thin
one is a data problem that implies nothing about the promotion.

### Still owed, and now recorded where the phase gate will check

`PLACEBO_OVERLAP` and `ROI_UNBOUNDED` remain unfireable. They are recorded in
`docs/plan.md` against **Task 4.5** and **Task 5.2**, in each task's **Done
when**, so each phase gate checks that the code fires through `run_audit()`
rather than only that the detection exists.

---

## Task 4.1 — The baseline, and the leak that changed its feature set

Reproduce: `promo.baseline.fit_baseline()`; boosters in `data/interim/baseline/`
and diagnostics in `data/interim/baseline_diagnostics.json`. Settled decisions
**9** and **10** above were both argued from this run; this section is the
record of the fit itself.

### The training frame, and what each exclusion cost

| | rows | units | sales value |
|---|---:|---:|---:|
| panel | 2,966,328 | | |
| − treated rows | −366,346 | −148,153 | −$217,440.07 |
| − weeks outside 18–101 (decision 5) | −248,233 | −41,052 | −$68,325.23 |
| **fitted on** | **2,351,749** | 545,252 | $920,602.93 |

86.99% of the training rows have zero units — the explicit demand zeros Phase
2.6 built inside the scope, kept because dropping them would refit the model on
"weeks the product happened to sell". **13.07% carry a mailer**, which
recomputes decision 8's 13.06% from the shipped panel rather than quoting it.

A caller-supplied frame containing treated rows raises `TreatedRowsError` and is
never filtered. Reading from the parquet, the selection happens in SQL and is
counted in the table above — explicit and recorded, which is a different thing
from silent.

### The fit

Fifteen features, LightGBM at `max_bin=63` / `num_leaves=63`, target
`log1p(units)`, inverted with `expm1` and floored at zero. Three independent
quantile fits at 0.1, 0.5, 0.9.

| feature | gain share |
|---|---:|
| `units_roll_mean_13` | 46.29% |
| `n_stores_carrying` | 30.74% |
| `units_roll_mean_8` | 6.45% |
| `store_traffic` | 5.96% |
| `in_mailer` | 2.18% |
| everything else (10 features) | 8.38% |

`is_holiday_week` contributes exactly zero, as it must — Task 2.6 recorded that
this dataset has no calendar anchor, so the flag is False everywhere.

**Quantile crossings are measured, not sorted away**: q10 > q50 on 0.041% of
rows and q50 > q90 on 0.323%, 0.363% either way. The three models are separate
fits, not a decomposition of one, so crossings are possible; re-ordering them
would make the interval look coherent while hiding that the fits disagree about
the conditional distribution.

### The backtest, and what it is not

Weeks 94–101 held out, 225,038 rows, trained on weeks 18–93. A time split, not a
random one: adjacent weeks of a product-store share their lagged units almost
exactly, so a random split scores the model on near-copies of its own training
rows.

| | |
|---|---:|
| MAE | 0.273 units |
| RMSE | 0.721 units |
| bias | +0.082 units (under-predicts) |
| mean actual / predicted | 0.239 / 0.157 |
| [q10, q90] empirical coverage | 92.73% against a nominal 80% |

**This is necessary and not sufficient, and must never be quoted as evidence the
effect estimate is right.** It is error on untreated weeks; it says nothing
about `ŷ(0)` where `D = 1`, which is the only quantity the baseline exists to
produce. The interval over-covers, which is the safe direction but still a
mis-calibration to carry into Task 5.2's ROI bootstrap.

### The treated group is split, not pooled

Decision 8 requires the two mechanics be reported apart. Over the estimation
window:

| stratum | rows | share | units |
|---|---:|---:|---:|
| `display_only` | 198,191 | 60.51% | 63,481 |
| `display_and_mailer` | 129,324 | **39.49%** | 72,603 |

`mechanic_strata()` emits the label, so Tasks 4.3 and 4.5 can report a display
effect and a joint effect rather than one number that is neither.
