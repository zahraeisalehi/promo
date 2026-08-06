# Promotional Intelligence — top-down build plan

Dunnhumby Complete Journey. Eight CSVs, 2,500 households, 102 weeks, basket-level.

Read this top to bottom once before starting. Then work one phase at a time, and do not begin a phase until the previous phase's **Done when** is literally true.

---

## The vision, stated once so it can be defended

**One sentence:** given a retailer's transaction history, the system says which promotions made money, which merely moved sales around, and which it cannot honestly evaluate at all.

**Three components, and what each is allowed to do:**

| Component | Job | Forbidden |
|---|---|---|
| Agent (read-only) | Explore the data, run the cleaning and derivation pipeline, decide which checks to run | Producing any number itself; writing to `data/raw` |
| Trained model | Estimate the counterfactual baseline — what sales would have been without the promotion | Seeing a promoted row during training |
| LLM calls | Turn computed diagnostics into sentences; guide the user; explain refusals | Computing, estimating, or inferring any number |

Your instinct to separate these is right, and the discipline that makes it work is that **the LLM never touches arithmetic**. It receives a JSON object of already-computed values and writes prose. That single rule is what lets you claim the system is trustworthy, and it is checkable on stage.

**The one adjustment I'd make to your framing:** you described the agent as doing analysis and then handing data to the model. Insert a gate between them. The agent's most valuable output is not clean data — it is a verdict on whether the question is answerable at all. If the treatment doesn't vary, or there's no comparable untreated unit, or the discount exceeds the margin, the model should never run. Build the refusal path before the model path.

**MVP mapping:**

| Deliverable | Produced by | Phase |
|---|---|---|
| MVP 01, 02 — incremental lift | baseline model + rollout | 4 |
| MVP 03 — ROI, margin | accounting layer | 5 |
| MVP 04 — cannibalisation | transfer matrix from baskets | 6 |
| MVP 05, 06 — ranking, recommender | shrinkage + response curves | 7 |
| Required final output — what to stop running | pattern search + shrunk ranking | 7 |

---

## Where this dataset sits in the chain

**This is sell-out data: retailer to end consumer.** Every row is a basket a
shopper actually paid for, at a store, on a day. It is not sell-in — we are not
looking at a manufacturer shipping cases to a distributor and guessing what
happened afterwards.

The participant guide warns that promotional analysis often cannot observe
end-consumer purchases, so lift measured at the shipment level confounds
retailer stock-building with genuine demand. **That warning does not bind here.**
There is no channel inventory between the promotion and the observation, so a
measured unit is a consumed unit, and there is no forward-buy artefact to strip
out.

State this as an advantage in the demo, in three parts:

1. **No sell-in confound.** Lift is demand, not stock movement.
2. **Basket-level granularity.** `BASKET_ID` is what makes Phase 6 identified at
   all. A project reading store totals or shipment volumes can assert
   cannibalisation but cannot measure it.
3. **Household continuity.** The same `household_key` recurs across 102 weeks,
   which is what makes repurchase cycles, pre-window substitute purchase, and
   post-promotion troughs observable rather than assumed.

The honest counterweight, stated in the same breath: this is one retailer's
2,500-household panel, so the *external* generalisation is limited even though
the *internal* observation is clean. Precision about consumption, not breadth of
market.

---

## Phase 0 — Environment

**Objective:** a repo where Claude Code has the context, rules, and guardrails to do the building.

### 0.1 System dependencies

```bash
# Debian / Parrot / Kali / Ubuntu
sudo apt update && sudo apt install -y python3 python3-venv python3-pip jq git
# macOS
brew install python jq git
```

Don't install a version-pinned package like `python3.11`. Debian-derived rolling distros (Parrot, Kali) ship a single current Python and have no `python3.11` package at all, which is what produces `Unable to locate package python3.11`. The project needs 3.11 or newer, not 3.11 exactly.

Check what you have:

```bash
python3 --version    # anything >= 3.11 is fine
```

If it reports 3.10 or older, install a newer one via `pyenv` rather than hunting for a distro package.

`jq` is not optional — the hooks parse JSON with it.

### 0.2 Repo

```bash
mkdir promo && cd promo && git init
mkdir -p promo app tests docs data/raw data/interim data/out scratch
touch promo/__init__.py tests/__init__.py
# copy the eight CSVs into data/raw/
```

### 0.3 Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn scipy statsmodels cvxpy \
            streamlit plotly pydantic pytest ruff anthropic python-dotenv
pip freeze > requirements.txt
```

The venv matters on Debian-derived systems for a second reason beyond isolation: recent Debian marks the system Python as externally managed, so a bare `pip install` outside a venv fails with `error: externally-managed-environment`. Activating `.venv` first avoids that entirely — never reach for `--break-system-packages` here.

If `cvxpy` fails to build a wheel on a very new Python, that's the one package worth checking early, since Step 2.3's shelf-price recovery depends on it. `pip install cvxpy` on its own will tell you quickly, and the fallback is a Condat TV1D implementation with no compiled dependency.

### 0.4 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
# unzip the scaffold bundle into the repo root, then:
chmod +x .claude/hooks/*.sh
echo "ANTHROPIC_API_KEY=sk-..." > .env
claude
```

Now verify the scaffold actually loaded. These are commands you type **into the Claude Code prompt itself** — the input line inside the running `claude` session, not your shell. Type each one and press Enter:

| Type this | You should see | If not |
|---|---|---|
| `/memory` | An editor listing `CLAUDE.md` as Project memory | The file isn't at the repo root, or you launched `claude` from a different directory |
| `/hooks` | A browser showing SessionStart, PreToolUse, PostToolUse, Stop, and a status line entry, each labelled `Project Settings` | `.claude/settings.json` is missing or has a JSON syntax error |
| `/agents` | `causal-reviewer` and `eval-runner` in the list | The files aren't in `.claude/agents/`, or their frontmatter is malformed |
| `/` alone | An autocomplete menu containing `gate`, `placebo`, and `stage` alongside the built-ins | The files aren't in `.claude/commands/` |

Press Escape to close any of these.

The path-scoped rules in `.claude/rules/` are the one thing `/memory` won't show yet — by design, they load only once Claude reads a file matching their `paths:` glob. To confirm they work, ask Claude to read a Python file under `promo/`, then run `/memory` again and check that `python-style.md` has appeared.

Finally, test the guardrail. Ask Claude in plain English:

> Write a test file to data/raw/scratch.txt

It should be denied with a message about `data/raw` being immutable. If the write succeeds instead, the hook isn't executable — run `chmod +x .claude/hooks/*.sh` and restart the session.

**Done when:** `pytest -q` runs (zero tests is fine), the eight CSVs are in `data/raw/`, and `/hooks` shows the PreToolUse hook. Test it by asking Claude to write a file into `data/raw` — it must be denied.

---

## Phase 1 — Know the data

**Objective:** a written description of what is actually in these files, including the things that will break naive analysis. No modelling, no cleaning yet. Just truth.

You cannot skip this. Every wrong assumption made here propagates into every number downstream.

### Task 1.1 — Shape and coverage

> **Prompt:** Write `scratch/explore_01.py` that loads all eight CSVs and reports for each: row count, column dtypes, null counts, and the range of every ID and date column. Specifically report the min and max of DAY and WEEK_NO in transaction_data, how many distinct households, products, stores, and baskets exist, and how many households appear in hh_demographic versus transaction_data. Print a summary; write nothing to disk.

Expect: DAY spans roughly 1–711, WEEK_NO 1–102, and demographics covering well under half the households. That last gap matters later — it means household-level segmentation is only available for a subset.

### Task 1.2 — The quantity column

> **Prompt:** In transaction_data, compute unit value as SALES_VALUE divided by QUANTITY. Report the distribution of QUANTITY and of unit value. Then find rows where unit value is below 0.05 and report what share of rows they are and what share of total QUANTITY they carry. Join to product.csv and report which DEPARTMENT and COMMODITY_DESC values dominate those rows.

You are looking for fuel and other volume-measured goods sharing a QUANTITY column with counted groceries. A tiny share of rows will carry most of the raw units at fractions of a cent. A model trained on raw units learns the price of petrol and calls it demand.

Record the finding. Do not drop anything yet — dropping happens in Phase 2 with a recorded before/after effect.

### Task 1.3 — The discount columns

> **Prompt:** Report the sign, distribution, and zero-share of RETAIL_DISC, COUPON_DISC, and COUPON_MATCH_DISC. Then test two candidate reconstructions of the regular price per unit:
> A: (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
> B: (SALES_VALUE + |RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|) / QUANTITY
> For a sample of 200 high-frequency PRODUCT_IDs, compute both across all transactions and report which produces a more stable modal price per product-week. Report the answer as evidence, not as a conclusion.

This is Beat 1 of the lecture made concrete: the regular price is not a field, it is the level the price returns to. The three discount columns are three different mechanics — retailer loyalty, manufacturer-funded coupon, and the retailer's match of that coupon — and they have different costs and different bearers. Deciding which belongs in the shelf price is a modelling choice you must make explicitly and record.

### Task 1.4 — The treatment candidates

> **Prompt:** For causal_data.csv, report the distinct values of `display` and `mailer` and their frequencies. For each, report what share of product-store-weeks are treated, and how that share varies across products, stores, and weeks. Then do the same for campaign membership from campaign_table.csv at the household level and coupon redemption from coupon_redempt.csv.

This is the most important task in Phase 1. You are looking for which treatment varies on an axis you can compare across. Expect display and mailer to vary across all three of product, store, and week — that is a real treatment. Expect campaign membership to be far more concentrated, and coupon redemption to be very sparse.

### Task 1.5 — Structural zeros

> **Prompt:** Build a household × week grid over all 102 weeks and report what share of household-weeks contain no transaction at all. Then, for a sample of 50 frequently-purchased COMMODITY_DESC values, report the median inter-purchase gap in weeks per household.

The no-trip share tells you how much of your "zero sales" is people not being in the shop rather than people declining to buy. The inter-purchase gaps give you the repurchase cycle per category, which is what sets your measurement horizon in Phase 4. Without it you bank the promotional peak and never see the trough.

**Done when:** `docs/data_findings.md` exists and answers, in writing: which treatment varies and on what axis; which regular-price reconstruction you chose and why; what share of units are volume-measured; the no-trip share; and the median repurchase cycle for your top categories.

---

## Phase 2 — Build the data layer

**Objective:** one clean panel, plus every derived variable the model needs, plus a data-honesty report. This is the biggest phase. Everything downstream reads its output.

### Task 2.1 — Ingest and schema contract

> **Prompt:** Write `promo/io.py` with a `load_raw()` that reads all eight CSVs from data/raw into a dict of DataFrames, enforcing dtypes explicitly (IDs as int64 or string, never float). Return `(dict, IngestReport)` where IngestReport is a pydantic model capturing row counts, date ranges, and which optional columns are present. Note explicitly in the report that no COGS or margin column exists anywhere in this dataset.

That last line matters. This dataset has no margin, which means true ROI is not computable and the honest output is the break-even margin instead. Encode that at ingest so it flows through as a refusal rather than surfacing as a crash in Phase 5.

### Task 2.2 — Transaction cleaning

> **Prompt:** Write `promo/clean.py`. Given raw transactions, produce a cleaned frame and a diagnostics dict. Steps: drop or flag rows with non-positive QUANTITY or SALES_VALUE; identify volume-measured rows using the unit-value threshold from docs/data_findings.md and flag rather than delete them, adding a boolean column; join DEPARTMENT and COMMODITY_DESC from product.csv. The diagnostics must record, for every exclusion, the row count and the share of total units and total sales value affected, before and after.

Rule: no silent filters. A filter whose effect is not recorded is indistinguishable from a bug.

### Task 2.3 — Price decomposition

> **Prompt:** Write `promo/prices.py`. Aggregate cleaned transactions to PRODUCT_ID × STORE_ID × WEEK_NO, computing units, sales value, and each discount component summed. Derive: paid price per unit; regular price per unit using the reconstruction chosen in Phase 1; discount depth as 1 minus paid over regular; and three separate depth components for loyalty, manufacturer coupon, and coupon match. Add a three-valued `price_status` flag per product: `bounded` if the product is on deal in more than 90% of its priced weeks, since its depth is then ordinal only; `insufficient_support` if it has fewer than 8 distinct priced weeks, since then nothing can be said about its depth in either direction; `identified` otherwise.

**Support is tested before depth, and the order carries the argument.** A deal
share computed on one or two weeks is not an ordinal depth — it is noise, and
calling it `bounded` claims a diagnosis the data cannot support. A product seen
once, on deal, has a deal share of 1.0 and is not a perpetually-discounted
product. Count support in **distinct weeks**, not product-store-weeks: ten
stores carrying a product in one week is one week of price history, not ten.
Phase 3 refuses the two under different reason codes; see Task 3.5 for why the
distinction is a product decision and not a taxonomy preference.

### Task 2.4 — Price index and deflation

> **Prompt:** Write `build_price_index()` in promo/prices.py: a weekly index built from unpromoted rows only, as a chained geometric mean of per-product price relatives. Apply it to produce real price columns alongside nominal. Report the total index drift across the 102 weeks.

Expect near-flat drift on this US dataset — that is the correct result and you should report it. Build the module anyway: your pitch targets Iranian retail, where the same estimator is load-bearing, and a module that returns ~1.0 here and is essential there is a strength, not waste. Say that explicitly in the demo.

### Task 2.5 — The treatment panel

> **Prompt:** Write `promo/treatment.py`. Join causal_data to the product-store-week panel. Create `on_display` and `in_mailer` booleans plus the raw categorical codes preserved as separate columns. Create a combined `treated` flag per the treatment definition, taking the definition as a parameter rather than hardcoding it. Weeks and product-stores absent from causal_data are untreated, not missing — assert this is a defensible assumption and record the share it affects.

### Task 2.6 — Derived model variables

> **Prompt:** Write `promo/features.py` producing, per product-store-week: lags of units at 1, 2, 4, and 52 weeks; rolling means over 4, 8, and 13 weeks; week-of-year and a holiday flag; category units excluding the focal product; store total traffic that week; the product's price relative to its category median price; number of stores carrying the product that week; and the weekly price index. Every lag must be computed within product-store, never across.

These are the extra variables that let the baseline generalise across campaign types. The category-excluding-self and store-traffic controls are what let it distinguish "this product rose" from "everything rose."

### Task 2.7 — Availability and horizon

> **Prompt:** Write `promo/quality.py`. Produce a household-week shopping flag, a structural-versus-sampling zero classifier, and a per-commodity repurchase cycle from median inter-purchase gaps. Write the repurchase cycles to data/interim/repurchase_cycles.parquet.

**Done when:** `data/interim/panel.parquet` exists at product × store × week with all derived features, `data/interim/quality.json` records every exclusion and its effect, and a test asserts no feature column is computed from post-treatment data.

---

## Phase 3 — The feasibility gate

**Objective:** for any proposed campaign, a verdict of measurable, bounded, or not identified — before any model runs. This phase is small in code and large in value.

### Task 3.1 — Variation

> **Prompt:** Write `promo/audit.py` with `variation_axes(panel)`: for each of product, store, and week, report the share of units that are fully treated, fully untreated, and mixed. The usable axis is the one with meaningful mixed mass.

### Task 3.2 — Overlap

> **Prompt:** Add `overlap(panel)`: fit a gradient-boosting classifier predicting `treated` from the covariates, cross-validated, and report AUC plus the share of observations with propensity below 0.02 or above 0.98. Also report the top five features by importance so a near-perfect AUC can be diagnosed as leakage rather than genuine non-overlap.

### Task 3.3 — Collisions and horizon

> **Prompt:** Add checks for: product-store-weeks where display and mailer both fire and are being treated as one effect; and campaigns whose measurement horizon is shorter than the commodity's repurchase cycle.

### Task 3.4 — Break-even

> **Prompt:** Add `kappa_star(depth, margin)` returning depth over margin, with margin as a required user-supplied parameter. When margin is None, return None and the reason code NO_MARGIN. Add a sweep that, for a given campaign's observed depth, tabulates the required incremental share across assumed margins from 10% to 50% — on the same nine-point grid Task 5.2 defines (5-point steps), so the gate's sweep and the accounting table never disagree.

Since this dataset has no COGS, that sweep *is* your MVP 03 answer, and it is an honest one: here is the margin your promotion needed to beat.

### Task 3.5 — The refusal engine

> **Prompt:** Write `promo/gates.py` with a `GateResult` pydantic model (gate, status, reason_code, detail, message) and a deterministic message template for each reason code: NO_VARIATION, NO_OVERLAP, LEAKED_FEATURE, DEPTH_BOUNDED, INSUFFICIENT_SUPPORT, KAPPA_IMPOSSIBLE, NO_MARGIN, PLACEBO_OVERLAP, OVERLAPPING_TREATMENTS, ROI_UNBOUNDED, HORIZON_TOO_SHORT. Write a `run_audit()` that returns a list of GateResults and short-circuits the pipeline on any refuse. Add tests that construct data guaranteed to trigger each code.

**`INSUFFICIENT_SUPPORT` and `DEPTH_BOUNDED` are separate codes and must stay
separate.** Task 2.3 assigns every product one of three `price_status` values,
and the last two are different diagnoses that a category manager acts on
differently:

| status | reason code | what it says | what to do about it |
|---|---|---|---|
| `identified` | — | depth is cardinal | measure it |
| `bounded` | `DEPTH_BOUNDED` | on deal in over 90% of its priced weeks, so the regular price is barely observed and depth is **ordinal only** | rank this product's deals against each other; stop running it at every depth if you want its shelf price back |
| `insufficient_support` | `INSUFFICIENT_SUPPORT` | fewer than 8 distinct priced weeks, so **nothing** can be said about depth in either direction | this is a data problem, not a pricing problem — no action on the promotion is implied |

Collapsing the two would tell a manager that a product they have run four times
is permanently discounted. On this dataset the distinction is not marginal:
**63,929 products (69.6%) are `insufficient_support` and only 3,445 (3.7%) are
`bounded`.** Under a two-way rule, 14,772 of the thin products would have been
labelled `bounded` — four times the genuine count — because a product seen once,
on deal, has a deal share of 1.0.

The size of each group in money is the reason neither code is a footnote:
`identified` covers 26.7% of products but **75.5% of sales value**, `bounded`
3.7% and 15.0%, `insufficient_support` 69.6% and **9.6%**. A refusal that covers
seven products in ten but under a tenth of the money is exactly the shape a
useful long-tail refusal should have, and the audit should say so rather than
report a bare product count.

Both codes carry the product's `n_weeks_priced` and `deal_share` in `detail`, so
the refusal message states the evidence rather than only the verdict.

**Done when:** `run_audit()` returns a full verdict for a real campaign, every reason code has a test that fires it, and the pipeline provably stops on refuse.

This is your first demoable checkpoint. If everything after this failed, you would still have the strongest version of the lecture's own argument.

---

## Phase 4 — The baseline and incremental lift (MVP 01, 02)

**Objective:** a defensible counterfactual, and incremental units with an interval and a placebo band.

### Task 4.1 — Train

> **Prompt:** Write `promo/baseline.py`. Fit LightGBM on `treated == 0` rows only, predicting log1p(units) from the Phase 2 features. Raise if treated rows are present in the training frame. Add quantile variants at 0.1, 0.5, 0.9.

### Task 4.2 — Rollout

> **Prompt:** Add `rollout(model, history, exog_weeks)` producing a multi-week counterfactual where each week's prediction feeds back as the next week's lag, never the observed value. Write `tests/test_rollout_contamination.py` proving the naive version shrinks the measured effect as window length grows and the rollout version does not.

### Task 4.3 — Lift with horizon

> **Prompt:** Write `estimate_lift(campaign)` summing residuals over a window extended past campaign end by that commodity's repurchase cycle. Return gross incremental during the window, post-window residual, net incremental, and a retention ratio of net over gross.

### Task 4.4 — Synthetic truth

> **Prompt:** Write `tests/synthetic.py` generating panels with a known effect using multiplicative seasonality and AR(1) noise — deliberately not a tree-shaped process. Run the full pipeline across true effects of 0, 0.05, 0.15, 0.30 and report bias and interval coverage.

### Task 4.5 — Placebo

> **Prompt:** Write `promo/validate.py` with a placebo harness over at least 300 never-treated windows. Return the distribution and a helper that flags whether a given estimate falls inside it.

**Done when:** the τ=0 synthetic case recovers approximately zero, the placebo band is computed and stored, and every reported lift carries an interval and its placebo comparison.

---

## Phase 5 — Accounting (MVP 03)

> **Prompt:** Write `promo/accounting.py`. Compute the subsidy — the discount paid on all units sold during the promotion, not only the incremental ones. Compute promotional profit as incremental margin minus total promotional cost, parameterised on an assumed margin. Compute ROI as a bootstrap interval from the baseline's quantile draws, returning ROI_UNBOUNDED when the denominator interval spans zero. Never average ROIs across campaigns; aggregate components and divide once.

### NO_MARGIN is the condition that produces the right answer, not a gap in it

`promo/io.py` establishes at ingest that no cost or margin column exists in any
of the eight tables — it searches all 46 column names for `cogs`, `margin`,
`cost`, `profit`, and `gross`, finds nothing, and sets the reason code
`NO_MARGIN` (`data/interim/ingest_report.json`). Treat that as settled and final
for this dataset.

**It cannot be closed by working harder.** Margin requires the cost of goods —
what the retailer paid its supplier. Transaction data records what the *shopper*
paid. The two numbers sit on opposite sides of the retailer, and no amount of
price reconstruction crosses that line: every figure recoverable from
`SALES_VALUE`, `RETAIL_DISC`, `COUPON_DISC`, and `COUPON_MATCH_DISC` is a fact
about the shopper-facing price, and none of them is a fact about supplier cost.
A margin here would have to be invented. This project's whole claim is that it
does not invent numbers, so it does not invent this one.

**So the deliverable changes shape, and is better for it.** Rather than one ROI
resting on a fabricated margin, MVP 03 ships two objects that are computable
without any margin at all:

1. **The break-even margin per campaign** — the gross margin the promotion would
   have needed to clear its own discount.
2. **A sensitivity table** — incremental profit across assumed margins from 10%
   to 50%, so a reader who knows their own margin reads their answer off the
   table.

A merchant knows their margin; the dataset does not. Handing them the threshold
and letting them apply their own number is more useful than handing them a point
estimate built on a guess, and it is the only version of this output that is
true. Say exactly that on stage. `NO_MARGIN` is not an apology.

### Task 5.1 — Total promotional cost has two components, not one

Promotional cost is **discount subsidy plus free goods**. Both are money the
retailer gave away to run the promotion; counting only the first understates the
denominator of every ROI in the system.

> **Prompt:** In `promo/accounting.py`, compute promotional cost as two separately reported components summed into one total:
> - **`subsidy`** — the discount paid on all units sold during the promotion, from the Phase 2.3 discount decomposition, kept split by mechanic (loyalty, manufacturer coupon, coupon match) because the cost bearer differs.
> - **`free_goods`** — the giveaway lines: rows with `QUANTITY > 0` and `SALES_VALUE == 0`. Value them at the product's reconstructed regular price per unit from Phase 2.3, not at zero, and not at the paid price. Report the units given away, the count of rows, and the imputed value. Where no regular price is recoverable for that product, report those units as `free_goods_unpriced` and refuse to fold them into the total rather than valuing them at zero.
>
> Return `promo_cost_total = subsidy + free_goods`, with both components and the unpriced residual visible in the diagnostics dict. ROI, break-even margin, and the margin-sensitivity sweep all use the total, never the subsidy alone.

**Why these rows are a cost and not noise.** Phase 1 found **4,451 rows with
positive `QUANTITY` and zero `SALES_VALUE`, carrying 4,544 units**
(`docs/data_findings.md`). They are a rounding error in units — 0.00% of the
260.7M total — and they were *correctly* excluded from the volume-measured
analysis, because they are a different phenomenon from the KIOSK-GAS fuel rows.
That exclusion was about **units**. This is about **money**. A buy-one-get-one
free arm, a sampling giveaway, or a store-funded free line costs the retailer the
full regular price of every unit handed over, and the transaction file records it
as revenue zero. Left out, the promotion looks cheaper than it was.

Two consequences to encode:

1. **Do not reuse the Phase 2.2 volume-measured exclusion here.** That filter
   drops rows for being unit-incomparable. These rows must survive into the
   accounting layer as a cost line even though they contribute no revenue. If
   `clean.py` has already removed them, the accounting layer reads them back from
   the pre-exclusion frame and says so in its diagnostics.
2. **Free goods are a cost, never a lift.** Units given away are not incremental
   demand. Assert that `free_goods` units never enter the Phase 4 lift numerator;
   a test should fail if they do.

Report the two components separately in the UI as well as summed. A campaign
whose cost is 90% free goods is a structurally different instrument from one
that is 90% price discount, and the recommendation in Phase 7 depends on telling
them apart.

### Task 5.2 — Break-even margin and the sensitivity table

> **Prompt:** In `promo/accounting.py`, add `breakeven_margin(campaign)` returning `m_star = promo_cost_total / incremental_revenue`, where `promo_cost_total` is the subsidy plus free goods from Task 5.1 and `incremental_revenue` is the Phase 4 incremental units valued at the promoted price. Return it as an **interval**, not a point: the numerator is known and the denominator is estimated, so propagate the lift interval through the ratio. Return `None` with reason code `ROI_UNBOUNDED` when the incremental-revenue interval spans zero — a promotion with no measurable lift has no finite break-even margin, and reporting a large one implies a precision that is not there.
>
> Then add `sensitivity_table(campaign)` returning incremental profit at assumed gross margins of **10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, and 50%** — nine columns, 5-point steps. Each cell is `m * incremental_revenue - promo_cost_total`, in currency, signed. Carry the lift interval into each cell so a cell whose sign is uncertain is marked as such rather than shown as a confident positive. The table is the campaign's answer; the break-even margin is where its sign flips.

**Read the table, not a single number.** The nine columns exist so that the
reader supplies the one fact the dataset cannot. A merchant on a 22% margin
looks at the 20% and 25% columns and knows. Nothing in the pipeline needs to
guess on their behalf, and the table makes the guess unnecessary rather than
hidden.

**The identity that ties this to Phase 3.** Task 3.4's `kappa_star(depth, margin)`
— the incremental share a campaign needs to break even — is the same statement
seen from the other side: `kappa_star(m) = m_star / m`. Requiring more than 100%
of promoted units to be incremental is exactly the condition that the break-even
margin exceeds the assumed margin. Task 3.4's sweep and this table therefore use
the identical 10–50% grid in 5-point steps, and this task is the authority on
that grid. The generalisation here is that `m_star` includes free goods, which a
depth-over-margin ratio does not.

**Flag any campaign whose break-even margin exceeds 50%.** No plausible grocery
gross margin clears it, so the campaign is arithmetically unprofitable before
any measurement question is asked — every cell in its sensitivity row is
negative, by construction. Emit the existing `KAPPA_IMPOSSIBLE` reason code
rather than inventing a new one: `m_star > 0.5` and `kappa_star(0.5) > 1` are
the same sentence, by the identity above. Surface these campaigns in the
Portfolio page as a distinct group, and feed them to Task 7.4, where a campaign
that cannot pay at any believable margin is the cheapest stop recommendation the
system can make — it needs no counterfactual to defend it.

### Task 5.3 — A supplied margin is an assumption, and is labelled as one

The system accepts a margin from the user. It never forgets where it came from.

> **Prompt:** Give every accounting entry point an optional `margin` parameter and a `margin_source` field taking `None`, `"supplied"`, or `"derived"` — with `"derived"` unreachable on this dataset and present only so a future dataset carrying COGS does not need the field invented. When `margin` is None, return break-even and the sensitivity table, and return `None` with reason code `NO_MARGIN` for any figure that requires a margin. When `margin` is supplied, compute those figures and stamp every one of them — profit, ROI, ranking position, response-curve crossing, stop recommendation — with `conditional_on_margin = <value>` and `margin_source = "supplied"`. Add a test asserting that no figure computed from a supplied margin can be serialised without that stamp.

Three rules follow, and none of them is optional:

1. **A supplied margin never replaces the measured objects.** The break-even
   margin and the sensitivity table still ship. They are what the data
   establishes; the supplied figure is what the user asserted.
2. **`promo/narrate.py` must say so in words.** A verdict resting on a supplied
   margin reads "at the 30% margin you supplied, this campaign returned …",
   never "this campaign returned …". The LLM receives `margin_source` in its
   JSON and has a fixed clause for it.
3. **The UI marks it.** Any figure carrying `conditional_on_margin` renders with
   the assumption visible next to it, through the same shared component that
   renders `GateResult`, so an assumption looks as deliberate as a refusal.

The failure this prevents is specific and easy to walk into: a user types 30%
into a box during the demo, and four screens later a ranked list of ROIs looks
like a measurement. It is not one. It is arithmetic conditional on a number the
user made up, and the system must keep saying so.

**Done when:** every campaign has a break-even margin reported as an interval, a nine-column sensitivity table across 10–50% in 5-point steps, and either an ROI interval or a stated refusal; total promotional cost is reported as subsidy plus free goods, with the split visible and the unpriced residual stated rather than silently zeroed; campaigns with a break-even margin above 50% are flagged `KAPPA_IMPOSSIBLE`; and no figure derived from a supplied margin can leave the module without carrying `margin_source` and `conditional_on_margin`.

---

## Phase 6 — Cannibalisation (MVP 04)

You have BASKET_ID, so this is identified. Most teams working from store totals cannot do this at all.

> **Prompt:** Write `promo/transfer.py`. Using baskets, for a promoted product identify households that purchased a substitute in the same COMMODITY_DESC in their pre-window trips and the promoted product during the window. Build T where T[i][j] is units moved from j to i. Assert mass conservation: row sums total equals column sums total. Return gains, losses, and the matrix, and label the cell-level split in the output metadata as a stated convention rather than an identified quantity.

> **Prompt:** Then write `decompose(campaign)` returning `delta_q = s + (g - l)` where s comes from the Phase 4 counterfactual. Add a test that fails if any code path subtracts cannibalisation from lift.

**Done when:** the decomposition runs end to end, mass conservation asserts pass, and expansion and redistribution are reported as separate quantities.

---

## Phase 7 — Ranking, pattern search, and recommendation (MVP 05, 06)

### Task 7.1 — Ranking

> **Prompt:** Write `promo/decide.py`. Rank campaigns on James-Stein shrunk estimates rather than raw ones, and publish the shrunk value as the expectation. Fit a lift-versus-depth response curve per commodity and locate where marginal return crosses zero. Add an MDE calculator over holdout fraction, cluster count, cluster size, and intra-cluster correlation.

### Task 7.2 — Cross-campaign pattern search

Ranking answers *which campaign* did well. It does not answer *what kind of
promotion* does well, and the second question is the one a merchant can act on
next quarter. Search for structure across campaigns on four axes.

> **Prompt:** Write `promo/patterns.py` with `search_patterns(campaign_results, panel)`. Partition the evaluated campaigns along four pre-declared axes and, within each segment, report the pooled effect — aggregate the components and divide once, never an average of per-campaign ratios. The axes:
>
> - **Discount depth** — banded from the Phase 2.3 depth, e.g. 0–10%, 10–20%, 20–30%, 30–50%, 50%+. Exclude `bounded` products from depth bands entirely, since their depth is ordinal only; report how many campaigns that removes.
> - **Timing** — week-of-year block, holiday proximity, campaign length in weeks, and position relative to the commodity's repurchase cycle (shorter than one cycle, one to two, longer).
> - **Product** — `DEPARTMENT`, `COMMODITY_DESC`, price tier relative to category median, and repurchase-cycle band (fast versus slow movers).
> - **Store segment** — store traffic tier, store count carrying the product, and treatment intensity (how much of the store's assortment was on deal that week).
>
> Return one row per (axis, segment) with: campaign count, distinct products, distinct stores, pooled lift, pooled promotional cost, break-even margin, and the verdict from Task 7.3. Return `(DataFrame, dict)` like every other stage.

**The declaration rule.** The axis and band definitions are written down *before*
the effects are looked at, and the file records the full list of segments tested,
including the ones that came back null. A pattern search whose denominator of
attempted tests is unrecorded is not a search, it is a story. The count of tests
performed is what Task 7.3 needs.

### Task 7.3 — The coincidence test

With few campaigns per segment, the strongest-looking pattern is exactly what you
would expect from noise plus a wide search. Every claimed pattern must clear four
hurdles, all reported, before it can be called a pattern.

> **Prompt:** Add `test_pattern(segment, campaign_results, seed)` to `promo/patterns.py` returning a `PatternVerdict` pydantic model (axis, segment, n_campaigns, effect, hurdles dict, status, reason_code, message). The four hurdles:
>
> 1. **Effective sample size.** `n` is the number of **independent campaigns** in the segment, not product-store-weeks. Product-store-weeks within one campaign are one draw, not thousands. If `n_campaigns < 5`, return status `INSUFFICIENT` with reason code `FEW_CAMPAIGNS` and stop — no p-value is computed, because computing one would give it a credibility the sample cannot support.
> 2. **Leave-one-campaign-out stability.** Recompute the segment effect dropping each campaign in turn. Report the min, max, and sign consistency across the `n` refits. A pattern that reverses sign or loses its ranking when any single campaign is removed is one campaign, not a pattern. Fail the hurdle if the sign is not consistent across all leave-one-out refits.
> 3. **Search-corrected permutation null.** Permute the segment labels across campaigns `B >= 2000` times with an explicit `seed` and `np.random.default_rng`. On each draw, recompute the effect for **every** segment on that axis and record the *maximum* absolute effect across segments. Compare the observed effect against that max-statistic null. This corrects for having searched many segments and reporting the winner; an uncorrected per-segment p-value is not acceptable here. Report the corrected p and the number of segments the correction covers.
> 4. **Temporal hold-out.** Refit the pattern on campaigns in the first half of the window and check whether it holds, with the same sign, on campaigns in the second half. Report both halves' effects and their campaign counts. If either half has fewer than three campaigns, report `HOLDOUT_UNAVAILABLE` rather than a pass.
>
> Status is `HOLDS` only when hurdles 2, 3, and 4 all pass and hurdle 1 is satisfied. `SUGGESTIVE` when the permutation survives but stability or hold-out does not. `COINCIDENCE` when the permutation does not survive. `INSUFFICIENT` when there are too few campaigns to ask. Add reason codes `FEW_CAMPAIGNS`, `UNSTABLE_TO_LEAVE_ONE_OUT`, `FAILS_PERMUTATION`, `HOLDOUT_UNAVAILABLE` to `promo/gates.py` and give each a deterministic message template.

> **Prompt:** Write `tests/test_pattern_coincidence.py`. Generate campaign result sets under a null where segment membership is assigned at random and no true segment effect exists, and assert that across many seeds the `HOLDS` rate is at or below the nominal level — that is, the search does not manufacture patterns from noise. Then generate a set with one real, planted segment effect and assert it is recovered as `HOLDS` at a reasonable effect size. Also assert that a segment carrying a single dominant campaign is caught by hurdle 2 rather than reported as a pattern.

The synthetic-null test is the point of this task. The search is the thing most
likely to produce a confident wrong answer on stage, so it gets a harness that
proves its false-positive rate, in the same way the placebo band proves the
estimator's zero.

Every pattern shown in the UI displays its `n_campaigns`, its leave-one-out
range, and its corrected p. A pattern with `SUGGESTIVE` or `INSUFFICIENT` status
is shown with that label, not hidden and not promoted.

### Task 7.4 — The required final output: what to stop running

The deliverable is not complete without a named recommendation to **stop**. A
system that only ever ranks winners has not been asked to take a risk.

> **Prompt:** Add `recommend_stop(campaign_results, patterns)` to `promo/decide.py`, returning at least one named promotion or promotion type to discontinue, together with the evidence chain that supports it. Each recommendation returns: the target (a specific campaign, or a segment from Task 7.2 named in plain language such as "30%+ depth on fast-moving GROCERY in high-traffic stores"); the shrunk lift estimate with its interval; the placebo band it was compared against; the total promotional cost including free goods; the break-even margin; the margin range over which it loses money; the cannibalisation split from Phase 6; the `PatternVerdict` if the target is a type rather than a single campaign; and the gate verdicts that let it be evaluated at all.

**The bar for naming something.** A stop recommendation is supported when at
least one of these holds, and the output states which:

- The shrunk lift interval **overlaps or sits inside the placebo band** — the
  promotion is not distinguishable from doing nothing, while the subsidy and free
  goods are certain.
- The break-even margin **exceeds the plausible margin range** for that
  department, so the campaign cannot have paid under any assumption in the
  sensitivity table. Name the range and its source as an assumption, since this
  dataset has no COGS.
- `delta_q` is **dominated by redistribution rather than expansion** — the
  promotion moved sales between products at full subsidy cost without growing the
  category.
- The response curve from Task 7.1 shows the depth run is **past the point where
  marginal return crosses zero**, in which case the recommendation is to reduce
  depth to the crossing point rather than to stop entirely, and it says so.

**The bar for refusing to name one.** If no campaign or type clears any of the
above, the honest output is a refusal with a reason code, plus the closest
candidate and the specific evidence that is missing — not silence, and not a
recommendation stretched to fill the slot. That refusal is a legitimate final
answer under this project's thesis, but it must be reached, not defaulted to. In
practice on this dataset, expect at least one deep-discount segment to fail the
placebo comparison.

Never justify a stop recommendation by intuition, by category priors, or by "this
kind of promotion usually underperforms." Every clause in the output traces to a
computed value, and `promo/narrate.py` receives it as JSON and turns it into
sentences without adding a number.

**Done when:** the ranking is shrunk, the response curves are fitted, the MDE calculator runs, `search_patterns()` returns verdicts across all four axes with the coincidence harness passing its synthetic-null test, and `recommend_stop()` names at least one promotion or promotion type to stop with its full evidence chain — or refuses with a reason code and the missing evidence stated.

---

## Phase 8 — Agent, UI, chatbot

Only now. Building this earlier is the most common way hackathon projects fail.

### Task 8.1 — Orchestration

> **Prompt:** Write `promo/agent.py` exposing each pipeline stage as a typed tool function, plus `analyse(campaign_spec)` that runs ingest, clean, derive, audit, and stops on any refuse, returning a structured result object containing every GateResult and every computed metric. This is deterministic Python — no LLM calls in this module.

### Task 8.2 — UI

> **Prompt:** Write `app/app.py` in Streamlit with five pages reading only from data/out: Audit (the gate verdicts and data-honesty report — this is the landing page), Portfolio (ranked measurable campaigns, with unmeasurable ones listed separately by reason code), Campaign (counterfactual chart with band, placebo distribution with the estimate marked, expansion versus redistribution, a promotional-cost bar split into subsidy and free goods, break-even table), Patterns (the Task 7.2 segment table across all four axes, each row showing its `PatternVerdict` status, `n_campaigns`, leave-one-out range, and corrected p — `SUGGESTIVE` and `INSUFFICIENT` rows visible and labelled, never filtered out), and Planner (depth response curve, MDE holdout designer, and the Task 7.4 stop recommendation rendered as a named target with its full evidence chain). Render GateResult and PatternVerdict through one shared component so refusals look like a deliberate product state everywhere.

### Task 8.3 — Narration

> **Prompt:** Write `promo/narrate.py` calling the Anthropic API with a system prompt forbidding computation: it receives a JSON diagnostics object and returns two or three sentences, saying a value cannot be reported when it is null. Use it for campaign verdicts, refusal messages, and a chat panel scoped to the campaign currently on screen. Cache generated refusal messages per reason code so the demo works without network.

**Done when:** you can open the app, land on the audit, drill into a refused campaign, drill into an accepted one, and ask the chat why it worked.

---

## Cut order

If you fall behind, remove in this order and no other:

1. The planner's optimisation, replaced by a top-3 heuristic list
2. The chat panel, keeping only pre-generated verdict paragraphs
3. The transfer matrix, reporting gains and losses without cell-level flows
4. Confidence intervals on the recommender, keeping them on the estimates
5. The pattern search's store-segment and timing axes, keeping depth and product — but never the coincidence test on the axes that remain

Never cut: the placebo harness, the refusal engine, the rollout, the data-honesty report, or the stop recommendation of Task 7.4. Those five are the argument. Everything else is decoration on it.

If time collapses hard, Task 7.4 can name a single campaign rather than a type, since a single campaign needs only the placebo and break-even evidence and not the pattern search. It cannot be cut to nothing.

---

## Demo order

Audit page and the refusal → the data-honesty finding about volume-measured units → an accepted campaign's counterfactual → the placebo band with the estimate outside it → expansion versus redistribution → break-even margin table with subsidy and free goods split → the pattern that survived the coincidence test → **the promotion we are telling you to stop, and the four numbers that say so** → the holdout designer.

Four minutes, all six MVPs, in order. The stop recommendation is the close, not
the holdout designer — a named thing to kill lands harder than a methodological
suggestion, and it is the one slide where the placebo band, the free-goods cost,
and the redistribution split all pay off at once. Open the sell-out point early,
when the counterfactual first appears: what we measure is what a person carried
out of the shop, so there is no shipment inventory hiding between the promotion
and the number.
