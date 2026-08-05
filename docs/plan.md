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

> **Prompt:** Write `promo/prices.py`. Aggregate cleaned transactions to PRODUCT_ID × STORE_ID × WEEK_NO, computing units, sales value, and each discount component summed. Derive: paid price per unit; regular price per unit using the reconstruction chosen in Phase 1; discount depth as 1 minus paid over regular; and three separate depth components for loyalty, manufacturer coupon, and coupon match. Add an `identified` versus `bounded` flag per product: bounded if the product is on deal in more than 90% of its weeks, since its depth is then ordinal only.

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

> **Prompt:** Add `kappa_star(depth, margin)` returning depth over margin, with margin as a required user-supplied parameter. When margin is None, return None and the reason code NO_MARGIN. Add a sweep that, for a given campaign's observed depth, tabulates the required incremental share across assumed margins from 10% to 50%.

Since this dataset has no COGS, that sweep *is* your MVP 03 answer, and it is an honest one: here is the margin your promotion needed to beat.

### Task 3.5 — The refusal engine

> **Prompt:** Write `promo/gates.py` with a `GateResult` pydantic model (gate, status, reason_code, detail, message) and a deterministic message template for each reason code: NO_VARIATION, NO_OVERLAP, LEAKED_FEATURE, DEPTH_BOUNDED, KAPPA_IMPOSSIBLE, NO_MARGIN, PLACEBO_OVERLAP, OVERLAPPING_TREATMENTS, ROI_UNBOUNDED, HORIZON_TOO_SHORT. Write a `run_audit()` that returns a list of GateResults and short-circuits the pipeline on any refuse. Add tests that construct data guaranteed to trigger each code.

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

> **Prompt:** Write `promo/accounting.py`. Compute the subsidy — the discount paid on all units sold during the promotion, not only the incremental ones. Compute promotional profit as incremental margin minus subsidy, parameterised on an assumed margin. Compute ROI as a bootstrap interval from the baseline's quantile draws, returning ROI_UNBOUNDED when the denominator interval spans zero. Never average ROIs across campaigns; aggregate components and divide once.

**Done when:** every campaign has a break-even margin, a margin-sensitivity table, and either an ROI interval or a stated refusal.

---

## Phase 6 — Cannibalisation (MVP 04)

You have BASKET_ID, so this is identified. Most teams working from store totals cannot do this at all.

> **Prompt:** Write `promo/transfer.py`. Using baskets, for a promoted product identify households that purchased a substitute in the same COMMODITY_DESC in their pre-window trips and the promoted product during the window. Build T where T[i][j] is units moved from j to i. Assert mass conservation: row sums total equals column sums total. Return gains, losses, and the matrix, and label the cell-level split in the output metadata as a stated convention rather than an identified quantity.

> **Prompt:** Then write `decompose(campaign)` returning `delta_q = s + (g - l)` where s comes from the Phase 4 counterfactual. Add a test that fails if any code path subtracts cannibalisation from lift.

**Done when:** the decomposition runs end to end, mass conservation asserts pass, and expansion and redistribution are reported as separate quantities.

---

## Phase 7 — Ranking and recommendation (MVP 05, 06)

> **Prompt:** Write `promo/decide.py`. Rank campaigns on James-Stein shrunk estimates rather than raw ones, and publish the shrunk value as the expectation. Fit a lift-versus-depth response curve per commodity and locate where marginal return crosses zero. Add an MDE calculator over holdout fraction, cluster count, cluster size, and intra-cluster correlation.

**Done when:** the ranking is shrunk, the response curves are fitted, and the MDE calculator runs — giving you a recommendation whose headline is "hold out 20% next time."

---

## Phase 8 — Agent, UI, chatbot

Only now. Building this earlier is the most common way hackathon projects fail.

### Task 8.1 — Orchestration

> **Prompt:** Write `promo/agent.py` exposing each pipeline stage as a typed tool function, plus `analyse(campaign_spec)` that runs ingest, clean, derive, audit, and stops on any refuse, returning a structured result object containing every GateResult and every computed metric. This is deterministic Python — no LLM calls in this module.

### Task 8.2 — UI

> **Prompt:** Write `app/app.py` in Streamlit with four pages reading only from data/out: Audit (the gate verdicts and data-honesty report — this is the landing page), Portfolio (ranked measurable campaigns, with unmeasurable ones listed separately by reason code), Campaign (counterfactual chart with band, placebo distribution with the estimate marked, expansion versus redistribution, subsidy bar, break-even table), and Planner (depth response curve and MDE holdout designer). Render GateResult through one shared component so refusals look like a deliberate product state everywhere.

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

Never cut: the placebo harness, the refusal engine, the rollout, or the data-honesty report. Those four are the argument. Everything else is decoration on it.

---

## Demo order

Audit page and the refusal → the data-honesty finding about volume-measured units → an accepted campaign's counterfactual → the placebo band with the estimate outside it → expansion versus redistribution → break-even margin table → the holdout designer as the closing recommendation.

Four minutes, all six MVPs, in order.
