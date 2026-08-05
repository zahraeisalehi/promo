# Promotional Intelligence

Measures whether retail promotions actually made money, by estimating what sales would have been without them. Hackathon project on the Dunnhumby Complete Journey dataset: 2,500 households, 102 weeks, basket-level, eight CSVs in `data/raw/`.

## The thesis this code exists to defend

A number returned without identification is worse than no number. The system must refuse to produce an estimate it cannot defend, and say why in plain language. Refusal is a product state, not an error.

## Build order

Phases are sequential. Do not start one before the previous phase's **Done when** in `docs/plan.md` is literally true.

0 environment · 1 know the data · 2 data layer · 3 feasibility gate · 4 baseline and lift · 5 accounting · 6 cannibalisation · 7 ranking · 8 agent, UI, chatbot

The gate comes before the model on purpose. The most valuable output of the pipeline is a verdict on whether the question is answerable, not a number.

## Dataset facts that are already settled

Do not re-litigate these; they were decided in Phase 1 and are recorded in `docs/data_findings.md`.

- **The treatment is `display` and `mailer` from `causal_data.csv`**, which vary across product, store, and week. Campaign membership and coupon redemption are household-targeted and mostly fail the variation test.
- **The panel grain is PRODUCT_ID × STORE_ID × WEEK_NO.** Household-level work is a separate, optional axis.
- **There is no COGS or margin column anywhere.** True ROI is therefore not computable. The honest MVP 03 output is the break-even margin plus a sensitivity table. Never impute a margin.
- **`QUANTITY` mixes counted and volume-measured goods.** A small share of rows carries most of the raw units at fractions of a cent. Flag, never silently drop.
- **`SALES_VALUE` is what the shopper paid.** The regular price is reconstructed, not read. `RETAIL_DISC`, `COUPON_DISC`, and `COUPON_MATCH_DISC` are three different mechanics with different cost bearers — keep them separate.
- **Baskets exist**, so cannibalisation is identifiable. Store-total-only projects cannot do this.
- Demographics cover well under half of households. Anything using `hh_demographic` runs on a subset and must say so.

## Data access

- **Never read `data/raw/transaction_data.csv` directly.** It has 2.6M rows and will exhaust memory.
- Use `data/interim/transactions.parquet` via DuckDB for aggregates.
- Use `data/interim/transactions_sample.parquet` (300k rows) for distribution and profiling questions.
- Always `SET memory_limit='2GB'` and `threads=2` on the DuckDB connection.

## Non-negotiable invariants

- Train the baseline on `treated == 0` rows only. Raise if treated rows reach the fit — never filter silently, because that hides a caller bug.
- Multi-week windows use recursive rollout: feed predicted counterfactuals back as lags, never observed values.
- `delta_q = s + (g - l)`. Expansion and redistribution are added, never netted. Subtracting cannibalisation from a lift double-counts.
- Deflate to real terms before any price logic. Expect near-flat drift on this dataset; the module exists because the pitch targets high-inflation retail.
- Measurement windows extend past campaign end by the commodity's repurchase cycle. Otherwise you bank the peak and never see the trough.
- No feature may be derived from the promotion or from post-treatment data.
- ROI is a ratio of two estimates: report an interval, never a point, never an average of ratios.
- Every estimate ships with the placebo band it was compared against.

## Layout

```
promo/        library code, one module per pipeline stage
app/          Streamlit, reads data/out/ only
tests/        pytest, includes the synthetic-truth harness
docs/         plan.md, runbook.md, data_findings.md
data/raw/     the eight CSVs, immutable
data/interim/ per-stage parquet + diagnostics
data/out/     what the app reads
scratch/      exploration, gitignored
```

## Conventions

- Python 3.11+, pandas, venv at `.venv`. Tests: `pytest -q`.
- Every stage function returns `(DataFrame, dict)` — data plus diagnostics. Diagnostics are returned, never printed.
- Gate failures return a `GateResult` from `promo/gates.py`, never raise.
- Randomness takes an explicit `seed` and uses `np.random.default_rng`. No global seeding.
- Plotting lives in `app/`, never in `promo/`.
- No notebooks in the repo.

## The three components, and what each may do

- **Agent** (`promo/agent.py`): orchestrates deterministic stages, decides which checks to run, stops on refusal. Contains no LLM calls.
- **Model** (`promo/baseline.py`): estimates the counterfactual. Never sees a promoted row in training.
- **LLM** (`promo/narrate.py`): receives a JSON object of already-computed values and returns sentences. It never computes, estimates, or infers a number. If a value is null it says so and why.

## Working style

- Read the relevant section of `docs/plan.md` before implementing a stage; it specifies the intended maths and the acceptance condition.
- When maths and convenience conflict, take the maths and say so.
- Prefer adding a diagnostic over adding a fallback. Silent repair is the failure mode this project is about.
- Every exclusion records its effect on row count, unit total, and sales value, before and after.
