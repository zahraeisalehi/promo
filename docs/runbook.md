# Promotional Intelligence — technical build runbook

A step-by-step stack build-up that produces the narrative in order: honest data → identified treatment → believable counterfactual → attributed units → money → decision, with a refusal engine wired across every gate.

The build order matters. Each step consumes the artifact the previous one emits. Do not start the model before Gate 2 passes — that is the whole argument of the project.

---

## Step 0 — Environment and repo

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy polars duckdb pyarrow \
            lightgbm scikit-learn statsmodels scipy cvxpy \
            streamlit plotly pydantic pytest anthropic python-dotenv
```

Layout:

```
promo/
  data/raw/            uploads, never modified
  data/interim/        parquet after each gate
  data/out/            results tables the UI reads
  promo/
    io.py              load + schema contract
    prices.py          Gate 1: deflation, TV filter, depth, mechanics
    quality.py         Gate 1: units, zeros, censoring
    estimand.py        the declaration object
    audit.py           Gate 2: variation, overlap, kappa*
    baseline.py        counterfactual engine + rollout
    validate.py        synthetic truth + placebo
    transfer.py        redistribution matrix
    accounting.py      subsidy, break-even, ROI intervals
    decide.py          shrinkage, ranking, MDE
    gates.py           the refusal engine
    narrate.py         LLM layer
  app/app.py           Streamlit
  tests/
```

Use DuckDB over the parquet files rather than holding panels in memory. Every gate writes a parquet and a JSON diagnostic; the app only ever reads `data/out/`. That separation is what lets you demo even if a later stage breaks.

---

## Step 1 — Ingest with a schema contract

Do not accept a dataframe. Accept a contract, and fail loudly.

```python
# promo/io.py
from pydantic import BaseModel
import pandas as pd

REQUIRED = ["txn_id","household_id","store_id","sku","date",
            "quantity","amount_paid","discount_amount"]
OPTIONAL = ["loyalty_discount","coupon_discount","manufacturer_funding",
            "category","brand","unit_of_measure","cogs","stock_on_hand"]

class IngestReport(BaseModel):
    n_rows: int
    missing_required: list[str]
    present_optional: list[str]
    date_min: str
    date_max: str
    n_sku: int
    n_household: int

def load(path: str) -> tuple[pd.DataFrame, IngestReport]:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    rep = IngestReport(
        n_rows=len(df), missing_required=missing,
        present_optional=[c for c in OPTIONAL if c in df.columns],
        date_min=str(df["date"].min()), date_max=str(df["date"].max()),
        n_sku=df["sku"].nunique(), n_household=df["household_id"].nunique())
    return df, rep
```

`cogs` being absent is not an error — it is a downstream refusal. Record it here so Gate 3 can refuse with a reason rather than crashing.

---

## Step 2 — Deflate to real terms

Nominal prices under Iranian inflation break the shelf-price estimator, so this runs **before** any price logic.

Prefer an internal index built from your own non-promoted rows over official CPI, because it matches your basket and has no publication lag.

```python
# promo/prices.py
def build_internal_index(df, base_period):
    unp = df[df["discount_amount"] == 0].copy()
    unp["unit_price"] = unp["amount_paid"] / unp["quantity"]
    # geometric mean of per-SKU price relatives, chained monthly
    piv = (unp.groupby(["sku","period"])["unit_price"].median()
              .unstack("period").sort_index(axis=1))
    rel = (piv / piv.shift(1, axis=1)).apply(lambda c: np.exp(np.log(c.dropna()).mean()))
    idx = rel.fillna(1.0).cumprod()
    return idx / idx.loc[base_period]

def deflate(df, idx, base_period):
    df = df.merge(idx.rename("cpi"), left_on="period", right_index=True, how="left")
    for c in ["amount_paid","discount_amount"]:
        df[f"real_{c}"] = df[c] / df["cpi"]
    return df
```

Two things to hold onto:

- Chained geometric means of price relatives survive SKUs entering and leaving the assortment; a simple mean of levels does not.
- Compute the index from unpromoted rows only, or your deflator absorbs your treatment.

Carry both nominal and real columns forward. Depth and κ* are ratios and use real; reported revenue uses nominal same-period.

Also derive an inflation-expectation control for the baseline model later:

```python
df["cpi_mom"] = df["cpi"].pct_change()
df["cpi_mom_3m"] = df["cpi_mom"].rolling(3).mean()
```

Without it, pre-emptive stockpiling ahead of a price rise gets attributed to your campaign as pull-forward.

---

## Step 3 — Recover the shelf price

Aggregate to SKU × store × week on real unit prices, then run the total-variation fit. The constraint `u ≥ p` encodes that discounts only cut.

```python
import cvxpy as cp

def tv_shelf_price(p, lam=0.5):
    n = len(p)
    u = cp.Variable(n)
    D = np.diff(np.eye(n), axis=0)
    obj = cp.Minimize(0.5*cp.sum_squares(p - u) + lam*cp.norm1(D @ u))
    cp.Problem(obj, [u >= p]).solve(solver=cp.ECOS)
    return np.asarray(u.value)
```

Run per SKU × store series. For a few thousand series cvxpy is fine; if it stalls, swap in a Condat TV1D solve followed by an isotonic-style projection onto `u ≥ p`, iterated a handful of times.

Tune `λ` by requiring the recovered series to reproduce known regular prices on a hand-checked sample. A quick sanity check: the fitted series should have far fewer distinct levels than the raw one. If it has almost as many, `λ` is too low and you are fitting wobble.

If the retailer recorded discount separately, you get the shelf price by arithmetic and use TV only as a cross-check:

```python
df["p_reg_arith"] = (df["amount_paid"] + df["discount_amount"]) / df["quantity"]
```

Agreement between the arithmetic and TV estimates is a strong slide.

---

## Step 4 — Depth, and the identification flag

```python
def depth_and_flag(series_real_price, p_hat0, promo_share_threshold=0.9):
    delta = 1 - series_real_price / p_hat0
    on_deal = (delta > 0.01)
    if on_deal.mean() > promo_share_threshold:
        return delta, "bounded"     # never observed at shelf price
    return delta, "identified"
```

Every SKU carries `identified` or `bounded` from here on. Anything downstream that averages across SKUs must filter on this flag or state that it is mixing the two. This is the deck's warning about knowing which products are identified before you average them together, made mechanical.

---

## Step 5 — Split the mechanics

```python
df["r"] = df.get("loyalty_discount", 0)
df["c"] = df.get("coupon_discount", 0)
df["m_fund"] = df.get("manufacturer_funding", 0)
df["tpr"] = df["discount_amount"] - df[["r","c","m_fund"]].sum(axis=1)
```

Then force a choice. The pipeline takes `treatment_mechanic` as a required argument and builds `D` from that component alone. Loyalty in particular usually fails Gate 2 because it is always on for members — a price level, not a promotion. Let the audit say so rather than silently modelling it.

---

## Step 6 — Quantity units, zeros, censoring

Three checks, each emitting a diagnostic rather than silently mutating data.

```python
def unit_audit(df):
    df["unit_value"] = df["amount_paid"] / df["quantity"]
    tail = df.nlargest(int(0.02*len(df)), "quantity")
    return {"top2pct_share_of_units": tail["quantity"].sum()/df["quantity"].sum(),
            "min_unit_value": float(df["unit_value"].min()),
            "suspect_uom": df.loc[df["unit_value"] < 0.01, "sku"].unique().tolist()}
```

If a small share of rows carries most of the units at fractions of a cent, you have volume-measured goods sharing a column with counted goods. Drop or winsorise — and record the drop and its effect in the report. A silent filter is indistinguishable from a bug.

For zeros, build the household-week frame explicitly and carry an availability flag:

```python
frame = household_week_grid(df)                    # every household × week
frame["shopped"] = frame["n_trips"] > 0
frame["structural_zero"] = ~frame["shopped"]       # exclude or model separately
frame["sampling_zero"] = frame["shopped"] & (frame["units"] == 0)
```

Set the counterfactual to zero where nothing could have been bought. Report the share of household-weeks with no trip at all — on the deck's panel that was around 40%, and treating those as observed zeros teaches the model that promotions coincide with people not being in the shop.

For censoring, if `stock_on_hand` exists, flag stockout weeks and exclude them from training. If it does not, you cannot fix it — but state the direction: stockouts are likelier during promotions, so you under-record demand where the effect is strongest and the promotion looks weaker than it was. A signed bias statement beats silence.

---

## Step 7 — Declare the estimand

One object, written before anything is fitted, and printed on every output.

```python
class Estimand(BaseModel):
    unit: str            # household | sku | store | store_week
    treatment: str       # tpr | coupon | loyalty | manufacturer
    comparison: str      # treated vs untreated at same time, etc.
    period: tuple[str,str]
    horizon_weeks: int   # must cover the repurchase cycle
    outcome: str         # units | real_revenue | category_units
```

`horizon_weeks` is where pull-forward lives. Set it to at least the category's repurchase cycle or you bank the peak and never see the trough.

---

## Step 8 — Find the axis of variation

```python
def variation_axes(panel, D="treated"):
    out = {}
    for axis in ["household_id","sku","week"]:
        share = panel.groupby(axis)[D].mean()
        out[axis] = {"share_all_on": float((share > 0.99).mean()),
                     "share_all_off": float((share < 0.01).mean()),
                     "usable": float(((share > 0.01) & (share < 0.99)).mean())}
    return out
```

The axis with meaningful `usable` mass is your unit of analysis. If every axis is all-on or all-off, the run stops here with a refusal: the treatment is the calendar.

---

## Step 9 — Overlap

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

def overlap(X, D):
    e = cross_val_predict(HistGradientBoostingClassifier(), X, D,
                          cv=5, method="predict_proba")[:,1]
    return {"auc": roc_auc_score(D, e),
            "share_extreme": float(((e < 0.02) | (e > 0.98)).mean()),
            "e": e}
```

AUC near 1.0 means no overlap — the estimator would be extrapolating your model rather than measuring an effect. Before refusing, check the feature list: hand the classifier something that encodes the treatment and it scores near-perfectly for a trivial reason. The refusal message should name both possibilities.

Keep `e` — you will want it for trimming or weighting if you go beyond a plain baseline model.

---

## Step 10 — κ\* before any model

```python
def kappa_star(discount_per_unit, margin_per_unit):
    if margin_per_unit is None: return None
    return discount_per_unit / margin_per_unit
```

`κ* = d/m` is the share of promoted sales that must be genuinely new. If `κ* ≥ 1`, no level of success makes the promotion profitable — arithmetically unprofitable before measurement. Run this across the client's forward calendar. It is the cheapest analysis in the pipeline and needs no counterfactual at all.

---

## Step 11 — The refusal engine

Every gate returns the same object, and the app renders it identically wherever it appears.

```python
class GateResult(BaseModel):
    gate: str
    status: str           # pass | bounded | refuse
    reason_code: str | None
    detail: dict
    message: str          # plain language, written by narrate.py
```

Reason codes worth implementing, each mapped to a sentence:

| code | trigger |
|---|---|
| `NO_VARIATION` | no axis has usable mixed mass |
| `NO_OVERLAP` | AUC > 0.95 and extreme-propensity share high |
| `LEAKED_FEATURE` | AUC > 0.95 but a feature correlates > 0.99 with D |
| `DEPTH_BOUNDED` | SKU on deal above threshold share |
| `KAPPA_IMPOSSIBLE` | κ* ≥ 1 |
| `NO_MARGIN` | cogs absent — units and κ* only |
| `PLACEBO_OVERLAP` | estimate inside the placebo band |
| `OVERLAPPING_TREATMENTS` | two campaigns in one cell |
| `ROI_UNBOUNDED` | denominator interval spans zero |
| `HORIZON_TOO_SHORT` | horizon < repurchase cycle |

The pipeline runner short-circuits on any `refuse` and returns the partial results computed so far. Partial output plus a stated reason is the product; a complete-looking output with a hidden failure is the thing you are arguing against.

---

## Step 12 — The counterfactual engine

Train on unpromoted rows only.

```python
import lightgbm as lgb

FEATURES = ["lag1","lag2","lag4","lag52","roll4","roll8","roll13",
            "week_of_year","jalali_month","is_nowruz","is_ramadan","is_yalda",
            "real_price","price_vs_ref","cat_units_ex_self",
            "n_active_stores","cpi_mom_3m","sku_id","category_id"]

def fit_baseline(panel):
    tr = panel[panel["treated"] == 0]
    return lgb.train(
        {"objective":"regression","learning_rate":0.05,"num_leaves":63,
         "min_data_in_leaf":40,"verbose":-1},
        lgb.Dataset(tr[FEATURES], np.log1p(tr["units"]),
                    categorical_feature=["sku_id","category_id","jalali_month"]),
        num_boost_round=800)
```

Use the Persian calendar fields, not Gregorian week numbers, or Nowruz lands in a different week each year and the model never learns it.

For intervals, fit the same design three times with `objective="quantile"` at `alpha` 0.1, 0.5, 0.9.

---

## Step 13 — Recursive rollout

The single most important twenty lines in the project. Without it, multi-week campaigns measure smaller every week they run.

```python
def rollout(model, hist, exog_weeks):
    """hist: dict of lag values from real, pre-promotion history."""
    preds, h = [], dict(hist)
    for z in exog_weeks:                      # one row of exogenous features per week
        x = build_features(h, z)
        yhat = np.expm1(model.predict(x[FEATURES])[0])
        preds.append(yhat)
        h = shift_lags(h, yhat)               # feed the prediction back, not the actual
    return np.array(preds)
```

`shift_lags` must push `yhat` into `lag1` and cascade — never the observed promoted value. Rolling means recompute from the counterfactual series too.

This is not circular: every parameter was fitted on unpromoted weeks, and the rollout only applies what was learned. The honest cost is compounding error down the horizon, which is why the next step validates the whole rollout rather than one-week error.

---

## Step 14 — Synthetic truth

Generate data with a **different** process than LightGBM assumes, or you have only shown the model can recover its own assumptions.

```python
def simulate(n_sku=200, n_weeks=104, tau=0.15, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    out = []
    for j in range(n_sku):
        level = rng.lognormal(4, 0.5)
        season = 1 + 0.3*np.sin(2*np.pi*t/52 + rng.uniform(0, 2*np.pi))
        trend = np.exp(rng.normal(0, 0.002)*t)
        ar = np.zeros(n_weeks); eps = rng.normal(0, 0.08, n_weeks)
        for k in range(1, n_weeks): ar[k] = 0.6*ar[k-1] + eps[k]   # AR noise, not tree-shaped
        base = level*season*trend*np.exp(ar)
        D = np.zeros(n_weeks, bool); start = rng.integers(30, 90)
        D[start:start+3] = True
        y = base*(1 + tau*D)
        out.append(pd.DataFrame({"sku":j,"week":t,"units":y,"treated":D,"true_base":base}))
    return pd.concat(out)
```

Run the full pipeline on this and check that recovered τ̂ brackets the τ you chose. Report bias and coverage. Extend it to cases you care about: add a pull-forward term that suppresses the three weeks after treatment, and confirm your horizon logic recovers the net rather than the peak.

This harness doubles as your test suite — wire it into pytest and you get regression safety for free.

---

## Step 15 — The placebo distribution

```python
def placebo(panel, model, n=300, seed=0):
    rng = np.random.default_rng(seed)
    clean = panel[panel["ever_treated"] == 0]
    ests = []
    for _ in range(n):
        sku = rng.choice(clean["sku"].unique())
        wk  = rng.choice(clean.query("sku == @sku")["week"].unique()[10:-6])
        ests.append(estimate_effect(clean, model, sku, wk, weeks=3))
    return np.array(ests)
```

The band this produces is where your zero actually sits, and it is almost never at zero. Draw the real estimate on the same axis in the UI. If it lands inside, the gate returns `PLACEBO_OVERLAP` — a diagnosis that this comparison carries no signal, not a verdict that the effect is absent. Word the message that way; the distinction is the deck's and judges will listen for it.

---

## Step 16 — The transfer matrix

Only from baskets. From store totals you can learn what a product lost, not where the units went.

```python
def transfer_matrix(baskets, promoted_sku, category_skus):
    """T[i,j] = units moved from j to i because i was promoted."""
    # switching evidence: households that bought j in their pre-window
    # and bought i during the promo window, weighted by choice-prob change
    T = np.zeros((len(category_skus), len(category_skus)))
    ...
    np.fill_diagonal(T, 0)
    return T

g = T.sum(axis=1)      # gains, row sums
l = T.sum(axis=0)      # losses, column sums
assert abs(g.sum() - l.sum()) < 1e-6      # conserves mass
```

Assert mass conservation in code — it is a real invariant, since T is built from choice probabilities and probabilities sum to one. A failing assert means your switching estimator is leaking.

Then keep expansion and redistribution separate:

```python
delta_q = s + (g - l)      # s from Step 13, NOT from T
```

Never subtract cannibalisation from a lift that already nets it out. And expose in the UI that row and column sums are identified while the cell-level split is a stated convention — a different rule moves mass between cells and changes no total.

---

## Step 17 — Accounting

```python
def promo_pnl(Q, kappa, m, d):
    return kappa*Q*m - (1-kappa)*Q*d          # margin on new units − subsidy

def roi_interval(inc_margin_draws, cost_draws, alpha=0.05):
    r = inc_margin_draws / cost_draws
    lo, hi = np.percentile(r, [100*alpha/2, 100*(1-alpha/2)])
    unbounded = (cost_draws.min() <= 0 <= cost_draws.max())
    return {"lo": lo, "hi": hi, "unbounded": bool(unbounded)}
```

Bootstrap both numerator and denominator jointly from the baseline's quantile draws. If the denominator interval spans zero, return `ROI_UNBOUNDED` and say so — that is more useful than a tidy number that is wrong. Never average ROIs across campaigns; aggregate the components and divide once.

If `cogs` is missing, return incremental units and κ* and refuse the profit figures. Reporting the margin your promotion would need to beat is the honest output; inventing one is the dangerous one.

---

## Step 18 — Ranking without the winner's curse

```python
def james_stein_shrink(est, se):
    grand = np.average(est, weights=1/se**2)
    tau2 = max(np.var(est) - np.mean(se**2), 1e-9)
    w = tau2 / (tau2 + se**2)
    return grand + w*(est - grand)
```

Rank on shrunk estimates, then publish the shrunk value as the expectation — not the raw estimate that made you pick it. Skip that and your recommendations underperform their forecasts on average, every time, by construction.

---

## Step 19 — The holdout designer

```python
def mde(sigma, K, n, rho, varrho, t_a=1.96, t_b=0.84):
    deff = 1 + (n-1)*rho
    return (t_a + t_b)*sigma*np.sqrt(deff / (K*n*varrho*(1-varrho)))
```

Expose it as a slider over `varrho`. The curve is minimised at a balanced split, and a 5% holdout costs roughly 2.3× the detectable effect of a 50/50 split while 20% costs about 1.25×. Show the user that the small holdout that feels cheap is priced in lost sales, while its real price is in the effects they can no longer see.

This is your closing recommendation and arguably the product's most valuable output.

---

## Step 20 — The app

Streamlit, four pages, reading only `data/out/`.

```python
# app/app.py
import streamlit as st, pandas as pd
st.set_page_config(page_title="Promotional Intelligence", layout="wide")
page = st.sidebar.radio("", ["Audit","Portfolio","Campaign","Planner"])
res = pd.read_parquet("data/out/campaigns.parquet")
gates = pd.read_json("data/out/gates.json")
```

- **Audit** — the six veto questions with pass/bounded/refuse chips, the data-honesty report, and the κ* scan of the forward calendar. Land here, not on a leaderboard: it establishes the argument before showing any number.
- **Portfolio** — measurable campaigns ranked on shrunk estimates; unmeasurable ones listed separately with reason codes rather than hidden.
- **Campaign** — actual vs counterfactual with the rollout band, the placebo distribution with the estimate marked on it, the expansion/redistribution split, the subsidy bar, and the ROI interval.
- **Planner** — κ* calculator, discount-depth response curve, and the MDE holdout designer.

Render `GateResult` with one shared component so a refusal looks like a deliberate product state everywhere it appears, never like an error.

---

## Step 21 — The narration layer

The model receives computed numbers and returns sentences. It never computes.

```python
# promo/narrate.py
import anthropic, json
client = anthropic.Anthropic()

SYSTEM = """You explain promotion analytics results to a category manager.
You receive a JSON object of already-computed diagnostics.
Never compute, estimate, or infer any number that is not in the JSON.
If a value is null, say the system cannot report it and why.
Two or three sentences. Plain language. No hedging beyond what the JSON states."""

def explain(payload: dict) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400, system=SYSTEM,
        messages=[{"role":"user","content":json.dumps(payload)}])
    return "".join(b.text for b in r.content if b.type == "text")
```

Use it for three things only: the campaign verdict paragraph, the refusal message for each reason code, and the chat panel scoped to whatever campaign is on screen. Pre-generate the refusal sentences per reason code and cache them, so a demo without network still shows the full behaviour.

---

## Step 22 — Rehearse the demo against the gates

1. Load the data. Show the ingest report and the unit audit finding the volume-measured rows.
2. Show the deflation chart — nominal vs real — and state that the shelf-price estimator is not identified without it.
3. Open the audit page. Land on a campaign the system refuses, and read the reason.
4. Move to one it accepts. Show the counterfactual with the rollout band, then the placebo distribution with the estimate sitting outside it.
5. Show expansion and redistribution side by side, and say why they are added rather than netted.
6. Show the subsidy bar and the ROI interval. If margin is missing, show κ* and the refusal.
7. Close on the holdout designer: hold out 20% next time and most of this machinery becomes unnecessary.

---

## Time budget

| Hours | Work |
|---|---|
| 0–4 | Steps 0–2. Ingest, contract, deflation. |
| 4–9 | Steps 3–6. Shelf price, depth flags, mechanics, quality diagnostics. |
| 9–13 | Steps 7–11. Estimand, variation, overlap, κ*, refusal engine. **Demoable here.** |
| 13–19 | Steps 12–15. Baseline, rollout, synthetic truth, placebo. |
| 19–23 | Steps 16–19. Transfer matrix (if baskets), accounting, shrinkage, MDE. |
| 23–29 | Steps 20–21. App and narration. |
| 29–32 | Step 22. Rehearsal, caching, fallbacks. |

If you fall behind, cut the transfer matrix first and report gains and losses only, stating that the cell split is unidentified from your data. Cut narration second. Never cut the placebo or the refusal engine — they are the argument.
