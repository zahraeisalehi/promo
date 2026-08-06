"""Task 4.4 — a panel whose truth is known, and what the pipeline recovers from it.

Backtest error says nothing about `ŷ(0)` where `D = 1`. The only way to learn
whether the estimator recovers an effect is to build a world where the effect is
known and ask it.

**The process is deliberately not tree-shaped.** Units are a product of a
lognormal level, a multiplicative seasonal cycle, a slow trend, and AR(1) noise
in logs. A gradient-boosted tree approximates none of those exactly — it fits
axis-aligned steps to a smooth multiplicative surface. Validating a tree on
tree-generated data is the inverse crime: it proves the estimator can recover its
own assumptions and nothing else. Do not "simplify" the generator.

What it produces, and why each piece is there:

- **A single commodity with several products across several stores**, so
  `category_units_ex_focal` means something and the focal product's own units
  are excluded from it by construction.
- **Exogenous no-sale weeks.** A cell that does not sell has no price, so
  `price_rel_category` is null exactly where `units == 0` — the shape of settled
  decision 9's leak, reproduced so the replacement feature is what gets tested.
  Being exogenous, a no-sale week is a zero in both worlds and contributes
  nothing to the truth.
- **A mailer that actually does something**, on 13% of control weeks and 40% of
  treated ones, matching what Task 3.3 measured. Settled decision 8 says the
  baseline must condition on it; a generator where it had no effect could not
  tell whether that mattered.
- **Store traffic driven by store units**, so it is a genuine *mediator*: the
  promotion moves it. That is what makes the contemporaneous-block axis a real
  question rather than a formality.
- **Treatment assigned on the product's level**, so promoted and unpromoted
  cells differ in an observable way and the baseline has something to condition
  on. Assignment independent of demand would make overlap trivial and the test
  easy for the wrong reason.

The truth is exact. `base_units` is the counterfactual: what the cell would have
sold with the mailer that ran and without the display. True incremental units
are `observed - base` summed over whatever window is being asked about, so
`gross`, `post` and `net` all have known values and the estimate can be scored
against each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from promo.baseline import (
    BASELINE_FEATURES,
    DEFAULT_TARGET,
    MEDIATOR_FEATURES,
    add_price_history,
    fit_baseline,
    rollout,
)
from promo.lift import LiftCampaign, estimate_lift

__all__ = [
    "CAMPAIGN_WEEKS",
    "CONTEMPORANEOUS_BLOCK",
    "EXPOSURE_PATTERNS",
    "HORIZON_WEEKS",
    "TRUE_EFFECTS",
    "counterfactual_ratios",
    "cycles_table",
    "exposure_grid",
    "feature_set",
    "recover",
    "recovery_grid",
    "simulate",
    "sparsity_grid",
    "target_grid",
]

#: The three features measured in week w that the promotion can itself move.
#: Task 3.2 found them carrying 77.5% of the overlap classifier's gain; Task
#: 2.6 flagged them as possible mediators. The two-by-two turned that worry into
#: a measurement and settled decision 13 dropped them, so the canonical list
#: lives in `promo.baseline` and this is an alias — a second copy would drift
#: from the thing it is meant to be testing.
CONTEMPORANEOUS_BLOCK: tuple[str, ...] = MEDIATOR_FEATURES

#: The true effects the plan names.
TRUE_EFFECTS: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30)

#: Late enough that `units_lag_52` exists for the window and its history.
CAMPAIGN_WEEKS: tuple[int, int] = (71, 74)
HORIZON_WEEKS: int = 3

#: How many of the four campaign weeks each participating cell runs. `full` is
#: the clean case; `uneven` reproduces the real panel's spread, where 43.9% of
#: treated cells ran one week of four and 18.9% ran throughout — the exposure
#: settled decision 11 averages over.
EXPOSURE_PATTERNS: dict[str, tuple[float, ...]] = {
    "full": (0.0, 0.0, 0.0, 1.0),
    "uneven": (0.439, 0.213, 0.159, 0.189),
}

COMMODITY = "SYNTH"


def _draw_exposure(rng: np.random.Generator, pattern: str, campaign_length: int) -> int:
    """How many of the campaign's weeks one participating cell runs.

    The stored weights describe a four-week campaign, which is the shape the
    real panel was measured on. A shorter campaign cannot give a cell four
    weeks, so the weights are truncated and renormalised rather than sampled
    from blindly — drawing four weeks out of a two-week campaign raised, which
    is how the clamp came to be written.
    """
    if pattern == "full":
        return campaign_length
    weights = np.array(EXPOSURE_PATTERNS[pattern], dtype=float)[:campaign_length]
    if weights.sum() <= 0:
        return campaign_length
    return int(rng.choice(np.arange(1, len(weights) + 1), p=weights / weights.sum()))


def simulate(
    *,
    tau: float = 0.15,
    n_products: int = 8,
    n_stores: int = 12,
    n_weeks: int = 90,
    campaign: tuple[int, int] = CAMPAIGN_WEEKS,
    payback: float = 0.0,
    payback_weeks: int = HORIZON_WEEKS,
    exposure: str = "full",
    treated_share: float = 0.55,
    mailer_effect: float = 0.10,
    no_sale_rate: float = 0.10,
    ar_phi: float = 0.6,
    ar_sigma: float = 0.18,
    seasonal_amplitude: float = 0.30,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A panel with a known effect, and the truth that goes with it.

    Args:
        tau: the multiplicative display effect on treated cell-weeks.
        payback: multiplicative suppression in the weeks after a cell's last
            treated week — a pull-forward. 0.0 leaves the tail clean, which
            makes `net` equal `gross` in truth; a positive value is what the
            horizon rule exists for.
        exposure: a key of `EXPOSURE_PATTERNS`. `uneven` gives participating
            cells one to four campaign weeks in the proportions the real panel
            shows.
        treated_share: share of product-stores that take part at all.
        mailer_effect: the mailer's own multiplicative effect. Non-zero on
            purpose — settled decision 8 is only testable if the mechanic the
            baseline conditions on does something.
        no_sale_rate: exogenous probability a cell-week does not sell. Drives
            the null pattern in `price_rel_category`.
        seed: explicit, per the project's randomness rule.

    Returns:
        `(panel, truth)`. `panel` carries every column the Phase 4 pipeline
        reads, plus `base_units` — the exact counterfactual — and `true_*`
        columns. `truth` holds the window totals the estimate is scored on.
    """
    if exposure not in EXPOSURE_PATTERNS:
        raise ValueError(
            f"exposure must be one of {sorted(EXPOSURE_PATTERNS)}, got {exposure!r}"
        )
    rng = np.random.default_rng(seed)
    first, last = campaign
    weeks = np.arange(1, n_weeks + 1)
    campaign_weeks = np.arange(first, last + 1)

    frames = []
    for product in range(n_products):
        level = float(rng.lognormal(3.2, 0.45))
        phase = float(rng.uniform(0, 2 * np.pi))
        trend = float(rng.normal(0.0, 0.0015))
        # Multiplicative seasonality and a slow trend: smooth, and nothing a
        # tree can represent without stepping through it.
        season = 1.0 + seasonal_amplitude * np.sin(2 * np.pi * weeks / 52.0 + phase)
        drift = np.exp(trend * weeks)
        regular_price = float(rng.uniform(1.5, 6.0))

        for store in range(n_stores):
            store_factor = float(rng.lognormal(0.0, 0.25))
            # AR(1) in logs. The persistence is what makes a lag informative,
            # and what a contaminated lag destroys.
            noise = rng.normal(0.0, ar_sigma, n_weeks)
            ar = np.zeros(n_weeks)
            for k in range(1, n_weeks):
                ar[k] = ar_phi * ar[k - 1] + noise[k]

            sold = rng.random(n_weeks) > no_sale_rate
            base = level * store_factor * season * drift * np.exp(ar) * sold

            # Assignment leans on the product's level, so treated and untreated
            # cells differ in a way the lags can see. Random assignment would
            # make overlap perfect and the recovery test too easy.
            propensity = treated_share * (0.6 + 0.8 * (level / (level + 20.0)))
            takes_part = rng.random() < min(0.95, propensity)
            treated = np.zeros(n_weeks, dtype=bool)
            if takes_part:
                n_treated_weeks = _draw_exposure(rng, exposure, len(campaign_weeks))
                chosen = rng.choice(
                    campaign_weeks, size=n_treated_weeks, replace=False
                )
                treated[chosen - 1] = True

            mailer = rng.random(n_weeks) < np.where(treated, 0.40, 0.13)

            effect = np.ones(n_weeks)
            effect *= 1.0 + tau * treated
            effect *= 1.0 + mailer_effect * mailer
            if payback and treated.any():
                tail = np.arange(
                    int(weeks[treated][-1]) + 1,
                    int(weeks[treated][-1]) + 1 + payback_weeks,
                )
                tail = tail[tail <= n_weeks]
                effect[tail - 1] *= 1.0 - payback

            # The mailer is part of both worlds: the counterfactual asks what
            # the cell would have done with the mailer that ran and without the
            # display, so it multiplies the base and not only the observation.
            base_units = base * (1.0 + mailer_effect * mailer)
            observed = base * effect

            frames.append(
                pd.DataFrame(
                    {
                        "PRODUCT_ID": 1000 + product,
                        "STORE_ID": 300 + store,
                        "WEEK_NO": weeks,
                        "COMMODITY_DESC": COMMODITY,
                        "units": observed,
                        "base_units": base_units,
                        "treated": treated,
                        "in_mailer": mailer,
                        "sold": sold,
                        "regular_price": np.where(sold, regular_price, np.nan),
                    }
                )
            )

    panel = pd.concat(frames, ignore_index=True)
    panel["sales_value"] = panel["units"] * panel["regular_price"].fillna(0.0)
    panel["true_residual"] = panel["units"] - panel["base_units"]
    panel = _add_features(panel)

    truth = _truth(panel, campaign=campaign, horizon_weeks=payback_weeks)
    truth.update(
        {
            "tau": tau,
            "payback": payback,
            "exposure": exposure,
            "seed": seed,
            "products": n_products,
            "stores": n_stores,
            "weeks": n_weeks,
        }
    )
    return panel, truth


def _add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Phase 2.6's feature definitions, applied to the generated panel.

    The definitions are copied deliberately: the point of the harness is to test
    the *estimator* against a process it does not assume, not to re-test the
    feature builder. Anything that diverges from `promo/features.py` here would
    make the recovery numbers describe a pipeline nobody runs.
    """
    out = panel.sort_values(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).copy()
    group = out.groupby(["PRODUCT_ID", "STORE_ID"], observed=True)["units"]
    for lag in (1, 2, 4, 52):
        out[f"units_lag_{lag}"] = group.shift(lag)
    for window in (4, 8, 13):
        out[f"units_roll_mean_{window}"] = group.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )

    store_week = out.groupby(["STORE_ID", "WEEK_NO"], observed=True)["units"]
    category_total = store_week.transform("sum")
    out["category_units_ex_focal"] = category_total - out["units"]
    # Traffic is trips, and a single product is a small part of a store's week.
    # It has to *move* with the promotion — that is what makes the
    # contemporaneous block a mediator and the axis a real question — but the
    # focal product's share of it must be realistic. An earlier version made
    # traffic a near-multiple of store units, which handed the model a
    # non-recursive readout of the current week's demand: with it the rollout
    # could not drift, without it tau_hat came back at 1.07 against a truth of
    # 0.30. That is the generator answering the question, not the estimator.
    rng = np.random.default_rng(20240404)
    store_base = out["STORE_ID"].map(
        {s: float(v) for s, v in zip(
            sorted(out["STORE_ID"].unique()),
            rng.uniform(300.0, 700.0, out["STORE_ID"].nunique()),
            strict=True,
        )}
    )
    out["store_traffic"] = (
        store_base * np.exp(rng.normal(0.0, 0.10, len(out))) + 0.5 * category_total
    )
    out["n_stores_carrying"] = out.groupby(
        ["PRODUCT_ID", "WEEK_NO"], observed=True
    )["sold"].transform("sum")

    category_price = out.groupby(["COMMODITY_DESC", "WEEK_NO"], observed=True)[
        "regular_price"
    ].transform("median")
    out["price_rel_category"] = out["regular_price"] / category_price
    out["week_of_year"] = ((out["WEEK_NO"] - 1) % 52 + 1).astype("int16")
    out["is_holiday_week"] = False
    out["price_index"] = 1.0 + 0.0004 * out["WEEK_NO"]

    return add_price_history(out.reset_index(drop=True))


def _truth(
    panel: pd.DataFrame, *, campaign: tuple[int, int], horizon_weeks: int
) -> dict[str, Any]:
    """The known answer, over the cells the membership rule selects.

    Scored over the same cells `estimate_lift` measures — every product-store
    treated at least once in the campaign weeks — because a bias against a
    differently-scoped truth would be an artefact of the comparison rather than
    of the estimator. Settled decision 11 is the rule being scored, not
    sidestepped.
    """
    first, last = campaign
    promoted = panel["WEEK_NO"].between(first, last)
    took_part = (
        panel.loc[promoted & panel["treated"], ["PRODUCT_ID", "STORE_ID"]]
        .drop_duplicates()
    )
    cells = panel.merge(took_part, on=["PRODUCT_ID", "STORE_ID"], how="inner")

    in_campaign = cells["WEEK_NO"].between(first, last)
    in_post = cells["WEEK_NO"].between(last + 1, last + horizon_weeks)
    gross = float(cells.loc[in_campaign, "true_residual"].sum())
    post = float(cells.loc[in_post, "true_residual"].sum())

    exposure = (
        cells.loc[in_campaign]
        .groupby(["PRODUCT_ID", "STORE_ID"], observed=True)["treated"]
        .sum()
    )
    return {
        "pairs": len(took_part),
        "gross_incremental": gross,
        "post_window_residual": post,
        "net_incremental": gross + post,
        "counterfactual_units_campaign": float(
            cells.loc[in_campaign, "base_units"].sum()
        ),
        "treated_pair_weeks": int(cells.loc[in_campaign, "treated"].sum()),
        "exposure_histogram": {
            int(k): int(v) for k, v in exposure.value_counts().sort_index().items()
        },
    }


def cycles_table(horizon_weeks: int = HORIZON_WEEKS) -> pd.DataFrame:
    """A repurchase-cycle table for the synthetic commodity."""
    return pd.DataFrame(
        [
            {
                "COMMODITY_DESC": COMMODITY,
                "horizon_weeks": horizon_weeks,
                "low_support": False,
            }
        ]
    )


def feature_set(contemporaneous: bool = False) -> tuple[str, ...]:
    """The baseline's features, with or without the mediator block.

    `BASELINE_FEATURES` no longer contains the block — settled decision 13 —
    so `contemporaneous=True` adds it back rather than the reverse. The default
    follows the shipped model.
    """
    if contemporaneous:
        return (*BASELINE_FEATURES, *CONTEMPORANEOUS_BLOCK)
    return BASELINE_FEATURES


def recover(
    *,
    tau: float = 0.15,
    identity: bool = False,
    contemporaneous: bool = False,
    target: str = DEFAULT_TARGET,
    seed: int = 0,
    campaign: tuple[int, int] = CAMPAIGN_WEEKS,
    horizon_weeks: int = HORIZON_WEEKS,
    n_estimators: int = 200,
    check_drift: bool = True,
    **simulate_kwargs: Any,
) -> dict[str, Any]:
    """One replication: generate a world, run the pipeline, score the answer.

    Returns a flat row — the configuration, the truth, the estimate, the signed
    error, and whether the interval covered the truth — so a grid of these is a
    DataFrame without further shaping.
    """
    panel, truth = simulate(
        tau=tau, seed=seed, campaign=campaign, payback_weeks=horizon_weeks,
        **simulate_kwargs,
    )
    controls = panel.loc[~panel["treated"]].reset_index(drop=True)
    model, _ = fit_baseline(
        controls,
        features=feature_set(contemporaneous),
        include_identity=identity,
        target=target,
        week_range=None,
        n_estimators=n_estimators,
        num_leaves=31,
        min_data_in_leaf=20,
        backtest_weeks=0,
        seed=seed,
    )
    spec = LiftCampaign(
        name=f"synthetic-tau-{tau}", commodity=COMMODITY, weeks=campaign
    )
    _, diag = estimate_lift(
        spec,
        model,
        panel,
        cycles_table(horizon_weeks),
        check_drift=check_drift,
    )
    lift = diag["lift"]

    row: dict[str, Any] = {
        "tau": tau,
        "identity": identity,
        "contemporaneous": contemporaneous,
        "target": target,
        "exposure": truth["exposure"],
        "payback": truth["payback"],
        "seed": seed,
        "pairs": truth["pairs"],
        "treated_pair_weeks": truth["treated_pair_weeks"],
        "counterfactual_units_true": truth["counterfactual_units_campaign"],
        "counterfactual_units_hat": lift["counterfactual_units"]["campaign"],
    }
    for field in ("gross_incremental", "post_window_residual", "net_incremental"):
        estimate = lift[field]
        true_value = truth[field]
        low, high = lift[f"{field}_interval"]
        row[f"{field}_true"] = true_value
        row[f"{field}_hat"] = estimate
        row[f"{field}_error"] = estimate - true_value
        row[f"{field}_covered"] = bool(low <= true_value <= high)
        row[f"{field}_width"] = high - low
    # Scaled by the counterfactual, so errors are comparable across effect
    # sizes and across panels of different size.
    row["gross_error_share"] = (
        row["gross_incremental_error"] / truth["counterfactual_units_campaign"]
        if truth["counterfactual_units_campaign"]
        else None
    )
    row["tau_hat"] = (
        lift["gross_incremental"] / lift["counterfactual_units"]["campaign"]
        if lift["counterfactual_units"]["campaign"]
        else None
    )
    drift = diag["drift_check"]
    row["drift_units"] = drift.get("residual_units_first_weeks")
    row["drift_exceeds_gross"] = drift.get("exceeds_gross")
    return row


def recovery_grid(
    taus: tuple[float, ...] = TRUE_EFFECTS,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    identity: tuple[bool, ...] = (False, True),
    contemporaneous: tuple[bool, ...] = (False, True),
    **recover_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every cell of the two-by-two, at every true effect, over several seeds.

    Coverage is a frequency, so it needs replications: one seed per cell would
    report either 0% or 100% and mean neither.
    """
    rows = [
        recover(tau=tau, identity=ident, contemporaneous=contemp, seed=seed,
                **recover_kwargs)
        for tau in taus
        for ident in identity
        for contemp in contemporaneous
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    return frame, summarise(frame, ["tau", "identity", "contemporaneous"])


def exposure_grid(
    *,
    tau: float = 0.15,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    **recover_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Settled decision 11's check: does uneven exposure bias recovery?

    Same truth, same estimator, two worlds — one where every participating cell
    runs all four campaign weeks, one where they run one to four in the
    proportions the real panel shows. If the membership rule biases the
    estimate, the two disagree, and the direction is the finding.
    """
    rows = [
        recover(tau=tau, seed=seed, exposure=pattern, **recover_kwargs)
        for pattern in EXPOSURE_PATTERNS
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    return frame, summarise(frame, ["exposure"])


def sparsity_grid(
    *,
    tau: float = 0.30,
    rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    seeds: tuple[int, ...] = (0, 1, 2),
    **recover_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recovery against the share of weeks that do not sell.

    Not in the plan's brief, and reported anyway, because it turned out to
    dominate every axis that was: under the `log1p` target recovery degraded
    sharply with sparsity while the two-by-two barely moved. `expm1` of a
    conditional mean in log space understates an outcome with mass at zero, the
    rollout fed that understatement back as next week's lag, and the
    counterfactual spiralled down — which shows up as *inflated* lift, since the
    residual is measured against it.

    **The real panel is 87% zero rows**, so this is the axis that says what the
    estimator does where it actually runs. It is now the regression check on the
    Poisson default rather than a diagnosis of a live defect: pass
    `target="log1p"` to see the original.
    """
    rows = [
        recover(tau=tau, seed=seed, no_sale_rate=rate, **recover_kwargs)
        for rate in rates
        for seed in seeds
    ]
    frame = pd.DataFrame(rows).assign(
        no_sale_rate=[r for r in rates for _ in seeds]
    )
    return frame, summarise(frame, ["no_sale_rate"])


def target_grid(
    *,
    tau: float = 0.0,
    targets: tuple[str, ...] = ("log1p", "log1p_smearing", "tweedie", "poisson"),
    rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    seeds: tuple[int, ...] = (0, 1, 2),
    **recover_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Which inversion recovers a null effect on a zero-inflated panel.

    The gate condition for Phase 4 is that tau = 0 comes back as approximately
    zero. `log1p` does not manage it once the outcome has mass at zero, and the
    two candidates fix the problem in different places: one corrects the
    retransformation after the fact, the other never transforms at all.
    Reported at every sparsity, because the defect is a function of sparsity
    and a comparison at one rate would be a comparison of nothing.
    """
    rows = [
        {
            **recover(tau=tau, seed=seed, target=target, no_sale_rate=rate,
                      **recover_kwargs),
            "no_sale_rate": rate,
        }
        for target in targets
        for rate in rates
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    return frame, summarise(frame, ["target", "no_sale_rate"])


def counterfactual_ratios(
    *,
    target: str = DEFAULT_TARGET,
    no_sale_rate: float = 0.10,
    seed: int = 0,
    campaign: tuple[int, int] = CAMPAIGN_WEEKS,
    horizon_weeks: int = HORIZON_WEEKS,
    n_estimators: int = 200,
    **simulate_kwargs: Any,
) -> pd.DataFrame:
    """One-step and rollout counterfactual against truth, week by week.

    Separates the two halves of the defect. The one-step column is the
    retransformation bias alone — the model on observed features, no feedback.
    The rollout column is that bias compounding through the recursion. A fix
    that only helps the second is not a fix.
    """
    panel, _ = simulate(
        tau=0.0, seed=seed, campaign=campaign, no_sale_rate=no_sale_rate,
        payback_weeks=horizon_weeks, **simulate_kwargs,
    )
    controls = panel.loc[~panel["treated"]].reset_index(drop=True)
    model, _ = fit_baseline(
        controls,
        features=feature_set(False),
        target=target,
        week_range=None,
        n_estimators=n_estimators,
        num_leaves=31,
        min_data_in_leaf=20,
        backtest_weeks=0,
        seed=seed,
    )
    first, last = campaign
    pairs = panel.loc[
        panel["treated"] & panel["WEEK_NO"].between(first, last),
        ["PRODUCT_ID", "STORE_ID"],
    ].drop_duplicates()
    cells = panel.merge(pairs, on=["PRODUCT_ID", "STORE_ID"])
    history = cells[cells["WEEK_NO"] < first]
    window = cells[cells["WEEK_NO"].between(first, last + horizon_weeks)]

    path, _ = rollout(model, history, window)
    scored = window.assign(one_step=model.predict(window)).merge(
        path[["PRODUCT_ID", "STORE_ID", "WEEK_NO", "counterfactual_units"]],
        on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
    )
    by_week = scored.groupby("WEEK_NO").agg(
        truth=("base_units", "sum"),
        one_step=("one_step", "sum"),
        rollout=("counterfactual_units", "sum"),
    )
    by_week["one_step_ratio"] = by_week["one_step"] / by_week["truth"]
    by_week["rollout_ratio"] = by_week["rollout"] / by_week["truth"]
    return by_week.assign(target=target, no_sale_rate=no_sale_rate).reset_index()


def summarise(frame: pd.DataFrame, by: list[str]) -> dict[str, Any]:
    """Bias and coverage per group, plus what they are and are not."""
    grouped = frame.groupby(by, observed=True)
    table = grouped.agg(
        replications=("seed", "count"),
        gross_true=("gross_incremental_true", "mean"),
        gross_hat=("gross_incremental_hat", "mean"),
        gross_bias=("gross_incremental_error", "mean"),
        gross_bias_share=("gross_error_share", "mean"),
        gross_coverage=("gross_incremental_covered", "mean"),
        gross_width=("gross_incremental_width", "mean"),
        net_bias=("net_incremental_error", "mean"),
        net_coverage=("net_incremental_covered", "mean"),
        tau_hat=("tau_hat", "mean"),
        drift_units=("drift_units", "mean"),
    ).reset_index()

    return {
        "by": by,
        "table": table.to_dict(orient="records"),
        "definitions": {
            "gross_bias": (
                "mean of (estimate - truth) in units, over replications. Signed: "
                "positive means the estimator claims more incremental units than "
                "the world contained."
            ),
            "gross_bias_share": (
                "the same error as a share of the true counterfactual units, so "
                "it is comparable across effect sizes and panel sizes."
            ),
            "gross_coverage": (
                "the share of replications whose [q10, q90] interval contained "
                "the truth. Nominal is 0.80 — the interval is the baseline's own "
                "quantile band, so this measures whether that band means what it "
                "says on this process."
            ),
            "tau_hat": (
                "recovered gross over recovered counterfactual units — the "
                "effect the pipeline would report, against the tau that "
                "generated the world."
            ),
            "drift_units": (
                "the pre-campaign drift check from Task 4.3, averaged. It is the "
                "estimator's error on weeks where nothing happened, and it "
                "explains most of what bias is not sampling noise."
            ),
        },
        "not_a_placebo_band": (
            "Coverage here is against a known truth on a synthetic world. It "
            "says whether the estimator recovers an effect it was given, not "
            "whether a real estimate is distinguishable from noise. Task 4.5 "
            "owns that."
        ),
    }
