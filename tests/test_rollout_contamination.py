"""Tests for Task 4.2, the recursive rollout.

The file is named for the one test that matters: **contamination**. A naive
multi-week counterfactual feeds the observed units of week *w-1* into week *w*'s
lag, and inside a campaign those observed units carry the promotion. The
counterfactual then chases the bump, the residual shrinks, and the measured
effect falls the longer the campaign runs — a campaign that kept working looks
like one that stopped.

`test_the_naive_version_decays_and_the_rollout_does_not` measures that on a
panel with a known effect, at every window length from one week to eight, and
asserts the decay exists in one mode and not the other. The rest of the file
checks the mechanics the argument rests on: that lags come from the predicted
path, that rolling means are recomputed rather than reused, and that a gap in
the weeks raises instead of quietly redefining what "last week" means.

The generator here is deliberately small and local. `tests/synthetic.py` and its
recovery harness are Task 4.4's, not this file's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from promo.baseline import (
    RECURSIVE_FEATURES,
    NonContiguousWeeksError,
    add_price_history,
    fit_baseline,
    rollout,
)

#: A short, lag-heavy feature set. The point of this file is what happens to a
#: lag under feedback, so the fit is given features that make the lag matter and
#: nothing that would obscure it.
FEATURES = (
    "units_lag_1",
    "units_lag_2",
    "units_lag_4",
    "units_roll_mean_4",
    "week_of_year",
)

CAMPAIGN_START = 41
TAU = 0.30


def _ar_panel(
    n_pairs: int = 24,
    n_weeks: int = 56,
    *,
    tau: float = TAU,
    campaign_start: int = CAMPAIGN_START,
    campaign_end: int | None = None,
    phi: float = 0.75,
    seed: int = 0,
) -> pd.DataFrame:
    """An AR(1) panel with a known multiplicative effect from `campaign_start`.

    Units are persistent in their own deviation from level, which is what makes
    a lag informative — and therefore what makes contaminating it costly. The
    process is autoregressive with seasonality, not a tree, so the model cannot
    recover it exactly; the test is about the *shape* of the bias with window
    length, not about point accuracy.
    """
    rng = np.random.default_rng(seed)
    end = n_weeks if campaign_end is None else campaign_end
    rows = []
    for pair in range(n_pairs):
        level = float(rng.uniform(20.0, 60.0))
        deviation = 0.0
        for week in range(1, n_weeks + 1):
            deviation = phi * deviation + rng.normal(0.0, 0.18)
            season = 1.0 + 0.15 * np.sin(2 * np.pi * week / 52.0)
            clean = max(0.0, level * season * (1.0 + deviation))
            treated = campaign_start <= week <= end
            rows.append(
                {
                    "PRODUCT_ID": 1000 + pair % 6,
                    "STORE_ID": 300 + pair // 6,
                    "WEEK_NO": week,
                    "clean_units": clean,
                    "units": clean * (1.0 + tau) if treated else clean,
                    "treated": treated,
                    "in_mailer": False,
                    "week_of_year": (week - 1) % 52 + 1,
                }
            )
    panel = pd.DataFrame(rows)
    panel["sales_value"] = panel["units"] * 2.0
    return _add_lags(panel)


def _add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Phase 2.6's lagged block, computed the same way, over observed units."""
    out = panel.sort_values(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).copy()
    group = out.groupby(["PRODUCT_ID", "STORE_ID"], observed=True)["units"]
    for lag in (1, 2, 4):
        out[f"units_lag_{lag}"] = group.shift(lag)
    out["units_roll_mean_4"] = group.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return out.reset_index(drop=True)


def _fitted(panel: pd.DataFrame, seed: int = 0, features: tuple[str, ...] = FEATURES):
    controls = panel.loc[~panel["treated"]].reset_index(drop=True)
    model, _ = fit_baseline(
        controls,
        features=features,
        week_range=None,
        n_estimators=120,
        num_leaves=15,
        min_data_in_leaf=20,
        backtest_weeks=0,
        seed=seed,
    )
    return model


def _measured_effect(
    model, panel: pd.DataFrame, horizon: int, feedback: str
) -> float:
    """Total observed over total counterfactual, minus one, over the window.

    A ratio of sums, never a mean of ratios: the aggregate is what a campaign
    report states, and averaging per-row ratios would weight a one-unit week the
    same as a hundred-unit one.
    """
    window = range(CAMPAIGN_START, CAMPAIGN_START + horizon)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].isin(window)]

    path, _ = rollout(model, history, exog, feedback=feedback)
    joined = path.merge(
        exog[["PRODUCT_ID", "STORE_ID", "WEEK_NO", "units"]],
        on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
        validate="one_to_one",
    )
    return float(
        joined["units"].sum() / joined["counterfactual_units"].sum() - 1.0
    )


# --- the contamination itself ------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_naive_version_decays_and_the_rollout_does_not(seed: int):
    """The argument for the whole module, at eight horizons and three panels.

    Run over several panels on purpose: an assertion tuned to one seed proves
    the seed, not the estimator.
    """
    panel = _ar_panel(seed=seed)
    model = _fitted(panel, seed=seed)
    horizons = list(range(1, 9))

    naive = [_measured_effect(model, panel, h, "observed") for h in horizons]
    recursive = [_measured_effect(model, panel, h, "recursive") for h in horizons]

    # Week one has no feedback yet — nothing has been fed back — so the two
    # modes must agree exactly. If they differ here, the difference below is
    # some other bug and not contamination.
    assert naive[0] == pytest.approx(recursive[0], abs=1e-12)

    # The naive version shrinks. Every horizon past the first is below it, the
    # trend is downward, and by eight weeks well over a third of the effect has
    # been eaten by the model chasing its own contaminated lag.
    assert all(v < naive[0] for v in naive[1:])
    assert naive[7] < 0.6 * naive[0]
    naive_slope = float(np.polyfit(horizons, naive, 1)[0])
    assert naive_slope < -0.005

    # The rollout does not shrink. That is the claim being tested — not that it
    # is flat: see the drift test below for the error it does have.
    assert recursive[7] > 0.9 * recursive[0]
    recursive_slope = float(np.polyfit(horizons, recursive, 1)[0])
    assert recursive_slope > naive_slope + 0.01

    # And it is the naive one that is wrong: the true effect is TAU throughout.
    assert abs(recursive[7] - TAU) < abs(naive[7] - TAU)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_rollouts_own_error_points_the_other_way(seed: int):
    """The rollout is unbiased by contamination, not error-free.

    Compounding is the price of not contaminating the lags: each step's
    features are built from earlier predictions, so the counterfactual drifts.
    On this generator it drifts *down*, which pushes the measured effect *up* —
    the opposite direction to contamination. Asserted rather than mentioned, so
    that nobody reads "the rollout does not shrink" as "the rollout is right".
    Quantifying it against a known truth is Task 4.4's job.
    """
    panel = _ar_panel(seed=seed)
    model = _fitted(panel, seed=seed)

    short = _measured_effect(model, panel, 1, "recursive")
    long = _measured_effect(model, panel, 8, "recursive")

    assert long >= short
    # Bounded, though: the drift is a fraction of the effect, not a multiple.
    assert long < 2.0 * short


def test_the_gap_between_the_two_grows_with_the_window():
    """The bias is a function of horizon, which is why one week never shows it."""
    panel = _ar_panel(seed=2)
    model = _fitted(panel, seed=2)

    gaps = [
        _measured_effect(model, panel, h, "recursive")
        - _measured_effect(model, panel, h, "observed")
        for h in (1, 4, 8)
    ]
    assert gaps[0] == pytest.approx(0.0, abs=1e-12)
    assert gaps[2] > gaps[1] > 0


# --- the mechanics the argument rests on -------------------------------------


def test_next_weeks_lag_is_this_weeks_prediction_never_the_observation():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=3)
    model = _fitted(panel, seed=3)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 5)]

    path, diag = rollout(model, history, exog)

    for _, block in path.groupby(["PRODUCT_ID", "STORE_ID"], observed=True):
        block = block.sort_values("WEEK_NO")
        predicted = block["counterfactual_units"].to_numpy()
        lag_1 = block["units_lag_1"].to_numpy()
        # Step t's lag_1 is step t-1's prediction, to the last bit.
        assert np.allclose(lag_1[1:], predicted[:-1], rtol=0, atol=1e-12)

    observed = exog.set_index(["PRODUCT_ID", "STORE_ID", "WEEK_NO"])["units"]
    later = path[path["step"] > 0].set_index(["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
    shifted = observed.groupby(level=[0, 1]).shift(1).reindex(later.index)
    # ...and it is not the observed value, which under a real effect differs.
    assert not np.allclose(later["units_lag_1"], shifted, rtol=0, atol=1e-9)

    assert set(diag["features"]["recursive"]) == set(FEATURES) & set(
        RECURSIVE_FEATURES
    )


def test_rolling_means_are_recomputed_from_the_counterfactual_path():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=4)
    model = _fitted(panel, seed=4)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 5)]

    path, _ = rollout(model, history, exog)

    for key, block in path.groupby(["PRODUCT_ID", "STORE_ID"], observed=True):
        block = block.sort_values("WEEK_NO")
        past = history[
            (history["PRODUCT_ID"] == key[0]) & (history["STORE_ID"] == key[1])
        ].sort_values("WEEK_NO")
        series = dict(zip(past["WEEK_NO"], past["units"], strict=True))
        for _, row in block.iterrows():
            week = int(row["WEEK_NO"])
            expected = np.mean(
                [series[w] for w in range(week - 4, week) if w in series]
            )
            assert row["units_roll_mean_4"] == pytest.approx(expected, rel=1e-9)
            series[week] = row["counterfactual_units"]


def test_a_gap_in_the_weeks_raises_rather_than_redefining_last_week():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=5)
    model = _fitted(panel, seed=5)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]

    inside = panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 5)
    with pytest.raises(NonContiguousWeeksError, match="not consecutive"):
        rollout(model, history, panel[inside & (panel["WEEK_NO"] != CAMPAIGN_START + 2)])

    detached = panel["WEEK_NO"].between(CAMPAIGN_START + 3, CAMPAIGN_START + 6)
    with pytest.raises(NonContiguousWeeksError, match="history ends at week"):
        rollout(model, history, panel[detached])


def test_ragged_windows_across_product_stores():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=6)
    model = _fitted(panel, seed=6)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]

    inside = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 5)]
    cut_short = inside["PRODUCT_ID"] == inside["PRODUCT_ID"].min()
    exog = inside[~cut_short | (inside["WEEK_NO"] <= CAMPAIGN_START + 1)]

    path, diag = rollout(model, history, exog)
    assert len(path) == len(exog)
    assert diag["steps"] == 6
    counts = path.groupby(["PRODUCT_ID", "STORE_ID"], observed=True).size()
    assert set(counts) == {2, 6}


def test_carried_features_hold_their_last_pre_window_value():
    # The real column, derived the real way, so the test exercises the
    # settled-decision-9 replacement rather than a stand-in named like it.
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=7)
    panel["price_rel_category"] = 1.0 + panel["WEEK_NO"] / 1000.0
    panel = add_price_history(panel)
    model = _fitted(panel, seed=7, features=(*FEATURES, "price_rel_category_lag"))

    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)]

    path, diag = rollout(model, history, exog.drop(columns=["price_rel_category_lag"]))

    assert diag["features"]["carried"] == ["price_rel_category_lag"]
    assert "decision 9" in diag["carried_note"]
    assert len(path) == len(exog)


def test_supplying_the_price_column_is_not_enough_to_use_it():
    """The safe reading is the default, because the natural call is the risky one.

    Handing `rollout` a slice of the panel supplies the observed in-window
    `price_rel_category_lag` without the caller thinking about it. Its
    availability is post-treatment (settled decision 9), so the column stays
    carried unless `carry=()` says otherwise — and the diagnostics name the
    override when it happens.
    """
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=8)
    panel["price_rel_category"] = 1.0 + panel["WEEK_NO"] / 1000.0
    panel = add_price_history(panel)
    model = _fitted(panel, seed=8, features=(*FEATURES, "price_rel_category_lag"))

    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)]

    default, diag = rollout(model, history, exog)
    assert diag["features"]["carried"] == ["price_rel_category_lag"]
    assert diag["features"]["forced_carry_overrode_exog"] == [
        "price_rel_category_lag"
    ]

    opted_in, opt_diag = rollout(model, history, exog, carry=())
    assert "price_rel_category_lag" in opt_diag["features"]["exogenous"]
    assert opt_diag["features"]["carried"] == []

    # The values the model was given are what differ, and the path shows them:
    # frozen by default, walking with the window when the caller opts in.
    #
    # Frozen at the *last history row's* value of the feature, which is a
    # generic rule and not a per-column one — the column is itself a lag, so
    # that value is week 39's price, one week staler than the last observable
    # pre-window price. Immaterial, and preferable to a carry rule that has to
    # know the semantics of every column it carries.
    frozen = 1.0 + (CAMPAIGN_START - 2) / 1000.0
    assert np.allclose(default["price_rel_category_lag"], frozen)
    assert opted_in["price_rel_category_lag"].nunique() == 4


def test_in_mailer_is_taken_from_the_window_not_from_history():
    """Settled decision 8: the mailer that actually ran, and no display."""
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=9)
    model = _fitted(panel, seed=9, features=(*FEATURES, "in_mailer"))

    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)].copy()
    exog["in_mailer"] = True

    _, diag = rollout(model, history, exog)
    assert "in_mailer" in diag["features"]["exogenous"]
    assert "in_mailer" not in diag["features"]["carried"]


def test_the_naive_mode_is_labelled_biased_and_needs_observed_units():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=10)
    model = _fitted(panel, seed=10)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)]

    _, diag = rollout(model, history, exog, feedback="observed")
    assert diag["biased"] is True
    assert "never produce a reported estimate" in diag["why_biased"]

    _, clean = rollout(model, history, exog)
    assert "biased" not in clean

    with pytest.raises(KeyError, match="observed units back as lags"):
        rollout(model, history, exog.drop(columns=["units"]), feedback="observed")

    with pytest.raises(ValueError, match="feedback must be"):
        rollout(model, history, exog, feedback="naive")


def test_quantile_paths_feed_their_own_predictions_back():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=11)
    model = _fitted(panel, seed=11)
    history = panel[panel["WEEK_NO"] < CAMPAIGN_START]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)]

    low, diag = rollout(model, history, exog, quantile=0.1)
    high, _ = rollout(model, history, exog, quantile=0.9)

    assert diag["quantile"] == 0.1
    assert low["counterfactual_units"].sum() < high["counterfactual_units"].sum()
    # Each path is its own recursion: the q10 path's lag is the q10 prediction.
    for _, block in low.groupby(["PRODUCT_ID", "STORE_ID"], observed=True):
        block = block.sort_values("WEEK_NO")
        assert np.allclose(
            block["units_lag_1"].to_numpy()[1:],
            block["counterfactual_units"].to_numpy()[:-1],
            rtol=0,
            atol=1e-12,
        )


def test_missing_history_for_a_key_is_named():
    panel = _ar_panel(n_pairs=6, n_weeks=50, seed=12)
    model = _fitted(panel, seed=12)
    history = panel[
        (panel["WEEK_NO"] < CAMPAIGN_START) & (panel["PRODUCT_ID"] != 1000)
    ]
    exog = panel[panel["WEEK_NO"].between(CAMPAIGN_START, CAMPAIGN_START + 3)]

    with pytest.raises(KeyError, match="no history for product-store"):
        rollout(model, history, exog)
