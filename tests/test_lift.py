"""Tests for Task 4.3, incremental units over a window that outlasts the cycle.

The fixture panel carries a known peak and a known trough: units are lifted
during the promoted weeks and depressed for a few weeks afterwards, which is
what a pull-forward looks like. That makes the four reported numbers checkable
by hand — and makes the netting rule testable, since a window that closes at
campaign end sees only the good half.

The heavy test runs one real campaign through the shipped model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.baseline import add_price_history, fit_baseline
from promo.lift import (
    LiftCampaign,
    NoCellsError,
    campaign_cells,
    estimate_lift,
    resolve_horizon,
    write_diagnostics,
)

PANEL = Path("data/interim/panel.parquet")
BASELINE = Path("data/interim/baseline")
CYCLES = Path("data/interim/repurchase_cycles.parquet")

#: Lag features only. `week_of_year` is deliberately absent: in a fixture with
#: one campaign it is a time index, and the campaign's own weeks never appear in
#: training because they are the treated rows the baseline must not see. The
#: model then has a hole exactly where it is asked to predict, and the rollout
#: falls into it — the counterfactual collapses to under half of truth. That is
#: an artefact of a single-campaign fixture, not of the estimator, so it is kept
#: out of the panel rather than allowed to masquerade as drift.
FEATURES = ("units_lag_1", "units_lag_2", "units_roll_mean_4")

CAMPAIGN_WEEKS = (41, 44)
TROUGH_WEEKS = 3
N_PAIRS = 40
WINDOW_WEEKS = CAMPAIGN_WEEKS[1] - CAMPAIGN_WEEKS[0] + 1 + TROUGH_WEEKS


def real_data(fn):
    """Marks a test that reads the real panel, model, and cycles."""
    missing = not (PANEL.exists() and BASELINE.exists() and CYCLES.exists())
    return pytest.mark.skipif(missing, reason="run Tasks 2.6 and 4.1 first")(
        pytest.mark.heavy(fn)
    )


def _cycles(horizon: int = TROUGH_WEEKS, commodity: str = "SOUP") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"COMMODITY_DESC": commodity, "horizon_weeks": horizon, "low_support": False},
            {"COMMODITY_DESC": "OTHER", "horizon_weeks": 2, "low_support": True},
        ]
    )


def _panel(
    n_pairs: int = N_PAIRS,
    n_weeks: int = 54,
    *,
    peak: float = 0.40,
    trough: float = -0.25,
    campaign: tuple[int, int] = CAMPAIGN_WEEKS,
    trough_weeks: int = TROUGH_WEEKS,
    commodity: str = "SOUP",
    seed: int = 0,
) -> pd.DataFrame:
    """A panel with a promoted peak and the payback trough that follows it.

    The trough is the whole point: a campaign that pulls demand forward looks
    profitable right up until the weeks after it are counted.
    """
    rng = np.random.default_rng(seed)
    first, last = campaign
    rows = []
    for pair in range(n_pairs):
        level = float(rng.uniform(20.0, 50.0))
        deviation = 0.0
        for week in range(1, n_weeks + 1):
            deviation = 0.6 * deviation + rng.normal(0.0, 0.10)
            clean = max(0.0, level * (1.0 + deviation))
            promoted = first <= week <= last
            paying_back = last < week <= last + trough_weeks
            effect = peak if promoted else (trough if paying_back else 0.0)
            rows.append(
                {
                    "PRODUCT_ID": 1000 + pair % 4,
                    "STORE_ID": 300 + pair // 4,
                    "WEEK_NO": week,
                    "COMMODITY_DESC": commodity,
                    "units": clean * (1.0 + effect),
                    "clean_units": clean,
                    "treated": promoted,
                    "in_mailer": False,
                    "week_of_year": (week - 1) % 52 + 1,
                    "price_rel_category": 1.0,
                }
            )
    panel = pd.DataFrame(rows)
    panel["sales_value"] = panel["units"] * 2.0
    panel = add_price_history(panel)
    return _add_lags(panel)


def _add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).copy()
    group = out.groupby(["PRODUCT_ID", "STORE_ID"], observed=True)["units"]
    for lag in (1, 2):
        out[f"units_lag_{lag}"] = group.shift(lag)
    out["units_roll_mean_4"] = group.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return out.reset_index(drop=True)


def _model(panel: pd.DataFrame, seed: int = 0):
    controls = panel.loc[~panel["treated"]].reset_index(drop=True)
    model, _ = fit_baseline(
        controls,
        features=FEATURES,
        week_range=None,
        n_estimators=120,
        num_leaves=15,
        min_data_in_leaf=20,
        backtest_weeks=0,
        seed=seed,
    )
    return model


def _campaign(**kwargs) -> LiftCampaign:
    base = {
        "name": "soup-week-41",
        "commodity": "SOUP",
        "weeks": CAMPAIGN_WEEKS,
    }
    return LiftCampaign(**{**base, **kwargs})


def _estimate(panel: pd.DataFrame, campaign=None, seed: int = 0, **kwargs):
    return estimate_lift(
        campaign or _campaign(),
        _model(panel, seed=seed),
        panel,
        _cycles(),
        **kwargs,
    )


# --- the four numbers ---------------------------------------------------------


def test_the_peak_the_trough_and_what_survives():
    panel = _panel(seed=1)
    residuals, diag = _estimate(panel, seed=1)
    lift = diag["lift"]

    # The generator put a peak in the promoted weeks and a trough after them,
    # and the estimate finds both with the right signs.
    assert lift["gross_incremental"] > 0
    assert lift["post_window_residual"] < 0
    assert lift["net_incremental"] < lift["gross_incremental"]
    assert 0.0 < lift["retention_ratio"] < 1.0

    # Every promoted and post week of every campaign pair, and nothing else.
    assert set(residuals["phase"]) == {"campaign", "post"}
    assert residuals["WEEK_NO"].min() == CAMPAIGN_WEEKS[0]
    assert residuals["WEEK_NO"].max() == CAMPAIGN_WEEKS[1] + TROUGH_WEEKS
    assert len(residuals) == N_PAIRS * WINDOW_WEEKS


def test_net_is_gross_plus_post_not_gross_minus_post():
    """The netting rule, asserted rather than trusted.

    `delta_q = s + (g - l)` in Phase 6 and net = gross + post here are the same
    discipline: a signed component is added. Subtracting a negative tail would
    report the payback period as a second helping of lift.
    """
    panel = _panel(seed=2)
    _, diag = _estimate(panel, seed=2)
    lift = diag["lift"]

    gross = lift["gross_incremental"]
    post = lift["post_window_residual"]
    assert lift["net_incremental"] == pytest.approx(gross + post, abs=1e-6)
    # The sign check that catches the subtraction: with a negative tail, the
    # wrong version is larger than gross, and this one is smaller.
    assert lift["net_incremental"] < gross
    assert lift["net_incremental"] != pytest.approx(gross - post, abs=1e-6)


def test_retention_is_computed_once_per_path_and_never_averaged():
    panel = _panel(seed=3)
    _, diag = _estimate(panel, seed=3)
    lift = diag["lift"]

    for path in ("point", "low", "high"):
        block = lift["by_path"][path]
        if block["retention_ratio"] is None:
            continue
        assert block["retention_ratio"] == pytest.approx(
            block["net_incremental"] / block["gross_incremental"], rel=1e-6
        )

    ratios = [
        lift["by_path"][p]["retention_ratio"]
        for p in ("low", "point", "high")
        if lift["by_path"][p]["retention_ratio"] is not None
    ]
    interval = lift["retention_ratio_interval"]
    assert interval == [min(ratios), max(ratios)]
    # The mean of the path ratios is not what is reported.
    assert lift["retention_ratio"] == lift["by_path"]["point"]["retention_ratio"]


def test_a_window_that_stops_at_campaign_end_sees_only_the_good_half():
    """Why the horizon rule exists, measured on the same panel."""
    panel = _panel(seed=4)
    model = _model(panel, seed=4)

    _, full_diag = estimate_lift(_campaign(), model, panel, _cycles())
    short, short_diag = estimate_lift(
        _campaign(horizon_weeks=0), model, panel, _cycles()
    )

    assert short_diag["horizon"]["status"] == "HORIZON_TOO_SHORT"
    assert short_diag["horizon"]["shortfall_weeks"] == TROUGH_WEEKS
    # Same gross, because the promoted weeks are the same weeks.
    assert short_diag["lift"]["gross_incremental"] == pytest.approx(
        full_diag["lift"]["gross_incremental"], rel=1e-9
    )
    # But the short window reports the peak as the whole story.
    assert short_diag["lift"]["net_incremental"] > full_diag["lift"][
        "net_incremental"
    ]
    assert (short["phase"] == "post").sum() == 0
    assert "biased upward" in short_diag["horizon"]["consequence"]


# --- the interval -------------------------------------------------------------


def test_the_band_inverts_and_brackets_the_point_estimate():
    panel = _panel(seed=5)
    residuals, diag = _estimate(panel, seed=5)
    lift = diag["lift"]

    # A high counterfactual makes a low residual.
    assert (residuals["counterfactual_low"] <= residuals["counterfactual_high"]).all()
    assert (residuals["residual_low"] <= residuals["residual_high"]).all()

    for field in ("gross_incremental", "post_window_residual", "net_incremental"):
        low, high = lift[f"{field}_interval"]
        assert low <= lift[field] <= high

    assert "not a placebo comparison" in diag["interval_note"]


def test_quantiles_none_gives_a_point_estimate_with_no_band():
    panel = _panel(seed=6)
    residuals, diag = _estimate(panel, seed=6, quantiles=None)

    assert "residual_low" not in residuals.columns
    assert "gross_incremental_interval" not in diag["lift"]
    assert diag["lift"]["gross_incremental"] > 0


# --- the horizon --------------------------------------------------------------


def test_the_horizon_defaults_to_the_commodity_cycle():
    weeks, diag = resolve_horizon(_campaign(), _cycles(horizon=6))
    assert weeks == 6
    assert diag["source"] == "repurchase cycle"
    assert diag["status"] == "OK"


def test_a_supplied_horizon_is_honoured_and_checked():
    weeks, diag = resolve_horizon(_campaign(horizon_weeks=9), _cycles(horizon=6))
    assert weeks == 9
    assert diag["source"] == "campaign"
    assert diag["status"] == "OK"

    weeks, short = resolve_horizon(_campaign(horizon_weeks=2), _cycles(horizon=6))
    assert weeks == 2
    assert short["status"] == "HORIZON_TOO_SHORT"
    assert short["shortfall_weeks"] == 4


def test_an_unrecorded_commodity_is_unknown_not_fine():
    weeks, diag = resolve_horizon(
        _campaign(commodity="NOT IN THE TABLE"), _cycles()
    )
    assert weeks is None
    assert diag["status"] == "UNKNOWN_CYCLE"
    assert "not a pass" in diag["consequence"]


def test_no_horizon_at_all_refuses_to_guess():
    panel = _panel(seed=7)
    campaign = _campaign(commodity="NOT IN THE TABLE")
    with pytest.raises(ValueError, match="no horizon"):
        estimate_lift(campaign, _model(panel, seed=7), panel, _cycles())


# --- the window's edges -------------------------------------------------------


def test_a_truncated_post_window_is_recorded():
    # Campaign ends two weeks before the panel does, but the cycle needs three.
    panel = _panel(n_weeks=46, campaign=(41, 44), seed=8)
    _, diag = _estimate(panel, seed=8)

    post = diag["cells"]["post_window"]
    assert post["required_weeks"] == 3
    assert post["available_weeks"] == 2
    assert post["truncated"] is True
    assert "biased upward" in post["consequence"]


def test_no_post_window_at_all_withholds_the_retention_ratio():
    """net == gross would read as 'retained everything'. It means 'never seen'."""
    panel = _panel(n_weeks=44, campaign=(41, 44), seed=9)
    _, diag = _estimate(panel, seed=9)
    lift = diag["lift"]

    assert diag["cells"]["post_window"]["available_weeks"] == 0
    assert lift["net_incremental"] == pytest.approx(lift["gross_incremental"])
    assert lift["retention_ratio"] is None
    assert "never observed" in lift["retention_ratio_absent"]


def test_a_non_positive_gross_withholds_the_retention_ratio():
    panel = _panel(peak=-0.30, trough=0.0, seed=10)
    _, diag = _estimate(panel, seed=10)
    lift = diag["lift"]

    assert lift["gross_incremental"] < 0
    assert lift["retention_ratio"] is None
    assert "not positive" in lift["retention_ratio_absent"]


# --- which cells are the campaign ---------------------------------------------


def test_the_campaign_is_the_pairs_that_ran_it():
    panel = _panel(seed=11)
    # One product never runs the display; it must not be in the measurement.
    panel.loc[panel["PRODUCT_ID"] == 1003, "treated"] = False
    cells, diag = campaign_cells(_campaign(), panel, horizon_weeks=TROUGH_WEEKS)

    assert 1003 not in set(cells["PRODUCT_ID"])
    assert diag["pairs"] == N_PAIRS - N_PAIRS // 4
    assert "treated at least once" in diag["membership_rule"]
    assert diag["treated_share_of_campaign_weeks"] == 1.0


def test_an_untreated_week_inside_the_window_still_counts_for_a_pair_that_ran():
    panel = _panel(seed=12)
    panel.loc[
        (panel["PRODUCT_ID"] == 1000) & (panel["WEEK_NO"] == 43), "treated"
    ] = False
    residuals, diag = _estimate(panel, seed=12)

    week_43 = residuals[
        (residuals["PRODUCT_ID"] == 1000) & (residuals["WEEK_NO"] == 43)
    ]
    assert len(week_43) == N_PAIRS // 4
    assert not week_43["treated"].any()
    assert (week_43["phase"] == "campaign").all()
    assert diag["cells"]["treated_share_of_campaign_weeks"] < 1.0


def test_stores_and_products_narrow_the_campaign():
    panel = _panel(seed=13)
    cells, diag = campaign_cells(
        _campaign(products=(1000, 1001), stores=(300,)),
        panel,
        horizon_weeks=TROUGH_WEEKS,
    )
    assert set(cells["PRODUCT_ID"]) == {1000, 1001}
    assert set(cells["STORE_ID"]) == {300}
    assert diag["pairs"] == 2


def test_a_campaign_that_matched_nothing_raises_rather_than_returning_zero():
    panel = _panel(seed=14)
    with pytest.raises(NoCellsError, match="not a lift of zero"):
        campaign_cells(_campaign(weeks=(20, 22)), panel, horizon_weeks=TROUGH_WEEKS)


# --- the counterfactual it is built on ----------------------------------------


def test_the_counterfactual_is_the_recursive_rollout():
    panel = _panel(seed=15)
    _, diag = _estimate(panel, seed=15)

    rollout_diag = diag["counterfactual"]["rollout"]
    assert rollout_diag["feedback"] == "recursive"
    assert "biased" not in rollout_diag
    # The post window is rolled out in the same recursion as the campaign, so
    # the trough is measured against a counterfactual that never saw the peak.
    assert rollout_diag["steps"] == WINDOW_WEEKS


def test_the_drift_check_measures_the_same_window_before_the_campaign():
    """A rollout over weeks where nothing happened should return nothing."""
    panel = _panel(seed=17)
    _, diag = _estimate(panel, seed=17)
    drift = diag["drift_check"]

    assert drift["ran"] is True
    assert drift["weeks"] == [CAMPAIGN_WEEKS[0] - WINDOW_WEEKS, CAMPAIGN_WEEKS[0] - 1]
    assert drift["clean"] is True
    assert "Task 4.5 owns that" in drift["not_the_placebo_band"]

    # It is small next to the campaign's own gross — which is the only reason
    # the gross figure means anything.
    assert abs(drift["residual_units"]) < 0.5 * diag["lift"]["gross_incremental"]
    assert drift["exceeds_gross"] is False
    assert "not sufficient" in drift["reading"]


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_a_campaign_with_no_effect_cannot_be_told_from_drift(seed: int):
    """What the check is for, on a panel where the truth is zero.

    Both figures are then the estimator talking, so they come out the same size
    and which one is larger is a coin flip — that indistinguishability is the
    finding, and asserting a direction here would only pin the seed. The flags
    still have to agree with the arithmetic, and the sentence with the flags.
    """
    panel = _panel(peak=0.0, trough=0.0, seed=seed)
    _, diag = _estimate(panel, seed=seed)
    lift, drift = diag["lift"], diag["drift_check"]

    # The estimate is not large next to the drift, compared over the same
    # number of steps. Only the upper bound is a claim: a ratio far below one
    # means drift dominates the estimate outright, which is the same finding
    # more emphatically, not a failure of it.
    ratio = abs(lift["gross_incremental"]) / abs(drift["residual_units_first_weeks"])
    assert ratio < 4.0

    assert drift["exceeds_gross"] == (
        abs(drift["residual_units_first_weeks"]) >= abs(lift["gross_incremental"])
    )
    assert drift["exceeds_net"] == (
        abs(drift["residual_units"]) >= abs(lift["net_incremental"])
    )
    expected = (
        "should not be acted on"
        if drift["exceeds_gross"] or drift["exceeds_net"]
        else "not sufficient"
    )
    assert expected in drift["reading"]


def test_drift_larger_than_the_estimate_is_said_out_loud():
    """The sentence itself, on numbers chosen rather than sampled."""
    from promo.lift import _compare_drift_to_gross

    swamped = {
        "ran": True,
        "residual_units": 300.0,
        "residual_units_first_weeks": 120.0,
    }
    _compare_drift_to_gross(swamped, gross=10.0, net=12.0)
    assert swamped["exceeds_gross"] is True
    assert swamped["exceeds_net"] is True
    assert "should not be acted on" in swamped["reading"]
    assert "Task 4.5" in swamped["reading"]

    resolvable = {
        "ran": True,
        "residual_units": 30.0,
        "residual_units_first_weeks": 12.0,
    }
    _compare_drift_to_gross(resolvable, gross=400.0, net=380.0)
    assert resolvable["exceeds_gross"] is False
    assert resolvable["exceeds_net"] is False
    assert "necessary" in resolvable["reading"]

    # A check that could not run says nothing rather than something reassuring.
    skipped = {"ran": False, "why": "not enough history"}
    _compare_drift_to_gross(skipped, gross=1.0, net=1.0)
    assert "reading" not in skipped


def test_the_drift_check_says_when_it_cannot_run_or_is_contaminated():
    early = _panel(campaign=(8, 11), seed=18)
    _, diag = estimate_lift(
        _campaign(weeks=(8, 11)), _model(early, seed=18), early, _cycles()
    )
    assert diag["drift_check"]["ran"] is False
    assert "history" in diag["drift_check"]["why"]

    panel = _panel(seed=19)
    # An earlier promotion inside the drift window: the residual there is no
    # longer drift alone, and the check has to say so rather than report it.
    panel.loc[panel["WEEK_NO"].between(36, 37), "treated"] = True
    _, dirty = _estimate(panel, seed=19)
    assert dirty["drift_check"]["clean"] is False
    assert "upper bound on drift" in dirty["drift_check"]["contaminated"]


def test_drift_checking_can_be_turned_off():
    panel = _panel(seed=20)
    _, diag = _estimate(panel, seed=20, check_drift=False)
    assert diag["drift_check"] == {"ran": False, "why": "check_drift=False"}


def test_write_diagnostics_round_trips(tmp_path):
    import json

    panel = _panel(seed=16)
    _, diag = _estimate(panel, seed=16)
    path = write_diagnostics(diag, tmp_path / "lift_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "estimate_lift"


# --- the real panel -----------------------------------------------------------


@real_data
def test_a_real_campaign_runs_end_to_end():
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    product, commodity = con.execute(
        f"""
        SELECT PRODUCT_ID, any_value(COMMODITY_DESC)
        FROM read_parquet('{PANEL.as_posix()}')
        WHERE treated AND WEEK_NO BETWEEN 80 AND 83
        GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1
        """
    ).fetchone()
    con.close()

    campaign = LiftCampaign(
        name="real", commodity=commodity, product=int(product), weeks=(80, 83)
    )
    residuals, diag = estimate_lift(campaign, BASELINE, PANEL, CYCLES)

    assert diag["horizon"]["source"] == "repurchase cycle"
    assert diag["horizon"]["required_weeks"] >= 1
    horizon = diag["horizon"]["horizon_weeks"]
    assert residuals["WEEK_NO"].max() == 83 + horizon
    assert set(residuals["phase"]) == {"campaign", "post"}
    assert diag["cells"]["pairs"] > 0

    lift = diag["lift"]
    low, high = lift["net_incremental_interval"]
    assert low <= lift["net_incremental"] <= high
    assert lift["net_incremental"] == pytest.approx(
        lift["gross_incremental"] + lift["post_window_residual"], abs=1e-6
    )
