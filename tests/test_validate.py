"""Tests for Task 4.5, the placebo distribution and the gate it owes.

Three things are being checked and they are different kinds of claim.

**The band is built correctly**: never-treated cells only, size-matched draws,
at least 300 windows or an exception. These are correctness tests.

**The gate fires and does not fire**, per the gate-authoring skill: one case
constructed so an estimate lands inside the band, one so it lands outside, both
through `run_audit()`. A gate that has never fired in a test does not work.

**The band on real data says what it says.** The heavy test records the shape of
the null on the shipped model rather than asserting a number — the point of a
placebo is that its value is discovered, not specified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.baseline import add_price_history, fit_baseline
from promo.gates import GATE_ORDER, CampaignSpec, run_audit
from promo.lift import LiftCampaign
from promo.validate import (
    DEFAULT_ALPHA,
    DEFAULT_POOL,
    MIN_WINDOWS,
    InsufficientPlaceboError,
    band_for_campaign,
    inside_band,
    never_treated_cells,
    placebo_band,
    placebo_pool,
    summarise_band,
    write_band,
)

PANEL = Path("data/interim/panel.parquet")
BASELINE = Path("data/interim/baseline")
CYCLES = Path("data/interim/repurchase_cycles.parquet")

FEATURES = ("units_lag_1", "units_lag_2", "units_roll_mean_4")


def real_data(fn):
    missing = not (PANEL.exists() and BASELINE.exists() and CYCLES.exists())
    return pytest.mark.skipif(missing, reason="run Tasks 2.6 and 4.1 first")(
        pytest.mark.heavy(fn)
    )


def _panel(
    n_pairs: int = 60,
    n_weeks: int = 60,
    *,
    treated_pairs: int = 20,
    campaign: tuple[int, int] = (40, 43),
    tau: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A panel where some cells are promoted and the rest never are.

    `tau` is the effect on the treated cells. Zero by default: most of these
    tests are about the band's construction, not about detecting anything.
    """
    rng = np.random.default_rng(seed)
    first, last = campaign
    rows = []
    for pair in range(n_pairs):
        level = float(rng.uniform(15.0, 45.0))
        deviation = 0.0
        promoted_cell = pair < treated_pairs
        for week in range(1, n_weeks + 1):
            deviation = 0.6 * deviation + rng.normal(0.0, 0.12)
            clean = max(0.0, level * (1.0 + deviation))
            treated = promoted_cell and first <= week <= last
            rows.append(
                {
                    "PRODUCT_ID": 1000 + pair % 10,
                    "STORE_ID": 300 + pair // 10,
                    "WEEK_NO": week,
                    "COMMODITY_DESC": "SOUP",
                    "units": clean * (1.0 + tau) if treated else clean,
                    "treated": treated,
                    "in_mailer": False,
                    "week_of_year": (week - 1) % 52 + 1,
                    "price_rel_category": 1.0,
                }
            )
    panel = pd.DataFrame(rows)
    panel["sales_value"] = panel["units"] * 2.0
    panel = add_price_history(panel)
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
        n_estimators=60,
        num_leaves=15,
        min_data_in_leaf=20,
        backtest_weeks=0,
        seed=seed,
    )
    return model


def _band(panel: pd.DataFrame, *, n_cells: int = 8, seed: int = 0, **kwargs):
    defaults = {
        "n_cells": n_cells,
        "campaign_length": 4,
        "horizon_weeks": 3,
        "n_windows": MIN_WINDOWS,
        "week_range": None,
        "pool": "never_treated",
        "seed": seed,
    }
    return placebo_band(_model(panel, seed=seed), panel, **{**defaults, **kwargs})


# --- the pool -----------------------------------------------------------------


def test_only_never_treated_cells_are_in_the_pool():
    panel = _panel(seed=1)
    cells, diag = never_treated_cells(panel, week_range=None)

    ever = panel.loc[panel["treated"], ["PRODUCT_ID", "STORE_ID"]].drop_duplicates()
    pool = cells[["PRODUCT_ID", "STORE_ID"]].drop_duplicates()
    overlap = pool.merge(ever, on=["PRODUCT_ID", "STORE_ID"])
    assert overlap.empty
    assert not cells["treated"].any()

    assert diag["pairs"] == len(pool)
    assert diag["panel_pairs_ever_treated"] == len(ever)
    assert "still paying it back" in diag["rule"]


def test_a_cell_treated_in_any_week_is_excluded_from_every_week():
    """The rule that matters: untreated-in-this-window is not good enough."""
    panel = _panel(seed=2)
    cells, _ = never_treated_cells(panel, week_range=None)

    # Pick a cell promoted only in weeks 40-43. None of its 60 weeks may appear.
    promoted = panel.loc[panel["treated"]].iloc[0]
    match = cells[
        (cells["PRODUCT_ID"] == promoted["PRODUCT_ID"])
        & (cells["STORE_ID"] == promoted["STORE_ID"])
    ]
    assert match.empty


# --- the ever-treated pool, which is the default ------------------------------


def test_the_default_pool_is_ever_treated_cells():
    """Settled by measurement: the campaign's cells are ever-treated cells."""
    assert DEFAULT_POOL == "ever_treated"
    panel = _panel(seed=30)
    pool = placebo_pool(panel, week_range=None)

    ever = panel.loc[panel["treated"], ["PRODUCT_ID", "STORE_ID"]].drop_duplicates()
    assert len(pool.pairs) == len(ever)
    assert set(map(tuple, pool.pairs.to_numpy())) == set(map(tuple, ever.to_numpy()))
    assert "AUC 0.70" in pool.diagnostics["why"]


def test_eligibility_excludes_the_window_and_the_payback_before_it():
    """The exclusion that is easy to forget is the run-up, not the window."""
    panel = _panel(seed=31)
    pool = placebo_pool(panel, week_range=None)
    # The fixture treats weeks 40-43 on every pool cell.
    span, guard = 7, 3

    # A window overlapping the promotion: nobody is drawable.
    assert not pool.eligible(42, span, guard).any()
    # A window starting right after it: still nobody, because weeks 40-43 are
    # inside the guard and their payback lands in the window.
    assert not pool.eligible(44, span, guard).any()
    # Far enough past the payback, everyone is drawable again.
    assert pool.eligible(47, span, guard).all()
    # And before it.
    assert pool.eligible(20, span, guard).all()


def test_a_never_treated_pool_is_always_eligible():
    panel = _panel(seed=32)
    pool = placebo_pool(panel, "never_treated", week_range=None)
    assert pool.eligible(42, 7, 3).all()
    assert pool.treated_by_week.size == 0
    assert "not because it is the right null" in pool.diagnostics["why"]


def test_an_unknown_pool_kind_is_refused():
    with pytest.raises(ValueError, match="kind must be one of"):
        placebo_pool(_panel(seed=33), "sometimes_treated", week_range=None)


def test_the_ever_treated_band_records_how_many_cells_were_drawable():
    panel = _panel(n_pairs=80, treated_pairs=60, seed=34)
    _, diag = _band(panel, n_cells=4, seed=34, pool="ever_treated")

    assert diag["pool_kind"] == "ever_treated"
    eligibility = diag["eligibility"]
    assert eligibility["guard_weeks"] == 3
    assert eligibility["eligible_cells_per_usable_start"]["min"] >= 4
    assert "payback" in eligibility["why"]
    # Weeks around the fixture's promotion cannot seat a draw, so they are
    # dropped up front rather than silently shrinking the band.
    assert eligibility["usable_starts"] < eligibility["candidate_starts"]
    assert "cluster in time" in eligibility["starts_note"]


def test_a_pool_too_saturated_to_seat_a_draw_says_so():
    """Every cell promoted in the same weeks leaves no clean wide window."""
    panel = _panel(n_pairs=20, treated_pairs=20, n_weeks=26, campaign=(18, 21), seed=36)
    with pytest.raises(InsufficientPlaceboError, match="no week can seat"):
        _band(panel, n_cells=20, seed=36, pool="ever_treated",
              campaign_length=4, horizon_weeks=3, min_history_weeks=13)


def test_a_pool_too_small_for_the_campaign_raises():
    panel = _panel(n_pairs=30, treated_pairs=10, seed=35)
    with pytest.raises(InsufficientPlaceboError, match="pool holds only"):
        _band(panel, n_cells=20, seed=35, pool="ever_treated")


# --- the draws ----------------------------------------------------------------


def test_every_draw_has_the_shape_it_was_asked_for():
    panel = _panel(seed=3)
    draws, diag = _band(panel, n_cells=8)

    assert len(draws) == MIN_WINDOWS
    assert (draws["cells"] == 8).all()
    assert diag["shape"]["cells_per_draw"] == 8
    assert diag["shape"]["campaign_length_weeks"] == 4
    assert diag["shape"]["horizon_weeks"] == 3
    assert diag["shape"]["window_length_weeks"] == 7
    assert "different statistic" in diag["shape"]["why_matched"]


def test_the_band_is_wider_when_the_draws_are_smaller():
    """Size matching is not decoration: the band's width depends on it.

    A band drawn on fewer cells is wider *per cell* and narrower in absolute
    units, and comparing a campaign's total against either mismatched version
    would be arithmetic between different statistics.
    """
    panel = _panel(n_pairs=80, treated_pairs=20, seed=4)
    small, _ = _band(panel, n_cells=4, seed=4)
    large, _ = _band(panel, n_cells=16, seed=4)

    assert small["gross"].std() < large["gross"].std()
    # Per cell it goes the other way, which is the whole point.
    assert (small["gross"] / 4).std() > (large["gross"] / 16).std()


def test_the_truth_in_every_draw_is_zero_so_the_band_should_contain_it():
    panel = _panel(seed=5)
    _, diag = _band(panel, seed=5)
    band = diag["band"]

    assert band["low"] < band["high"]
    assert band["zero_inside"]
    # And the centre is almost never exactly at zero — the runbook's line.
    assert not band["median_is_zero"]
    assert "where the estimator puts zero" in band["why_the_median_matters"]


def test_a_band_below_the_window_floor_raises():
    panel = _panel(seed=6)
    with pytest.raises(InsufficientPlaceboError, match="below the 300-window floor"):
        _band(panel, n_windows=50)


def test_asking_for_more_cells_than_were_never_treated_raises():
    panel = _panel(n_pairs=30, treated_pairs=25, seed=7)
    with pytest.raises(InsufficientPlaceboError, match="size-matched draw is impossible"):
        _band(panel, n_cells=20)


def test_a_window_that_does_not_fit_raises():
    panel = _panel(n_weeks=20, seed=8)
    with pytest.raises(InsufficientPlaceboError, match="no valid start"):
        _band(panel, campaign_length=8, horizon_weeks=8, min_history_weeks=13)


def test_the_same_seed_gives_the_same_band():
    panel = _panel(seed=9)
    a, _ = _band(panel, seed=3)
    b, _ = _band(panel, seed=3)
    assert np.allclose(a["gross"], b["gross"])


def test_net_is_gross_plus_post_in_the_placebo_too():
    panel = _panel(seed=10)
    draws, _ = _band(panel, seed=10)
    assert np.allclose(draws["net"], draws["gross"] + draws["post"])


# --- the helper ---------------------------------------------------------------


def test_inside_band_reports_where_the_estimate_sits():
    panel = _panel(seed=11)
    draws, diag = _band(panel, seed=11)
    band = diag["band"]

    inside, evidence = inside_band(band["median"], draws)
    assert inside is True
    assert evidence["p_value"] > 0.5
    assert "cannot separate the promotion" in evidence["meaning"]

    outside, evidence_out = inside_band(band["high"] * 100 + 1000, draws)
    assert outside is False
    assert evidence_out["p_value"] < 0.05
    assert "not sufficient" in evidence_out["meaning"]


def test_the_helper_accepts_a_summarised_band_without_the_draws():
    panel = _panel(seed=12)
    draws, _ = _band(panel, seed=12)
    band = summarise_band(draws)

    inside, evidence = inside_band(band["median"], band)
    assert inside is True
    # No draws, so no empirical p — absent rather than invented.
    assert "p_value" not in evidence


def test_the_band_can_be_written_and_read(tmp_path):
    import json

    panel = _panel(seed=13)
    draws, diag = _band(panel, seed=13)
    paths = write_band(draws, diag, tmp_path)

    assert pd.read_parquet(paths["draws"]).shape[0] == MIN_WINDOWS
    assert json.loads(paths["band"].read_text())["stage"] == "placebo_band"


# --- the gate, per the gate-authoring skill -----------------------------------


def test_placebo_overlap_fires_through_run_audit():
    """An estimate inside the band must refuse, with the right wording."""
    panel = _panel(seed=14)
    draws, diag = _band(panel, seed=14)
    estimate = diag["band"]["median"]

    results, audit = run_audit(
        CampaignSpec(name="c"),
        panel,
        run_overlap=False,
        stop_on_refuse=False,
        estimate=estimate,
        placebo=draws,
    )
    placebo = next(r for r in results if r.gate == "placebo")

    assert placebo.reason_code == "PLACEBO_OVERLAP"
    assert placebo.status == "refuse"
    assert "PLACEBO_OVERLAP" in audit["refusals"]
    assert audit["verdict"] == "not identified"
    # The distinction this project turns on.
    assert "not evidence that the promotion did nothing" in placebo.message


def test_placebo_overlap_does_not_fire_on_an_estimate_outside_the_band():
    panel = _panel(seed=15)
    draws, diag = _band(panel, seed=15)
    estimate = diag["band"]["high"] * 100 + 1000

    results, audit = run_audit(
        CampaignSpec(name="c"),
        panel,
        run_overlap=False,
        stop_on_refuse=False,
        estimate=estimate,
        placebo=draws,
    )
    placebo = next(r for r in results if r.gate == "placebo")

    assert placebo.reason_code is None
    assert placebo.status == "pass"
    assert "PLACEBO_OVERLAP" not in audit["refusals"]
    # A pass here is necessary and not sufficient, and says so.
    assert "not sufficient" in placebo.message
    assert "optimistic" in placebo.message


def test_without_an_estimate_the_gate_says_the_comparison_was_not_made():
    """An absent refusal must not read as a clean bill."""
    panel = _panel(seed=16)
    results, _ = run_audit(
        CampaignSpec(name="c"), panel, run_overlap=False, stop_on_refuse=False
    )
    placebo = next(r for r in results if r.gate == "placebo")

    assert placebo.status == "pass"
    assert placebo.detail["compared"] is False
    assert "has not been shown to be distinguishable" in placebo.message


def test_the_placebo_gate_is_in_the_running_order():
    assert "placebo" in GATE_ORDER
    assert GATE_ORDER[-1] == "placebo"


# --- the real panel -----------------------------------------------------------


@real_data
def test_the_real_band_is_built_and_stored():
    """The deliverable: 300 windows on the shipped model, written to disk."""
    campaign = LiftCampaign(
        name="demo",
        commodity="WATER - CARBONATED/FLVRD DRINK",
        product=834117,
        weeks=(80, 83),
    )
    draws, diag = band_for_campaign(
        campaign, BASELINE, PANEL, CYCLES, n_cells=101, n_windows=MIN_WINDOWS
    )

    assert len(draws) == MIN_WINDOWS
    assert diag["shape"]["cells_per_draw"] == 101
    assert diag["shape"]["campaign_length_weeks"] == 4
    assert diag["shape"]["horizon_weeks"] == 6
    assert diag["pool"]["panel_pairs_never_treated"] == 14_601
    assert diag["band"]["low"] < diag["band"]["high"]
    assert diag["model"]["target"] == "poisson"
    assert "in-sample" in diag["in_sample_caveat"] or "trains on" in (
        diag["in_sample_caveat"]
    )

    write_band(draws, diag)
    assert Path("data/interim/placebo_band.parquet").exists()


@real_data
def test_the_real_campaign_is_compared_against_its_own_band():
    """The bottled-water campaign, against a band matched to its shape."""
    from promo.lift import estimate_lift

    campaign = LiftCampaign(
        name="demo",
        commodity="WATER - CARBONATED/FLVRD DRINK",
        product=834117,
        weeks=(80, 83),
    )
    _, lift_diag = estimate_lift(campaign, BASELINE, PANEL, CYCLES)
    pairs = lift_diag["cells"]["pairs"]
    draws, _ = band_for_campaign(
        campaign, BASELINE, PANEL, CYCLES, n_cells=pairs, n_windows=MIN_WINDOWS
    )

    gross = lift_diag["lift"]["gross_incremental"]
    inside, evidence = inside_band(gross, draws)

    # The estimate is a couple of units on a band hundreds wide — it cannot
    # be seen. Asserted because it is the finding, not an accident of seed.
    assert inside is True
    assert evidence["p_value"] > DEFAULT_ALPHA
