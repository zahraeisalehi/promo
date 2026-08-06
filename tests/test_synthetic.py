"""Tests for Task 4.4 — the generator, the harness, and what they found.

Two kinds of test live here and they are not the same kind of claim.

**Correctness tests** pin the generator and the harness: the truth is exact, the
features match Phase 2.6's definitions, the exposure patterns are what they say,
and the process is the one the plan asked for rather than something a tree can
memorise. These should hold forever.

**Characterisation tests** pin what the pipeline currently *does*: it recovers a
known effect cleanly on a dense panel and badly on a sparse one. They encode a
defect, not a requirement. If the estimator is fixed they will fail, and the
right response is to update them together with the record in
`docs/data_findings.md` — not to loosen them.

The full recovery grid is marked `slow`: it fits a few hundred models and takes
minutes. `pytest -m slow tests/test_synthetic.py` runs it and rewrites the
report in `data/interim/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.baseline import DEFAULT_TARGET
from tests.synthetic import (
    CAMPAIGN_WEEKS,
    CONTEMPORANEOUS_BLOCK,
    EXPOSURE_PATTERNS,
    HORIZON_WEEKS,
    TRUE_EFFECTS,
    exposure_grid,
    feature_set,
    recover,
    recovery_grid,
    simulate,
    sparsity_grid,
)

OUT = Path("data/interim")


# --- the generator ------------------------------------------------------------


def test_the_truth_is_exact_and_only_the_display_is_in_it():
    panel, truth = simulate(tau=0.25, seed=0)

    treated_selling = panel["treated"] & (panel["units"] > 0)
    ratio = (
        panel.loc[treated_selling, "units"] / panel.loc[treated_selling, "base_units"]
    )
    assert np.allclose(ratio, 1.25)
    # Untreated weeks are their own counterfactual, to the last bit.
    untreated = ~panel["treated"]
    assert np.allclose(
        panel.loc[untreated, "units"], panel.loc[untreated, "base_units"]
    )
    assert truth["gross_incremental"] == pytest.approx(
        float(panel.loc[panel["treated"], "true_residual"].sum())
    )


def test_the_counterfactual_keeps_the_mailer_that_ran():
    """Settled decision 8, in the generator: the mailer is in both worlds."""
    panel, _ = simulate(tau=0.0, mailer_effect=0.5, seed=1)
    assert np.allclose(panel["units"], panel["base_units"])

    mailed = panel["in_mailer"] & (panel["units"] > 0)
    assert panel.loc[mailed, "units"].mean() > panel.loc[~mailed, "units"].mean()


def test_the_mailer_shares_match_what_task_33_measured():
    panel, _ = simulate(seed=2)
    control = panel.loc[~panel["treated"], "in_mailer"].mean()
    treated = panel.loc[panel["treated"], "in_mailer"].mean()
    assert control == pytest.approx(0.13, abs=0.02)
    assert treated == pytest.approx(0.40, abs=0.03)


def test_no_sale_weeks_reproduce_the_decision_9_null_pattern():
    panel, _ = simulate(no_sale_rate=0.15, seed=3)

    no_sale = panel["units"] == 0
    assert no_sale.mean() == pytest.approx(0.15, abs=0.02)
    # Null exactly where nothing sold — the shape of the leak decision 9 names.
    assert (panel.loc[no_sale, "price_rel_category"].isna()).all()
    assert (panel.loc[~no_sale, "price_rel_category"].notna()).all()
    # And the replacement is available where history exists, which is the point
    # of having a replacement at all.
    late = panel["WEEK_NO"] > 10
    assert panel.loc[late, "price_rel_category_lag"].notna().mean() > 0.95


def test_the_process_is_not_tree_shaped():
    """Multiplicative seasonality and AR(1) noise, as the plan requires."""
    panel, _ = simulate(tau=0.0, no_sale_rate=0.0, ar_phi=0.7, seed=4)
    one = panel[(panel["PRODUCT_ID"] == 1000) & (panel["STORE_ID"] == 300)]
    logs = np.log(one.sort_values("WEEK_NO")["units"].to_numpy())

    # AR(1): the series is strongly autocorrelated at lag 1 in logs.
    deviations = logs - logs.mean()
    rho = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]
    assert rho > 0.4

    # Seasonality: a 52-week cycle is present across the panel, and it is
    # multiplicative, so it shows up in logs as a sinusoid rather than a step.
    weekly = np.log(panel.groupby("WEEK_NO")["units"].mean().to_numpy())
    weeks = np.arange(1, len(weekly) + 1)
    design = np.column_stack(
        [np.ones_like(weeks), np.sin(2 * np.pi * weeks / 52), np.cos(2 * np.pi * weeks / 52)]
    )
    fitted = design @ np.linalg.lstsq(design, weekly, rcond=None)[0]
    explained = 1 - ((weekly - fitted) ** 2).sum() / ((weekly - weekly.mean()) ** 2).sum()
    assert explained > 0.3


def test_the_features_match_the_phase_2_6_definitions():
    panel, _ = simulate(seed=5)
    one = panel[(panel["PRODUCT_ID"] == 1000) & (panel["STORE_ID"] == 300)].sort_values(
        "WEEK_NO"
    )
    units = one["units"].astype("float64")

    assert np.allclose(one["units_lag_1"], units.shift(1), equal_nan=True)
    assert np.allclose(one["units_lag_52"], units.shift(52), equal_nan=True)
    assert np.allclose(
        one["units_roll_mean_4"],
        units.shift(1).rolling(4, min_periods=1).mean(),
        equal_nan=True,
    )
    # category_units_ex_focal excludes the focal product by construction.
    store_week = panel[(panel["STORE_ID"] == 300) & (panel["WEEK_NO"] == 40)]
    focal = store_week[store_week["PRODUCT_ID"] == 1000].iloc[0]
    assert focal["category_units_ex_focal"] == pytest.approx(
        store_week["units"].sum() - focal["units"]
    )


def test_traffic_moves_with_the_promotion_but_is_mostly_not_the_focal_product():
    """A mediator has to mediate, and a leak has to stay a trickle."""
    panel, _ = simulate(tau=0.30, seed=6)
    share = (0.5 * panel["units"] / panel["store_traffic"]).mean()
    assert 0.005 < share < 0.05

    assert "store_traffic" in CONTEMPORANEOUS_BLOCK
    assert set(feature_set(False)).isdisjoint(CONTEMPORANEOUS_BLOCK)
    assert set(CONTEMPORANEOUS_BLOCK) <= set(feature_set(True))


def test_exposure_patterns_are_what_they_claim():
    full, truth_full = simulate(exposure="full", seed=7)
    assert set(truth_full["exposure_histogram"]) == {4}

    _, truth_uneven = simulate(exposure="uneven", seed=7)
    histogram = truth_uneven["exposure_histogram"]
    assert set(histogram) <= {1, 2, 3, 4}
    total = sum(histogram.values())
    # The real panel's spread: 43.9% of cells run one week of four.
    assert histogram[1] / total == pytest.approx(0.44, abs=0.15)
    assert histogram.get(4, 0) / total < histogram[1] / total

    assert full["treated"].sum() > sum(
        v * k for k, v in histogram.items()
    ) * 0.9  # full exposure treats strictly more cell-weeks


def test_a_campaign_shorter_than_the_pattern_clamps_instead_of_raising():
    """Regression: drawing four treated weeks from a two-week campaign raised."""
    panel, truth = simulate(campaign=(71, 72), exposure="uneven", seed=8)
    assert set(truth["exposure_histogram"]) <= {1, 2}
    assert panel.loc[panel["treated"], "WEEK_NO"].between(71, 72).all()


# --- the harness ---------------------------------------------------------------


def test_a_replication_scores_itself_against_the_truth():
    row = recover(tau=0.15, seed=0, no_sale_rate=0.0)

    for field in ("gross_incremental", "post_window_residual", "net_incremental"):
        assert row[f"{field}_error"] == pytest.approx(
            row[f"{field}_hat"] - row[f"{field}_true"]
        )
        assert isinstance(row[f"{field}_covered"], bool)
    assert row["tau_hat"] is not None
    # The harness defaults follow the shipped model: identity off (decision 10),
    # mediator block off (decision 13).
    assert row["identity"] is False and row["contemporaneous"] is False


def test_with_no_payback_the_truth_has_no_trough():
    _, truth = simulate(tau=0.20, payback=0.0, seed=9)
    assert truth["post_window_residual"] == pytest.approx(0.0)
    assert truth["net_incremental"] == pytest.approx(truth["gross_incremental"])

    _, with_payback = simulate(tau=0.20, payback=0.25, seed=9)
    assert with_payback["post_window_residual"] < 0
    assert with_payback["net_incremental"] < with_payback["gross_incremental"]


# --- what the pipeline actually recovers ---------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.30])
def test_recovery_is_clean_on_a_dense_panel(tau: float):
    """The estimator works where its own arithmetic is not against it.

    With no zero-inflation the `log1p`/`expm1` round trip is close to unbiased,
    the rollout does not compound, and both a null effect and a large one come
    back near the truth. This is the control condition for the failure below.
    """
    rows = [recover(tau=tau, seed=seed, no_sale_rate=0.0) for seed in (0, 1)]
    recovered = np.mean([r["tau_hat"] for r in rows])

    assert recovered == pytest.approx(tau, abs=0.12)
    assert all(r["gross_incremental_covered"] for r in rows)


@pytest.mark.parametrize("no_sale_rate", [0.10, 0.20])
def test_the_default_target_recovers_zero_on_a_sparse_panel(no_sale_rate: float):
    """Phase 4's gate condition, at the sparsity that used to break it.

    A true effect of zero has to come back as approximately zero. Under the
    `log1p` target it came back as +0.28 of counterfactual units at 10% no-sale
    weeks and +0.44 at 20%; the Poisson default brings both inside 0.10.
    """
    rows = [recover(tau=0.0, seed=s, no_sale_rate=no_sale_rate) for s in (0, 1, 2)]
    bias = np.mean([r["gross_error_share"] for r in rows])

    assert abs(bias) < 0.10
    assert all(r["gross_incremental_covered"] for r in rows)


def test_the_log1p_path_still_carries_the_defect_it_was_kept_for():
    """Why the default changed, kept runnable so the claim stays checkable.

    `expm1(E[log1p(y)])` understates a conditional mean with mass at zero, and
    the rollout compounds it. This is the comparison that moved the default, not
    a requirement of the pipeline — if someone repairs the log1p path, this test
    fails and should be retired along with the target.
    """
    dense = [recover(tau=0.0, seed=s, no_sale_rate=0.0, target="log1p") for s in (0, 1)]
    sparse = [
        recover(tau=0.0, seed=s, no_sale_rate=0.10, target="log1p") for s in (0, 1)
    ]
    fixed = [recover(tau=0.0, seed=s, no_sale_rate=0.10) for s in (0, 1)]

    dense_bias = np.mean([r["gross_error_share"] for r in dense])
    sparse_bias = np.mean([r["gross_error_share"] for r in sparse])
    fixed_bias = np.mean([r["gross_error_share"] for r in fixed])

    # The defect is a function of sparsity, which is why a dense panel hid it.
    assert abs(dense_bias) < 0.05
    assert sparse_bias > 0.15
    assert abs(fixed_bias) < abs(sparse_bias) / 3

    # The interval covered the truth even when the point estimate was wrong by
    # a quarter of the counterfactual. A band wide enough to always be right is
    # how a broken point estimate passes a coverage check.
    assert all(r["gross_incremental_covered"] for r in sparse)


# --- the full report -----------------------------------------------------------


@pytest.mark.slow
def test_the_recovery_report_runs_and_is_written():
    """The deliverable: bias and coverage across every axis the task owes."""
    grid, grid_summary = recovery_grid(seeds=(0, 1, 2, 3, 4))
    exposure, exposure_summary = exposure_grid(seeds=(0, 1, 2, 3, 4))
    sparsity, sparsity_summary = sparsity_grid(seeds=(0, 1, 2))

    assert set(grid["tau"]) == set(TRUE_EFFECTS)
    assert set(grid["identity"]) == {True, False}
    assert set(grid["contemporaneous"]) == {True, False}
    assert set(exposure["exposure"]) == set(EXPOSURE_PATTERNS)

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            grid.assign(grid="two_by_two"),
            exposure.assign(grid="exposure"),
            sparsity.assign(grid="sparsity"),
        ],
        ignore_index=True,
    )
    frame.to_parquet(OUT / "synthetic_recovery.parquet", index=False)
    (OUT / "synthetic_recovery.json").write_text(
        json.dumps(
            {
                "stage": "synthetic_recovery",
                # The target is stamped on the report because it went stale
                # once: a grid measured under the log1p defect reads exactly
                # like one measured under the shipped model.
                "target": DEFAULT_TARGET,
                "campaign_weeks": list(CAMPAIGN_WEEKS),
                "horizon_weeks": HORIZON_WEEKS,
                "true_effects": list(TRUE_EFFECTS),
                "two_by_two": grid_summary,
                "exposure": exposure_summary,
                "sparsity": sparsity_summary,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    assert (OUT / "synthetic_recovery.parquet").exists()
