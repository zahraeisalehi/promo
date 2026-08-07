"""Tests for Task 7.1 — ranking, the response curve, and the MDE calculator.

The three parts fail in different ways, so they are tested for different things.

Shrinkage is tested for **the property it exists to produce**: a lucky outlier
with a wide standard error must be pulled in further than a precise one, and the
published number must be the shrunk one. Testing the arithmetic alone would pass
an implementation that shrank the wrong way.

The response curve is tested for **what it refuses**: a turning point outside
the depths actually run, or fitted from too few of them, is an extrapolation and
must be labelled rather than returned bare.

The MDE calculator is tested for **the shape of the trade** — minimised at a
balanced split, rising steeply as the holdout shrinks — because that shape is
the recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from promo.decide import (
    james_stein_shrink,
    mde,
    mde_grid,
    rank_campaigns,
    response_curve,
    write_diagnostics,
)


def _separable() -> pd.DataFrame:
    """Real spread, tight errors: the campaigns can be told apart."""
    return pd.DataFrame(
        {
            "campaign": list("ABCDE"),
            "lift": [100.0, 60.0, 55.0, 50.0, 45.0],
            "se": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


# --- shrinkage ----------------------------------------------------------------


def test_a_noisy_outlier_is_pulled_in_further_than_a_precise_one():
    """The property shrinkage exists for, not just its arithmetic."""
    # Real spread, so tau2 is positive and shrinkage is live; the two
    # outliers sit the same distance from the pack but differ in precision.
    estimates = np.array([100.0, 100.0, 50.0, 50.0, 50.0, 50.0])
    errors = np.array([30.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    shrunk, diag = james_stein_shrink(estimates, errors)
    assert not diag["collapsed_to_grand_mean"]

    pulled_noisy = 100.0 - shrunk[0]
    pulled_precise = 100.0 - shrunk[1]
    assert pulled_noisy > pulled_precise


def test_shrinkage_moves_every_estimate_towards_the_grand_mean():
    frame = _separable()
    shrunk, diag = james_stein_shrink(frame["lift"], frame["se"])
    grand = diag["grand_mean"]

    for raw, pulled in zip(frame["lift"], shrunk, strict=True):
        assert abs(pulled - grand) <= abs(raw - grand) + 1e-9
        # and never overshoots past the mean
        assert (raw - grand) * (pulled - grand) >= -1e-9


def test_a_zero_standard_error_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        james_stein_shrink(np.array([1.0, 2.0]), np.array([1.0, 0.0]))


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="differ in shape"):
        james_stein_shrink(np.array([1.0, 2.0]), np.array([1.0]))


# --- ranking ------------------------------------------------------------------


def test_the_published_expectation_is_the_shrunk_value_not_the_raw_one():
    """Publishing the raw estimate for the shrunk winner restores the bias."""
    ranked, diag = rank_campaigns(_separable())

    top = ranked.iloc[0]
    assert top["expectation"] == pytest.approx(top["shrunk"])
    assert top["expectation"] != pytest.approx(top["lift"])
    assert top["expectation"] < top["lift"]      # the winner is pulled down
    assert diag["published_as_expectation"] == "shrunk"
    assert diag["ranked_on"] == "shrunk"
    assert "underperforms its own forecast" in diag["publish_rule"]


def test_the_ranking_is_ordered_by_the_shrunk_estimate():
    ranked, _ = rank_campaigns(_separable())
    assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
    assert ranked["shrunk"].is_monotonic_decreasing


def test_a_lucky_outlier_can_lose_the_top_spot_to_a_steadier_campaign():
    """The winner's curse, made concrete."""
    frame = pd.DataFrame(
        {
            "campaign": ["lucky", "steady", "c", "d", "e", "f"],
            "lift": [90.0, 70.0, 50.0, 48.0, 46.0, 44.0],
            "se": [30.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    ranked, diag = rank_campaigns(frame)

    assert diag["raw_top"] == "lucky"
    assert diag["shrunk_top"] == "steady"
    assert diag["top_changed"] is True
    # The lucky campaign is shrunk far harder than the steady one.
    shrinkage = ranked.set_index("campaign")["shrinkage"]
    assert shrinkage["lucky"] > shrinkage["steady"]


def test_when_the_spread_is_all_noise_no_ranking_is_published():
    """A ranking that is not there must not look like one.

    With the spread smaller than sampling noise predicts, every shrunk value is
    the grand mean and the order between them is floating-point dust. The
    campaigns tie at rank 1 rather than being numbered 1..n.
    """
    frame = pd.DataFrame(
        {
            "campaign": list("ABCDE"),
            "lift": [100.0, 60.0, 55.0, 50.0, 45.0],
            "se": [40.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    ranked, diag = rank_campaigns(frame)

    assert diag["shrinkage"]["collapsed_to_grand_mean"] is True
    assert diag["ranking_meaningful"] is False
    assert diag["top_changed"] is False
    assert list(ranked["rank"]) == [1, 1, 1, 1, 1]
    assert not ranked["rank_meaningful"].any()
    assert "floating-point dust" in diag["why_not_rankable"]


def test_a_missing_column_is_named():
    with pytest.raises(KeyError, match="'se' is not a column"):
        rank_campaigns(_separable().drop(columns=["se"]))


# --- the response curve -------------------------------------------------------


def _concave(depths, peak=0.25) -> pd.DataFrame:
    lift = [-100 * (d - peak) ** 2 + 30 for d in depths]
    return pd.DataFrame(
        {"COMMODITY_DESC": ["SOUP"] * len(depths), "depth": depths, "lift": lift}
    )


def test_the_crossing_is_where_marginal_return_reaches_zero():
    curve, diag = response_curve(_concave([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]))
    row = curve.iloc[0]

    assert row["peak_depth"] == pytest.approx(0.25, abs=1e-6)
    assert bool(row["concave"]) is True
    assert bool(row["within_support"]) is True
    assert row["reason"] is None
    assert diag["groups_with_usable_crossing"] == 1


def test_a_crossing_outside_the_observed_depths_is_labelled_an_extrapolation():
    """A concave fit always has a turning point; support is what makes it real."""
    curve, diag = response_curve(_concave([0.05, 0.08, 0.11, 0.14], peak=0.60))
    row = curve.iloc[0]

    assert row["peak_depth"] > row["depth_max"]
    assert bool(row["within_support"]) is False
    assert "extrapolation" in row["reason"]
    assert diag["groups_with_usable_crossing"] == 0


def test_too_few_distinct_depths_returns_no_crossing_and_says_why():
    curve, _ = response_curve(_concave([0.10, 0.20, 0.30]), min_points=4)
    row = curve.iloc[0]

    assert row["peak_depth"] is None
    assert "three determine a parabola exactly" in row["reason"]


def test_a_convex_fit_is_a_minimum_and_is_labelled_as_one():
    frame = pd.DataFrame(
        {
            "COMMODITY_DESC": ["SOUP"] * 5,
            "depth": [0.05, 0.10, 0.15, 0.20, 0.25],
            "lift": [30.0, 18.0, 14.0, 18.0, 30.0],
        }
    )
    curve, diag = response_curve(frame)
    row = curve.iloc[0]

    assert bool(row["concave"]) is False
    assert "minimum rather than a point of diminishing return" in row["reason"]
    assert diag["groups_with_usable_crossing"] == 0


def test_the_curve_says_it_is_observational():
    _, diag = response_curve(_concave([0.05, 0.10, 0.15, 0.20, 0.25]))
    assert "not an experiment" in diag["observational"]
    assert "chosen for reasons that may themselves relate to lift" in (
        diag["observational"]
    )


def test_each_commodity_gets_its_own_curve():
    soup = _concave([0.05, 0.10, 0.15, 0.20, 0.25], peak=0.20)
    bread = _concave([0.05, 0.10, 0.15, 0.20, 0.25], peak=0.15)
    bread["COMMODITY_DESC"] = "BREAD"
    curve, _ = response_curve(pd.concat([soup, bread], ignore_index=True))

    assert len(curve) == 2
    peaks = curve.set_index("COMMODITY_DESC")["peak_depth"]
    assert peaks["SOUP"] == pytest.approx(0.20, abs=1e-6)
    assert peaks["BREAD"] == pytest.approx(0.15, abs=1e-6)


# --- the MDE calculator -------------------------------------------------------


def test_the_detectable_effect_is_smallest_at_a_balanced_split():
    table, diag = mde_grid(sigma=1.0, clusters=100, cluster_size=50, icc=0.05)

    assert diag["cheapest_to_detect"] == 0.50
    assert table["mde"].idxmin() == table.index[-1]
    assert table["vs_balanced"].iloc[-1] == pytest.approx(1.0)


def test_a_small_holdout_costs_what_the_runbook_says_it_costs():
    """5% needs about 2.3x the effect, 20% about 1.25x."""
    table, _ = mde_grid(sigma=1.0, clusters=100, cluster_size=50, icc=0.05)
    ratios = table.set_index("holdout_fraction")["vs_balanced"]

    assert ratios[0.05] == pytest.approx(2.29, abs=0.05)
    assert ratios[0.20] == pytest.approx(1.25, abs=0.02)
    assert ratios.is_monotonic_decreasing


def test_clustering_shrinks_the_effective_sample():
    """Ignoring the design effect promises more power than the test has."""
    _, independent = mde_grid(sigma=1.0, clusters=100, cluster_size=50, icc=0.0)
    _, clustered = mde_grid(sigma=1.0, clusters=100, cluster_size=50, icc=0.05)

    assert independent["design_effect"] == pytest.approx(1.0)
    assert clustered["design_effect"] == pytest.approx(1 + 49 * 0.05)
    assert clustered["effective_sample"] < independent["effective_sample"]
    assert clustered["balanced_mde"] > independent["balanced_mde"]


def test_more_clusters_beat_bigger_clusters_under_correlation():
    """Where the design effect bites: 100x50 detects less than 500x10."""
    few_big = mde(1.0, clusters=100, cluster_size=50, icc=0.10, holdout_fraction=0.5)
    many_small = mde(1.0, clusters=500, cluster_size=10, icc=0.10, holdout_fraction=0.5)
    assert many_small < few_big


def test_a_degenerate_split_is_refused():
    for fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            mde(1.0, 100, 50, 0.05, fraction)


def test_an_impossible_icc_is_refused():
    with pytest.raises(ValueError, match="icc must be between 0 and 1"):
        mde(1.0, 100, 50, 1.5, 0.5)


def test_the_trade_is_stated_in_words():
    _, diag = mde_grid(sigma=1.0, clusters=100, cluster_size=50, icc=0.05)
    assert "cheap in forgone promoted sales" in diag["the_trade"]
    assert "more power than it has" in diag["design_effect_note"]


def test_write_diagnostics_round_trips(tmp_path):
    import json

    _, diag = rank_campaigns(_separable())
    path = write_diagnostics(diag, tmp_path / "rank.json")
    assert json.loads(path.read_text())["stage"] == "rank_campaigns"
