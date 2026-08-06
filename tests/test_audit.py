"""Tests for Task 3.1, the variation audit.

Fixture panels are built so that each axis's classification is obvious by
inspection, including the case the module exists to catch: an axis whose levels
are mostly mixed while the mixed levels hold almost no demand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.audit import (
    AXES,
    DEFAULT_COVARIATES,
    LEAKAGE_AUC,
    MEANINGFUL_MIXED_SHARE,
    PROPENSITY_HIGH,
    PROPENSITY_LOW,
    UnobservedRowsError,
    collisions,
    horizon_check,
    overlap,
    variation_axes,
    write_diagnostics,
)

PANEL = Path("data/interim/panel.parquet")

_NO_DATA = not PANEL.exists()


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast
    pass excludes it with -m "not heavy", and it is skipped outright when
    the artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="run Phase 2 first")(
        pytest.mark.heavy(fn)
    )


def _panel(rows: list[dict]) -> pd.DataFrame:
    base = {
        "PRODUCT_ID": 1,
        "STORE_ID": 1,
        "WEEK_NO": 1,
        "units": 1,
        "treated": False,
        "treatment_observed": True,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _cell(axes: pd.DataFrame, axis: str, klass: str) -> pd.Series:
    return axes[(axes["axis"] == axis) & (axes["class"] == klass)].iloc[0]


def test_every_axis_and_class_is_reported() -> None:
    axes, _ = variation_axes(_panel([{"treated": True}]))
    assert set(axes["axis"]) == set(AXES)
    assert set(axes["class"]) == {"fully_treated", "fully_untreated", "mixed"}
    assert len(axes) == 3 * 3


def test_classes_partition_the_panel() -> None:
    """Every row and unit lands in exactly one class per axis."""
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "units": 5, "treated": True},
            {"PRODUCT_ID": 1, "WEEK_NO": 2, "units": 3, "treated": False},
            {"PRODUCT_ID": 2, "WEEK_NO": 1, "units": 7, "treated": False},
        ]
    )
    axes, diag = variation_axes(panel)
    for axis in AXES:
        block = axes[axes["axis"] == axis]
        assert block["rows"].sum() == diag["panel"]["rows"]
        assert block["units"].sum() == diag["panel"]["units"]
        assert block["units_share"].sum() == pytest.approx(1.0)


def test_a_product_treated_in_some_weeks_only_is_mixed() -> None:
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "units": 5, "treated": True},
            {"PRODUCT_ID": 1, "WEEK_NO": 2, "units": 5, "treated": False},
        ]
    )
    axes, _ = variation_axes(panel)
    assert _cell(axes, "product", "mixed")["levels"] == 1
    assert _cell(axes, "product", "mixed")["units"] == 10


def test_a_product_always_treated_is_fully_treated() -> None:
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": w, "units": 4, "treated": True}
            for w in (1, 2, 3)
        ]
    )
    axes, _ = variation_axes(panel)
    assert _cell(axes, "product", "fully_treated")["levels"] == 1
    assert _cell(axes, "product", "mixed")["levels"] == 0
    assert _cell(axes, "product", "fully_treated")["units_share"] == pytest.approx(1.0)


def test_a_product_never_treated_is_fully_untreated() -> None:
    panel = _panel([{"PRODUCT_ID": 1, "WEEK_NO": w} for w in (1, 2)])
    axes, _ = variation_axes(panel)
    assert _cell(axes, "product", "fully_untreated")["levels"] == 1
    assert _cell(axes, "product", "fully_untreated")["units_share"] == pytest.approx(1.0)


def test_axes_are_classified_independently() -> None:
    """One product, two stores; only one store ever treated.

    The product is mixed (it is treated somewhere and not elsewhere), store 1 is
    fully treated, store 2 fully untreated, and the week is mixed.
    """
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 1, "units": 6, "treated": True},
            {"PRODUCT_ID": 1, "STORE_ID": 2, "WEEK_NO": 1, "units": 4},
        ]
    )
    axes, _ = variation_axes(panel)
    assert _cell(axes, "product", "mixed")["levels"] == 1
    assert _cell(axes, "store", "fully_treated")["levels"] == 1
    assert _cell(axes, "store", "fully_untreated")["levels"] == 1
    assert _cell(axes, "store", "mixed")["levels"] == 0
    assert _cell(axes, "week", "mixed")["levels"] == 1


def test_units_weighting_can_disagree_with_level_counts() -> None:
    """The failure this module exists to catch.

    Nine products are mixed but tiny; one product is fully treated and carries
    almost all the demand. By level count the panel looks 90% mixed; by units it
    is 1% mixed, and the units figure is the one that matters.
    """
    rows = []
    for p in range(1, 10):
        rows += [
            {"PRODUCT_ID": p, "WEEK_NO": 1, "units": 1, "treated": True},
            {"PRODUCT_ID": p, "WEEK_NO": 2, "units": 1, "treated": False},
        ]
    rows.append({"PRODUCT_ID": 99, "WEEK_NO": 1, "units": 1782, "treated": True})

    axes, diag = variation_axes(_panel(rows))
    mixed = _cell(axes, "product", "mixed")
    assert mixed["levels_share"] == pytest.approx(0.9)
    assert mixed["units_share"] == pytest.approx(0.01)
    # The verdict must follow the unit mass, not the level count.
    assert "product" not in diag["usable_axes"]


def test_threshold_is_a_parameter_and_is_recorded() -> None:
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "units": 1, "treated": True},
            {"PRODUCT_ID": 1, "WEEK_NO": 2, "units": 1},
            {"PRODUCT_ID": 2, "WEEK_NO": 1, "units": 8, "treated": True},
        ]
    )
    _, diag_l = variation_axes(panel, meaningful_mixed_share=0.1)
    _, diag_s = variation_axes(panel, meaningful_mixed_share=0.5)
    assert "product" in diag_l["usable_axes"]      # 20% of units are mixed
    assert "product" not in diag_s["usable_axes"]
    assert diag_l["threshold"] == 0.1
    assert diag_s["threshold"] == 0.5


def test_best_axis_is_the_one_with_the_most_mixed_units() -> None:
    """Week mixes fully; store only partly; product not at all."""
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 1, "units": 5, "treated": True},
            {"PRODUCT_ID": 2, "STORE_ID": 2, "WEEK_NO": 1, "units": 4},
            {"PRODUCT_ID": 3, "STORE_ID": 2, "WEEK_NO": 1, "units": 1, "treated": True},
        ]
    )
    _, diag = variation_axes(panel)
    assert diag["mixed_units_share"]["week"] == pytest.approx(1.0)
    assert diag["mixed_units_share"]["store"] == pytest.approx(0.5)
    assert diag["mixed_units_share"]["product"] == pytest.approx(0.0)
    assert diag["best_axis"] == "week"
    assert diag["leading_axes"] == ["week"]


def test_a_tie_is_reported_rather_than_broken_by_axis_order() -> None:
    """Both rows sit in one store and one week, so those two axes tie at 1.0."""
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 1, "units": 5, "treated": True},
            {"PRODUCT_ID": 2, "STORE_ID": 1, "WEEK_NO": 1, "units": 5},
        ]
    )
    _, diag = variation_axes(panel)
    assert diag["leading_axes"] == ["store", "week"]
    assert diag["best_axis"] is None      # not silently the first in AXES order
    assert "tie" in diag["verdict"]


def test_no_mixed_mass_anywhere_is_reported_not_hidden() -> None:
    panel = _panel([{"PRODUCT_ID": p, "WEEK_NO": 1, "treated": True} for p in (1, 2)])
    _, diag = variation_axes(panel)
    assert diag["usable_axes"] == []
    assert diag["leading_axes"] == []
    assert diag["best_axis"] is None
    assert diag["verdict"] == "no axis has any mixed mass"


def test_unobserved_rows_are_refused_not_counted_as_untreated() -> None:
    """Task 2.5: absence outside the log's coverage is not a control."""
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "treated": True},
            {"PRODUCT_ID": 2, "WEEK_NO": 1, "treatment_observed": False},
        ]
    )
    with pytest.raises(UnobservedRowsError, match="unobserved, not untreated"):
        variation_axes(panel)

    # ...but the audit can still be run deliberately.
    _, diag = variation_axes(panel, require_observed=False)
    assert diag["panel"]["rows"] == 2


def test_treatment_column_is_a_parameter() -> None:
    panel = _panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "units": 5, "treated": True},
            {"PRODUCT_ID": 1, "WEEK_NO": 2, "units": 5, "treated": False},
        ]
    )
    panel["in_mailer"] = True  # mailer never varies here
    _, by_display = variation_axes(panel, treatment_column="treated")
    _, by_mailer = variation_axes(panel, treatment_column="in_mailer")
    assert by_display["mixed_units_share"]["product"] == pytest.approx(1.0)
    assert by_mailer["mixed_units_share"]["product"] == pytest.approx(0.0)
    assert by_mailer["treatment_column"] == "in_mailer"


def test_unknown_treatment_column_raises() -> None:
    with pytest.raises(KeyError):
        variation_axes(_panel([{"treated": True}]), treatment_column="on_promo")


# --------------------------------------------------------------------------
# The real panel.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_axes():
    return variation_axes(PANEL)


@pytest.fixture(scope="module")
def real_overlap():
    return overlap(PANEL)


@real_data
def test_real_panel_is_fully_inside_the_treatment_envelope(real_axes) -> None:
    """It must not raise: Task 2.6's scope is the envelope."""
    _, diag = real_axes
    assert diag["panel"]["products"] == 300
    assert diag["panel"]["stores"] == 115
    assert diag["panel"]["weeks"] == 93


@real_data
def test_real_classes_partition_the_panel(real_axes) -> None:
    axes, diag = real_axes
    for axis in AXES:
        block = axes[axes["axis"] == axis]
        assert block["rows"].sum() == diag["panel"]["rows"]
        assert block["units"].sum() == diag["panel"]["units"]


@real_data
def test_real_week_axis_mixes_almost_completely(real_axes) -> None:
    """Task 1.4 found 91.18% of weeks switch; on the scoped panel every week
    holds both kinds, so the contemporaneous cross-section is available."""
    axes, _ = real_axes
    assert _cell(axes, "week", "mixed")["units_share"] > 0.95


@real_data
def test_real_marginal_audit_cannot_separate_display_from_mailer() -> None:
    """The limitation, asserted so it cannot be forgotten and mis-cited.

    Task 1.4 chose display because it varies across stores *within a week for
    the same product* — 65.34% against mailer's 2.28%. That is a joint
    condition. This audit is marginal: a store counts as mixed if anything was
    treated there while anything else was not, which both treatments satisfy
    almost everywhere. The near-tie below is the evidence that decision 4 must
    keep resting on Task 1.4, not on this module.
    """
    _, display = variation_axes(PANEL, treatment_column="on_display")
    _, mailer = variation_axes(PANEL, treatment_column="in_mailer")
    assert display["mixed_units_share"]["store"] > 0.99
    assert mailer["mixed_units_share"]["store"] > 0.99
    assert display["limitation"].startswith("Marginal, not joint")


@real_data
def test_real_verdict_and_threshold_are_recorded(real_axes) -> None:
    _, diag = real_axes
    assert diag["threshold"] == MEANINGFUL_MIXED_SHARE
    assert diag["best_axis"] in AXES
    assert diag["usable_axes"]


@real_data
def test_real_diagnostics_are_json_serialisable(real_axes, tmp_path: Path) -> None:
    import json

    _, diag = real_axes
    path = write_diagnostics(diag, tmp_path / "variation.json")
    assert json.loads(path.read_text())["stage"] == "variation_axes"


# --------------------------------------------------------------------------
# Task 3.2 — overlap.
# --------------------------------------------------------------------------


def _overlap_panel(n: int, *, rule, seed: int = 0) -> pd.DataFrame:
    """A panel whose treatment is generated by `rule(covariates, rng)`."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "PRODUCT_ID": rng.integers(1, 30, n),
            "STORE_ID": rng.integers(1, 8, n),
            "WEEK_NO": rng.integers(1, 40, n),
            "units": rng.integers(0, 20, n),
            "treatment_observed": True,
        }
    )
    for column in DEFAULT_COVARIATES:
        frame[column] = rng.normal(size=n).astype("float32")
    frame["treated"] = rule(frame, rng)
    return frame


def _fast(frame, **kwargs):
    kwargs.setdefault("n_folds", 3)
    kwargs.setdefault("n_estimators", 30)
    return overlap(frame, **kwargs)


def test_random_treatment_is_unpredictable_and_well_overlapped() -> None:
    """Assignment independent of every covariate: AUC near a coin flip."""
    frame = _overlap_panel(3000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    _, diag = _fast(frame)
    assert 0.4 < diag["auc"] < 0.6
    assert diag["diagnosis"] == "WELL_OVERLAPPED"
    assert diag["propensity_extremes"]["outside_share"] < 0.05


def test_a_leaked_covariate_is_caught_and_named() -> None:
    """The failure the top-five table exists to diagnose.

    One covariate *is* the treatment. AUC goes near 1 and that feature carries
    nearly all the gain, which is the signature of leakage rather than of two
    genuinely different populations.
    """
    frame = _overlap_panel(3000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    frame["store_traffic"] = frame["treated"].astype("float32")  # a copy of y
    imp, diag = _fast(frame)
    assert diag["auc"] > 0.99
    assert diag["diagnosis"] == "LEAKAGE_SUSPECTED"
    assert diag["top_features"][0]["feature"] == "store_traffic"
    assert diag["top_feature_gain_share"] > 0.9
    assert imp.iloc[0]["feature"] == "store_traffic"


def test_separable_populations_are_not_called_leakage() -> None:
    """Many covariates each contributing: non-overlap, not a leaked column.

    The distinction matters because the responses are opposite — a refusal
    versus a bug — so a high AUC with diffuse importance must not be labelled
    leakage.
    """
    def rule(f, rng):
        score = sum(f[c] for c in DEFAULT_COVARIATES[:6])
        return score > score.median()

    frame = _overlap_panel(3000, rule=rule)
    _, diag = _fast(frame)
    assert diag["auc"] >= LEAKAGE_AUC
    assert diag["top_feature_gain_share"] < 0.5
    assert diag["diagnosis"] == "NON_OVERLAP_SUSPECTED"


def test_propensity_extremes_are_reported_on_both_sides() -> None:
    """Perfect separation, fitted long enough for the probabilities to saturate."""
    frame = _overlap_panel(3000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    frame["store_traffic"] = frame["treated"].astype("float32")
    _, diag = _fast(frame, n_estimators=300)
    e = diag["propensity_extremes"]
    assert e["low"] == PROPENSITY_LOW
    assert e["high"] == PROPENSITY_HIGH
    assert e["outside_share"] > 0.9
    assert e["below_low"] > 0 and e["above_high"] > 0
    assert e["outside_share"] == pytest.approx(
        e["below_low_share"] + e["above_high_share"]
    )


def test_extreme_share_depends_on_the_fit_and_says_so() -> None:
    """The same data, two fit lengths, different extreme counts.

    Worth pinning: the propensity-extreme share is not a pure property of the
    data. A short fit shrinks every probability towards the base rate and can
    report zero rows outside the bounds on perfectly separable data.
    """
    frame = _overlap_panel(3000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    frame["store_traffic"] = frame["treated"].astype("float32")
    _, short = _fast(frame, n_estimators=30)
    _, long = _fast(frame, n_estimators=300)
    assert short["auc"] > 0.99 and long["auc"] > 0.99   # both separate perfectly
    assert short["propensity_extremes"]["outside_share"] == 0.0
    assert long["propensity_extremes"]["outside_share"] > 0.9
    assert "depend on the model" in short["propensity_extremes"]["caveat"]


def test_every_covariate_is_ranked_not_just_the_top_five() -> None:
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    imp, diag = _fast(frame)
    assert list(imp["feature"]) and set(imp["feature"]) == set(DEFAULT_COVARIATES)
    assert len(diag["top_features"]) == 5
    assert imp["gain"].is_monotonic_decreasing
    assert imp["gain_share"].sum() == pytest.approx(1.0)


def test_identifiers_are_not_covariates_by_default() -> None:
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    _, diag = _fast(frame)
    for identifier in ("PRODUCT_ID", "STORE_ID", "WEEK_NO"):
        assert identifier not in diag["covariates"]
    assert "memorises" in diag["identifiers_excluded"]


def test_grouped_folds_keep_a_product_store_whole() -> None:
    """A random split would put near-duplicate weeks on both sides."""
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    _, grouped = _fast(frame, cv="group")
    _, random = _fast(frame, cv="random")
    assert grouped["cv"]["scheme"] == "group"
    assert grouped["cv"]["grouped_by"] == "PRODUCT_ID x STORE_ID"
    assert random["cv"]["scheme"] == "random"
    assert random["cv"]["grouped_by"] is None


def test_constant_covariates_are_reported() -> None:
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    frame["is_holiday_week"] = False
    _, diag = _fast(frame)
    assert "is_holiday_week" in diag["constant_covariates"]


def test_a_treatment_that_never_varies_is_refused() -> None:
    frame = _overlap_panel(500, rule=lambda f, rng: np.ones(len(f), dtype=bool))
    with pytest.raises(ValueError, match="does not vary"):
        _fast(frame)


def test_unknown_cv_scheme_raises() -> None:
    frame = _overlap_panel(500, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    with pytest.raises(ValueError, match="cv must be"):
        _fast(frame, cv="timeseries")


def test_missing_covariate_raises() -> None:
    frame = _overlap_panel(500, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    with pytest.raises(KeyError, match="not columns of the panel"):
        _fast(frame, covariates=["units_lag_1", "nonexistent"])


def test_unobserved_rows_are_refused_by_overlap_too() -> None:
    frame = _overlap_panel(500, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    frame.loc[frame.index[:10], "treatment_observed"] = False
    with pytest.raises(UnobservedRowsError, match="unobserved as untreated"):
        _fast(frame)


def test_results_are_reproducible_under_a_seed() -> None:
    """Same seed, same answer — on both fold schemes."""
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    for scheme in ("group", "random"):
        _, first = _fast(frame, seed=7, cv=scheme)
        _, second = _fast(frame, seed=7, cv=scheme)
        assert first["auc"] == second["auc"], scheme


def test_the_seed_moves_random_folds_and_not_grouped_ones() -> None:
    """Grouped folds are deterministic, so the seed cannot change them.

    GroupKFold does not shuffle, and the learner has no stochastic component at
    these settings — no bagging, no feature subsampling. The seed is therefore
    inert under cv="group" and only bites under cv="random", which is worth
    knowing before anyone reads a seed sweep as a robustness check.
    """
    frame = _overlap_panel(2000, rule=lambda f, rng: rng.random(len(f)) < 0.3)
    _, g7 = _fast(frame, seed=7, cv="group")
    _, g8 = _fast(frame, seed=8, cv="group")
    _, r7 = _fast(frame, seed=7, cv="random")
    _, r8 = _fast(frame, seed=8, cv="random")
    assert g7["auc"] == g8["auc"]
    assert r7["auc"] != r8["auc"]


@real_data
def test_real_overlap_is_not_leakage(real_overlap) -> None:
    """AUC well below the leakage bar, and gain spread across features."""
    _, diag = real_overlap
    assert diag["auc"] < LEAKAGE_AUC
    assert diag["diagnosis"] == "SEPARABLE_BUT_PLAUSIBLE"
    assert diag["top_feature_gain_share"] < 0.5


@real_data
def test_real_no_region_is_treatment_saturated(real_overlap) -> None:
    """Nothing sits above 0.98, so every treated row has untreated company."""
    _, diag = real_overlap
    e = diag["propensity_extremes"]
    assert e["above_high"] == 0
    assert e["untreated_above_high"] == 0
    assert 0.0 < e["below_low_share"] < 0.15


@real_data
def test_real_folds_agree(real_overlap) -> None:
    """A fold that disagreed would mean the split, not the data, drove the AUC."""
    _, diag = real_overlap
    spread = max(diag["auc_by_fold"]) - min(diag["auc_by_fold"])
    assert spread < 0.01


@real_data
def test_real_overlap_diagnostics_are_json_serialisable(
    real_overlap, tmp_path: Path
) -> None:
    import json

    _, diag = real_overlap
    path = write_diagnostics(diag, tmp_path / "overlap.json")
    assert json.loads(path.read_text())["stage"] == "overlap"


# --------------------------------------------------------------------------
# Task 3.3 — collisions and horizon.
# --------------------------------------------------------------------------


def _collision_panel(rows: list[dict]) -> pd.DataFrame:
    base = {
        "PRODUCT_ID": 1,
        "STORE_ID": 1,
        "WEEK_NO": 1,
        "units": 1,
        "treated": False,
        "in_mailer": False,
        "treatment_observed": True,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_all_four_cells_are_reported() -> None:
    panel = _collision_panel(
        [
            {"treated": True, "in_mailer": True, "units": 4},
            {"treated": True, "in_mailer": False, "units": 3},
            {"treated": False, "in_mailer": True, "units": 2},
            {"treated": False, "in_mailer": False, "units": 1},
        ]
    )
    cells, _ = collisions(panel)
    by_cell = cells.set_index("cell")
    assert set(by_cell.index) == {
        "treated_with_secondary",
        "treated_clean",
        "control_with_secondary",
        "control_clean",
    }
    assert by_cell.loc["treated_with_secondary", "units"] == 4
    assert by_cell.loc["control_with_secondary", "units"] == 2
    assert cells["rows"].sum() == 4
    assert cells["units_share"].sum() == pytest.approx(1.0)


def test_collision_share_is_of_treated_rows() -> None:
    panel = _collision_panel(
        [{"treated": True, "in_mailer": True}]
        + [{"treated": True, "in_mailer": False}] * 3
        + [{"treated": False, "in_mailer": False}] * 6
    )
    _, diag = collisions(panel)
    assert diag["collision"]["rows"] == 1
    assert diag["collision"]["share_of_treated"] == pytest.approx(0.25)


def test_contaminated_controls_are_counted_separately() -> None:
    """The failure mode the plan's single collision count would miss.

    An untreated row carrying a mailer is a promoted row sitting in the control
    group. It biases the counterfactual up and the effect towards zero, which is
    the opposite direction to a collision on a treated row.
    """
    panel = _collision_panel(
        [{"treated": True, "in_mailer": False}] * 2
        + [{"treated": False, "in_mailer": True}] * 3
        + [{"treated": False, "in_mailer": False}] * 5
    )
    _, diag = collisions(panel)
    assert diag["collision"]["rows"] == 0            # no treated row collides
    assert diag["contaminated_controls"]["rows"] == 3
    assert diag["contaminated_controls"]["share_of_controls"] == pytest.approx(3 / 8)
    # Contamination alone must still trip the status.
    assert diag["status"] == "OVERLAPPING_TREATMENTS"


def test_a_clean_panel_is_separable() -> None:
    panel = _collision_panel(
        [{"treated": True, "in_mailer": False}] * 3
        + [{"treated": False, "in_mailer": False}] * 7
    )
    _, diag = collisions(panel)
    assert diag["status"] == "SEPARABLE"
    assert diag["collision"]["rows"] == 0
    assert diag["contaminated_controls"]["rows"] == 0


def test_collision_threshold_is_a_parameter() -> None:
    panel = _collision_panel(
        [{"treated": True, "in_mailer": True}]
        + [{"treated": True, "in_mailer": False}] * 9
        + [{"treated": False, "in_mailer": False}] * 10
    )
    _, strict = collisions(panel, max_collision_share=0.05)
    _, lenient = collisions(panel, max_collision_share=0.5)
    assert strict["status"] == "OVERLAPPING_TREATMENTS"   # 10% collides
    assert lenient["status"] == "SEPARABLE"
    assert strict["threshold"] == 0.05


def test_collisions_refuse_unobserved_rows() -> None:
    panel = _collision_panel(
        [{"treated": True}, {"treated": False, "treatment_observed": False}]
    )
    with pytest.raises(UnobservedRowsError):
        collisions(panel)


def _cycles(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["COMMODITY_DESC", "horizon_weeks", "low_support"]
    )


def test_a_short_horizon_is_flagged_with_its_shortfall() -> None:
    campaigns = pd.DataFrame(
        {"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [1]}
    )
    checked, diag = horizon_check(campaigns, _cycles([("SOUP", 5, False)]))
    assert checked.iloc[0]["status"] == "HORIZON_TOO_SHORT"
    assert checked.iloc[0]["shortfall_weeks"] == 4
    assert diag["too_short"] == 1
    assert diag["max_shortfall_weeks"] == 4


def test_an_adequate_horizon_passes() -> None:
    campaigns = pd.DataFrame({"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [9]})
    checked, diag = horizon_check(campaigns, _cycles([("SOUP", 3, False)]))
    assert checked.iloc[0]["status"] == "OK"
    assert pd.isna(checked.iloc[0]["shortfall_weeks"])
    assert diag["too_short"] == 0


def test_an_exactly_equal_horizon_passes() -> None:
    """The rule is 'at least the cycle', so equality clears it."""
    campaigns = pd.DataFrame({"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [3]})
    checked, _ = horizon_check(campaigns, _cycles([("SOUP", 3, False)]))
    assert checked.iloc[0]["status"] == "OK"


def test_an_unknown_cycle_is_not_a_pass() -> None:
    """A commodity with no recorded cycle must not read as 'long enough'."""
    campaigns = pd.DataFrame({"COMMODITY_DESC": ["GHOST"], "horizon_weeks": [99]})
    checked, diag = horizon_check(campaigns, _cycles([("SOUP", 3, False)]))
    assert checked.iloc[0]["status"] == "UNKNOWN_CYCLE"
    assert diag["unknown_cycle"] == 1
    assert diag["too_short"] == 0


def test_a_null_cycle_is_also_unknown() -> None:
    campaigns = pd.DataFrame({"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [4]})
    cycles = pd.DataFrame(
        {"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [pd.NA], "low_support": [True]}
    )
    checked, _ = horizon_check(campaigns, cycles)
    assert checked.iloc[0]["status"] == "UNKNOWN_CYCLE"


def test_low_support_cycles_are_checked_but_flagged() -> None:
    campaigns = pd.DataFrame(
        {"COMMODITY_DESC": ["SOUP", "BREAD"], "horizon_weeks": [1, 9]}
    )
    checked, diag = horizon_check(
        campaigns, _cycles([("SOUP", 5, True), ("BREAD", 3, False)])
    )
    assert list(checked["status"]) == ["HORIZON_TOO_SHORT", "OK"]
    assert bool(checked.iloc[0]["low_support"]) is True
    assert diag["on_a_low_support_cycle"] == 1


def test_horizon_check_does_not_duplicate_campaigns() -> None:
    campaigns = pd.DataFrame(
        {"COMMODITY_DESC": ["SOUP", "SOUP", "BREAD"], "horizon_weeks": [1, 9, 2]}
    )
    checked, diag = horizon_check(
        campaigns, _cycles([("SOUP", 5, False), ("BREAD", 3, False)])
    )
    assert len(checked) == 3
    assert diag["campaigns"] == 3


def test_horizon_check_requires_its_columns() -> None:
    with pytest.raises(KeyError, match="horizon_weeks"):
        horizon_check(
            pd.DataFrame({"COMMODITY_DESC": ["SOUP"]}), _cycles([("SOUP", 3, False)])
        )
    with pytest.raises(KeyError, match="not a column of the cycles"):
        horizon_check(
            pd.DataFrame({"COMMODITY_DESC": ["SOUP"], "horizon_weeks": [3]}),
            pd.DataFrame({"COMMODITY_DESC": ["SOUP"]}),
        )


@real_data
def test_real_display_and_mailer_collide_substantially() -> None:
    cells, diag = collisions(PANEL)
    assert diag["status"] == "OVERLAPPING_TREATMENTS"
    assert 0.3 < diag["collision"]["share_of_treated"] < 0.5
    assert 0.1 < diag["contaminated_controls"]["share_of_controls"] < 0.2
    assert cells["rows"].sum() == diag["rows"]


@real_data
def test_real_horizon_check_runs_against_the_recorded_cycles() -> None:
    """A two-week window fails for a fast commodity and for a slow one."""
    cycles = Path("data/interim/repurchase_cycles.parquet")
    if not cycles.exists():
        pytest.skip("run Task 2.7 first")
    campaigns = pd.DataFrame(
        {
            "COMMODITY_DESC": ["FLUID MILK PRODUCTS", "SOUP", "FLUID MILK PRODUCTS"],
            "horizon_weeks": [2, 2, 12],
        }
    )
    checked, diag = horizon_check(campaigns, cycles)
    assert checked.iloc[0]["status"] == "HORIZON_TOO_SHORT"   # milk needs 3
    assert checked.iloc[2]["status"] == "OK"
    assert diag["too_short"] >= 1
