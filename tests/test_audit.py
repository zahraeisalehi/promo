"""Tests for Task 3.1, the variation audit.

Fixture panels are built so that each axis's classification is obvious by
inspection, including the case the module exists to catch: an axis whose levels
are mostly mixed while the mixed levels hold almost no demand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.audit import (
    AXES,
    MEANINGFUL_MIXED_SHARE,
    UnobservedRowsError,
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
