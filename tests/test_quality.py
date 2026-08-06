"""Tests for Task 2.7: availability, zero classification, and the horizon.

Fixture tests use household histories small enough that every gap and median can
be checked by hand. The heavy tests assert the module reproduces Task 1.5's
figures on the real file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from promo.quality import (
    STRUCTURAL_STATES,
    ZERO_STATES,
    build_quality_report,
    classify_zeros,
    household_week_flags,
    repurchase_cycles,
)

CLEAN = Path("data/interim/transactions_clean.parquet")
PRODUCT = Path("data/raw/product.csv")

_NO_DATA = not CLEAN.exists() or not PRODUCT.exists()


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast
    pass excludes it with -m "not heavy", and it is skipped outright when
    the artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="run Task 2.2 first")(
        pytest.mark.heavy(fn)
    )


def _tx(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """(household_key, WEEK_NO, PRODUCT_ID) triples, all usable."""
    return pd.DataFrame(
        [
            {"household_key": h, "WEEK_NO": w, "PRODUCT_ID": p, "usable": True}
            for h, w, p in rows
        ]
    )


def _prod(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["PRODUCT_ID", "COMMODITY_DESC"])


# --------------------------------------------------------------------------
# The household-week shopping flag.
# --------------------------------------------------------------------------


def test_flags_cover_every_household_week() -> None:
    tx = _tx([(1, 2, 10), (2, 5, 10)])
    flags, diag = household_week_flags(tx, week_min=1, week_max=6)
    assert len(flags) == 2 * 6
    assert diag["households"] == 2
    assert diag["weeks"] == 6


def test_shopped_marks_only_weeks_with_a_transaction() -> None:
    tx = _tx([(1, 2, 10), (1, 4, 10)])
    flags, _ = household_week_flags(tx, week_min=1, week_max=5)
    shopped = flags.set_index("WEEK_NO")["shopped"]
    assert list(shopped) == [False, True, False, True, False]


def test_in_span_excludes_weeks_before_entry_and_after_exit() -> None:
    """A week before a household's first trip is not a refused purchase."""
    tx = _tx([(1, 3, 10), (1, 5, 10)])
    flags, _ = household_week_flags(tx, week_min=1, week_max=7)
    in_span = flags.set_index("WEEK_NO")["in_span"]
    assert list(in_span) == [False, False, True, True, True, False, False]


def test_within_span_no_trip_share_is_the_honest_one() -> None:
    tx = _tx([(1, 3, 10), (1, 5, 10)])
    _, diag = household_week_flags(tx, week_min=1, week_max=7)
    # Raw: 5 of 7 weeks have no trip. Within span (weeks 3-5): only week 4.
    assert diag["raw"]["no_trip"] == 5
    assert diag["within_span"]["household_weeks"] == 3
    assert diag["within_span"]["no_trip"] == 1
    assert diag["within_span"]["no_trip_share"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# The zero classifier.
# --------------------------------------------------------------------------


def test_four_states_are_assigned_correctly() -> None:
    """One household: entered week 2, left week 5, bought SOUP in week 3."""
    tx = _tx([(1, 2, 20), (1, 3, 10), (1, 5, 20)])
    product = _prod([(10, "SOUP"), (20, "BREAD")])
    out, diag = classify_zeros("SOUP", tx, product, week_min=1, week_max=6)
    state = out.set_index("WEEK_NO")["state"]

    assert state[1] == "out_of_panel"      # before entry
    assert state[2] == "no_buy_on_trip"    # in shop, bought bread not soup
    assert state[3] == "bought"
    assert state[4] == "no_trip"           # in panel, no trip
    assert state[5] == "no_buy_on_trip"
    assert state[6] == "out_of_panel"      # after exit
    assert set(diag["states"]) == set(ZERO_STATES)


def test_only_no_buy_on_trip_is_a_sampling_zero() -> None:
    tx = _tx([(1, 2, 20), (1, 3, 10), (1, 5, 20)])
    product = _prod([(10, "SOUP"), (20, "BREAD")])
    out, diag = classify_zeros("SOUP", tx, product, week_min=1, week_max=6)
    structural = out.set_index("WEEK_NO")["structural"]

    assert bool(structural[1]) is True   # out_of_panel
    assert bool(structural[4]) is True   # no_trip
    assert bool(structural[2]) is False  # no_buy_on_trip carries demand info
    assert bool(structural[3]) is False  # bought is not a zero at all
    assert diag["zeros"]["sampling"] == 2
    assert diag["zeros"]["structural"] == 3
    assert set(STRUCTURAL_STATES) == {"no_trip", "out_of_panel"}


def test_classifier_matches_the_requested_commodity_only() -> None:
    tx = _tx([(1, 1, 10), (1, 2, 20)])
    product = _prod([(10, "SOUP"), (20, "BREAD")])
    soup, _ = classify_zeros("SOUP", tx, product, week_min=1, week_max=2)
    bread, _ = classify_zeros("BREAD", tx, product, week_min=1, week_max=2)
    assert soup.set_index("WEEK_NO")["state"][1] == "bought"
    assert bread.set_index("WEEK_NO")["state"][1] == "no_buy_on_trip"


# --------------------------------------------------------------------------
# The repurchase cycle.
# --------------------------------------------------------------------------


@pytest.fixture
def gaps_fixture():
    """Household 1 buys SOUP in weeks 1, 3, 7 -> gaps 2, 4 -> hh median 3.

    Household 2 buys in weeks 1, 2, 3 -> gaps 1, 1 -> hh median 1.
    Median of household medians = 2. Pooled gaps are [2, 4, 1, 1], median 1.5.
    """
    tx = _tx(
        [(1, 1, 10), (1, 3, 10), (1, 7, 10)]
        + [(2, 1, 10), (2, 2, 10), (2, 3, 10)]
    )
    return tx, _prod([(10, "SOUP")])


def test_horizon_uses_household_medians_not_the_pooled_median(gaps_fixture) -> None:
    tx, product = gaps_fixture
    cycles, _ = repurchase_cycles(tx, product, out_path=None)
    row = cycles[cycles.COMMODITY_DESC == "SOUP"].iloc[0]
    assert row["hh_median_gap"] == pytest.approx(2.0)
    assert row["pooled_median_gap"] == pytest.approx(1.5)
    assert int(row["horizon_weeks"]) == 2  # from the household view, not 2 from 1.5


def test_horizon_rounds_up(gaps_fixture) -> None:
    """A 2.5-week cycle must not become a 2-week window."""
    tx = _tx([(1, 1, 10), (1, 3, 10), (2, 1, 10), (2, 4, 10)])
    cycles, _ = repurchase_cycles(tx, _prod([(10, "SOUP")]), out_path=None)
    row = cycles.iloc[0]
    assert row["hh_median_gap"] == pytest.approx(2.5)
    assert int(row["horizon_weeks"]) == 3


def test_single_week_pairs_contribute_no_gap(gaps_fixture) -> None:
    """They are the slowest buyers, which is why every horizon is a floor."""
    tx = _tx([(1, 1, 10), (1, 3, 10), (2, 5, 10)])
    cycles, diag = repurchase_cycles(tx, _prod([(10, "SOUP")]), out_path=None)
    row = cycles.iloc[0]
    assert int(row["n_pairs"]) == 2
    assert int(row["single_week_pairs"]) == 1
    assert int(row["n_gaps"]) == 1  # only household 1 contributes
    assert "floor" in diag["bias"]


def test_commodity_with_no_gap_gets_no_invented_horizon() -> None:
    tx = _tx([(1, 1, 10)])
    cycles, _ = repurchase_cycles(tx, _prod([(10, "SOUP")]), out_path=None)
    row = cycles.iloc[0]
    assert pd.isna(row["horizon_weeks"])
    assert bool(row["low_support"]) is True


def test_low_support_is_flagged_not_dropped(gaps_fixture) -> None:
    tx, product = gaps_fixture
    cycles, diag = repurchase_cycles(
        tx, product, min_gap_events=100, out_path=None
    )
    assert bool(cycles.iloc[0]["low_support"]) is True
    assert diag["low_support"] == 1
    assert len(cycles) == 1  # flagged, still present


def test_writes_repurchase_cycles_parquet(gaps_fixture, tmp_path: Path) -> None:
    tx, product = gaps_fixture
    out = tmp_path / "repurchase_cycles.parquet"
    cycles, diag = repurchase_cycles(tx, product, out_path=out)
    assert diag["written_to"] == str(out)
    assert len(pd.read_parquet(out)) == len(cycles)


# --------------------------------------------------------------------------
# The quality report.
# --------------------------------------------------------------------------


def test_report_names_missing_stages_rather_than_hiding_them(tmp_path: Path) -> None:
    report, diag = build_quality_report(tmp_path, out_path=tmp_path / "quality.json")
    assert diag["sources_missing"]
    assert diag["sources_present"] == []
    for entry in report["sources"].values():
        assert entry["present"] is False
        assert "re-run that stage" in entry["consequence"]


def test_report_collects_exclusions_from_stage_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "clean_diagnostics.json").write_text(json.dumps({
        "filters": [{
            "name": "volume_measured", "action": "exclude",
            "definition": "DEPARTMENT IN (...)",
            "attributed_to_this_stage": {"rows": 10, "units": 99, "sales_value": 5.0},
            "share_of_all": {"rows_share": 0.1},
            "before": {"rows": 100}, "after": {"rows": 90},
        }]
    }))
    (tmp_path / "prices_diagnostics.json").write_text(json.dumps({
        "exclusions": [{
            "name": "retail_disc_surcharge", "action": "exclude",
            "definition": "RETAIL_DISC > 0",
            "effect": {"rows": 6}, "before": {"rows": 90}, "after": {"rows": 84},
        }]
    }))
    report, diag = build_quality_report(tmp_path, out_path=tmp_path / "quality.json")
    assert diag["exclusions_recorded"] == 2
    names = [e["name"] for e in report["exclusions"]]
    assert names == ["volume_measured", "retail_disc_surcharge"]
    assert report["exclusions"][0]["stage"] == "clean"
    assert report["exclusions"][1]["stage"] == "prices"


def test_report_is_written_as_json(tmp_path: Path) -> None:
    out = tmp_path / "quality.json"
    _, diag = build_quality_report(tmp_path, out_path=out)
    assert diag["written_to"] == str(out)
    assert json.loads(out.read_text())["report"] == "Phase 2 data honesty"


def test_report_lists_unresolved_limits(tmp_path: Path) -> None:
    report, _ = build_quality_report(tmp_path, out_path=tmp_path / "quality.json")
    joined = " ".join(report["unresolved"])
    assert "Stockouts are unobservable" in joined
    assert "not stocked" in joined


# --------------------------------------------------------------------------
# The real file.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_flags():
    return household_week_flags(CLEAN)


@real_data
def test_real_no_trip_shares_match_task_1_5(real_flags) -> None:
    _, diag = real_flags
    assert diag["households"] == 2_500
    assert diag["household_weeks"] == 255_000
    assert diag["raw"]["no_trip_share"] == pytest.approx(0.5138, abs=0.001)
    assert diag["within_span"]["no_trip_share"] == pytest.approx(0.4457, abs=0.001)
    assert diag["trip_definition"] == "any transaction"


@real_data
def test_real_usable_filter_would_erase_real_trips() -> None:
    """A fuel-only visit is a trip. Filtering to usable rows loses 2,827 of them."""
    _, any_row = household_week_flags(CLEAN)
    _, usable = household_week_flags(CLEAN, usable_only=True)
    lost = any_row["raw"]["with_a_trip"] - usable["raw"]["with_a_trip"]
    assert lost == 2_827
    assert usable["raw"]["no_trip_share"] > any_row["raw"]["no_trip_share"]


@real_data
def test_real_entry_matches_task_1_5(real_flags) -> None:
    _, diag = real_flags
    assert diag["entry"]["median_first_week"] == 11
    assert diag["entry"]["median_last_week"] == 101


@real_data
def test_real_repurchase_cycles_match_task_1_5() -> None:
    cycles, diag = repurchase_cycles(CLEAN, PRODUCT, out_path=None)
    assert diag["commodities"] > 250
    by_name = cycles.set_index("COMMODITY_DESC")
    # Task 1.5 reported a median-of-household-medians of 3.0 for milk.
    milk = by_name.loc["FLUID MILK PRODUCTS"]
    assert milk["hh_median_gap"] == pytest.approx(3.0)
    assert int(milk["horizon_weeks"]) == 3
    assert milk["pooled_median_gap"] == pytest.approx(2.0)
    # The typical commodity's horizon is the 3-week floor Task 1.5 named.
    assert diag["horizon_weeks"]["median"] >= 3


@real_data
def test_real_single_week_share_matches_task_1_5_on_its_own_basis() -> None:
    """Task 1.5 quoted 15.53% on the top 50 commodities, not on all 306.

    The all-commodity figure is twice as high because the long tail is mostly
    one-off purchases. Both are asserted so the bases cannot be confused again.
    """
    _, diag = repurchase_cycles(CLEAN, PRODUCT, out_path=None)
    assert diag["single_week_pairs_share_top_50"] == pytest.approx(0.1553, abs=0.01)
    assert diag["single_week_pairs_share"] == pytest.approx(0.3149, abs=0.01)


@real_data
def test_real_structural_share_dominates_for_a_common_commodity() -> None:
    """Task 1.5: two thirds of milk's zeros are weeks with no trip at all."""
    _, diag = classify_zeros("FLUID MILK PRODUCTS", CLEAN, PRODUCT)
    assert diag["zeros"]["structural_share"] > 0.6
