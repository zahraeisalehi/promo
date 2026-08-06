"""Tests for Task 2.6, the derived model variables.

The fixture tests build a panel whose lags and windows are known by hand, so an
off-by-one in a window frame fails loudly. The two leakage rules named in the
module docstring each get a test that would fail if the rule were dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.features import (
    CONTEMPORANEOUS_FEATURES,
    LAGGED_FEATURES,
    LAGS,
    ROLLING_WINDOWS,
    build_feature_panel,
    write_diagnostics,
)

PANEL = Path("data/interim/panel_treated.parquet")
CLEAN = Path("data/interim/transactions_clean.parquet")
PRODUCT = Path("data/raw/product.csv")
CAUSAL = Path("data/raw/causal_data.csv")

_NO_DATA = not all(p.exists() for p in (PANEL, CLEAN, PRODUCT, CAUSAL))


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast pass
    excludes it with -m "not heavy", and it is skipped outright when the
    artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="run Tasks 2.2-2.5 first")(
        pytest.mark.heavy(fn)
    )


def _treated_panel(rows: list[dict]) -> pd.DataFrame:
    """A minimal Task 2.5-shaped panel."""
    base = {
        "PRODUCT_ID": 1,
        "STORE_ID": 1,
        "WEEK_NO": 1,
        "n_rows": 1,
        "units": 1,
        "sales_value": 1.0,
        "regular_price": 1.0,
        "real_regular_price": 1.0,
        "paid_price": 1.0,
        "depth": 0.0,
        "price_status": "identified",
        "price_index": 1.0,
        "on_display": False,
        "in_mailer": False,
        "treatment_observed": True,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _causal(tmp_path: Path, rows: list[tuple]) -> Path:
    path = tmp_path / "causal_data.csv"
    pd.DataFrame(
        rows or [(-1, -1, 9, "9", "A")],
        columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
    ).to_csv(path, index=False)
    return path


def _product(tmp_path: Path, rows: list[tuple]) -> Path:
    path = tmp_path / "product.csv"
    pd.DataFrame(rows, columns=["PRODUCT_ID", "COMMODITY_DESC"]).to_csv(
        path, index=False
    )
    return path


def _transactions(tmp_path: Path, rows: list[tuple]) -> Path:
    path = tmp_path / "clean.parquet"
    pd.DataFrame(
        rows, columns=["STORE_ID", "WEEK_NO", "BASKET_ID", "usable"]
    ).to_parquet(path, index=False)
    return path


@pytest.fixture
def simple(tmp_path: Path):
    """One product, one store, weeks 1-6 with a gap at week 3.

    units: w1=5, w2=7, w3=(absent -> 0), w4=9, w5=11, w6=13.
    """
    panel = _treated_panel(
        [
            {"WEEK_NO": w, "units": u, "sales_value": float(u), "on_display": True}
            for w, u in [(1, 5), (2, 7), (4, 9), (5, 11), (6, 13)]
        ]
    )
    return {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, w, "9", "A") for w in range(1, 7)]),
        "product": _product(tmp_path, [(1, "SOUP")]),
        "transactions": _transactions(
            tmp_path, [(1, w, 100 + w, True) for w in range(1, 7)]
        ),
    }


def _build(fix, **kwargs):
    kwargs.setdefault("n_products", 10)
    return build_feature_panel(
        fix["panel"], fix["transactions"], fix["product"], fix["causal"], **kwargs
    )


def test_zeros_are_filled_across_the_week_grid(simple) -> None:
    out, diag = _build(simple)
    assert len(out) == 6  # weeks 1-6, no gap
    assert sorted(out["WEEK_NO"]) == [1, 2, 3, 4, 5, 6]
    week3 = out[out.WEEK_NO == 3].iloc[0]
    assert int(week3["units"]) == 0
    assert bool(week3["zero_filled"]) is True
    assert diag["grid"]["zero_filled_rows"] == 1


def test_lags_are_true_week_shifts_not_row_shifts(simple) -> None:
    """The zero week must occupy its slot, or every later lag is wrong."""
    out, _ = _build(simple)
    by_week = out.set_index("WEEK_NO")
    assert by_week.loc[4, "units_lag_1"] == 0  # week 3 was zero, not week 2's 7
    assert by_week.loc[4, "units_lag_2"] == 7
    assert by_week.loc[5, "units_lag_1"] == 9
    assert pd.isna(by_week.loc[1, "units_lag_1"])


def test_rolling_mean_excludes_the_current_week(simple) -> None:
    """Leakage rule 1: a window containing week w hands over the answer."""
    out, _ = _build(simple)
    by_week = out.set_index("WEEK_NO")
    # weeks 1-3 are 5, 7, 0 -> mean 4. Including week 4 would give 5.25.
    assert by_week.loc[4, "units_roll_mean_4"] == pytest.approx(4.0)
    assert by_week.loc[2, "units_roll_mean_4"] == pytest.approx(5.0)
    assert pd.isna(by_week.loc[1, "units_roll_mean_4"])


def test_lags_never_cross_a_product_store_boundary(tmp_path: Path) -> None:
    panel = _treated_panel(
        [
            {
                "PRODUCT_ID": p,
                "STORE_ID": s,
                "WEEK_NO": w,
                "units": 10 * p + w,
                "on_display": True,
            }
            for p in (1, 2)
            for s in (1, 2)
            for w in (1, 2, 3)
        ]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP"), (2, "SOUP")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, _ = _build(fix)
    # Every group's first week must have a null lag: nothing leaked in from the
    # previous product-store block.
    first = out[out.WEEK_NO == 1]
    assert first["units_lag_1"].isna().all()
    assert len(first) == 4


def test_price_rel_category_uses_regular_not_paid_price(tmp_path: Path) -> None:
    """Leakage rule 2: a paid price is low *because* the product is on deal."""
    panel = _treated_panel(
        [
            {
                "PRODUCT_ID": 1,
                "WEEK_NO": 1,
                "regular_price": 10.0,
                "paid_price": 2.0,
                "depth": 0.8,
                "on_display": True,
            },
            {
                "PRODUCT_ID": 2,
                "WEEK_NO": 1,
                "regular_price": 10.0,
                "paid_price": 10.0,
                "on_display": True,
            },
        ]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP"), (2, "SOUP")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, _ = _build(fix)
    # Both sit at the category median on regular price, despite an 80% markdown
    # on one. Paid price would have given 0.2 against 1.0.
    assert out["price_rel_category"].round(6).nunique() == 1
    assert out["price_rel_category"].iloc[0] == pytest.approx(1.0)


def test_category_units_exclude_the_focal_product(tmp_path: Path) -> None:
    panel = _treated_panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": 1, "units": 5, "on_display": True},
            {"PRODUCT_ID": 2, "WEEK_NO": 1, "units": 3, "on_display": True},
            {"PRODUCT_ID": 3, "WEEK_NO": 1, "units": 4, "on_display": True},
        ]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP"), (2, "SOUP"), (3, "BREAD")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, _ = _build(fix)
    by_product = out.set_index("PRODUCT_ID")["category_units_ex_focal"]
    assert by_product[1] == 3  # SOUP total 8, minus its own 5
    assert by_product[2] == 5
    assert by_product[3] == 0  # only BREAD in its category


def test_store_traffic_counts_baskets_not_units(tmp_path: Path) -> None:
    panel = _treated_panel([{"WEEK_NO": 1, "units": 500, "on_display": True}])
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP")]),
        "transactions": _transactions(
            tmp_path,
            [(1, 1, 10, True), (1, 1, 10, True), (1, 1, 11, True), (1, 1, 12, False)],
        ),
    }
    out, _ = _build(fix)
    # Two distinct usable baskets, not four rows and not 500 units.
    assert out["store_traffic"].iloc[0] == 2


def test_week_of_year_cycles_at_52(tmp_path: Path) -> None:
    panel = _treated_panel(
        [{"WEEK_NO": w, "units": 1, "on_display": True} for w in (1, 52, 53, 104)]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, _ = _build(fix)
    by_week = out.set_index("WEEK_NO")["week_of_year"]
    assert by_week[1] == 1
    assert by_week[52] == 52
    assert by_week[53] == 1
    assert by_week[104] == 52


def test_holiday_flag_is_empty_and_says_so(simple) -> None:
    """No calendar anchor exists, so none is invented."""
    out, diag = _build(simple)
    assert "is_holiday_week" in out.columns
    assert not out["is_holiday_week"].any()
    flag = diag["features"]["holiday_flag"]
    assert flag["populated"] is False
    assert flag["weeks_supplied"] == []
    assert "no calendar dates" in flag["why_empty"]


def test_holiday_flag_populates_when_weeks_are_supplied(simple) -> None:
    out, diag = _build(simple, holiday_weeks={2, 5})
    assert set(out.loc[out["is_holiday_week"], "WEEK_NO"]) == {2, 5}
    assert diag["features"]["holiday_flag"]["populated"] is True
    assert diag["features"]["holiday_flag"]["weeks_supplied"] == [2, 5]


def test_zero_filled_rows_get_their_treatment_looked_up(tmp_path: Path) -> None:
    """A product can be on display and sell nothing that week."""
    panel = _treated_panel(
        [{"WEEK_NO": w, "units": 5, "on_display": True} for w in (1, 3)]
    )
    fix = {
        "panel": panel,
        # Week 2 has no sale but the log says it was on display.
        "causal": _causal(tmp_path, [(1, 1, w, "9", "A") for w in (1, 2, 3)]),
        "product": _product(tmp_path, [(1, "SOUP")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, diag = _build(fix)
    week2 = out[out.WEEK_NO == 2].iloc[0]
    assert bool(week2["zero_filled"]) is True
    assert int(week2["units"]) == 0
    assert bool(week2["treated"]) is True  # not defaulted to untreated
    assert diag["treatment_rederived"]["treated_rows_with_zero_units"] == 1


def test_scope_keeps_only_ever_treated_products(tmp_path: Path) -> None:
    panel = _treated_panel(
        [
            {"PRODUCT_ID": 1, "WEEK_NO": w, "units": 5, "on_display": True}
            for w in (1, 2)
        ]
        + [{"PRODUCT_ID": 2, "WEEK_NO": w, "units": 99} for w in (1, 2)]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(1, "SOUP"), (2, "SOUP")]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    out, diag = _build(fix)
    # Product 2 is never treated, so it cannot inform a treatment effect.
    assert set(out["PRODUCT_ID"]) == {1}
    assert diag["scope"]["n_products"] == 1
    assert 0 < diag["scope"]["coverage"]["sales_value_share"] < 1


def test_scope_size_is_a_parameter(tmp_path: Path) -> None:
    panel = _treated_panel(
        [
            {
                "PRODUCT_ID": p,
                "WEEK_NO": w,
                "units": 10 - p,
                "n_rows": 10 - p,
                "on_display": True,
            }
            for p in (1, 2, 3)
            for w in (1, 2)
        ]
    )
    fix = {
        "panel": panel,
        "causal": _causal(tmp_path, [(1, 1, 1, "9", "A")]),
        "product": _product(tmp_path, [(p, "SOUP") for p in (1, 2, 3)]),
        "transactions": _transactions(tmp_path, [(1, 1, 1, True)]),
    }
    two, diag = _build(fix, n_products=2)
    assert set(two["PRODUCT_ID"]) == {1, 2}  # the two most frequent
    assert diag["scope"]["n_products"] == 2


def test_price_index_is_present_on_zero_filled_rows(simple) -> None:
    """It is a property of the week, not of the sale."""
    out, _ = _build(simple)
    assert out["price_index"].notna().all()
    assert out.loc[out["zero_filled"], "price_index"].notna().all()


def test_every_declared_feature_column_exists(simple) -> None:
    out, diag = _build(simple)
    for column in (*LAGGED_FEATURES, *CONTEMPORANEOUS_FEATURES):
        assert column in out.columns, column
    assert diag["features"]["lags"] == list(LAGS)
    assert diag["features"]["rolling_windows"] == list(ROLLING_WINDOWS)


def test_writes_parquet_when_asked(simple, tmp_path: Path) -> None:
    out_path = tmp_path / "panel.parquet"
    frame, diag = _build(simple, out_path=out_path)
    assert diag["written_to"] == str(out_path)
    assert len(pd.read_parquet(out_path)) == len(frame)


# --------------------------------------------------------------------------
# The real panel.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_features():
    return build_feature_panel(PANEL, CLEAN, PRODUCT, CAUSAL)


@real_data
def test_real_scope_is_within_budget(real_features) -> None:
    out, diag = real_features
    scope = diag["scope"]
    assert scope["n_products"] == 300
    assert scope["n_stores"] == 115
    assert scope["n_weeks"] == 93
    assert len(out) == diag["grid"]["rows"]
    # Far below the 5.5 billion cells a full grid would need.
    assert len(out) < 5_000_000


@real_data
def test_real_scope_coverage_is_recorded(real_features) -> None:
    """A scope whose coverage is not recorded is a silent filter."""
    _, diag = real_features
    coverage = diag["scope"]["coverage"]
    assert 0.1 < coverage["transactions_share"] < 0.5
    assert 0.1 < coverage["sales_value_share"] < 0.5


@real_data
def test_real_grid_is_complete_and_unique(real_features) -> None:
    out, diag = real_features
    assert not out.duplicated(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).any()
    assert len(out) == diag["grid"]["carried_pairs"] * diag["grid"]["weeks"]
    counts = out.groupby(["PRODUCT_ID", "STORE_ID"], observed=True).size()
    assert counts.nunique() == 1  # every pair has every week


@real_data
def test_real_lags_match_a_hand_computed_shift(real_features) -> None:
    """The leakage assertion, re-run independently of the module's own check."""
    out, _ = real_features
    key = out[["PRODUCT_ID", "STORE_ID"]].drop_duplicates().iloc[0]
    block = out[
        (out["PRODUCT_ID"] == key["PRODUCT_ID"]) & (out["STORE_ID"] == key["STORE_ID"])
    ].sort_values("WEEK_NO")
    units = block["units"].astype("float64")
    for k in LAGS:
        assert np.allclose(
            units.shift(k),
            block[f"units_lag_{k}"].astype("float64"),
            equal_nan=True,
        ), k


@real_data
def test_real_lag_52_missingness_is_structural(real_features) -> None:
    """93 weeks of scope means the first 52 cannot have a 52-week lag."""
    out, diag = real_features
    missing = diag["features"]["missingness"]["units_lag_52"]
    assert 0.5 < missing < 0.6
    weeks = out.loc[out["units_lag_52"].notna(), "WEEK_NO"]
    assert int(weeks.min()) == int(out["WEEK_NO"].min()) + 52


@real_data
def test_real_price_index_covers_every_row(real_features) -> None:
    out, diag = real_features
    assert out["price_index"].notna().all()
    assert diag["features"]["missingness"].get("price_index", 0.0) == 0.0


@real_data
def test_real_holiday_flag_is_unpopulated(real_features) -> None:
    out, diag = real_features
    assert not out["is_holiday_week"].any()
    assert diag["features"]["holiday_flag"]["populated"] is False


@real_data
def test_real_treated_rows_include_zero_sale_weeks(real_features) -> None:
    """The reason treatment is re-derived rather than carried over."""
    _, diag = real_features
    assert diag["treatment_rederived"]["treated_rows_with_zero_units"] > 0


@real_data
def test_real_diagnostics_are_json_serialisable(real_features, tmp_path: Path) -> None:
    import json

    _, diag = real_features
    path = write_diagnostics(diag, tmp_path / "feature_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "build_feature_panel"
