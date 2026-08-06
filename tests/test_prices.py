"""Tests for Task 2.3 price decomposition.

The fixture tests build product-store-weeks whose reconstructed price is known
by hand, including the three cases `data_findings.md:440` names as hazards: a
positive `RETAIL_DISC`, a zero-quantity group, and float equality. The
`real_data` tests then check the stage against the file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.prices import (
    BOUNDED_THRESHOLD,
    MIN_MATCHED_PAIRS,
    MIN_PRICED_WEEKS,
    PRICE_TOLERANCE,
    build_price_index,
    build_price_panel,
    deflate_prices,
    level_test,
    write_diagnostics,
)

CLEAN = Path("data/interim/transactions_clean.parquet")

_NO_DATA = not CLEAN.exists()


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast
    pass excludes it with -m "not heavy", and it is skipped outright when
    the artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="run Task 2.2 first to build the cleaned parquet")(
        pytest.mark.heavy(fn)
    )



def _rows(records: list[dict]) -> pd.DataFrame:
    """Build a cleaned-shaped frame; unnamed fields take harmless defaults."""
    base = {
        "household_key": 1,
        "BASKET_ID": 1,
        "DAY": 1,
        "WEEK_NO": 1,
        "PRODUCT_ID": 1,
        "STORE_ID": 1,
        "QUANTITY": 1,
        "SALES_VALUE": 1.0,
        "RETAIL_DISC": 0.0,
        "COUPON_DISC": 0.0,
        "COUPON_MATCH_DISC": 0.0,
        "usable": True,
    }
    return pd.DataFrame([{**base, **r} for r in records])


@pytest.fixture
def simple() -> pd.DataFrame:
    """One product in two stores.

    Store 1 week 1: 2 units, paid 8.00, retail disc -2.00 → regular 5.00,
        paid 4.00, depth 0.20, all of it loyalty.
    Store 1 week 2: 2 units, paid 10.00, no discount → depth 0.00.
    Store 2 week 1: 1 unit, paid 3.00, coupon -1.00, match -0.50 →
        base 3.50, regular 3.50, paid 3.00, depth 1/7, all of it match;
        depth_manufacturer 1/3.5, excluded from depth.
    """
    return _rows(
        [
            {"STORE_ID": 1, "WEEK_NO": 1, "QUANTITY": 2, "SALES_VALUE": 8.0,
             "RETAIL_DISC": -2.0},
            {"STORE_ID": 1, "WEEK_NO": 2, "QUANTITY": 2, "SALES_VALUE": 10.0},
            {"STORE_ID": 2, "WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 3.0,
             "COUPON_DISC": -1.0, "COUPON_MATCH_DISC": -0.5},
        ]
    )


def _build(frame: pd.DataFrame, **kwargs):
    return build_price_panel(frame, run_level_test=False, **kwargs)


def test_reconstruction_a_prices_match_hand_arithmetic(simple) -> None:
    panel, _ = _build(simple)
    row = panel[(panel.STORE_ID == 1) & (panel.WEEK_NO == 1)].iloc[0]
    assert row["paid_price"] == pytest.approx(4.0)
    assert row["regular_price"] == pytest.approx(5.0)
    assert row["depth"] == pytest.approx(0.2, abs=1e-6)


def test_coupon_disc_is_excluded_from_the_shelf_price(simple) -> None:
    """Decision 3: the manufacturer coupon never moved the retailer's price."""
    panel, _ = _build(simple)
    row = panel[panel.STORE_ID == 2].iloc[0]
    # Reconstruction B would give (3.0 + 1.0 + 0.5) / 1 = 4.50.
    assert row["regular_price"] == pytest.approx(3.5)
    assert row["depth"] == pytest.approx(0.5 / 3.5, abs=1e-6)
    assert row["depth_manufacturer"] == pytest.approx(1.0 / 3.5, abs=1e-6)


def test_depth_equals_loyalty_plus_match_never_manufacturer(simple) -> None:
    panel, _ = _build(simple)
    priced = panel[~panel["price_undefined"]]
    residual = (
        priced["depth"] - (priced["depth_loyalty"] + priced["depth_match"])
    ).abs()
    assert float(residual.max()) < 1e-6
    assert (priced["depth_manufacturer"] > 0).any()  # fires, but stays out of depth


def test_surcharge_rows_are_excluded_not_absolute_valued() -> None:
    """Requirement 1: abs() on a positive RETAIL_DISC invents a discount."""
    frame = _rows(
        [
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 4.0, "RETAIL_DISC": 1.0},
            {"WEEK_NO": 2, "QUANTITY": 1, "SALES_VALUE": 4.0, "RETAIL_DISC": -1.0},
        ]
    )
    panel, diag = _build(frame)
    assert set(panel["WEEK_NO"]) == {2}  # the surcharge week is gone entirely
    exclusion = diag["exclusions"][0]
    assert exclusion["name"] == "retail_disc_surcharge"
    assert exclusion["effect"]["rows"] == 1
    assert exclusion["before"]["rows"] - exclusion["effect"]["rows"] == (
        exclusion["after"]["rows"]
    )
    # Had abs() been applied, the surcharge row would price at 5.00 and read as
    # a 20% discount. The surviving week is the genuine one.
    assert panel.iloc[0]["regular_price"] == pytest.approx(5.0)


def test_zero_quantity_group_yields_null_not_infinity() -> None:
    """Requirement 2: guard the divide."""
    frame = _rows([{"QUANTITY": 0, "SALES_VALUE": 0.0}])
    panel, diag = _build(frame)
    assert bool(panel.iloc[0]["price_undefined"]) is True
    assert pd.isna(panel.iloc[0]["paid_price"])
    assert pd.isna(panel.iloc[0]["regular_price"])
    assert pd.isna(panel.iloc[0]["depth"])
    assert diag["price_undefined"]["rows"] == 1


def test_free_goods_group_has_no_reconstructable_price() -> None:
    """Units bought, nothing paid, nothing discounted: the base is zero."""
    frame = _rows([{"QUANTITY": 3, "SALES_VALUE": 0.0}])
    panel, _ = _build(frame)
    assert bool(panel.iloc[0]["price_undefined"]) is True
    assert pd.isna(panel.iloc[0]["regular_price"])
    assert int(panel.iloc[0]["units"]) == 3  # the units are still reported


def test_on_deal_uses_tolerance_not_equality() -> None:
    """Requirement 3: a float-noise depth is not a deal."""
    frame = _rows(
        [
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 10.0,
             "RETAIL_DISC": -1e-13},
            {"WEEK_NO": 2, "QUANTITY": 1, "SALES_VALUE": 10.0, "RETAIL_DISC": -1.0},
        ]
    )
    panel, _ = _build(frame)
    by_week = panel.set_index("WEEK_NO")["on_deal"]
    assert bool(by_week[1]) is False
    assert bool(by_week[2]) is True
    assert PRICE_TOLERANCE > 1e-13


def test_bounded_flag_marks_a_product_always_on_deal() -> None:
    always = [
        {"PRODUCT_ID": 1, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 4.0,
         "RETAIL_DISC": -1.0}
        for w in range(1, 11)
    ]
    sometimes = [
        {"PRODUCT_ID": 2, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 5.0,
         "RETAIL_DISC": -1.0 if w <= 5 else 0.0}
        for w in range(1, 11)
    ]
    panel, diag = _build(_rows(always + sometimes))
    status = panel.groupby("PRODUCT_ID")["price_status"].first()
    assert status[1] == "bounded"
    assert status[2] == "identified"
    assert diag["product_status"]["thresholds"]["bounded_threshold"] == (
        BOUNDED_THRESHOLD
    )


def test_bounded_threshold_is_a_parameter() -> None:
    frame = _rows(
        [
            {"WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 4.0,
             "RETAIL_DISC": -1.0 if w <= 8 else 0.0}
            for w in range(1, 11)
        ]
    )
    lenient, _ = _build(frame, bounded_threshold=0.9)
    strict, _ = _build(frame, bounded_threshold=0.7)
    assert lenient["price_status"].iloc[0] == "identified"
    assert strict["price_status"].iloc[0] == "bounded"


# --------------------------------------------------------------------------
# The three-way status: identified, bounded, insufficient_support.
# --------------------------------------------------------------------------


def _weeks(product: int, n: int, *, on_deal: bool) -> list[dict]:
    return [
        {
            "PRODUCT_ID": product,
            "WEEK_NO": w,
            "QUANTITY": 1,
            "SALES_VALUE": 4.0,
            "RETAIL_DISC": -1.0 if on_deal else 0.0,
        }
        for w in range(1, n + 1)
    ]


def test_thin_product_is_insufficient_support_not_bounded() -> None:
    """One week, on deal, is not evidence of a perpetual deal."""
    panel, _ = _build(_rows(_weeks(1, 1, on_deal=True)))
    assert panel["price_status"].iloc[0] == "insufficient_support"


def test_thin_product_off_deal_is_also_insufficient_support() -> None:
    """Nor is one undiscounted week evidence of an identified depth."""
    panel, _ = _build(_rows(_weeks(1, 1, on_deal=False)))
    assert panel["price_status"].iloc[0] == "insufficient_support"


def test_support_is_tested_before_depth() -> None:
    """A product failing support never reaches the bounded test."""
    thin = _weeks(1, MIN_PRICED_WEEKS - 1, on_deal=True)
    thick = _weeks(2, MIN_PRICED_WEEKS, on_deal=True)
    panel, diag = _build(_rows(thin + thick))
    status = panel.groupby("PRODUCT_ID")["price_status"].first()
    assert status[1] == "insufficient_support"
    assert status[2] == "bounded"
    # Both have a deal share of 1.0; only the supported one is called bounded.
    assert diag["product_status"]["insufficient_support_detail"][
        "would_have_been_bounded"
    ] == 1


def test_all_three_statuses_are_counted() -> None:
    frame = _rows(
        _weeks(1, 10, on_deal=True)              # bounded
        + _weeks(2, 1, on_deal=True)             # insufficient_support
        + [
            {"PRODUCT_ID": 3, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 5.0,
             "RETAIL_DISC": -1.0 if w <= 5 else 0.0}
            for w in range(1, 11)
        ]                                        # identified
    )
    _, diag = _build(frame)
    counts = diag["product_status"]["products"]
    assert counts == {"bounded": 1, "insufficient_support": 1, "identified": 1}
    coverage = diag["product_status"]["coverage"]
    assert set(coverage) == {"identified", "bounded", "insufficient_support"}
    for entry in coverage.values():
        assert 0.0 <= entry["sales_value_share"] <= 1.0
    assert sum(e["products"] for e in coverage.values()) == 3


def test_min_priced_weeks_is_a_parameter_and_is_recorded() -> None:
    frame = _rows(_weeks(1, 5, on_deal=False))
    lenient, diag_l = _build(frame, min_priced_weeks=4)
    strict, diag_s = _build(frame, min_priced_weeks=6)
    assert lenient["price_status"].iloc[0] == "identified"
    assert strict["price_status"].iloc[0] == "insufficient_support"
    assert diag_l["product_status"]["thresholds"]["min_priced_weeks"] == 4
    assert diag_s["product_status"]["thresholds"]["min_priced_weeks"] == 6
    assert diag_s["product_status"]["thresholds"]["min_priced_weeks_rationale"]


def test_support_counts_distinct_weeks_not_product_store_weeks() -> None:
    """Ten stores in one week is one week of price history, not ten."""
    frame = _rows(
        [
            {"PRODUCT_ID": 1, "STORE_ID": s, "WEEK_NO": 1, "QUANTITY": 1,
             "SALES_VALUE": 4.0}
            for s in range(1, 11)
        ]
    )
    panel, _ = _build(frame)
    assert int(panel["n_psw_priced"].iloc[0]) == 10
    assert int(panel["n_weeks_priced"].iloc[0]) == 1
    assert panel["price_status"].iloc[0] == "insufficient_support"


def test_unpriced_product_is_insufficient_support_and_counted() -> None:
    """A product with no reconstructable price at all is a subset, not a fourth."""
    frame = _rows([{"PRODUCT_ID": 1, "QUANTITY": 3, "SALES_VALUE": 0.0}])
    panel, diag = _build(frame)
    assert panel["price_status"].iloc[0] == "insufficient_support"
    detail = diag["product_status"]["insufficient_support_detail"]
    assert detail["unpriced_products"] == 1


def test_units_and_sales_are_conserved_by_aggregation(simple) -> None:
    panel, diag = _build(simple)
    assert int(panel["units"].sum()) == diag["totals_after"]["units"]
    assert float(panel["sales_value"].sum()) == pytest.approx(
        diag["totals_after"]["sales_value"], abs=0.01
    )


def test_panel_has_one_row_per_group_and_no_zero_rows(simple) -> None:
    panel, _ = _build(simple)
    assert len(panel) == 3
    assert not panel.duplicated(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).any()
    assert (panel["n_rows"] > 0).all()


def test_level_test_recovers_a_correct_reconstruction() -> None:
    """A group with both kinds of row: A should hit the observed price exactly."""
    frame = _rows(
        [
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 5.0},  # undiscounted
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 4.0, "RETAIL_DISC": -1.0},
        ]
    )
    result = level_test(frame)
    psw = result["grains"]["product_store_week"]
    assert psw["groups"] == 1
    assert psw["median_abs_error"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert result["verdict"]["status"] == "A_NOT_WORSE"
    assert result["verdict"]["reopens_decision_3"] is False


def test_level_test_prefers_a_when_a_manufacturer_coupon_is_present() -> None:
    """B adds back a coupon that never moved the shelf price, and overshoots."""
    frame = _rows(
        [
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 5.0},
            {"WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 4.0,
             "RETAIL_DISC": -1.0, "COUPON_DISC": -2.0},
        ]
    )
    psw = level_test(frame)["grains"]["product_store_week"]
    assert psw["median_abs_error"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert psw["median_abs_error"]["B"] == pytest.approx(2.0, abs=1e-6)
    assert psw["closer_share"]["A"] == 1.0


def test_level_test_reports_not_run_when_no_group_supports_it() -> None:
    frame = _rows([{"QUANTITY": 1, "SALES_VALUE": 4.0, "RETAIL_DISC": -1.0}])
    result = level_test(frame)
    assert result["grains"]["product_store_week"]["groups"] == 0
    assert result["verdict"]["status"] == "NOT_RUN"


def test_writes_parquet_when_asked(simple, tmp_path: Path) -> None:
    out = tmp_path / "prices.parquet"
    panel, diag = _build(simple, out_path=out)
    assert diag["written_to"] == str(out)
    assert len(pd.read_parquet(out)) == len(panel)


# --------------------------------------------------------------------------
# Task 2.4 — the price index and deflation.
# --------------------------------------------------------------------------


def _flat(product: int, price: float, weeks: int = 12) -> list[dict]:
    """A product sold undiscounted at one constant price for `weeks` weeks."""
    return [
        {"PRODUCT_ID": product, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": price}
        for w in range(1, weeks + 1)
    ]


def test_constant_prices_give_a_flat_index() -> None:
    panel, _ = _build(_rows(_flat(1, 4.0) + _flat(2, 9.0)))
    index, diag = build_price_index(panel, min_matched=1)
    assert index["price_index"].round(9).nunique() == 1
    assert index["price_index"].iloc[0] == pytest.approx(1.0)
    assert diag["drift"]["full_span"]["total_drift"] == pytest.approx(0.0, abs=1e-9)


def test_uniform_inflation_is_recovered_exactly() -> None:
    """Every product up 10% a week for ten weeks: the index must find 1.1^9."""
    rows = []
    for product, base in ((1, 4.0), (2, 9.0)):
        rows += [
            {"PRODUCT_ID": product, "WEEK_NO": w, "QUANTITY": 1,
             "SALES_VALUE": base * 1.1 ** (w - 1)}
            for w in range(1, 11)
        ]
    panel, _ = _build(_rows(rows))
    index, diag = build_price_index(panel, min_matched=1)
    assert index["link"].iloc[1:].round(9).nunique() == 1
    assert float(index["link"].iloc[1]) == pytest.approx(1.1)
    assert index["price_index"].iloc[-1] == pytest.approx(1.1**9)
    assert diag["drift"]["full_span"]["total_drift"] == pytest.approx(1.1**9 - 1)


def test_index_is_a_geometric_not_arithmetic_mean() -> None:
    """One product doubles, one halves: a geometric mean returns exactly 1."""
    rows = [
        {"PRODUCT_ID": 1, "WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 4.0},
        {"PRODUCT_ID": 1, "WEEK_NO": 2, "QUANTITY": 1, "SALES_VALUE": 8.0},
        {"PRODUCT_ID": 2, "WEEK_NO": 1, "QUANTITY": 1, "SALES_VALUE": 8.0},
        {"PRODUCT_ID": 2, "WEEK_NO": 2, "QUANTITY": 1, "SALES_VALUE": 4.0},
    ]
    panel, _ = _build(_rows(rows))
    index, _ = build_price_index(panel, min_matched=1)
    # An arithmetic mean of relatives would give (2 + 0.5) / 2 = 1.25.
    assert float(index["link"].iloc[1]) == pytest.approx(1.0)


def test_promoted_rows_never_enter_the_index() -> None:
    """A deep markdown in one week must not read as deflation."""
    clean = _rows(_flat(1, 4.0) + _flat(2, 4.0))
    promoted = _rows(
        [
            {"PRODUCT_ID": 3, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 1.0,
             "RETAIL_DISC": -3.0}
            for w in range(1, 13)
        ]
    )
    panel, _ = _build(pd.concat([clean, promoted], ignore_index=True))
    _, diag = build_price_index(panel, min_matched=1)
    assert diag["drift"]["full_span"]["total_drift"] == pytest.approx(0.0, abs=1e-9)
    # The promoted product contributed no pairs at all.
    assert diag["relatives"]["pairs"] == 2 * 11


def test_unmatched_products_cannot_move_the_index() -> None:
    """A new expensive product appearing mid-span is assortment, not inflation."""
    rows = _flat(1, 4.0) + [
        {"PRODUCT_ID": 2, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 50.0}
        for w in range(6, 13)
    ]
    panel, _ = _build(_rows(rows))
    index, _ = build_price_index(panel, min_matched=1)
    assert index["price_index"].round(9).nunique() == 1


def test_store_mix_does_not_masquerade_as_a_price_move() -> None:
    """Same product, two stores, two prices; the cheap store stops selling.

    At product grain the unit value jumps. At product-store grain — the default
    — nothing moved, because nothing did.
    """
    rows = [
        {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": w, "QUANTITY": 1,
         "SALES_VALUE": 10.0}
        for w in range(1, 13)
    ] + [
        {"PRODUCT_ID": 1, "STORE_ID": 2, "WEEK_NO": w, "QUANTITY": 1,
         "SALES_VALUE": 2.0}
        for w in range(1, 7)
    ]
    panel, _ = _build(_rows(rows))
    by_store, _ = build_price_index(panel, grain="product_store", min_matched=1)
    by_product, _ = build_price_index(panel, grain="product", min_matched=1)
    assert by_store["price_index"].round(9).nunique() == 1
    assert float(by_product["price_index"].iloc[-1]) > 1.5


def test_thin_link_is_imputed_and_named() -> None:
    """A week resting on too few pairs is flagged, not quietly believed."""
    panel, _ = _build(_rows(_flat(1, 4.0)))
    index, diag = build_price_index(panel, min_matched=5)
    assert diag["support"]["weeks_link_imputed"] == 11
    assert diag["support"]["imputed_weeks"] == list(range(2, 13))
    assert (index.loc[index["link_imputed"], "link"] == 1.0).all()
    # The base week has no predecessor and is not an imputation.
    assert bool(index["link_imputed"].iloc[0]) is False


def test_grain_must_be_one_of_two() -> None:
    panel, _ = _build(_rows(_flat(1, 4.0)))
    with pytest.raises(ValueError, match="grain must be one of"):
        build_price_index(panel, grain="household")


def test_deflation_adds_real_columns_and_keeps_nominal() -> None:
    rows = []
    for product, base in ((1, 4.0), (2, 9.0)):
        rows += [
            {"PRODUCT_ID": product, "WEEK_NO": w, "QUANTITY": 1,
             "SALES_VALUE": base * 1.1 ** (w - 1)}
            for w in range(1, 11)
        ]
    panel, _ = _build(_rows(rows))
    index, _ = build_price_index(panel, min_matched=1)
    deflated, diag = deflate_prices(panel, index)

    assert "paid_price" in deflated  # nominal survives untouched
    assert deflated["paid_price"].equals(panel["paid_price"])
    assert diag["nominal_preserved"] is True
    # A price rising exactly with the index is flat in real terms.
    real = deflated[deflated["PRODUCT_ID"] == 1].sort_values("WEEK_NO")
    assert real["real_paid_price"].round(6).nunique() == 1


def test_deflation_leaves_depth_unchanged() -> None:
    """Depth is a within-week ratio, so the index cancels."""
    rows = [
        {"PRODUCT_ID": 1, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 4.0}
        for w in range(1, 11)
    ] + [
        {"PRODUCT_ID": 2, "WEEK_NO": w, "QUANTITY": 1, "SALES_VALUE": 4.0,
         "RETAIL_DISC": -1.0}
        for w in range(1, 11)
    ]
    panel, _ = _build(_rows(rows))
    index, _ = build_price_index(panel, min_matched=1)
    deflated, _ = deflate_prices(panel, index)
    assert deflated["depth"].equals(panel["depth"])


def test_deflation_refuses_a_week_with_no_index_value() -> None:
    panel, _ = _build(_rows(_flat(1, 4.0)))
    index, _ = build_price_index(panel, min_matched=1)
    with pytest.raises(AssertionError, match="no index value"):
        deflate_prices(panel, index.iloc[:3])


# --------------------------------------------------------------------------
# The real file.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_panel():
    return build_price_panel(CLEAN)


@real_data
def test_real_panel_conserves_units_and_sales(real_panel) -> None:
    panel, diag = real_panel
    assert int(panel["units"].sum()) == diag["totals_after"]["units"]
    assert float(panel["sales_value"].sum()) == pytest.approx(
        diag["totals_after"]["sales_value"], abs=0.01
    )


@real_data
def test_real_surcharge_exclusion_is_tiny_and_recorded(real_panel) -> None:
    """Task 1.3 found 36 positive RETAIL_DISC rows; few survive 2.2's cascade."""
    _, diag = real_panel
    effect = diag["exclusions"][0]["effect"]
    assert 0 < effect["rows"] <= 36


@real_data
def test_real_depth_identity_holds(real_panel) -> None:
    panel, _ = real_panel
    priced = panel[~panel["price_undefined"]]
    residual = (
        priced["depth"].astype("float64")
        - (
            priced["depth_loyalty"].astype("float64")
            + priced["depth_match"].astype("float64")
        )
    ).abs()
    assert float(residual.max()) < 1e-6


@real_data
def test_real_depth_median_matches_phase_1(real_panel) -> None:
    """Task 1.3 put the median loyalty depth near 24.6% where it fires."""
    _, diag = real_panel
    median = diag["depth"]["components"]["loyalty_median_where_fires"]
    assert 0.15 < median < 0.35


@real_data
def test_real_status_split_separates_thin_from_perpetual(real_panel) -> None:
    """The bounded count must no longer be dominated by one-week products."""
    panel, diag = real_panel
    ps = diag["product_status"]
    assert set(ps["products"]) <= {"identified", "bounded", "insufficient_support"}
    bounded = panel[panel["price_status"] == "bounded"]
    assert int(bounded["n_weeks_priced"].min()) >= MIN_PRICED_WEEKS
    # Most of what the old rule called bounded was actually thin.
    assert ps["insufficient_support_detail"]["would_have_been_bounded"] > (
        ps["products"]["bounded"]
    )
    # And the thin status costs many products but little money.
    thin = ps["coverage"]["insufficient_support"]
    assert thin["products_share"] > 0.5
    assert thin["sales_value_share"] < 0.2


@real_data
def test_real_index_rests_on_real_support(real_panel) -> None:
    panel, _ = real_panel
    index, diag = build_price_index(panel)
    assert diag["support"]["weeks_link_imputed"] == 0
    assert diag["support"]["matched_pairs_min"] >= MIN_MATCHED_PAIRS
    assert len(index) == 102
    assert index["price_index"].iloc[0] == pytest.approx(1.0)
    assert index["price_index"].notna().all()


@real_data
def test_real_drift_is_not_carried_by_outliers(real_panel) -> None:
    """The trimmed and untrimmed drifts must agree in sign and rough size."""
    panel, _ = real_panel
    _, diag = build_price_index(panel)
    full = diag["drift"]["full_span"]["total_drift"]
    trimmed = diag["robustness"]["trimmed_5pct_total_drift"]
    assert full > 0 and trimmed > 0
    assert abs(full - trimmed) < 0.05


@real_data
def test_real_pooled_and_matched_drift_disagree_in_sign(real_panel) -> None:
    """The composition finding, asserted so it cannot quietly go away.

    The pool of undiscounted observations roughly doubles across the span and
    the entrants are cheaper, so the naive average falls while a fixed item's
    price rises. If this ever stops holding, the index's interpretation changes
    and the write-up in docs/data_findings.md needs revisiting.
    """
    panel, _ = real_panel
    _, diag = build_price_index(panel)
    comp = diag["composition"]
    assert comp["pooled_drift"] < 0
    assert comp["balanced_drift"] > 0
    assert diag["drift"]["full_span"]["total_drift"] > 0
    assert comp["pool_size_late"] > comp["pool_size_early"]


@real_data
def test_real_deflation_leaves_depth_and_nominal_untouched(real_panel) -> None:
    panel, _ = real_panel
    index, _ = build_price_index(panel)
    deflated, diag = deflate_prices(panel, index)
    assert deflated["paid_price"].equals(panel["paid_price"])
    assert deflated["depth"].equals(panel["depth"])
    assert diag["effect"]["mean_paid_price_real"] < diag["effect"][
        "mean_paid_price_nominal"
    ]


@real_data
def test_real_level_test_ran_and_did_not_reopen_decision_3(real_panel) -> None:
    _, diag = real_panel
    verdict = diag["level_test"]["verdict"]
    assert verdict["status"] != "NOT_RUN"
    assert verdict["reopens_decision_3"] is False


@real_data
def test_real_diagnostics_are_json_serialisable(real_panel, tmp_path: Path) -> None:
    import json

    _, diag = real_panel
    path = write_diagnostics(diag, tmp_path / "prices_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "build_price_panel"
