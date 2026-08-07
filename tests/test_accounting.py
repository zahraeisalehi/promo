"""Tests for Task 5.1, the two components of promotional cost.

The fixture is small and hand-checkable: the subsidy and the free-goods value
are both known by arithmetic, so a sign error or a dropped mechanic fails here
rather than in a diagnostics file nobody reads.

One test does not assert what the plan asked for, and says why: free-good units
currently *do* reach the lift numerator, so a test claiming they never do would
be false. It measures the contamination instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.accounting import (
    MARGIN_GRID,
    MARGIN_REQUIRING_FIGURES,
    MARGIN_SOURCES,
    MECHANICS,
    UnpricedFreeGoodsError,
    UnstampedMarginError,
    adjust_lift_for_free_goods,
    assert_margin_stamped,
    breakeven_margin,
    campaign_accounting,
    free_goods,
    free_goods_in_lift,
    promo_cost,
    sensitivity_table,
    subsidy,
    write_diagnostics,
)

CLEAN = Path("data/interim/transactions_clean.parquet")
PRICES = Path("data/interim/prices.parquet")
PANEL = Path("data/interim/panel.parquet")


def real_data(fn):
    missing = not (CLEAN.exists() and PRICES.exists() and PANEL.exists())
    return pytest.mark.skipif(missing, reason="run Phase 2 first")(
        pytest.mark.heavy(fn)
    )


def _transactions() -> pd.DataFrame:
    """Four sold rows and two giveaways, all arithmetic doable by hand.

    Discounts arrive negative, as they do in the raw file.
    """
    rows = [
        # product 1: 10 units, 20.00 paid, 3.00 loyalty + 1.00 manufacturer
        {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 5, "QUANTITY": 10, "SALES_VALUE": 20.0,
             "RETAIL_DISC": -3.0, "COUPON_DISC": -1.0, "COUPON_MATCH_DISC": 0.0,
             "COMMODITY_DESC": "SOUP", "usable": True},
        # product 1, another store: 5 units, 2.00 match
        {"PRODUCT_ID": 1, "STORE_ID": 11, "WEEK_NO": 5, "QUANTITY": 5, "SALES_VALUE": 12.0,
             "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": -2.0,
             "COMMODITY_DESC": "SOUP", "usable": True},
        # product 2: no discount at all
        {"PRODUCT_ID": 2, "STORE_ID": 10, "WEEK_NO": 5, "QUANTITY": 4, "SALES_VALUE": 8.0,
             "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0,
             "COMMODITY_DESC": "SOUP", "usable": True},
        # product 1 in a later week, out of a week-scoped campaign
        {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 9, "QUANTITY": 7, "SALES_VALUE": 15.0,
             "RETAIL_DISC": -1.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0,
             "COMMODITY_DESC": "SOUP", "usable": True},
        # a giveaway of product 1 — priceable
        {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 5, "QUANTITY": 3, "SALES_VALUE": 0.0,
             "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0,
             "COMMODITY_DESC": "SOUP", "usable": True},
        # a giveaway of product 3 — no price anywhere, and not `usable`, which
        # must not keep it out: this is money, not units.
        {"PRODUCT_ID": 3, "STORE_ID": 10, "WEEK_NO": 5, "QUANTITY": 6, "SALES_VALUE": 0.0,
             "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0,
             "COMMODITY_DESC": "SOUP", "usable": False},
    ]
    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    """Product 1 at 2.00, product 2 at 3.00. Product 3 has no price at all."""
    return pd.DataFrame(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 5, "regular_price": 2.0},
            {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 6, "regular_price": 2.0},
            {"PRODUCT_ID": 2, "STORE_ID": 10, "WEEK_NO": 5, "regular_price": 3.0},
            {"PRODUCT_ID": 3, "STORE_ID": 10, "WEEK_NO": 5, "regular_price": None},
        ]
    )


# --- the subsidy --------------------------------------------------------------


def test_the_subsidy_is_positive_money_split_by_bearer():
    _, diag = subsidy(_transactions())

    # 3.00 loyalty + 1.00 later week, 1.00 manufacturer, 2.00 match.
    assert diag["by_mechanic"] == {
        "loyalty": 4.0,
        "manufacturer": 1.0,
        "coupon_match": 2.0,
    }
    assert diag["subsidy_total"] == 7.0
    assert set(MECHANICS) == set(diag["by_mechanic"])
    assert "cost bearer differs" in diag["why_split"]


def test_the_subsidy_is_charged_on_every_unit_not_the_incremental_ones():
    per_cell, diag = subsidy(_transactions())
    # 10 + 5 + 4 + 7 sold units; the giveaway rows carry no discount.
    assert diag["units"] == 26.0
    assert "not only on the incremental ones" in diag["on_all_units"]
    assert per_cell["subsidy"].sum() == pytest.approx(7.0)


def test_a_week_scope_narrows_the_subsidy():
    _, diag = subsidy(_transactions(), weeks=(5, 5))
    # Week 9's 1.00 loyalty drops out.
    assert diag["by_mechanic"]["loyalty"] == 3.0
    assert diag["subsidy_total"] == 6.0


# --- free goods ---------------------------------------------------------------


def test_giveaways_are_valued_at_the_regular_price_never_zero():
    _, diag = free_goods(_transactions(), _prices())

    assert diag["units"] == 9.0            # 3 of product 1 + 6 of product 3
    assert diag["priced"]["units"] == 3.0
    assert diag["priced"]["value"] == 6.0  # 3 units x 2.00 regular
    assert "never the paid price and never zero" in diag["valuation"]


def test_unpriceable_giveaways_are_refused_not_zeroed():
    _, diag = free_goods(_transactions(), _prices())

    assert diag["unpriced"]["units"] == 6.0
    assert diag["unpriced"]["products"] == 1
    assert diag["unpriced"]["value"] is None
    assert "rather than valued at zero" in diag["unpriced"]["why_not_zero"]


def test_the_usable_filter_is_not_applied_to_free_goods():
    """A free line costs the regular price whether or not its units compare."""
    _, diag = free_goods(_transactions(), _prices())
    # Product 3's giveaway has usable=False and must still be counted.
    assert diag["units"] == 9.0
    assert "claim about money" in diag["usable_filter_not_applied"]


def test_a_product_is_priced_from_its_own_history_not_its_own_cell():
    """The free line is often the only row in its cell, so pricing is by product."""
    transactions = _transactions()
    prices = _prices()
    # Remove product 1's week-5 price. Week 6 must still value the giveaway.
    prices = prices[~((prices.PRODUCT_ID == 1) & (prices.WEEK_NO == 5))]

    _, diag = free_goods(transactions, prices)
    assert diag["priced"]["value"] == 6.0


# --- the total ----------------------------------------------------------------


def test_the_total_is_both_components_and_reports_the_shortfall():
    per_cell, diag = promo_cost(_transactions(), _prices())

    assert diag["components"]["subsidy"] == 7.0
    assert diag["components"]["free_goods"] == 6.0
    assert diag["promo_cost_total"] == 13.0
    assert per_cell["promo_cost"].sum() == pytest.approx(13.0)
    # The total is a lower bound, and says by how much.
    assert diag["lower_bound_by"]["free_goods_unpriced_units"] == 6.0
    assert "Never the subsidy alone" in diag["for_downstream"]


def test_strict_refuses_a_total_that_would_be_a_lower_bound():
    with pytest.raises(UnpricedFreeGoodsError, match="no reconstructable price"):
        promo_cost(_transactions(), _prices(), strict=True)


def test_the_components_are_reported_apart_as_well_as_summed():
    """Phase 7 needs to tell a discount campaign from a giveaway campaign."""
    _, diag = promo_cost(_transactions(), _prices())
    shares = diag["components"]
    assert shares["subsidy_share"] + shares["free_goods_share"] == pytest.approx(1.0)
    assert "structurally different instrument" in diag["why_two_components"]


def test_write_diagnostics_round_trips(tmp_path):
    import json

    _, diag = promo_cost(_transactions(), _prices())
    path = write_diagnostics(diag, tmp_path / "cost.json")
    assert json.loads(path.read_text())["stage"] == "promo_cost"


# --- the double count the plan asked me to assert away ------------------------


def test_free_units_reaching_the_lift_numerator_are_measured_not_asserted():
    """The plan asks for a test that fails if giveaways enter the numerator.

    On this panel they do, so a test asserting they do not would be false. This
    one pins the *measurement* instead: the check exists, it reports the
    overlap, and it names the consequence. If Phase 2.2 is ever changed to keep
    giveaway units out of `units`, `clean` flips to True and this test still
    passes — at which point the stronger assertion becomes available.
    """
    panel = pd.DataFrame(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 10, "WEEK_NO": 5, "units": 13.0, "treated": True},
            {"PRODUCT_ID": 2, "STORE_ID": 10, "WEEK_NO": 5, "units": 4.0, "treated": False},
        ]
    )
    report = free_goods_in_lift(_transactions(), panel)

    assert report["units_in_treated_weeks"] == 3.0
    assert report["clean"] is False
    assert "CONTAMINATED" in report["status"]
    assert "both sides of the ratio" in report["why_it_matters"]


def test_a_panel_without_giveaway_units_reads_as_clean():
    panel = pd.DataFrame(
        [{"PRODUCT_ID": 2, "STORE_ID": 10, "WEEK_NO": 5, "units": 4.0, "treated": True}]
    )
    report = free_goods_in_lift(_transactions(), panel)
    assert report["units_in_treated_weeks"] == 0.0
    assert report["clean"] is True
    assert report["status"] == "clean"


# --- the real panel -----------------------------------------------------------


@real_data
def test_the_real_free_goods_are_the_phase_1_rows():
    _, diag = free_goods(CLEAN, PRICES)

    assert diag["rows"] == 4_451
    assert diag["units"] == 4_544.0
    assert diag["priced"]["units"] == 4_069.0
    assert diag["unpriced"]["units"] == 475.0
    assert diag["unpriced"]["products"] == 23
    assert diag["priced"]["value"] > 15_000


@real_data
def test_the_real_double_count_is_live_and_measured():
    report = free_goods_in_lift(CLEAN, PANEL)
    assert report["units_in_panel"] == 360.0
    assert report["units_in_treated_weeks"] == 85.0
    assert report["clean"] is False


# --- Task 5.2: break-even, the sensitivity table, and the ROI gate ------------


def test_the_break_even_margin_is_an_interval_not_a_point():
    """The numerator is known; only the denominator is estimated."""
    result = breakeven_margin(1000.0, 5000.0, interval=(4000.0, 6000.0))

    assert result["m_star"] == pytest.approx(0.20)
    low, high = result["m_star_interval"]
    # The module rounds to six places; the interval inverts because a ratio is
    # monotone in its denominator.
    assert low == pytest.approx(1000.0 / 6000.0, abs=1e-6)
    assert high == pytest.approx(1000.0 / 4000.0, abs=1e-6)
    assert result["reason_code"] is None
    assert "money that left the till" in result["numerator_is_known"]


def test_a_denominator_spanning_zero_returns_roi_unbounded():
    result = breakeven_margin(1000.0, 500.0, interval=(-200.0, 900.0))
    assert result["m_star"] is None
    assert result["reason_code"] == "ROI_UNBOUNDED"
    assert "no finite bound" in result["why"]


def test_a_break_even_above_fifty_percent_is_kappa_impossible():
    """m_star > 0.5 and kappa_star(0.5) > 1 are the same sentence."""
    result = breakeven_margin(1000.0, 1500.0)
    assert result["m_star"] == pytest.approx(2 / 3)
    assert result["reason_code"] == "KAPPA_IMPOSSIBLE"
    assert result["exceeds_plausible_margin"] is True
    assert "kappa_star(m) = m_star / m" in result["identity"]


def test_break_even_is_computable_where_lift_is_not_resolvable():
    """The point of the measure: it needs no counterfactual to be defined.

    A campaign whose lift cannot be separated from noise still has a known cost
    and an observed revenue, so the margin it *needed* is arithmetic. Only the
    interval — the part that comes from the estimate — is affected.
    """
    point_only = breakeven_margin(1000.0, 5000.0)
    assert point_only["m_star"] == pytest.approx(0.20)
    assert point_only["m_star_interval"] is None
    assert point_only["reason_code"] is None


def test_the_sensitivity_table_is_the_nine_point_grid_and_flips_at_m_star():
    table, diag = sensitivity_table(1000.0, 5000.0)

    assert list(table["margin"]) == list(MARGIN_GRID)
    assert len(table) == 9
    # m_star is 0.20, so the sign flips exactly there.
    assert table.loc[table.margin == 0.20, "incremental_profit"].iloc[0] == 0.0
    assert (table.loc[table.margin < 0.20, "incremental_profit"] < 0).all()
    assert (table.loc[table.margin > 0.20, "incremental_profit"] > 0).all()
    assert diag["profitable_from_margin"] == 0.25
    assert "MARGIN_GRID is imported" in diag["grid_authority"]


def test_a_cell_whose_sign_is_uncertain_is_marked_not_shown_as_confident():
    table, diag = sensitivity_table(1000.0, 5000.0, interval=(4000.0, 6000.0))
    uncertain = table.loc[~table["sign_certain"], "margin"]
    assert set(uncertain) == {0.20, 0.25}
    assert diag["uncertain_cells"] == 2


def test_the_grid_is_imported_from_audit_never_restated():
    from promo.audit import MARGIN_GRID as AUDIT_GRID

    assert MARGIN_GRID is AUDIT_GRID


# --- the reporting-time correction for contaminated units ---------------------


def test_the_free_goods_correction_is_exact_and_says_so():
    result = adjust_lift_for_free_goods(500.0, 85.0, interval=(300.0, 700.0))

    assert result["gross_incremental_adjusted"] == 415.0
    assert result["interval_adjusted"] == [215.0, 615.0]
    assert result["exact"] is True
    assert "identified individually by key and week" in result["why"]
    assert "clean.py" in result["structural_fix_deferred"]


def test_roi_unbounded_fires_through_run_audit():
    """The last owed reason code, per the gate-authoring skill."""
    from promo.gates import CampaignSpec, run_audit

    spanning = breakeven_margin(1000.0, 500.0, interval=(-200.0, 900.0))
    results, audit = run_audit(
        CampaignSpec(name="c"), _panel_for_audit(), run_overlap=False,
        stop_on_refuse=False, breakeven=spanning,
    )
    roi = next(r for r in results if r.gate == "roi")
    assert roi.reason_code == "ROI_UNBOUNDED"
    assert roi.status == "bounded"
    assert "ROI_UNBOUNDED" in audit["bounded"]
    assert "crosses zero" in roi.message


def test_roi_unbounded_does_not_fire_on_a_bounded_denominator():
    from promo.gates import CampaignSpec, run_audit

    bounded = breakeven_margin(1000.0, 5000.0, interval=(4000.0, 6000.0))
    results, _ = run_audit(
        CampaignSpec(name="c"), _panel_for_audit(), run_overlap=False,
        stop_on_refuse=False, breakeven=bounded,
    )
    roi = next(r for r in results if r.gate == "roi")
    assert roi.reason_code is None
    assert roi.status == "pass"
    assert "20.0%" in roi.message


def test_an_impossible_break_even_refuses_via_kappa_impossible():
    from promo.gates import CampaignSpec, run_audit

    results, _ = run_audit(
        CampaignSpec(name="c"), _panel_for_audit(), run_overlap=False,
        stop_on_refuse=False, breakeven=breakeven_margin(1000.0, 1500.0),
    )
    roi = next(r for r in results if r.gate == "roi")
    assert roi.reason_code == "KAPPA_IMPOSSIBLE"
    assert roi.status == "refuse"


def _panel_for_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": w, "units": 5.0,
             "sales_value": 10.0, "treated": w in (3, 4), "in_mailer": False,
             "price_status": "identified", "depth": 0.2,
             "treatment_observed": True, "on_display": w in (3, 4)}
            for w in range(1, 9)
        ]
    )


# --- campaign_accounting: the three per-campaign objects ----------------------


def _campaign():
    from promo.lift import LiftCampaign

    return LiftCampaign(name="demo", commodity="SOUP", product=1, weeks=(5, 5))


def _tx() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 5, "QUANTITY": 100,
             "SALES_VALUE": 200.0, "RETAIL_DISC": -40.0, "COUPON_DISC": 0.0,
             "COUPON_MATCH_DISC": 0.0, "COMMODITY_DESC": "SOUP", "usable": True}
        ]
    )


def _pr() -> pd.DataFrame:
    return pd.DataFrame(
        [{"PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 5, "regular_price": 3.0}]
    )


def _accounted(**kwargs):
    return campaign_accounting(
        _campaign(), gross_incremental=50.0, interval=(30.0, 70.0),
        transactions=_tx(), prices=_pr(), **kwargs,
    )


def test_a_campaign_gets_a_breakeven_interval_a_table_and_a_verdict():
    """Phase 5's first three conditions, in one object."""
    table, diag = _accounted()

    assert diag["breakeven"]["m_star"] is not None
    assert diag["breakeven"]["m_star_interval"] is not None      # an interval
    assert list(table["margin"]) == list(MARGIN_GRID)            # nine columns
    assert len(table) == 9
    assert diag["reason_code"] == "NO_MARGIN"                    # a stated refusal
    assert diag["promo_cost"]["promo_cost_total"] == 40.0


def test_an_unbounded_denominator_gives_a_stated_refusal_not_a_number():
    _, diag = campaign_accounting(
        _campaign(), gross_incremental=1.0, interval=(-50.0, 50.0),
        transactions=_tx(), prices=_pr(),
    )
    assert diag["breakeven"]["m_star"] is None
    assert diag["breakeven"]["reason_code"] == "ROI_UNBOUNDED"


def test_the_price_comes_from_the_campaigns_own_promoted_weeks():
    _, diag = _accounted()
    assert diag["promoted_price"] == pytest.approx(2.0)          # 200.0 / 100
    assert diag["incremental_revenue"] == pytest.approx(100.0)   # 50 x 2.00


# --- Task 5.3: a supplied margin is an assumption and is labelled as one ------


def test_without_a_margin_the_margin_figures_refuse_rather_than_guess():
    _, diag = _accounted()

    assert diag["conditional"] is None
    assert diag["margin_source"] is None
    assert diag["conditional_on_margin"] is None
    assert diag["reason_code"] == "NO_MARGIN"
    assert "no COGS or margin column exists" in diag["why_no_conditional"]


def test_a_supplied_margin_is_stamped_on_every_figure_it_touches():
    _, diag = _accounted(margin=0.30)
    cond = diag["conditional"]

    assert cond["margin_source"] == "supplied"
    assert cond["conditional_on_margin"] == 0.30
    for figure in MARGIN_REQUIRING_FIGURES:
        assert figure in cond
    assert cond["incremental_profit"] == pytest.approx(0.30 * 100.0 - 40.0)
    assert "margin you supplied" in cond["reads_as"]
    assert "not a measurement" in cond["not_a_measurement"].lower()


def test_a_supplied_margin_never_replaces_the_measured_objects():
    """Rule 1: break-even and the table are what the data establishes."""
    table, diag = _accounted(margin=0.30)

    assert diag["breakeven"]["m_star"] is not None
    assert len(table) == 9
    assert "never replaces them" in diag["measured_objects_always_ship"]


def test_no_margin_derived_figure_can_be_serialised_without_its_stamp():
    """The guarantee Task 5.3 asks for, enforced by shape.

    Margin-derived figures live only inside the stamped container, so the check
    passes by construction — and the checker is proven able to fail by handing
    it a figure placed anywhere else.
    """
    _, diag = _accounted(margin=0.30)
    assert_margin_stamped(diag)                    # the real object passes

    naked = {"campaign": "demo", "summary": {"roi": 1.4}}
    with pytest.raises(UnstampedMarginError, match="without margin_source"):
        assert_margin_stamped(naked)

    nested = {"campaigns": [{"incremental_profit": 10.0}]}
    with pytest.raises(UnstampedMarginError):
        assert_margin_stamped(nested)


def test_derived_is_a_legal_source_but_unreachable_on_this_dataset():
    """Present so a future dataset with COGS need not invent the field."""
    assert MARGIN_SOURCES == (None, "supplied", "derived")

    with pytest.raises(ValueError, match="margin_source must be one of"):
        _accounted(margin=0.30, margin_source="guessed")
    with pytest.raises(ValueError, match="no margin supplied"):
        _accounted(margin_source="supplied")


def test_a_margin_supplied_without_a_source_is_stamped_supplied_anyway():
    """The stamp is the point, so it is defaulted rather than trusted."""
    _, diag = _accounted(margin=0.25)
    assert diag["conditional"]["margin_source"] == "supplied"
