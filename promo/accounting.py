"""What a promotion cost the retailer. Two components, never one.

Task 5.1. Promotional cost is **discount subsidy plus free goods**, and the
second is the one that gets dropped. Counting only the first understates the
denominator of every ROI in the system, so both are computed here, reported
separately, and summed once.

**The subsidy is paid on every unit sold, not on the incremental ones.** A
shopper who would have bought anyway still takes the discount. Charging the
promotion only for the units it created is the single most common way to make a
promotion look profitable, and it is wrong: the money left the till either way.
The three mechanics stay split — `RETAIL_DISC` is the retailer's own loyalty
discount, `COUPON_DISC` is manufacturer-funded, `COUPON_MATCH_DISC` is the
retailer matching a manufacturer coupon — because **the cost bearer differs**,
and a campaign whose subsidy is mostly manufacturer money is a different
instrument from one the retailer funded itself.

**Free goods are valued at the regular price, never at zero and never at the
paid price.** A row with `QUANTITY > 0` and `SALES_VALUE == 0` is a unit handed
over: a BOGOF arm, a sampling giveaway, a store-funded free line. The
transaction file records it as revenue zero, which is true of the revenue and
false of the cost. On this dataset that is **4,451 rows carrying 4,544 units** —
a rounding error in volume, 0.00% of total units, and real money once priced.

**Where the price cannot be reconstructed, the units are refused, not zeroed.**
Valuing an unpriceable giveaway at zero would quietly restore the error this
module exists to fix. They come back as `free_goods_unpriced` and stay out of
the total, so a reader sees that the total is a lower bound and by how much.

**Do not reuse Phase 2.2's volume-measured exclusion here.** That filter drops
rows for being unit-incomparable, which is a statement about *units*. This is a
statement about *money*: a free line costs the full regular price whether or not
its units are comparable with anything. The free-good rows are read with the
`usable` filter deliberately not applied, and the diagnostics say so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.audit import MARGIN_GRID as _MARGIN_GRID
from promo.io import connect

__all__ = [
    "MARGIN_GRID",
    "MARGIN_REQUIRING_FIGURES",
    "MARGIN_SOURCES",
    "MECHANICS",
    "UnpricedFreeGoodsError",
    "UnstampedMarginError",
    "adjust_lift_for_free_goods",
    "assert_margin_stamped",
    "breakeven_margin",
    "campaign_accounting",
    "free_goods",
    "free_goods_in_lift",
    "promo_cost",
    "sensitivity_table",
    "subsidy",
    "write_diagnostics",
]

#: Where a margin came from. `"derived"` is unreachable on this dataset — no
#: COGS column exists in any of the eight tables (`promo/io.py` establishes it
#: at ingest) — and is present only so a future dataset carrying cost does not
#: need the field invented.
MARGIN_SOURCES: tuple[str | None, ...] = (None, "supplied", "derived")

#: Figures that cannot exist without a margin. Every one of them lives inside a
#: stamped container or not at all — see `stamp` in `campaign_accounting`.
MARGIN_REQUIRING_FIGURES: tuple[str, ...] = (
    "incremental_profit",
    "roi",
    "margin_headroom",
)

#: The assumed-margin grid, imported rather than restated. Task 5.2 is the
#: authority on it and Task 3.4's kappa sweep uses the same nine points, so two
#: tables keyed on margin cannot drift apart.
MARGIN_GRID: tuple[float, ...] = _MARGIN_GRID

#: The three discount columns, and who bears each. Kept apart through the whole
#: pipeline — settled decision 3 and the data-layer rules both require it.
MECHANICS: dict[str, str] = {
    "loyalty": "RETAIL_DISC",
    "manufacturer": "COUPON_DISC",
    "coupon_match": "COUPON_MATCH_DISC",
}

_KEY = ("PRODUCT_ID", "STORE_ID", "WEEK_NO")


class UnstampedMarginError(Exception):
    """A margin-derived figure appeared without its provenance.

    The failure this prevents is specific: a user types 30% into a box during a
    demo, and four screens later a ranked list of returns looks like a
    measurement. It is arithmetic conditional on a number the user made up, and
    every figure carrying it must keep saying so.
    """


class UnpricedFreeGoodsError(Exception):
    """Free goods were given away that no reconstructed price can value.

    Only raised when a caller asks for a total that pretends otherwise. The
    default is to report them and leave them out, because a cost total that
    silently swallows unpriceable giveaways is the understatement this module
    was written to prevent.
    """


def _source(
    frame: str | Path | pd.DataFrame, con: duckdb.DuckDBPyConnection, alias: str
) -> str:
    if isinstance(frame, pd.DataFrame):
        con.register(alias, frame)
        return alias
    return f"read_parquet('{Path(frame).as_posix()}')"


def _scope_sql(
    products: tuple[int, ...] | None,
    stores: tuple[int, ...] | None,
    weeks: tuple[int, int] | None,
    commodity: str | None,
) -> str:
    filters = []
    if products:
        filters.append(f"PRODUCT_ID IN ({','.join(str(int(p)) for p in products)})")
    if stores:
        filters.append(f"STORE_ID IN ({','.join(str(int(s)) for s in stores)})")
    if weeks:
        filters.append(f"WEEK_NO BETWEEN {int(weeks[0])} AND {int(weeks[1])}")
    if commodity:
        escaped = commodity.replace("'", "''")
        filters.append(f"COMMODITY_DESC = '{escaped}'")
    return " AND ".join(filters) if filters else "TRUE"


def subsidy(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    products: tuple[int, ...] | None = None,
    stores: tuple[int, ...] | None = None,
    weeks: tuple[int, int] | None = None,
    commodity: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Discount paid on **all** units sold in scope, split by who bore it.

    The three mechanics arrive negative in the raw file and are returned as
    positive money given away. Their sum is the subsidy; the split is kept
    because a manufacturer-funded discount is not the retailer's cost.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        src = _source(transactions, con, "_subsidy_tx")
        where = _scope_sql(products, stores, weeks, commodity)
        # `usable` applies here: a subsidy is a per-unit discount and the
        # volume-measured rows are the ones whose units cannot be compared.
        per_cell = con.execute(
            f"""
            SELECT PRODUCT_ID, STORE_ID, WEEK_NO,
                   SUM(QUANTITY)                      AS units,
                   SUM(SALES_VALUE)                   AS sales_value,
                   SUM(ABS(RETAIL_DISC))              AS loyalty,
                   SUM(ABS(COUPON_DISC))              AS manufacturer,
                   SUM(ABS(COUPON_MATCH_DISC))        AS coupon_match,
                   COUNT(*)                           AS n_rows
            FROM {src}
            -- Giveaway rows are excluded so the two components partition the
            -- rows rather than overlapping. Their discount is zero, so this
            -- moves no money; it keeps the same unit from being counted once
            -- as a discounted sale and again as a giveaway.
            WHERE usable AND NOT (QUANTITY > 0 AND SALES_VALUE = 0) AND ({where})
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """
        ).df()
    finally:
        if isinstance(transactions, pd.DataFrame):
            con.unregister("_subsidy_tx")
        if own:
            con.close()

    for name in MECHANICS:
        per_cell[name] = per_cell[name].fillna(0.0)
    per_cell["subsidy"] = per_cell[list(MECHANICS)].sum(axis=1)

    totals = {name: round(float(per_cell[name].sum()), 2) for name in MECHANICS}
    total = round(float(per_cell["subsidy"].sum()), 2)
    diagnostics = {
        "stage": "subsidy",
        "cells": len(per_cell),
        "rows": int(per_cell["n_rows"].sum()),
        "units": float(per_cell["units"].sum()),
        "sales_value": round(float(per_cell["sales_value"].sum()), 2),
        "by_mechanic": totals,
        "by_mechanic_share": {
            k: (round(v / total, 6) if total else None) for k, v in totals.items()
        },
        "subsidy_total": total,
        "columns": dict(MECHANICS),
        "excludes_free_goods": (
            "Rows with QUANTITY > 0 and SALES_VALUE = 0 are handled by "
            "free_goods() instead, so the two components partition the rows. "
            "Their discount is zero either way; the split stops one unit being "
            "reported as both a discounted sale and a giveaway."
        ),
        "on_all_units": (
            "The discount is charged on every unit sold in scope, not only on "
            "the incremental ones. A shopper who would have bought anyway still "
            "took the discount; the money left the till either way. Charging "
            "the promotion for incremental units alone is the commonest way to "
            "make one look profitable."
        ),
        "why_split": (
            "The cost bearer differs. RETAIL_DISC is the retailer's own loyalty "
            "discount, COUPON_DISC is manufacturer-funded, COUPON_MATCH_DISC is "
            "the retailer matching a manufacturer coupon. A campaign funded "
            "mostly by a supplier is a different instrument from one the "
            "retailer paid for."
        ),
    }
    return per_cell, diagnostics


def free_goods(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    prices: str | Path | pd.DataFrame = "data/interim/prices.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    products: tuple[int, ...] | None = None,
    stores: tuple[int, ...] | None = None,
    weeks: tuple[int, int] | None = None,
    commodity: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Units handed over for nothing, valued at the product's regular price.

    A giveaway is priced from **the product**, not from its own cell: the free
    line is often the only transaction for that product-store-week, so there is
    no price to reconstruct there — at product-store-week level 16% of these
    units come back unpriceable, at product level 10%. The product's median
    reconstructed regular price over every week it *was* priced is the closest
    honest figure.

    `usable` is deliberately not applied. That filter is about unit
    comparability; this is about money, and a free line costs the full regular
    price whether or not its units compare with anything.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        tx = _source(transactions, con, "_fg_tx")
        pr = _source(prices, con, "_fg_prices")
        where = _scope_sql(products, stores, weeks, commodity)
        per_cell = con.execute(
            f"""
            WITH fg AS (
                SELECT PRODUCT_ID, STORE_ID, WEEK_NO,
                       SUM(QUANTITY) AS units, COUNT(*) AS n_rows
                FROM {tx}
                WHERE QUANTITY > 0 AND SALES_VALUE = 0 AND ({where})
                GROUP BY 1, 2, 3
            ),
            product_price AS (
                SELECT PRODUCT_ID, median(regular_price) AS regular_price,
                       COUNT(*) AS priced_cells
                FROM {pr} WHERE regular_price IS NOT NULL GROUP BY 1
            )
            SELECT fg.*, p.regular_price, p.priced_cells
            FROM fg LEFT JOIN product_price p USING (PRODUCT_ID)
            ORDER BY 1, 2, 3
            """
        ).df()
        scoped_all = con.execute(
            f"SELECT COUNT(*) n, SUM(QUANTITY) u FROM {tx} WHERE usable AND ({where})"
        ).fetchone()
    finally:
        for alias in ("_fg_tx", "_fg_prices"):
            con.unregister(alias)
        if own:
            con.close()

    priced = per_cell["regular_price"].notna()
    per_cell["value"] = (per_cell["units"] * per_cell["regular_price"]).where(priced)

    valued = round(float(per_cell.loc[priced, "value"].sum()), 2)
    unpriced_units = float(per_cell.loc[~priced, "units"].sum())
    diagnostics = {
        "stage": "free_goods",
        "cells": len(per_cell),
        "rows": int(per_cell["n_rows"].sum()),
        "units": float(per_cell["units"].sum()),
        "priced": {
            "cells": int(priced.sum()),
            "units": float(per_cell.loc[priced, "units"].sum()),
            "value": valued,
        },
        "unpriced": {
            "cells": int((~priced).sum()),
            "units": unpriced_units,
            "products": int(per_cell.loc[~priced, "PRODUCT_ID"].nunique()),
            "value": None,
            "why_not_zero": (
                "No reconstructed regular price exists for these products "
                "anywhere in the panel, so their giveaway cannot be valued. "
                "They are reported and left out of the total rather than "
                "valued at zero — zeroing them would quietly restore the "
                "understatement this component exists to fix. The total is "
                "therefore a lower bound."
            ),
        },
        "share_of_scope_units": (
            round(float(per_cell["units"].sum()) / float(scoped_all[1]), 8)
            if scoped_all[1]
            else None
        ),
        "valuation": (
            "the product's median reconstructed regular price over every "
            "product-store-week where one exists — settled decision 3's "
            "reconstruction A, never the paid price and never zero"
        ),
        "why_product_level": (
            "A free line is often the only transaction for its "
            "product-store-week, so that cell has no price to reconstruct. "
            "Pricing from the product recovers most of them; pricing from the "
            "cell would leave 16% of units unvalued rather than 10%."
        ),
        "usable_filter_not_applied": (
            "Phase 2.2's volume-measured exclusion drops rows for being "
            "unit-incomparable, which is a claim about units. This is a claim "
            "about money: a free line costs the full regular price whether or "
            "not its units compare with anything."
        ),
    }
    return per_cell, diagnostics


def promo_cost(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    prices: str | Path | pd.DataFrame = "data/interim/prices.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    products: tuple[int, ...] | None = None,
    stores: tuple[int, ...] | None = None,
    weeks: tuple[int, int] | None = None,
    commodity: str | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`promo_cost_total = subsidy + free_goods`, both components visible.

    Args:
        strict: raise `UnpricedFreeGoodsError` when any giveaway could not be
            valued. Off by default — the honest default is to report the
            shortfall, not to refuse the whole total for it.

    Returns:
        `(per_cell, diagnostics)`. `per_cell` is one row per product-store-week
        with `subsidy`, `free_goods_value` and `promo_cost`. ROI, break-even and
        the margin sweep all take the **total**, never the subsidy alone.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        scope = {
            "products": products,
            "stores": stores,
            "weeks": weeks,
            "commodity": commodity,
        }
        sub, sub_diag = subsidy(transactions, con=con, **scope)
        gifts, gift_diag = free_goods(transactions, prices, con=con, **scope)
    finally:
        if own:
            con.close()

    if strict and gift_diag["unpriced"]["units"]:
        raise UnpricedFreeGoodsError(
            f"{gift_diag['unpriced']['units']:,.0f} free units across "
            f"{gift_diag['unpriced']['products']} products have no "
            f"reconstructable price, so the total would be a lower bound. Pass "
            f"strict=False to get it with the shortfall reported."
        )

    merged = sub.merge(
        gifts[[*_KEY, "units", "value"]].rename(
            columns={"units": "free_goods_units", "value": "free_goods_value"}
        ),
        on=list(_KEY),
        how="outer",
    )
    for column in ("subsidy", "free_goods_value", "free_goods_units", "units"):
        merged[column] = merged[column].fillna(0.0)
    merged["promo_cost"] = merged["subsidy"] + merged["free_goods_value"]

    subsidy_total = sub_diag["subsidy_total"]
    gifts_total = gift_diag["priced"]["value"]
    total = round(subsidy_total + gifts_total, 2)
    diagnostics = {
        "stage": "promo_cost",
        "scope": {k: (list(v) if isinstance(v, tuple) else v) for k, v in scope.items()},
        "promo_cost_total": total,
        "components": {
            "subsidy": subsidy_total,
            "free_goods": gifts_total,
            "subsidy_share": round(subsidy_total / total, 6) if total else None,
            "free_goods_share": round(gifts_total / total, 6) if total else None,
        },
        "lower_bound_by": {
            "free_goods_unpriced_units": gift_diag["unpriced"]["units"],
            "free_goods_unpriced_products": gift_diag["unpriced"]["products"],
        },
        "subsidy_detail": sub_diag,
        "free_goods_detail": gift_diag,
        "why_two_components": (
            "Both are money the retailer gave away to run the promotion. "
            "Counting only the discount understates the denominator of every "
            "ROI in the system. They are reported apart as well as summed "
            "because a campaign whose cost is mostly free goods is a "
            "structurally different instrument from one that is mostly price "
            "discount, and Phase 7's recommendation depends on telling them "
            "apart."
        ),
        "for_downstream": (
            "ROI, break-even margin and the margin sweep take promo_cost_total. "
            "Never the subsidy alone."
        ),
    }
    return merged, diagnostics


def free_goods_in_lift(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """How many giveaway units reach the panel, and so the lift numerator.

    **This should be zero and is not.** A unit handed over for nothing is a
    cost, not incremental demand; if it also lands in the panel's `units` it is
    counted twice — once as promotional cost here, once as lift in Phase 4.

    Measured rather than asserted, because the honest number on this panel is
    not zero and a test claiming otherwise would be false. The fix belongs in
    Phase 2.2, which decides what `usable` means, and changing it means
    rebuilding the panel and refitting the baseline.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        tx = _source(transactions, con, "_lift_tx")
        pa = _source(panel, con, "_lift_panel")
        row = con.execute(
            f"""
            WITH fg AS (
                SELECT PRODUCT_ID, STORE_ID, WEEK_NO, SUM(QUANTITY) AS free_units
                FROM {tx}
                WHERE QUANTITY > 0 AND SALES_VALUE = 0 AND usable
                GROUP BY 1, 2, 3
            )
            SELECT COUNT(*)                                          AS keys,
                   SUM(fg.free_units)                                AS units,
                   SUM(CASE WHEN p.treated THEN fg.free_units END)   AS treated_units,
                   SUM(CASE WHEN p.treated THEN 1 ELSE 0 END)        AS treated_keys
            FROM fg JOIN {pa} p USING (PRODUCT_ID, STORE_ID, WEEK_NO)
            """
        ).fetchone()
    finally:
        for alias in ("_lift_tx", "_lift_panel"):
            con.unregister(alias)
        if own:
            con.close()

    return {
        "keys_in_panel": int(row[0] or 0),
        "units_in_panel": float(row[1] or 0.0),
        "treated_keys": int(row[3] or 0),
        "units_in_treated_weeks": float(row[2] or 0.0),
        "clean": (row[2] or 0) == 0,
        "why_it_matters": (
            "A giveaway unit inside a treated week is counted as promotional "
            "cost by this module and as incremental demand by the Phase 4 "
            "lift. The same unit on both sides of the ratio inflates ROI twice "
            "over — once by raising the numerator, once by having been paid "
            "for in the denominator."
        ),
        "status": (
            "clean"
            if (row[2] or 0) == 0
            else "CONTAMINATED — the double count is live; see the module "
            "docstring for why it is measured here rather than asserted away"
        ),
    }


def adjust_lift_for_free_goods(
    gross_incremental: float,
    free_units_in_treated_weeks: float,
    *,
    interval: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Take giveaway units back out of a lift figure, at reporting time.

    Units handed over for nothing are a cost, not incremental demand. They
    currently reach the panel and so the lift numerator — see
    `free_goods_in_lift`. The structural fix is in `clean.py`'s definition of
    `usable` and was deferred because it needs a panel rebuild and a refit.

    **This correction is exact, not approximate.** The contaminated units are
    identified individually by key and week: they are the giveaway rows that
    join the panel in treated weeks. Subtracting them removes precisely the
    double-counted quantity, with no estimation involved. It is a reporting-time
    correction rather than a modelling one, and it is applied wherever a lift or
    ROI figure is shown.
    """
    adjusted = gross_incremental - free_units_in_treated_weeks
    result: dict[str, Any] = {
        "gross_incremental_raw": round(float(gross_incremental), 6),
        "free_units_removed": round(float(free_units_in_treated_weeks), 6),
        "gross_incremental_adjusted": round(float(adjusted), 6),
        "share_of_raw": (
            round(free_units_in_treated_weeks / gross_incremental, 6)
            if gross_incremental
            else None
        ),
        "exact": True,
        "why": (
            "Giveaway units are a cost, not incremental demand, and they reach "
            "the panel today. The units are identified individually by key and "
            "week, so removing them is exact rather than estimated."
        ),
        "structural_fix_deferred": (
            "The permanent fix belongs in clean.py's definition of `usable`. It "
            "was deferred because it requires rebuilding the panel and refitting "
            "the baseline, and the contamination is 1.9% of free-goods units and "
            "0.02% of promotional cost. free_goods_in_lift() keeps reporting "
            "CONTAMINATED until it is done."
        ),
    }
    if interval is not None:
        result["interval_raw"] = [float(interval[0]), float(interval[1])]
        result["interval_adjusted"] = [
            round(float(interval[0] - free_units_in_treated_weeks), 6),
            round(float(interval[1] - free_units_in_treated_weeks), 6),
        ]
    return result


def breakeven_margin(
    promo_cost_total: float,
    incremental_revenue: float,
    *,
    interval: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """`m_star = promo_cost_total / incremental_revenue`, as an interval.

    The margin the promotion needed to clear its own cost. The numerator is
    **known** — it is money that left the till, measured in Task 5.1 — and the
    denominator is **estimated**, so the ratio is reported as a range whenever
    a lift interval is supplied.

    **This is computable wherever lift is not.** It depends on observed depth
    and the cost actually paid, never on a counterfactual; the denominator is
    the only estimated part, and where it is unusable the function says so
    rather than declining to exist. That is the point of the measure: it states
    what a promotion needed to be true, without needing to know what it
    achieved.

    Returns a dict with `m_star`, its interval, and a `reason_code` of
    `ROI_UNBOUNDED` when the denominator interval spans zero, or
    `KAPPA_IMPOSSIBLE` when the break-even margin exceeds 50%.
    """
    spans_zero = interval is not None and interval[0] <= 0.0 <= interval[1]
    if spans_zero or incremental_revenue == 0:
        return {
            "m_star": None,
            "m_star_interval": None,
            "reason_code": "ROI_UNBOUNDED",
            "promo_cost_total": round(float(promo_cost_total), 2),
            "incremental_revenue": round(float(incremental_revenue), 2),
            "incremental_revenue_interval": (
                [float(interval[0]), float(interval[1])] if interval else None
            ),
            "why": (
                "The incremental-revenue interval crosses zero, so the ratio has "
                "no finite bound. Incremental units and the cost are still "
                "reported; only the ratio is undefined. A large break-even "
                "margin here would imply a precision that is not there."
            ),
        }

    m_star = promo_cost_total / incremental_revenue
    bounds = None
    if interval is not None:
        # A ratio is monotone in its denominator, so the interval's ends map to
        # the ratio's ends — but they swap when the denominator is negative.
        candidates = [promo_cost_total / e for e in interval if e != 0]
        bounds = [round(min(candidates), 6), round(max(candidates), 6)]

    impossible = m_star > 0.50
    return {
        "m_star": round(float(m_star), 6),
        "m_star_interval": bounds,
        "reason_code": "KAPPA_IMPOSSIBLE" if impossible else None,
        "promo_cost_total": round(float(promo_cost_total), 2),
        "incremental_revenue": round(float(incremental_revenue), 2),
        "incremental_revenue_interval": (
            [float(interval[0]), float(interval[1])] if interval else None
        ),
        "exceeds_plausible_margin": impossible,
        "identity": (
            "kappa_star(m) = m_star / m — Task 3.4's incremental share and this "
            "margin are the same statement from opposite sides. m_star > 0.5 "
            "and kappa_star(0.5) > 1 are one sentence: no plausible grocery "
            "gross margin clears it, so the campaign is arithmetically "
            "unprofitable before any measurement question is asked."
        ),
        "numerator_is_known": (
            "promo_cost_total is money that left the till — subsidy plus free "
            "goods, measured not estimated. Only the denominator carries "
            "uncertainty, which is why the interval comes from the lift."
        ),
    }


def sensitivity_table(
    promo_cost_total: float,
    incremental_revenue: float,
    *,
    interval: tuple[float, float] | None = None,
    margins: tuple[float, ...] = MARGIN_GRID,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Incremental profit at each assumed margin: `m * revenue - cost`.

    Nine columns, 10% to 50% in 5-point steps. The reader supplies the one fact
    the dataset cannot — their own margin — and reads their answer off the row.
    Where a lift interval is supplied, a cell whose **sign** is uncertain is
    marked `uncertain` rather than shown as a confident number.

    The break-even margin is where the sign flips, which makes this table and
    `breakeven_margin` two views of one calculation.
    """
    rows = []
    for margin in margins:
        profit = margin * incremental_revenue - promo_cost_total
        cell: dict[str, Any] = {
            "margin": margin,
            "incremental_profit": round(float(profit), 2),
        }
        if interval is not None:
            low = margin * min(interval) - promo_cost_total
            high = margin * max(interval) - promo_cost_total
            cell["profit_low"] = round(float(min(low, high)), 2)
            cell["profit_high"] = round(float(max(low, high)), 2)
            cell["sign_certain"] = bool(cell["profit_low"] * cell["profit_high"] > 0)
        rows.append(cell)

    table = pd.DataFrame(rows)
    positive = table.loc[table["incremental_profit"] > 0, "margin"]
    diagnostics = {
        "stage": "sensitivity_table",
        "margins": list(margins),
        "promo_cost_total": round(float(promo_cost_total), 2),
        "incremental_revenue": round(float(incremental_revenue), 2),
        "profitable_from_margin": (
            float(positive.min()) if len(positive) else None
        ),
        "all_negative": bool(len(positive) == 0),
        "uncertain_cells": (
            int((~table["sign_certain"]).sum()) if "sign_certain" in table else None
        ),
        "read_the_table": (
            "The nine columns exist so the reader supplies the one fact the "
            "dataset cannot. A merchant on 22% reads the 20% and 25% columns "
            "and knows. Nothing here guesses on their behalf."
        ),
        "grid_authority": (
            "MARGIN_GRID is imported from promo.audit, never restated, so Task "
            "3.4's kappa sweep and this table cannot drift apart."
        ),
    }
    return table, diagnostics


def assert_margin_stamped(payload: Any, *, _path: str = "") -> None:
    """Every margin-derived figure carries its provenance, or this raises.

    Walks a diagnostics structure and refuses any `MARGIN_REQUIRING_FIGURES`
    key that is not inside a mapping also carrying `margin_source` and
    `conditional_on_margin`. Called before serialisation, so a figure computed
    from an assumed margin cannot leave the module naked.
    """
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_margin_stamped(item, _path=f"{_path}[{i}]")
        return
    if not isinstance(payload, dict):
        return

    derived = [k for k in MARGIN_REQUIRING_FIGURES if k in payload]
    if derived:
        stamped = (
            "margin_source" in payload and "conditional_on_margin" in payload
        )
        if not stamped:
            raise UnstampedMarginError(
                f"{derived} appear at {_path or '<root>'} without "
                f"margin_source and conditional_on_margin beside them. A figure "
                f"computed from an assumed margin must carry the assumption "
                f"wherever it is shown, or a reader four screens later mistakes "
                f"arithmetic for a measurement."
            )
    for key, value in payload.items():
        assert_margin_stamped(value, _path=f"{_path}.{key}" if _path else str(key))


def campaign_accounting(
    campaign: Any,
    *,
    gross_incremental: float,
    interval: tuple[float, float] | None = None,
    promoted_price: float | None = None,
    margin: float | None = None,
    margin_source: str | None = None,
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    prices: str | Path | pd.DataFrame = "data/interim/prices.parquet",
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One campaign's whole accounting: cost, break-even, sensitivity, return.

    Composes Task 5.1's cost and Task 5.2's ratio into the object a campaign
    report needs, and applies Task 5.3's provenance rule to everything a margin
    touches.

    **The measured objects always ship.** The break-even margin and the
    nine-column table are what the data establishes; a supplied margin never
    replaces them, it only adds a conditional row beside them.

    Args:
        campaign: a `LiftCampaign`, or anything carrying `weeks`, and optionally
            `product`/`products`, `stores`, `commodity`.
        gross_incremental: the Phase 4 lift, in units. Adjust it for free goods
            first — see `adjust_lift_for_free_goods`.
        interval: the lift interval, which becomes the break-even interval.
        promoted_price: revenue per incremental unit. Measured from the
            campaign's own promoted weeks when omitted.
        margin: an assumed gross margin. **None is the honest default on this
            dataset** — there is no COGS column, so nothing derives it.
        margin_source: `"supplied"` when a caller provides `margin`. Defaulted
            for them rather than trusted, because the stamp is the point.

    Returns:
        `(sensitivity, diagnostics)`. Margin-derived figures live only inside
        `diagnostics["conditional"]`, which carries the stamp — so the guarantee
        is structural rather than a thing to remember.
    """
    if margin_source not in MARGIN_SOURCES:
        raise ValueError(
            f"margin_source must be one of {list(MARGIN_SOURCES)}, got "
            f"{margin_source!r}"
        )
    if margin is not None and margin_source is None:
        margin_source = "supplied"
    if margin is None and margin_source == "supplied":
        raise ValueError("margin_source='supplied' with no margin supplied")

    products = getattr(campaign, "product_ids", None) or (
        (campaign.product,) if getattr(campaign, "product", None) else None
    )
    weeks = tuple(campaign.weeks)
    scope = {
        "products": products,
        "stores": getattr(campaign, "stores", None),
        "weeks": weeks,
        "commodity": getattr(campaign, "commodity", None),
    }

    own = con is None
    con = connect() if con is None else con
    try:
        _, cost = promo_cost(transactions, prices, con=con, **scope)
        if promoted_price is None:
            src = _source(transactions, con, "_price_tx")
            row = con.execute(
                f"SELECT SUM(SALES_VALUE) AS v, SUM(QUANTITY) AS q FROM {src} "
                f"WHERE usable AND ({_scope_sql(**scope)})"
            ).fetchone()
            con.unregister("_price_tx") if isinstance(transactions, pd.DataFrame) else None
            promoted_price = float(row[0] / row[1]) if row and row[1] else 0.0
    finally:
        if own:
            con.close()

    total = cost["promo_cost_total"]
    revenue = gross_incremental * promoted_price
    revenue_interval = (
        (interval[0] * promoted_price, interval[1] * promoted_price)
        if interval is not None
        else None
    )
    breakeven = breakeven_margin(total, revenue, interval=revenue_interval)
    sensitivity, sensitivity_diag = sensitivity_table(
        total, revenue, interval=revenue_interval
    )

    diagnostics: dict[str, Any] = {
        "stage": "campaign_accounting",
        "campaign": getattr(campaign, "name", None),
        "scope": {k: (list(v) if isinstance(v, tuple) else v) for k, v in scope.items()},
        "promoted_price": round(float(promoted_price), 6),
        "incremental_units": round(float(gross_incremental), 6),
        "incremental_revenue": round(float(revenue), 2),
        "promo_cost": cost,
        "breakeven": breakeven,
        "sensitivity": sensitivity_diag,
        "margin_source": margin_source,
        "conditional_on_margin": margin,
        "measured_objects_always_ship": (
            "The break-even margin and the nine-column table are what the data "
            "establishes and are present whether or not a margin was supplied. "
            "A supplied margin adds a conditional figure beside them; it never "
            "replaces them."
        ),
    }

    if margin is None:
        diagnostics["conditional"] = None
        diagnostics["reason_code"] = "NO_MARGIN"
        diagnostics["why_no_conditional"] = (
            "No margin was supplied and none can be derived: no COGS or margin "
            "column exists in any of the eight tables, established at ingest. "
            "Profit and return are therefore not computed. The break-even "
            "margin and the sensitivity table are the answer this dataset "
            "supports, and a merchant reads their own margin off the table."
        )
    else:
        profit = margin * revenue - total
        diagnostics["conditional"] = {
            # The stamp lives on the same mapping as the figures, so the
            # guarantee holds by construction rather than by discipline.
            "margin_source": margin_source,
            "conditional_on_margin": margin,
            "incremental_profit": round(float(profit), 2),
            "roi": round(float(profit / total), 6) if total else None,
            "margin_headroom": (
                round(float(margin - breakeven["m_star"]), 6)
                if breakeven["m_star"] is not None
                else None
            ),
            "reads_as": (
                f"at the {margin:.0%} margin you supplied, this campaign "
                f"returned {profit:,.0f} in incremental profit"
            ),
            "not_a_measurement": (
                "Arithmetic conditional on a number the user supplied. The "
                "dataset carries no margin, so this figure is an assumption "
                "applied to a measurement, not a measurement."
            ),
        }

    assert_margin_stamped(diagnostics)
    return sensitivity, diagnostics


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return out
