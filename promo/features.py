"""Derived model variables at PRODUCT_ID x STORE_ID x WEEK_NO.

This is where the modelling panel is materialised, so it is also where the
scope from `CLAUDE.md` is applied. The full grid is 92,339 x 582 x 102 =
5.5 billion cells and is never constructed. Instead:

- the top **N products by purchase frequency**, default 300, chosen from the
  **ever-treated products only** — an untreated-only product cannot contribute
  to estimating a treatment effect, so it would consume the row budget and
  return nothing;
- the **115 stores** and **93 weeks** `causal_data` covers;
- **explicit zero rows inside that scope**, because a lag over a sparse panel is
  not a lag: without the zeros, "last week" silently means "the last week this
  product happened to sell".

Zeros are filled only for product-store pairs **observed at least once**. A pair
never observed is a product the store does not stock — a structural zero, not a
demand zero — and Task 1.5 recorded that this dataset cannot tell "not stocked"
from "stocked and unsold". Fabricating those rows would invent demand
observations for shelves that do not exist. Both counts are reported so the
choice is visible.

**Zero-filled rows get their treatment looked up, not assumed.** A product can
be on display and sell nothing, and `causal_data` is chain-wide so it records
exactly that. Task 2.5's join ran against observed rows only, so this module
re-derives treatment across the expanded grid through the same function rather
than defaulting the new rows to untreated.

Two leakage rules govern every column here, and both are tested:

1. **No feature may use the current week's outcome.** Rolling means are computed
   over weeks *w-k .. w-1*, never including *w*. A window that includes its own
   week hands the model the answer.
2. **No feature may be derived from the promotion.** `price_rel_category` uses
   the reconstructed *regular* price, never the paid price — a paid price is low
   precisely because the product is on deal, so a feature built on it encodes
   the treatment and calls itself a control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.io import connect
from promo.treatment import build_treatment_panel

__all__ = [
    "CONTEMPORANEOUS_FEATURES",
    "LAGGED_FEATURES",
    "LAGS",
    "ROLLING_WINDOWS",
    "SCOPE_PRODUCTS",
    "EmptyScopeError",
    "build_feature_panel",
    "write_diagnostics",
]


class EmptyScopeError(Exception):
    """The scope selected no products or no stores, so there is nothing to build."""

#: Default product budget. Configurable per CLAUDE.md.
SCOPE_PRODUCTS: int = 300

LAGS: tuple[int, ...] = (1, 2, 4, 52)
ROLLING_WINDOWS: tuple[int, ...] = (4, 8, 13)

#: Features that use only weeks strictly before w. Safe under any reading.
LAGGED_FEATURES: tuple[str, ...] = (
    *(f"units_lag_{k}" for k in LAGS),
    *(f"units_roll_mean_{w}" for w in ROLLING_WINDOWS),
)

#: Features measured in week w itself. Not future information — but see the
#: mediator warning in the diagnostics before using them in a counterfactual.
CONTEMPORANEOUS_FEATURES: tuple[str, ...] = (
    "week_of_year",
    "is_holiday_week",
    "category_units_ex_focal",
    "store_traffic",
    "price_rel_category",
    "n_stores_carrying",
    "price_index",
)

_KEY = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]


def build_feature_panel(
    panel: str | Path | pd.DataFrame = "data/interim/panel_treated.parquet",
    transactions: str | Path = "data/interim/transactions_clean.parquet",
    product: str | Path = "data/raw/product.csv",
    causal: str | Path = "data/raw/causal_data.csv",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    n_products: int = SCOPE_PRODUCTS,
    holiday_weeks: set[int] | None = None,
    out_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scope the panel, fill its zeros, and derive every model feature.

    Args:
        panel: the Task 2.5 treated panel.
        transactions: the Task 2.2 cleaned transactions, for store traffic.
        product: `product.csv`, for `COMMODITY_DESC`.
        causal: `causal_data.csv`, to re-derive treatment over the filled grid.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        n_products: product budget for the scope.
        holiday_weeks: `WEEK_NO` values to mark as holiday weeks. **This dataset
            carries no calendar dates**, so there is no way to derive them and
            none is invented: when omitted, `is_holiday_week` is False
            everywhere and the diagnostics record it as unpopulated.
        out_path: if given, the panel is also written there as parquet.

    Returns:
        `(panel, diagnostics)` at product x store x week over the scope, with
        explicit zeros and every derived feature.

    Raises:
        AssertionError: the grid is not one row per key, or a lag crossed a
            product-store boundary.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        return _build(
            panel, transactions, product, causal, con,
            n_products=n_products,
            holiday_weeks=holiday_weeks,
            out_path=out_path,
        )
    finally:
        if own:
            con.close()


def _build(
    panel: str | Path | pd.DataFrame,
    transactions: str | Path,
    product: str | Path,
    causal: str | Path,
    con: duckdb.DuckDBPyConnection,
    *,
    n_products: int,
    holiday_weeks: set[int] | None,
    out_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(panel, pd.DataFrame):
        con.register("_panel_frame", panel)
        panel_sql = "_panel_frame"
    else:
        panel_sql = f"read_parquet('{Path(panel).as_posix()}')"

    product_sql = f"read_csv_auto('{Path(product).as_posix()}')"
    tx_sql = f"read_parquet('{Path(transactions).as_posix()}')"

    scope, scope_diag = _scope(con, panel_sql, n_products)
    grid, grid_diag = _grid(con, panel_sql, scope)

    # Treatment is re-derived across the filled grid: a zero-sale week can still
    # have been on display, and Task 2.5's join only saw weeks with a sale.
    grid = grid.drop(
        columns=[
            c for c in ("on_display", "in_mailer", "display_code", "mailer_code",
                        "in_causal_data", "treatment_observed", "treated")
            if c in grid.columns
        ]
    )
    grid, treat_diag = build_treatment_panel(grid, causal, con=con)

    grid, feature_diag = _add_features(
        con, grid, panel_sql, product_sql, tx_sql, holiday_weeks
    )

    _assert_no_leakage(grid)

    diagnostics = {
        "stage": "build_feature_panel",
        "scope": scope_diag,
        "grid": grid_diag,
        "features": feature_diag,
        "treatment_rederived": {
            "why": (
                "A zero-sale week can still have been on display. Task 2.5 "
                "joined observed rows only, so the filled rows would otherwise "
                "default to untreated — a silent mislabelling of exactly the "
                "rows the baseline trains on."
            ),
            "treated_rows": treat_diag["treated"]["rows"],
            "treated_share": treat_diag["treated"]["rows_share"],
            "treated_rows_with_zero_units": int(
                (grid["treated"] & (grid["units"] == 0)).sum()
            ),
        },
        "leakage_rules": {
            "no_current_week_outcome": (
                "Rolling means span weeks w-k..w-1 and never include w. Lags are "
                "shifts on a complete grid, so lag_1 is genuinely last week."
            ),
            "no_promotion_derived_features": (
                "price_rel_category uses the reconstructed regular price, never "
                "the paid price. A paid price is low because the product is on "
                "deal, so a feature built on it encodes the treatment."
            ),
            "time_reference": {
                "lagged": list(LAGGED_FEATURES),
                "contemporaneous": list(CONTEMPORANEOUS_FEATURES),
            },
            "mediator_warning": (
                "The contemporaneous controls are not future information, but "
                "store_traffic and category_units_ex_focal can be affected by "
                "the promotion itself. Conditioning on a mediator absorbs part "
                "of the effect. Phase 3 should report the estimate with and "
                "without them; the split above exists so that is one argument."
            ),
        },
    }

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(path, index=False)
        diagnostics["written_to"] = str(path)

    return grid, diagnostics


def _scope(
    con: duckdb.DuckDBPyConnection, panel_sql: str, n_products: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Top N ever-treated products, the logged stores, the logged weeks."""
    products = con.execute(
        f"""
        WITH ever AS (
            SELECT PRODUCT_ID
            FROM {panel_sql}
            GROUP BY 1
            HAVING BOOL_OR(on_display) OR BOOL_OR(in_mailer)
        )
        SELECT p.PRODUCT_ID, SUM(p.n_rows) AS transactions
        FROM {panel_sql} p SEMI JOIN ever USING (PRODUCT_ID)
        GROUP BY 1 ORDER BY transactions DESC, PRODUCT_ID
        LIMIT {int(n_products)}
        """
    ).df()

    envelope = con.execute(
        f"""
        SELECT MIN(WEEK_NO) AS week_min, MAX(WEEK_NO) AS week_max
        FROM {panel_sql} WHERE treatment_observed
        """
    ).df().iloc[0]
    stores = con.execute(
        f"SELECT DISTINCT STORE_ID FROM {panel_sql} WHERE treatment_observed"
    ).df()["STORE_ID"]

    scope = {
        "product_ids": products["PRODUCT_ID"].tolist(),
        "store_ids": stores.tolist(),
        "week_min": int(envelope["week_min"]),
        "week_max": int(envelope["week_max"]),
    }
    # An empty scope is a caller error worth naming: without this it becomes an
    # `IN ()` and dies in the SQL parser, which says nothing useful.
    if not scope["product_ids"]:
        raise EmptyScopeError(
            "no ever-treated products in the panel, so the scope is empty and "
            "no treatment effect could be estimated from it"
        )
    if not scope["store_ids"]:
        raise EmptyScopeError(
            "no store has treatment_observed, so the scope is empty; check that "
            "Task 2.5 ran against a populated causal_data"
        )

    totals = con.execute(
        f"SELECT SUM(n_rows) AS tx, SUM(sales_value) AS sales, SUM(units) AS units "
        f"FROM {panel_sql}"
    ).df().iloc[0]
    inside = con.execute(
        f"""
        SELECT SUM(n_rows) AS tx, SUM(sales_value) AS sales, SUM(units) AS units,
               COUNT(*) AS rows
        FROM {panel_sql}
        WHERE PRODUCT_ID IN ({','.join(str(int(p)) for p in scope['product_ids'])})
          AND STORE_ID IN ({','.join(str(int(s)) for s in scope['store_ids'])})
          AND WEEK_NO BETWEEN {scope['week_min']} AND {scope['week_max']}
        """
    ).df().iloc[0]

    diag = {
        "rule": (
            f"top {n_products} products by transaction count, chosen from "
            f"ever-treated products only; the {len(scope['store_ids'])} stores "
            f"and weeks {scope['week_min']}-{scope['week_max']} that "
            f"causal_data covers"
        ),
        "why_ever_treated_only": (
            "An untreated-only product cannot contribute to estimating a "
            "treatment effect. It would consume the row budget and return "
            "nothing."
        ),
        "n_products": len(scope["product_ids"]),
        "n_stores": len(scope["store_ids"]),
        "n_weeks": scope["week_max"] - scope["week_min"] + 1,
        "min_transactions_per_product": int(products["transactions"].min()),
        "median_transactions_per_product": int(products["transactions"].median()),
        # What the scope buys, and what it costs. A scope whose coverage is not
        # recorded is indistinguishable from a silent filter.
        "coverage": {
            "observed_rows": int(inside["rows"]),
            "transactions_share": round(float(inside["tx"] / totals["tx"]), 6),
            "sales_value_share": round(float(inside["sales"] / totals["sales"]), 6),
            "units_share": round(float(inside["units"] / totals["units"]), 6),
        },
    }
    return scope, diag


def _grid(
    con: duckdb.DuckDBPyConnection, panel_sql: str, scope: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complete week grid over carried product-store pairs, zeros filled."""
    products = ",".join(str(int(p)) for p in scope["product_ids"])
    stores = ",".join(str(int(s)) for s in scope["store_ids"])
    where = (
        f"PRODUCT_ID IN ({products}) AND STORE_ID IN ({stores}) "
        f"AND WEEK_NO BETWEEN {scope['week_min']} AND {scope['week_max']}"
    )

    grid = con.execute(
        f"""
        WITH scoped AS (SELECT * FROM {panel_sql} WHERE {where}),
        pairs AS (SELECT DISTINCT PRODUCT_ID, STORE_ID FROM scoped),
        weeks AS (
            SELECT UNNEST(range({scope['week_min']}, {scope['week_max']} + 1))
                   AS WEEK_NO
        ),
        full_grid AS (SELECT * FROM pairs CROSS JOIN weeks)
        SELECT
            g.PRODUCT_ID, g.STORE_ID, g.WEEK_NO,
            COALESCE(s.units, 0)::BIGINT        AS units,
            COALESCE(s.sales_value, 0.0)        AS sales_value,
            COALESCE(s.n_rows, 0)::INTEGER      AS n_rows,
            (s.PRODUCT_ID IS NULL)              AS zero_filled,
            s.regular_price, s.real_regular_price, s.paid_price,
            s.depth, s.price_status,
            -- The price index is a property of the week, not of the sale, so
            -- it is joined by week. Carrying it through the row join would
            -- leave it null on every zero-filled row.
            i.price_index
        FROM full_grid g
        LEFT JOIN scoped s USING (PRODUCT_ID, STORE_ID, WEEK_NO)
        LEFT JOIN (
            SELECT WEEK_NO, any_value(price_index) AS price_index
            FROM {panel_sql} WHERE price_index IS NOT NULL GROUP BY 1
        ) i USING (WEEK_NO)
        ORDER BY PRODUCT_ID, STORE_ID, WEEK_NO
        """
    ).df()

    # A week in which nothing sold anywhere has no index observation. The index
    # is a chained *level*, so its value in such a week is the level it was
    # already at — Task 2.4 carries levels through imputed links in exactly the
    # same way. Carried, and counted, never silently.
    week_index = (
        grid[["WEEK_NO", "price_index"]]
        .drop_duplicates("WEEK_NO")
        .set_index("WEEK_NO")["price_index"]
        .sort_index()
    )
    weeks_missing = week_index.isna()
    if weeks_missing.any():
        week_index = week_index.ffill().bfill()
        grid["price_index"] = grid["WEEK_NO"].map(week_index)

    n_pairs = grid[["PRODUCT_ID", "STORE_ID"]].drop_duplicates().shape[0]
    n_weeks = scope["week_max"] - scope["week_min"] + 1
    assert len(grid) == n_pairs * n_weeks, (
        f"grid is {len(grid):,} rows for {n_pairs:,} pairs x {n_weeks} weeks"
    )
    assert not grid.duplicated(_KEY).any(), "grid has duplicate keys"

    filled = grid["zero_filled"]
    diag = {
        "carried_pairs": int(n_pairs),
        "weeks": int(n_weeks),
        "rows": len(grid),
        "observed_rows": int((~filled).sum()),
        "zero_filled_rows": int(filled.sum()),
        "zero_filled_share": round(float(filled.mean()), 6),
        "full_cross_product_rows": (
            len(scope["product_ids"]) * len(scope["store_ids"]) * n_weeks
        ),
        "why_not_the_full_cross_product": (
            "A product-store pair never observed is a shelf the store does not "
            "stock — a structural zero, not a demand zero. Task 1.5 recorded "
            "that this dataset cannot tell 'not stocked' from 'stocked and "
            "unsold', so those rows would be invented demand observations."
        ),
        "why_zeros_at_all": (
            "A lag over a sparse panel is not a lag. Without explicit zeros, "
            "units_lag_1 would mean 'the last week this product happened to "
            "sell', which is a different variable in every row."
        ),
        "price_index_weeks_carried": int(weeks_missing.sum()),
        "price_index_note": (
            "The index is joined by week, not carried through the row join, "
            "since it is a property of the week. Weeks in which nothing sold "
            "anywhere have no index observation; the chained level is carried "
            "across them, and the count is above. Zero on the real panel."
        ),
    }
    return grid, diag


def _add_features(
    con: duckdb.DuckDBPyConnection,
    grid: pd.DataFrame,
    panel_sql: str,
    product_sql: str,
    tx_sql: str,
    holiday_weeks: set[int] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every derived column, plus what each one is and where it came from."""
    con.register("_grid", grid)

    # Lags and rolling means, within product-store and never across. The window
    # frame ends at 1 PRECEDING so the current week's outcome is never in its
    # own feature.
    lag_terms = ", ".join(
        f"LAG(units, {k}) OVER w AS units_lag_{k}" for k in LAGS
    )
    roll_terms = ", ".join(
        f"AVG(units) OVER (PARTITION BY PRODUCT_ID, STORE_ID ORDER BY WEEK_NO "
        f"ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING) AS units_roll_mean_{w}"
        for w in ROLLING_WINDOWS
    )
    out = con.execute(
        f"""
        SELECT *, {lag_terms}, {roll_terms}
        FROM _grid
        WINDOW w AS (PARTITION BY PRODUCT_ID, STORE_ID ORDER BY WEEK_NO)
        ORDER BY PRODUCT_ID, STORE_ID, WEEK_NO
        """
    ).df()

    # Category aggregates come from the FULL panel, not the scope: "category
    # units" must mean the commodity, not the 300 products we happened to keep.
    category = con.execute(
        f"""
        WITH cat AS (
            SELECT p.*, TRIM(pr.COMMODITY_DESC) AS COMMODITY_DESC
            FROM {panel_sql} p
            LEFT JOIN {product_sql} pr USING (PRODUCT_ID)
        )
        SELECT COMMODITY_DESC, STORE_ID, WEEK_NO,
               SUM(units) AS category_units_total
        FROM cat GROUP BY 1, 2, 3
        """
    ).df()
    commodity = con.execute(
        f"SELECT PRODUCT_ID, TRIM(COMMODITY_DESC) AS COMMODITY_DESC FROM {product_sql}"
    ).df()
    category_price = con.execute(
        f"""
        WITH cat AS (
            SELECT p.regular_price, TRIM(pr.COMMODITY_DESC) AS COMMODITY_DESC,
                   p.WEEK_NO
            FROM {panel_sql} p
            LEFT JOIN {product_sql} pr USING (PRODUCT_ID)
            WHERE p.regular_price IS NOT NULL
        )
        SELECT COMMODITY_DESC, WEEK_NO,
               median(regular_price) AS category_median_price
        FROM cat GROUP BY 1, 2
        """
    ).df()

    # Store traffic is trips, not units: distinct baskets in the store that week.
    traffic = con.execute(
        f"""
        SELECT STORE_ID, WEEK_NO, COUNT(DISTINCT BASKET_ID) AS store_traffic
        FROM {tx_sql} WHERE usable GROUP BY 1, 2
        """
    ).df()

    # Breadth of distribution, from the full panel so it is not truncated by
    # the store scope.
    carrying = con.execute(
        f"""
        SELECT PRODUCT_ID, WEEK_NO, COUNT(DISTINCT STORE_ID) AS n_stores_carrying
        FROM {panel_sql} WHERE units > 0 GROUP BY 1, 2
        """
    ).df()

    out = out.merge(commodity, on="PRODUCT_ID", how="left", validate="many_to_one")
    out = out.merge(
        category, on=["COMMODITY_DESC", "STORE_ID", "WEEK_NO"], how="left",
        validate="many_to_one",
    )
    out = out.merge(
        category_price, on=["COMMODITY_DESC", "WEEK_NO"], how="left",
        validate="many_to_one",
    )
    out = out.merge(traffic, on=["STORE_ID", "WEEK_NO"], how="left", validate="many_to_one")
    out = out.merge(
        carrying, on=["PRODUCT_ID", "WEEK_NO"], how="left", validate="many_to_one"
    )

    out["category_units_total"] = out["category_units_total"].fillna(0)
    out["category_units_ex_focal"] = out["category_units_total"] - out["units"]
    out["n_stores_carrying"] = out["n_stores_carrying"].fillna(0)
    out["store_traffic"] = out["store_traffic"].fillna(0)

    # Regular price, never paid price: see the leakage rules in the docstring.
    out["price_rel_category"] = out["regular_price"] / out["category_median_price"]

    # Relative seasonal position. The dataset has no calendar anchor, so this is
    # a position in a 52-week cycle and not a true ISO week number: week 1 of
    # the file is week 1 of the cycle, whatever date that was.
    out["week_of_year"] = ((out["WEEK_NO"] - 1) % 52 + 1).astype("int16")

    supplied = holiday_weeks is not None
    out["is_holiday_week"] = (
        out["WEEK_NO"].isin(holiday_weeks) if supplied else False
    )

    for column in (*LAGGED_FEATURES, "category_units_ex_focal", "store_traffic",
                   "price_rel_category", "n_stores_carrying", "category_units_total"):
        out[column] = out[column].astype("float32")

    out = out.drop(columns=["category_median_price"])

    feature_columns = [*LAGGED_FEATURES, *CONTEMPORANEOUS_FEATURES]
    diag = {
        "columns": feature_columns,
        "lags": list(LAGS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "missingness": {
            c: round(float(out[c].isna().mean()), 6)
            for c in feature_columns
            if out[c].dtype.kind == "f"
        },
        "definitions": {
            "units_lag_k": "units k weeks earlier, within product-store",
            "units_roll_mean_w": "mean units over weeks w-W..w-1, excluding w",
            "week_of_year": (
                "position in a 52-week cycle, ((WEEK_NO - 1) % 52) + 1. NOT an "
                "ISO week: the dataset has no calendar anchor, so the phase "
                "relative to the real year is unknown."
            ),
            "category_units_ex_focal": (
                "COMMODITY_DESC units in that store-week, minus the focal "
                "product's own. Computed over the full panel, so 'category' "
                "means the commodity and not the 300 scoped products."
            ),
            "store_traffic": "distinct baskets in that store that week — trips",
            "price_rel_category": (
                "regular_price over the commodity's median regular price that "
                "week. Regular, not paid: a paid price is low because the "
                "product is on deal."
            ),
            "n_stores_carrying": (
                "distinct stores with a sale of that product that week, over "
                "the full panel"
            ),
            "price_index": "the Task 2.4 weekly index",
        },
        "holiday_flag": {
            "populated": supplied,
            "weeks_supplied": sorted(holiday_weeks) if supplied else [],
            "why_empty": None if supplied else (
                "This dataset carries no calendar dates — only DAY 1-711 and "
                "WEEK_NO 1-102 — so there is no way to know which week contains "
                "which holiday, and none was invented. Deriving the flag from "
                "demand spikes was rejected: that builds a feature out of the "
                "outcome it is meant to predict. Pass holiday_weeks to populate "
                "it if an external anchor becomes available."
            ),
        },
        "lag_52_note": (
            "The scope spans 93 weeks, so units_lag_52 is null for the first 52 "
            "of them by construction. It is available for roughly a third of "
            "the panel; see missingness."
        ),
        "price_rel_category_note": (
            "Null on every zero-filled row, which is most of the panel: a week "
            "with no sale has no observed price to compare. Left null rather "
            "than carried forward from the last priced week — that would be a "
            "modelling choice, and an imputed price is indistinguishable from "
            "an observed one once it is in the column. LightGBM splits on null "
            "natively, so the feature still contributes where it exists. If "
            "Phase 4 wants a filled version it should add it as a separate, "
            "named column."
        ),
    }
    return out, diag


def _assert_no_leakage(panel: pd.DataFrame) -> None:
    """No feature may carry information from week w or later.

    Checked directly rather than by inspection: for a sample of product-stores,
    recompute each lag and rolling mean from the units series by hand and
    compare. A window that slipped by one week fails here.
    """
    sample = panel.groupby(["PRODUCT_ID", "STORE_ID"], observed=True).head(1000)
    keys = sample[["PRODUCT_ID", "STORE_ID"]].drop_duplicates().head(25)
    for _, key in keys.iterrows():
        block = panel[
            (panel["PRODUCT_ID"] == key["PRODUCT_ID"])
            & (panel["STORE_ID"] == key["STORE_ID"])
        ].sort_values("WEEK_NO")
        units = block["units"].astype("float64")
        for k in LAGS:
            expected = units.shift(k)
            got = block[f"units_lag_{k}"].astype("float64")
            assert np.allclose(expected, got, equal_nan=True), (
                f"units_lag_{k} does not equal a {k}-week shift within "
                f"product-store {key['PRODUCT_ID']}/{key['STORE_ID']}"
            )
        for w in ROLLING_WINDOWS:
            expected = units.shift(1).rolling(w, min_periods=1).mean()
            got = block[f"units_roll_mean_{w}"].astype("float64")
            assert np.allclose(expected, got, equal_nan=True, atol=1e-5), (
                f"units_roll_mean_{w} includes the current week or spans the "
                f"wrong window"
            )


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
