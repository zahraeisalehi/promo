"""Price decomposition at PRODUCT_ID x STORE_ID x WEEK_NO.

The regular price is not a column in this dataset. It is reconstructed, and the
reconstruction is settled decision 3 in `docs/data_findings.md`:

    regular_price = (SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY

`COUPON_DISC` is excluded because it is manufacturer-funded and reimbursed, so
it never moved the price the retailer charged. That exclusion has a consequence
this module makes explicit rather than leaving to be discovered: `depth` is a
shelf-price depth, and the manufacturer coupon is *not* in it.
`depth_manufacturer` is reported beside it on the same base, and the identity

    depth == depth_loyalty + depth_match

holds to tolerance and is asserted. A reader who wants what the shopper saved
adds the third component; a reader who wants what the shelf said does not.

The three implementation requirements recorded at `data_findings.md:440` are
conditions on this file, not suggestions, and each is met here:

1. **No `abs()` on `RETAIL_DISC` row values.** 36 rows in the file carry a
   positive `RETAIL_DISC` — surcharges, not discounts. They are excluded before
   aggregation, with the exclusion's effect recorded like any other. Only after
   that exclusion is `abs()` applied, to sums that are then guaranteed negative.
2. **The divide is guarded.** A group with no units, or whose reconstruction
   base is zero, yields null — never an infinity, never a silent zero. Free
   goods make this a real case, not a defensive one.
3. **Prices are compared at tolerance, never with `==`.** DuckDB sums doubles in
   thread-arrival order, so exact equality on float aggregates is not
   reproducible across runs.

`level_test()` is the validity check `data_findings.md:456` records as owed by
this task: it asks whether the reconstruction's *level* is right, which the
Phase 1 stability evidence could not establish.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.io import connect

__all__ = [
    "BOUNDED_THRESHOLD",
    "MIN_MATCHED_PAIRS",
    "MIN_PRICED_WEEKS",
    "PRICE_TOLERANCE",
    "build_price_index",
    "build_price_panel",
    "deflate_prices",
    "level_test",
    "write_diagnostics",
]

#: A product on deal in more than this share of its priced product-store-weeks
#: has an ordinal depth only: its undiscounted price is never observed.
BOUNDED_THRESHOLD: float = 0.90

#: Distinct priced weeks a product needs before anything is claimed about its
#: depth in either direction. See the rationale recorded in the diagnostics.
MIN_PRICED_WEEKS: int = 8

#: Matched pairs a weekly link needs before it is believed. Below this the link
#: is set to 1.0 and the week is named in the diagnostics — never silently.
MIN_MATCHED_PAIRS: int = 30

#: Requirement 3. Never compare two float aggregates with `==`.
PRICE_TOLERANCE: float = 1e-9

_DEPTH_COLUMNS = ("depth", "depth_loyalty", "depth_match", "depth_manufacturer")


def _source_sql(
    cleaned: pd.DataFrame | str | Path, con: duckdb.DuckDBPyConnection
) -> tuple[str, dict[str, Any]]:
    if isinstance(cleaned, pd.DataFrame):
        con.register("_cleaned_frame", cleaned)
        return "_cleaned_frame", {"kind": "dataframe", "path": None}
    path = Path(cleaned)
    return f"read_parquet('{path.as_posix()}')", {"kind": "parquet", "path": str(path)}


def _totals(con: duckdb.DuckDBPyConnection, source: str, where: str) -> dict[str, float]:
    row = con.execute(
        f"""
        SELECT COUNT(*) AS rows, COALESCE(SUM(QUANTITY), 0) AS units,
               COALESCE(SUM(SALES_VALUE), 0) AS sales_value
        FROM {source} WHERE {where}
        """
    ).fetchone()
    return {
        "rows": int(row[0]),
        "units": int(row[1]),
        "sales_value": round(float(row[2]), 2),
    }


def build_price_panel(
    cleaned: pd.DataFrame | str | Path = "data/interim/transactions_clean.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    bounded_threshold: float = BOUNDED_THRESHOLD,
    min_priced_weeks: int = MIN_PRICED_WEEKS,
    out_path: str | Path | None = None,
    run_level_test: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate cleaned transactions to product-store-week and decompose price.

    Args:
        cleaned: the Task 2.2 output, as a parquet path or a DataFrame.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        bounded_threshold: deal share above which a product's depth is ordinal.
        min_priced_weeks: distinct priced weeks below which a product is
            `insufficient_support` — too thin to diagnose, tested before depth.
        out_path: if given, the panel is also written there as parquet.
        run_level_test: run the validity check owed by `data_findings.md:456`.

    Returns:
        `(panel, diagnostics)`. One row per observed PRODUCT_ID x STORE_ID x
        WEEK_NO among usable transactions, with paid and regular price, the
        depth and its three components, and the per-product identified/bounded
        flag. Never a zero row — explicit zeros are the panel builder's job, not
        this one's.

    Raises:
        AssertionError: the depth identity or unit conservation failed.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        return _build(
            cleaned, con, bounded_threshold=bounded_threshold,
            min_priced_weeks=min_priced_weeks, out_path=out_path,
            run_level_test=run_level_test,
        )
    finally:
        if own:
            con.close()


def _build(
    cleaned: pd.DataFrame | str | Path,
    con: duckdb.DuckDBPyConnection,
    *,
    bounded_threshold: float,
    min_priced_weeks: int,
    out_path: str | Path | None,
    run_level_test: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source, source_meta = _source_sql(cleaned, con)

    before = _totals(con, source, "usable")
    surcharge = _totals(con, source, "usable AND RETAIL_DISC > 0")
    after = _totals(con, source, "usable AND RETAIL_DISC <= 0")

    panel = con.execute(
        f"""
        WITH rows_in AS (
            SELECT * FROM {source} WHERE usable AND RETAIL_DISC <= 0
        ),
        grouped AS (
            SELECT
                PRODUCT_ID, STORE_ID, WEEK_NO,
                COUNT(*)::INTEGER          AS n_rows,
                SUM(QUANTITY)              AS units,
                SUM(SALES_VALUE)           AS sales_value,
                SUM(RETAIL_DISC)           AS retail_disc,
                SUM(COUPON_DISC)           AS coupon_disc,
                SUM(COUPON_MATCH_DISC)     AS coupon_match_disc
            FROM rows_in
            GROUP BY 1, 2, 3
        ),
        based AS (
            SELECT *,
                -- Requirement 1: abs() is applied to sums of rows already
                -- filtered to RETAIL_DISC <= 0, so it cannot flip a sign.
                sales_value + abs(retail_disc) + abs(coupon_match_disc)
                    AS regular_base
            FROM grouped
        )
        SELECT
            PRODUCT_ID, STORE_ID, WEEK_NO, n_rows, units, sales_value,
            retail_disc, coupon_disc, coupon_match_disc,
            CASE WHEN units > 0 THEN sales_value / units END AS paid_price,
            CASE WHEN units > 0 AND regular_base > 0
                 THEN regular_base / units END AS regular_price,
            -- Depth as the plan defines it, 1 - paid/regular, so that the
            -- component identity below is a real check and not a tautology.
            CASE WHEN units > 0 AND regular_base > 0
                 THEN 1 - (sales_value / units) / (regular_base / units)
            END AS depth,
            CASE WHEN regular_base > 0
                 THEN abs(retail_disc) / regular_base END AS depth_loyalty,
            CASE WHEN regular_base > 0
                 THEN abs(coupon_match_disc) / regular_base END AS depth_match,
            CASE WHEN regular_base > 0
                 THEN abs(coupon_disc) / regular_base END AS depth_manufacturer,
            (units <= 0 OR regular_base <= 0) AS price_undefined
        FROM based
        """
    ).df()

    panel["on_deal"] = (panel["depth"] > PRICE_TOLERANCE).fillna(False)

    products, product_diag = _product_status(
        panel, bounded_threshold, min_priced_weeks
    )
    panel = panel.merge(products, on="PRODUCT_ID", how="left", validate="many_to_one")

    _assert_invariants(panel, after)

    for column in _DEPTH_COLUMNS:
        panel[column] = panel[column].astype("float32")

    diagnostics: dict[str, Any] = {
        "stage": "build_price_panel",
        "source": source_meta,
        "reconstruction": {
            "decision": "A (settled decision 3, docs/data_findings.md)",
            "formula": (
                "(SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|) / QUANTITY"
            ),
            "coupon_disc_excluded": True,
            "why": (
                "COUPON_DISC is manufacturer-funded and reimbursed, so it never "
                "moved the price the retailer charged. depth is a shelf-price "
                "depth and does not include it; depth_manufacturer reports it "
                "separately on the same base."
            ),
        },
        "totals_before": before,
        "totals_after": after,
        "exclusions": [
            {
                "name": "retail_disc_surcharge",
                "action": "exclude",
                "definition": "RETAIL_DISC > 0 on a usable row",
                "reason": (
                    "Requirement 1: these are surcharges, not discounts. abs() "
                    "would silently turn them into discounts, so they are "
                    "removed before any abs() is taken."
                ),
                "effect": surcharge,
                "before": before,
                "after": after,
            }
        ],
        "panel": {
            "rows": len(panel),
            "products": int(panel["PRODUCT_ID"].nunique()),
            "stores": int(panel["STORE_ID"].nunique()),
            "weeks": int(panel["WEEK_NO"].nunique()),
            "week_min": int(panel["WEEK_NO"].min()),
            "week_max": int(panel["WEEK_NO"].max()),
            "rows_per_group_mean": round(float(panel["n_rows"].mean()), 3),
        },
        "price_undefined": _undefined_diag(panel),
        "depth": _depth_diag(panel),
        "product_status": product_diag,
        "notes": [
            (
                "No zero rows. This table holds observed product-store-weeks "
                "only; explicit zeros inside a declared scope belong to the "
                "panel builder, not here."
            ),
            (
                "Money and per-unit prices are float64; the four depth ratios "
                "are float32. The float32 rule targets the wide feature panel."
            ),
            (
                f"All price comparisons use a tolerance of {PRICE_TOLERANCE:g}. "
                f"Float aggregate equality is not reproducible across runs."
            ),
            (
                "No week window is applied. Settled decision 5 (weeks 18-101) is "
                "a scoping choice for the estimation panel, not a property of "
                "the price table."
            ),
        ],
    }

    if run_level_test:
        diagnostics["level_test"] = level_test(cleaned, con=con)

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path, index=False)
        diagnostics["written_to"] = str(path)

    return panel, diagnostics


def _product_status(
    panel: pd.DataFrame, bounded_threshold: float, min_priced_weeks: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per product: the deal share, and why its regular price may be unobserved.

    Three statuses, and the difference between the last two is the whole point
    of this function:

    - **identified** — enough priced weeks, and undiscounted weeks among them.
      Depth is a cardinal quantity.
    - **bounded** — enough priced weeks, but on deal in more than
      `bounded_threshold` of them. The regular price is barely observed, so
      depth is ordinal only: this product's deals can be ranked against each
      other and not priced against a shelf.
    - **insufficient_support** — fewer than `min_priced_weeks` priced weeks.
      Nothing can be said about this product's depth in either direction, and
      that is a different sentence from saying its depth is ordinal.

    `deal_share` is measured over the product's **priced** product-store-weeks —
    the panel grain — because an undefined price is not evidence either way. The
    distinct-week variant is carried alongside for comparison; the flag is
    driven by the panel-grain share. Support, by contrast, is counted in
    **distinct weeks**, because ten stores carrying a product in one week is one
    week of price history, not ten.
    """
    priced = panel[~panel["price_undefined"]]

    by_product = priced.groupby("PRODUCT_ID", observed=True).agg(
        n_psw_priced=("on_deal", "size"),
        n_psw_on_deal=("on_deal", "sum"),
    )
    weeks = priced.groupby("PRODUCT_ID", observed=True)["WEEK_NO"].nunique()
    weeks_on_deal = (
        priced[priced["on_deal"]].groupby("PRODUCT_ID", observed=True)["WEEK_NO"].nunique()
    )
    by_product["n_weeks_priced"] = weeks
    by_product["n_weeks_on_deal"] = weeks_on_deal.reindex(by_product.index).fillna(0)

    by_product["deal_share"] = by_product["n_psw_on_deal"] / by_product["n_psw_priced"]
    by_product["deal_share_weeks"] = (
        by_product["n_weeks_on_deal"] / by_product["n_weeks_priced"]
    )

    # Products with no priced week at all. A division that never happened must
    # not sweep them into "identified".
    unpriced = panel.loc[
        ~panel["PRODUCT_ID"].isin(by_product.index), "PRODUCT_ID"
    ].unique()
    if len(unpriced):
        extra = pd.DataFrame(
            {
                "n_psw_priced": 0,
                "n_psw_on_deal": 0,
                "n_weeks_priced": 0,
                "n_weeks_on_deal": 0,
                "deal_share": float("nan"),
                "deal_share_weeks": float("nan"),
            },
            index=pd.Index(unpriced, name="PRODUCT_ID"),
        )
        by_product = pd.concat([by_product, extra])

    # Support is asked before depth, and the order is the point. A deal share
    # computed on three weeks is not an ordinal depth, it is noise, so a product
    # that fails the support test never reaches the bounded test.
    thin = by_product["n_weeks_priced"] < min_priced_weeks
    bounded_mask = ~thin & (by_product["deal_share"] > bounded_threshold)
    by_product["price_status"] = pd.Series(
        "identified", index=by_product.index, dtype="string"
    ).mask(bounded_mask, "bounded").mask(thin, "insufficient_support")

    products = by_product.reset_index()
    for column in ("n_psw_priced", "n_psw_on_deal", "n_weeks_priced", "n_weeks_on_deal"):
        products[column] = products[column].astype("int32")
    for column in ("deal_share", "deal_share_weeks"):
        products[column] = products[column].astype("float32")
    products["price_status"] = products["price_status"].astype("string")

    counts = products["price_status"].value_counts()
    status = products["price_status"]
    thin_products = products[status == "insufficient_support"]

    # What the support test cost, in the only terms that matter: a status that
    # covers most products but little money is a different object from one that
    # covers little of both.
    weights = (
        panel.groupby("PRODUCT_ID", observed=True)[["units", "sales_value"]]
        .sum()
        .reindex(products["PRODUCT_ID"])
        .fillna(0.0)
    )
    total_units = float(weights["units"].sum())
    total_sales = float(weights["sales_value"].sum())

    def _weight(mask: pd.Series) -> dict[str, float]:
        selected = weights[mask.to_numpy()]
        return {
            "products": int(mask.sum()),
            "products_share": round(float(mask.mean()), 6),
            "units_share": (
                round(float(selected["units"].sum()) / total_units, 6)
                if total_units
                else 0.0
            ),
            "sales_value_share": (
                round(float(selected["sales_value"].sum()) / total_sales, 6)
                if total_sales
                else 0.0
            ),
        }

    diag = {
        "rules": {
            "insufficient_support": (
                f"n_weeks_priced < {min_priced_weeks} distinct priced weeks. "
                f"Tested first: nothing can be said about depth either way."
            ),
            "bounded": (
                f"deal_share > {bounded_threshold} over the product's priced "
                f"product-store-weeks, among products that pass the support "
                f"test. Depth is ordinal only."
            ),
            "identified": "everything else. Depth is cardinal.",
            "precedence": (
                "support, then depth. A deal share computed on a handful of "
                "weeks is not an ordinal depth; it is noise, and labelling it "
                "bounded would claim a diagnosis the data cannot support."
            ),
        },
        "thresholds": {
            "bounded_threshold": bounded_threshold,
            "min_priced_weeks": min_priced_weeks,
            "min_priced_weeks_rationale": (
                "Three independent reasons land on eight. (1) Task 1.3 required "
                "at least eight weeks of a product before scoring its price "
                "stability — the same question, so the same bar. (2) At the "
                "observed on-deal base rate of about 50%, eight consecutive "
                "on-deal weeks arise by chance with probability 0.5^8, under "
                "0.4%, so 'always on deal' is a real inference at eight weeks "
                "and not at three. (3) Task 2.6's middle rolling window is "
                "eight weeks; a product that cannot fill it has no rolling "
                "feature to be modelled on."
            ),
        },
        "products": {str(k): int(v) for k, v in counts.items()},
        "coverage": {
            name: _weight(status == name)
            for name in ("identified", "bounded", "insufficient_support")
        },
        "insufficient_support_detail": {
            "weeks_median": (
                int(thin_products["n_weeks_priced"].median())
                if len(thin_products)
                else None
            ),
            "unpriced_products": int((products["n_psw_priced"] == 0).sum()),
            "would_have_been_bounded": int(
                (
                    (status == "insufficient_support")
                    & (products["deal_share"] > bounded_threshold)
                ).sum()
            ),
            "would_have_been_identified": int(
                (
                    (status == "insufficient_support")
                    & ~(products["deal_share"] > bounded_threshold)
                ).sum()
            ),
            "note": (
                "The two counts above are what the old two-way rule would have "
                "said about these products. Neither claim was supportable: the "
                "first is a product seen once, on deal, which is not evidence "
                "of a perpetual deal, and the second is a product seen once, "
                "off deal, which is not evidence of an identified depth. "
                "unpriced_products have no reconstructable price at all and are "
                "a subset of this status, not a fourth one."
            ),
        },
        "deal_share_variants": {
            "used": "product-store-week for deal_share, distinct weeks for support",
            "weeks_variant_disagrees_on_products": int(
                (
                    (products["deal_share"] > bounded_threshold)
                    != (products["deal_share_weeks"] > bounded_threshold)
                ).sum()
            ),
        },
        "phase_3": (
            "insufficient_support must refuse under its own reason code, not "
            "DEPTH_BOUNDED. 'We cannot see this product enough to say anything' "
            "and 'this product's depth is ordinal only' are different diagnoses "
            "and a category manager acts differently on each."
        ),
    }
    return products, diag


def _undefined_diag(panel: pd.DataFrame) -> dict[str, Any]:
    undefined = panel["price_undefined"]
    return {
        "rows": int(undefined.sum()),
        "rows_share": round(float(undefined.mean()), 6),
        "units": int(panel.loc[undefined, "units"].sum()),
        "sales_value": round(float(panel.loc[undefined, "sales_value"].sum()), 2),
        "reason": (
            "Requirement 2: units <= 0 or a zero reconstruction base yields null, "
            "never an infinity and never a silent zero. A group of pure free "
            "goods has no price to reconstruct."
        ),
    }


def _depth_diag(panel: pd.DataFrame) -> dict[str, Any]:
    priced = panel[~panel["price_undefined"]]
    depth = priced["depth"]
    quantiles = depth.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    on_deal = priced["on_deal"]
    return {
        "priced_rows": len(priced),
        "on_deal_rows": int(on_deal.sum()),
        "on_deal_share": round(float(on_deal.mean()), 6),
        "quantiles_all": {str(k): round(float(v), 6) for k, v in quantiles.items()},
        "quantiles_on_deal": {
            str(k): round(float(v), 6)
            for k, v in depth[on_deal].quantile([0.25, 0.5, 0.75, 0.95]).items()
        },
        "at_or_above_100pct": int((depth >= 1 - PRICE_TOLERANCE).sum()),
        "components": {
            "depth_equals_loyalty_plus_match": True,
            "manufacturer_excluded_from_depth": True,
            "loyalty_median_where_fires": round(
                float(
                    priced.loc[priced["depth_loyalty"] > PRICE_TOLERANCE, "depth_loyalty"]
                    .median()
                ),
                6,
            ),
            "match_rows": int((priced["depth_match"] > PRICE_TOLERANCE).sum()),
            "manufacturer_rows": int(
                (priced["depth_manufacturer"] > PRICE_TOLERANCE).sum()
            ),
        },
    }


def _assert_invariants(panel: pd.DataFrame, after: dict[str, float]) -> None:
    """Mathematical invariants, checked rather than trusted."""
    priced = panel[~panel["price_undefined"]]
    residual = (
        priced["depth"] - (priced["depth_loyalty"] + priced["depth_match"])
    ).abs()
    worst = float(residual.max()) if len(residual) else 0.0
    assert worst < 1e-9, (
        f"depth != depth_loyalty + depth_match; worst residual {worst:.3e}. "
        f"The reconstruction and the components disagree."
    )

    units = int(panel["units"].sum())
    assert units == after["units"], (
        f"units not conserved by aggregation: {after['units']:,} in, {units:,} out"
    )
    sales = round(float(panel["sales_value"].sum()), 2)
    assert abs(sales - after["sales_value"]) < 0.01, (
        f"sales value not conserved: {after['sales_value']:,.2f} in, {sales:,.2f} out"
    )


# --------------------------------------------------------------------------
# Task 2.4 — the weekly price index, and deflation to real terms.
# --------------------------------------------------------------------------


def build_price_index(
    panel: pd.DataFrame,
    *,
    grain: str = "product_store",
    min_matched: int = MIN_MATCHED_PAIRS,
    trim: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A weekly price index from unpromoted rows, chained across matched pairs.

    A Jevons index: for each consecutive week pair, take the geometric mean of
    price relatives over units observed undiscounted in **both** weeks, then
    chain the weekly links into a level. The matched-pair construction is what
    makes it a price index rather than a spending average — an unmatched mean
    moves when the assortment changes, and this dataset's assortment changes
    constantly.

    **"Unpromoted" here means not on deal**, that is `depth <= 0` after Task
    2.3's reconstruction. It cannot yet mean "not on display or in the mailer":
    `causal_data` is not joined until Task 2.5, and this index has to exist
    before the price columns it deflates are used. The narrower reading is also
    the right one for a price index — a shelf price is a shelf price whether or
    not a sign was pointing at it, and what must be excluded is the markdown.

    Args:
        panel: the Task 2.3 price panel.
        grain: `"product_store"` matches within a store, so a shift in which
            stores sold a product cannot masquerade as a price move.
            `"product"` matches across stores and is reported as a robustness
            check. The plan says "per-product"; product-store is the finer grain
            that removes store-mix contamination, so it is the default.
        min_matched: a link resting on fewer pairs than this is not believed.
            The link is set to 1.0 and the week is named in the diagnostics.
        trim: symmetric share of extreme log relatives to drop per week. The
            default of 0.0 is the plan's plain geometric mean; a trimmed variant
            is always reported alongside as a robustness check.

    Returns:
        `(index, diagnostics)`. `index` has one row per week with the link, the
        chained level, the matched-pair count, and whether the link was imputed.

    Raises:
        ValueError: `grain` is not one of the two supported values.
    """
    keys = {
        "product_store": ["PRODUCT_ID", "STORE_ID"],
        "product": ["PRODUCT_ID"],
    }
    if grain not in keys:
        raise ValueError(f"grain must be one of {sorted(keys)}, got {grain!r}")
    key = keys[grain]

    unpromoted = panel[~panel["price_undefined"] & ~panel["on_deal"]]
    prices = (
        unpromoted.groupby([*key, "WEEK_NO"], observed=True)[["units", "sales_value"]]
        .sum()
        .reset_index()
    )
    # Unit value within the matching key: a quantity-weighted price, which is
    # the right average when the same product-store sells at one shelf price.
    prices["price"] = prices["sales_value"] / prices["units"]
    prices = prices[prices["price"] > 0]

    links, diag = _chain_links(prices, key, min_matched, trim)

    weeks = pd.Index(
        range(int(panel["WEEK_NO"].min()), int(panel["WEEK_NO"].max()) + 1),
        name="WEEK_NO",
    )
    index = links.reindex(weeks)
    index["n_matched"] = index["n_matched"].fillna(0).astype("int32")
    index["link_imputed"] = index["link"].isna() | index["link_imputed"].fillna(True)
    index["link"] = index["link"].fillna(1.0)
    # The base week has no predecessor, so its link is 1.0 by construction and
    # is not an imputation.
    base_week = int(weeks[0])
    index.loc[base_week, "link_imputed"] = False
    index["price_index"] = np.exp(np.log(index["link"]).cumsum())
    index = index.reset_index()

    diagnostics = {
        "stage": "build_price_index",
        "method": {
            "form": "Jevons — chained geometric mean of matched price relatives",
            "grain": grain,
            "base_week": base_week,
            "base_value": 1.0,
            "unpromoted_definition": (
                "depth <= 0 after Task 2.3's reconstruction. Not 'absent from "
                "causal_data' — that table is not joined until Task 2.5, and a "
                "shelf price is a shelf price whether or not a sign pointed at "
                "it. What must be excluded is the markdown."
            ),
            "min_matched": min_matched,
            "trim": trim,
        },
        "support": {
            "unpromoted_panel_rows": len(unpromoted),
            "matched_pairs_total": int(index["n_matched"].sum()),
            "matched_pairs_min": int(
                index.loc[index["WEEK_NO"] > base_week, "n_matched"].min()
            ),
            "matched_pairs_median": int(
                index.loc[index["WEEK_NO"] > base_week, "n_matched"].median()
            ),
            "weeks": len(index),
            "weeks_link_imputed": int(index["link_imputed"].sum()),
            "imputed_weeks": index.loc[index["link_imputed"], "WEEK_NO"].tolist(),
        },
        "drift": _drift_report(index),
        "composition": _composition_report(prices, key),
        **diag,
    }
    return index, diagnostics


def _composition_report(prices: pd.DataFrame, key: list[str]) -> dict[str, Any]:
    """Does the matched index move with the pool, or against it?

    Three measurements of the same 102 weeks that answer three different
    questions, and the gap between them is the diagnostic:

    - **matched** (the index): what happened to the price of a *fixed* item.
    - **pooled**: the geometric mean price of everything undiscounted that week,
      composition and all. This is what a naive average would report.
    - **balanced**: the geometric mean over units observed in most weeks, which
      holds composition fixed by brute force rather than by matching.

    If pooled and matched disagree in *sign*, the pool's composition is moving —
    and a price index that tracked the pool would be reporting the assortment,
    not the prices.
    """
    weeks = prices["WEEK_NO"]
    span = int(weeks.max() - weeks.min())
    window = max(1, span // 10)
    early = range(int(weeks.min()), int(weeks.min()) + window + 1)
    late = range(int(weeks.max()) - window, int(weeks.max()) + 1)

    def _drift(frame: pd.DataFrame) -> float | None:
        by_week = frame.groupby("WEEK_NO", observed=True)["price"].apply(
            lambda s: float(np.exp(np.log(s).mean()))
        )
        head = by_week.reindex(early).dropna()
        tail = by_week.reindex(late).dropna()
        if head.empty or tail.empty:
            return None
        return round(float(tail.mean() / head.mean() - 1), 6)

    observed_weeks = prices.groupby(key, observed=True)["WEEK_NO"].nunique()
    threshold = max(2, int(span * 0.4))
    balanced_keys = observed_weeks[observed_weeks >= threshold].index
    balanced = prices.set_index(key).loc[
        prices.set_index(key).index.isin(balanced_keys)
    ]

    counts = prices.groupby("WEEK_NO", observed=True).size()
    return {
        "compared_windows": {
            "early_weeks": [int(early[0]), int(early[-1])],
            "late_weeks": [int(late[0]), int(late[-1])],
        },
        "pooled_drift": _drift(prices),
        "balanced_drift": _drift(balanced.reset_index()),
        "balanced_threshold_weeks": threshold,
        "balanced_units": len(balanced_keys),
        "pool_size_early": int(counts.reindex(early).dropna().mean())
        if not counts.reindex(early).dropna().empty
        else None,
        "pool_size_late": int(counts.reindex(late).dropna().mean())
        if not counts.reindex(late).dropna().empty
        else None,
        "reading": (
            "Compare pooled_drift against the chained index. A pooled figure "
            "that moves the other way means the mix of what is sold undiscounted "
            "changed — on this dataset the observation pool roughly doubles as "
            "the household panel fills, and the entrants are cheaper items. The "
            "index deliberately does not see that, which is the point of "
            "matching. It is also why the pooled figure must never be quoted as "
            "inflation."
        ),
    }


def _chain_links(
    prices: pd.DataFrame, key: list[str], min_matched: int, trim: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Weekly links from matched consecutive-week pairs."""
    previous = prices.copy()
    previous["WEEK_NO"] = previous["WEEK_NO"] + 1
    matched = prices.merge(
        previous[[*key, "WEEK_NO", "price"]],
        on=[*key, "WEEK_NO"],
        suffixes=("", "_prev"),
        how="inner",
    )
    matched["log_relative"] = np.log(matched["price"] / matched["price_prev"])

    grouped = matched.groupby("WEEK_NO", observed=True)["log_relative"]
    plain = grouped.mean()
    counts = grouped.size()

    links = pd.DataFrame({"log_link": plain, "n_matched": counts})
    links["link_imputed"] = links["n_matched"] < min_matched
    links.loc[links["link_imputed"], "log_link"] = 0.0

    if trim > 0:
        links["log_link"] = _trimmed_means(matched, trim).reindex(links.index)
        links.loc[links["link_imputed"], "log_link"] = 0.0

    links["link"] = np.exp(links["log_link"])

    # The trimmed variant is always computed, so a reader can see whether the
    # headline drift is carried by a handful of extreme relatives.
    trimmed_link = np.exp(_trimmed_means(matched, 0.05))
    trimmed_drift = float(np.exp(np.log(trimmed_link).sum()) - 1)

    diag = {
        "relatives": {
            "pairs": len(matched),
            "log_relative_mean": round(float(matched["log_relative"].mean()), 8),
            "log_relative_sd": round(float(matched["log_relative"].std()), 6),
            "share_unchanged": round(
                float((matched["log_relative"].abs() < PRICE_TOLERANCE).mean()), 6
            ),
            "share_up": round(float((matched["log_relative"] > 0).mean()), 6),
            "share_down": round(float((matched["log_relative"] < 0).mean()), 6),
        },
        "robustness": {
            "trimmed_5pct_total_drift": round(trimmed_drift, 6),
            "note": (
                "A chained index is a sum of logs, so one extreme relative can "
                "move the level permanently. If the trimmed and untrimmed drifts "
                "disagree, the headline is carried by outliers and should not be "
                "reported as a price level."
            ),
        },
    }
    return links[["link", "n_matched", "link_imputed"]], diag


def _trimmed_means(matched: pd.DataFrame, trim: float) -> pd.Series:
    """Symmetrically trimmed mean of log relatives, per week."""

    def _trim(values: pd.Series) -> float:
        if len(values) < 3:
            return float(values.mean())
        low, high = values.quantile([trim, 1 - trim])
        kept = values[(values >= low) & (values <= high)]
        return float(kept.mean()) if len(kept) else float(values.mean())

    return matched.groupby("WEEK_NO", observed=True)["log_relative"].apply(_trim)


def _drift_report(index: pd.DataFrame) -> dict[str, Any]:
    """Total drift over the full span and over the estimation window."""
    levels = index.set_index("WEEK_NO")["price_index"]
    first, last = int(levels.index[0]), int(levels.index[-1])

    def _span(start: int, end: int) -> dict[str, Any]:
        if start not in levels.index or end not in levels.index:
            return {"weeks": None, "total_drift": None}
        total = float(levels[end] / levels[start] - 1)
        weeks = end - start
        return {
            "from_week": start,
            "to_week": end,
            "weeks": weeks,
            "total_drift": round(total, 6),
            "annualised": round(float((1 + total) ** (52 / weeks) - 1), 6) if weeks else None,
        }

    return {
        "full_span": _span(first, last),
        # Settled decision 5. The ramp weeks and the two calendar stubs sit
        # outside it, and they are where the index has least support.
        "estimation_window_18_101": _span(18, 101),
        "index_min": round(float(levels.min()), 6),
        "index_max": round(float(levels.max()), 6),
        "reading": (
            "The plan expected near-flat drift on this US dataset. Read the "
            "measured figure against `composition` and `robustness` before "
            "quoting it: a matched-pair index measures the price of a fixed "
            "item, which is not the same quantity as the average price paid, "
            "and on this dataset the two move in opposite directions."
        ),
    }


def deflate_prices(
    panel: pd.DataFrame,
    index: pd.DataFrame,
    *,
    base_week: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach the index and the real price columns alongside the nominal ones.

    Real prices are nominal divided by the index, rebased so the base week's
    real and nominal prices coincide. Nominal columns are never overwritten:
    Phase 5's accounting is in the currency the shopper actually paid, and only
    price *comparisons* across weeks need real terms.

    Args:
        panel: the Task 2.3 price panel.
        index: the output of `build_price_index`.
        base_week: week whose index is normalised to 1.0. Defaults to the
            index's own base.

    Returns:
        `(panel, diagnostics)` with `price_index`, `real_paid_price` and
        `real_regular_price` added.

    Raises:
        AssertionError: a week in the panel has no index value.
    """
    levels = index.set_index("WEEK_NO")["price_index"]
    if base_week is not None:
        assert base_week in levels.index, f"base week {base_week} is not in the index"
        levels = levels / levels[base_week]

    out = panel.copy()
    out["price_index"] = out["WEEK_NO"].map(levels).astype("float32")
    missing = int(out["price_index"].isna().sum())
    assert not missing, f"{missing:,} panel rows have no index value for their week"

    for nominal, real in (
        ("paid_price", "real_paid_price"),
        ("regular_price", "real_regular_price"),
    ):
        out[real] = out[nominal] / out["price_index"]

    priced = out[~out["price_undefined"]]
    diagnostics = {
        "stage": "deflate_prices",
        "base_week": int(base_week if base_week is not None else levels.index[0]),
        "columns_added": [
            "price_index",
            "real_paid_price",
            "real_regular_price",
        ],
        "nominal_preserved": True,
        "effect": {
            "mean_paid_price_nominal": round(float(priced["paid_price"].mean()), 6),
            "mean_paid_price_real": round(float(priced["real_paid_price"].mean()), 6),
            "largest_adjustment": round(
                float((out["price_index"].max() / out["price_index"].min()) - 1), 6
            ),
        },
        "note": (
            "Depth is a ratio of two prices in the same week, so deflation "
            "leaves it unchanged. That is a property worth stating rather than "
            "a coincidence: the index cancels, and Task 2.3's depth columns need "
            "no real counterpart."
        ),
    }
    return out, diagnostics


def level_test(
    cleaned: pd.DataFrame | str | Path = "data/interim/transactions_clean.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Is the reconstruction's *level* right, not merely its stability?

    The Phase 1 evidence for reconstruction A was that its prices are more
    concentrated than B's. Concentration is not correctness — a constant would
    score perfectly. This is the test `data_findings.md:456` records as owed:
    where a product-store-week contains both undiscounted and discounted rows,
    the undiscounted rows show the regular price directly, so the reconstruction
    computed from the discounted rows can be compared against it.

    Reconstruction B is scored on the identical groups, so the test
    discriminates between the two rather than only describing A.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        source, _ = _source_sql(cleaned, con)
        results = {
            grain: _level_test_at(con, source, grain)
            for grain in ("product_store_week", "product_store")
        }
    finally:
        if own:
            con.close()

    primary = results["product_store_week"]
    return {
        "question": (
            "Does the reconstructed regular price match the price actually paid "
            "on rows carrying no discount, for the same product and store?"
        ),
        "method": (
            "Within a group holding both kinds of row, observed_regular = "
            "SUM(SALES_VALUE)/SUM(QUANTITY) over rows with all three discounts "
            "zero. Reconstructed A and B are computed over that group's "
            "discounted rows only. Surcharge rows (RETAIL_DISC > 0) excluded."
        ),
        "grains": results,
        "verdict": _level_verdict(primary),
    }


def _level_test_at(
    con: duckdb.DuckDBPyConnection, source: str, grain: str
) -> dict[str, Any]:
    keys = {
        "product_store_week": "PRODUCT_ID, STORE_ID, WEEK_NO",
        "product_store": "PRODUCT_ID, STORE_ID",
    }[grain]
    row = con.execute(
        f"""
        WITH rows_in AS (
            SELECT *, (RETAIL_DISC = 0 AND COUPON_DISC = 0 AND COUPON_MATCH_DISC = 0)
                       AS undiscounted
            FROM {source} WHERE usable AND RETAIL_DISC <= 0
        ),
        grouped AS (
            SELECT {keys},
                SUM(CASE WHEN undiscounted THEN QUANTITY END)    AS clean_units,
                SUM(CASE WHEN undiscounted THEN SALES_VALUE END) AS clean_sales,
                SUM(CASE WHEN NOT undiscounted THEN QUANTITY END) AS disc_units,
                SUM(CASE WHEN NOT undiscounted THEN SALES_VALUE END) AS disc_sales,
                SUM(CASE WHEN NOT undiscounted THEN abs(RETAIL_DISC) END) AS rd,
                SUM(CASE WHEN NOT undiscounted THEN abs(COUPON_MATCH_DISC) END) AS cmd,
                SUM(CASE WHEN NOT undiscounted THEN abs(COUPON_DISC) END) AS cd
            FROM rows_in GROUP BY {keys}
        ),
        scored AS (
            SELECT
                (clean_sales / clean_units)                       AS observed,
                (disc_sales + rd + cmd) / disc_units              AS recon_a,
                (disc_sales + rd + cmd + cd) / disc_units         AS recon_b
            FROM grouped
            WHERE clean_units > 0 AND disc_units > 0 AND clean_sales > 0
        )
        SELECT
            COUNT(*)                                              AS n,
            median(abs(recon_a - observed))                       AS med_abs_a,
            median(abs(recon_b - observed))                       AS med_abs_b,
            median(recon_a - observed)                            AS med_signed_a,
            median(recon_b - observed)                            AS med_signed_b,
            avg(CASE WHEN abs(recon_a - observed) <= 0.01 THEN 1.0 ELSE 0.0 END) AS w1c_a,
            avg(CASE WHEN abs(recon_b - observed) <= 0.01 THEN 1.0 ELSE 0.0 END) AS w1c_b,
            avg(CASE WHEN abs(recon_a - observed) <= 0.05 THEN 1.0 ELSE 0.0 END) AS w5c_a,
            avg(CASE WHEN abs(recon_b - observed) <= 0.05 THEN 1.0 ELSE 0.0 END) AS w5c_b,
            avg(CASE WHEN abs(recon_a - observed) + 1e-9
                          < abs(recon_b - observed) THEN 1.0 ELSE 0.0 END)  AS a_closer,
            avg(CASE WHEN abs(recon_b - observed) + 1e-9
                          < abs(recon_a - observed) THEN 1.0 ELSE 0.0 END)  AS b_closer
        FROM scored
        """
    ).fetchone()
    n = int(row[0])
    if not n:
        return {"groups": 0}
    return {
        "groups": n,
        "median_abs_error": {"A": round(float(row[1]), 4), "B": round(float(row[2]), 4)},
        "median_signed_error": {
            "A": round(float(row[3]), 4),
            "B": round(float(row[4]), 4),
        },
        "within_1_cent": {"A": round(float(row[5]), 4), "B": round(float(row[6]), 4)},
        "within_5_cents": {"A": round(float(row[7]), 4), "B": round(float(row[8]), 4)},
        "closer_share": {"A": round(float(row[9]), 4), "B": round(float(row[10]), 4)},
    }


def _level_verdict(primary: dict[str, Any]) -> dict[str, Any]:
    if not primary.get("groups"):
        return {"status": "NOT_RUN", "detail": "no group held both kinds of row"}
    a, b = primary["median_abs_error"]["A"], primary["median_abs_error"]["B"]
    return {
        "status": "A_NOT_WORSE" if a <= b else "A_WORSE",
        "median_abs_error_A": a,
        "median_abs_error_B": b,
        "within_1_cent_A": primary["within_1_cent"]["A"],
        "reopens_decision_3": a > b,
        "detail": (
            "Decision 3 stands on level as well as on stability."
            if a <= b
            else "Reconstruction B is closer to the observed regular price. "
            "Decision 3 is reopened."
        ),
    }


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
