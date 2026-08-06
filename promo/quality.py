"""Availability, zero classification, and the measurement horizon.

Three questions this module answers, and one report it assembles.

**Was the household in the shop at all?** Half of all household-weeks contain no
transaction (51.38%), and Task 1.5 established that most of the zero mass in
household-level work is "not in the shop" rather than "declined to buy". A model
that treats the two alike is fitting shopping-trip frequency and calling it
demand.

**Which zeros carry demand information?** Four states, not two:

| state | meaning | zero kind |
|---|---|---|
| `bought` | purchased the commodity that week | not a zero |
| `no_buy_on_trip` | in the shop, did not buy it | **sampling** — a real decision |
| `no_trip` | in the panel, never entered a shop | **structural** |
| `out_of_panel` | before the first trip or after the last | **structural**, and not an observation at all |

Only `no_buy_on_trip` is evidence about demand. The plan asks for a
structural-versus-sampling classifier and that split is provided, but the four
states are kept underneath it: `out_of_panel` is a household that had not been
recruited yet, which is a different thing from one who shopped elsewhere that
week, and collapsing them hides the recruitment ramp.

**How long must a measurement window stay open?** The per-commodity repurchase
cycle, from median inter-purchase gaps. Task 1.5 settled two things about how to
read it, and both are honoured here:

- the **median of household medians**, not the pooled median, because the pooled
  figure is dominated by frequent buyers who contribute more gaps each;
- **round up**, because the estimate is biased short — 15.53% of
  household-commodity pairs bought in exactly one week and contribute no gap,
  and those are the slowest buyers. Every horizon here is a floor.

Nothing in this module reads the product-store-week panel. A store is open every
week, so the no-trip question does not arise there; the panel's own zeros are a
"stocked or not" question that Task 2.6 handles separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.io import connect

__all__ = [
    "STRUCTURAL_STATES",
    "ZERO_STATES",
    "build_quality_report",
    "classify_zeros",
    "household_week_flags",
    "repurchase_cycles",
    "write_quality_report",
]

#: The four states a household-week can be in for a given commodity.
ZERO_STATES: tuple[str, ...] = ("bought", "no_buy_on_trip", "no_trip", "out_of_panel")

#: Of those, the ones that carry no demand information.
STRUCTURAL_STATES: tuple[str, ...] = ("no_trip", "out_of_panel")

#: A commodity with fewer gap observations than this gets a horizon, but the
#: row is flagged: a median over a handful of gaps is not a cycle.
MIN_GAP_EVENTS: int = 30

_WEEK_MIN, _WEEK_MAX = 1, 102


def _tx_sql(transactions: str | Path | pd.DataFrame, con: duckdb.DuckDBPyConnection) -> str:
    if isinstance(transactions, pd.DataFrame):
        con.register("_tx_frame", transactions)
        return "_tx_frame"
    return f"read_parquet('{Path(transactions).as_posix()}')"


def _product_sql(product: str | Path | pd.DataFrame, con: duckdb.DuckDBPyConnection) -> str:
    if isinstance(product, pd.DataFrame):
        con.register("_product_frame", product)
        return "_product_frame"
    return f"read_csv_auto('{Path(product).as_posix()}')"


def household_week_flags(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    week_min: int = _WEEK_MIN,
    week_max: int = _WEEK_MAX,
    usable_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per household-week: did they shop, and were they in the panel?

    **A trip is any transaction, not any usable one.** The `usable` filter from
    Task 2.2 excludes fuel and zero-quantity rows because they are not
    comparable *demand*; but a household that bought only petrol was still in
    the shop and still able to buy groceries. Applying a demand filter to an
    availability question turns 2,827 real trips into no-trips. `usable_only`
    exists to measure that difference, not because it is ever the right default.

    Returns:
        `(flags, diagnostics)`. `flags` covers every household x week in the
        window with `shopped`, `in_span`, `first_week`, `last_week`.
    """
    own = con is None
    con = connect() if con is None else con
    where = "WHERE usable" if usable_only else ""
    try:
        source = _tx_sql(transactions, con)
        flags = con.execute(
            f"""
            WITH tx AS (SELECT * FROM {source} {where}),
            spans AS (
                SELECT household_key,
                       MIN(WEEK_NO) AS first_week,
                       MAX(WEEK_NO) AS last_week
                FROM tx GROUP BY 1
            ),
            weeks AS (
                SELECT UNNEST(range({week_min}, {week_max} + 1)) AS WEEK_NO
            ),
            shopped AS (
                SELECT DISTINCT household_key, WEEK_NO FROM tx
            )
            SELECT
                s.household_key, w.WEEK_NO,
                (sh.household_key IS NOT NULL) AS shopped,
                (w.WEEK_NO BETWEEN s.first_week AND s.last_week) AS in_span,
                s.first_week, s.last_week
            FROM spans s
            CROSS JOIN weeks w
            LEFT JOIN shopped sh
              ON sh.household_key = s.household_key AND sh.WEEK_NO = w.WEEK_NO
            ORDER BY s.household_key, w.WEEK_NO
            """
        ).df()
    finally:
        if own:
            con.close()

    households = int(flags["household_key"].nunique())
    weeks = int(flags["WEEK_NO"].nunique())
    in_span = flags["in_span"]
    shopped = flags["shopped"]

    diagnostics = {
        "stage": "household_week_flags",
        "households": households,
        "weeks": weeks,
        "household_weeks": len(flags),
        "raw": {
            "with_a_trip": int(shopped.sum()),
            "no_trip": int((~shopped).sum()),
            "no_trip_share": round(float((~shopped).mean()), 6),
        },
        "within_span": {
            "household_weeks": int(in_span.sum()),
            "share_of_grid": round(float(in_span.mean()), 6),
            "no_trip": int((in_span & ~shopped).sum()),
            "no_trip_share": round(
                float((in_span & ~shopped).sum() / in_span.sum()), 6
            ),
        },
        "entry": {
            "median_first_week": int(flags.groupby("household_key")["first_week"].first().median()),
            "median_last_week": int(flags.groupby("household_key")["last_week"].first().median()),
        },
        "trip_definition": (
            "any transaction" if not usable_only else "a usable transaction only"
        ),
        "note": (
            "The raw no-trip share mixes two zeros: a week before a household's "
            "first trip is not a refused purchase, it is a household not yet "
            "recruited. The within-span figure is the honest one."
        ),
        "trip_definition_note": (
            "A trip is any transaction. Counting only usable rows would drop "
            "households whose sole purchase that week was fuel or a "
            "zero-quantity line — they were in the shop, so their not buying "
            "the commodity is a real decision, not an absence."
        ),
    }
    return flags, diagnostics


def classify_zeros(
    commodity: str,
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    product: str | Path | pd.DataFrame = "data/raw/product.csv",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    week_min: int = _WEEK_MIN,
    week_max: int = _WEEK_MAX,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify every household-week for one commodity into the four states.

    One commodity at a time on purpose. The full household x week x commodity
    cube is 2,500 x 102 x 306 = 78 million cells and answers a question nobody
    asked; the classification is only ever needed for the commodity under
    measurement.

    Args:
        commodity: the `COMMODITY_DESC` to classify, matched after trimming.

    Returns:
        `(classified, diagnostics)` with a `state` column drawn from
        `ZERO_STATES` and a `structural` boolean.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        tx = _tx_sql(transactions, con)
        prod = _product_sql(product, con)
        flags, _ = household_week_flags(
            transactions, con=con, week_min=week_min, week_max=week_max
        )
        con.register("_flags", flags)
        bought = con.execute(
            f"""
            SELECT DISTINCT t.household_key, t.WEEK_NO
            FROM {tx} t
            JOIN {prod} p ON t.PRODUCT_ID = p.PRODUCT_ID
            WHERE t.usable AND TRIM(p.COMMODITY_DESC) = ?
            """,
            [commodity],
        ).df()
        con.register("_bought", bought)
        out = con.execute(
            """
            SELECT f.household_key, f.WEEK_NO, f.shopped, f.in_span,
                   (b.household_key IS NOT NULL) AS bought
            FROM _flags f
            LEFT JOIN _bought b
              ON b.household_key = f.household_key AND b.WEEK_NO = f.WEEK_NO
            ORDER BY f.household_key, f.WEEK_NO
            """
        ).df()
    finally:
        if own:
            con.close()

    state = pd.Series("out_of_panel", index=out.index, dtype="string")
    state = state.mask(out["in_span"] & ~out["shopped"], "no_trip")
    state = state.mask(out["in_span"] & out["shopped"], "no_buy_on_trip")
    state = state.mask(out["bought"], "bought")
    out["state"] = state.astype("string")
    out["structural"] = out["state"].isin(STRUCTURAL_STATES)

    counts = out["state"].value_counts()
    zeros = out[out["state"] != "bought"]
    diagnostics = {
        "stage": "classify_zeros",
        "commodity": commodity,
        "household_weeks": len(out),
        "states": {s: int(counts.get(s, 0)) for s in ZERO_STATES},
        "zeros": {
            "total": len(zeros),
            "structural": int(zeros["structural"].sum()),
            "structural_share": (
                round(float(zeros["structural"].mean()), 6) if len(zeros) else None
            ),
            "sampling": int((~zeros["structural"]).sum()),
        },
        "reading": (
            "Only no_buy_on_trip is evidence about demand. A structural share "
            "near two thirds is normal here even for the most-bought commodity "
            "in the file, and a model that treats those weeks as refusals is "
            "fitting shopping-trip frequency."
        ),
    }
    return out, diagnostics


def repurchase_cycles(
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    product: str | Path | pd.DataFrame = "data/raw/product.csv",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    min_gap_events: int = MIN_GAP_EVENTS,
    out_path: str | Path | None = "data/interim/repurchase_cycles.parquet",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-commodity repurchase cycle, and the horizon it implies.

    A purchase event is a household buying anything in a commodity in one week;
    a gap is the number of weeks to that household's next event in the same
    commodity.

    `horizon_weeks` is `ceil(median of household medians)` — the per-household
    view because the pooled median is dominated by frequent buyers, and rounded
    up because the estimate is biased short. It is a **floor**, not an estimate.

    Returns:
        `(cycles, diagnostics)`, one row per commodity, written to `out_path`.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        tx = _tx_sql(transactions, con)
        prod = _product_sql(product, con)
        cycles = con.execute(
            f"""
            WITH events AS (
                SELECT DISTINCT
                    TRIM(p.COMMODITY_DESC) AS COMMODITY_DESC,
                    t.household_key, t.WEEK_NO
                FROM {tx} t
                JOIN {prod} p ON t.PRODUCT_ID = p.PRODUCT_ID
                WHERE t.usable AND p.COMMODITY_DESC IS NOT NULL
                  AND TRIM(p.COMMODITY_DESC) <> ''
            ),
            gaps AS (
                SELECT COMMODITY_DESC, household_key,
                       WEEK_NO - LAG(WEEK_NO) OVER (
                           PARTITION BY COMMODITY_DESC, household_key
                           ORDER BY WEEK_NO
                       ) AS gap
                FROM events
            ),
            per_household AS (
                SELECT COMMODITY_DESC, household_key,
                       median(gap) AS hh_median_gap,
                       COUNT(*) AS n_gaps
                FROM gaps WHERE gap IS NOT NULL
                GROUP BY 1, 2
            ),
            pooled AS (
                SELECT COMMODITY_DESC,
                       COUNT(*) AS n_gaps,
                       median(gap) AS pooled_median_gap,
                       quantile_cont(gap, 0.75) AS gap_p75,
                       quantile_cont(gap, 0.90) AS gap_p90
                FROM gaps WHERE gap IS NOT NULL GROUP BY 1
            ),
            support AS (
                SELECT COMMODITY_DESC,
                       COUNT(*) AS n_events,
                       COUNT(DISTINCT household_key) AS n_households
                FROM events GROUP BY 1
            ),
            pairs AS (
                SELECT COMMODITY_DESC,
                       COUNT(*) AS n_pairs,
                       SUM(CASE WHEN weeks = 1 THEN 1 ELSE 0 END) AS single_week_pairs
                FROM (
                    SELECT COMMODITY_DESC, household_key,
                           COUNT(*) AS weeks
                    FROM events GROUP BY 1, 2
                ) GROUP BY 1
            )
            SELECT
                s.COMMODITY_DESC,
                s.n_events, s.n_households,
                COALESCE(po.n_gaps, 0) AS n_gaps,
                po.pooled_median_gap,
                (SELECT median(hh_median_gap) FROM per_household ph
                  WHERE ph.COMMODITY_DESC = s.COMMODITY_DESC) AS hh_median_gap,
                po.gap_p75, po.gap_p90,
                pa.n_pairs, pa.single_week_pairs
            FROM support s
            LEFT JOIN pooled po USING (COMMODITY_DESC)
            LEFT JOIN pairs pa USING (COMMODITY_DESC)
            ORDER BY s.n_events DESC
            """
        ).df()
    finally:
        if own:
            con.close()

    cycles["single_week_share"] = (
        cycles["single_week_pairs"] / cycles["n_pairs"]
    ).astype("float32")

    # The recorded choice: per-household median, rounded up. Where a commodity
    # has no gap at all, no horizon is invented — it is null and flagged.
    cycles["horizon_weeks"] = cycles["hh_median_gap"].map(
        lambda v: math.ceil(v) if pd.notna(v) else pd.NA
    ).astype("Int16")
    cycles["horizon_weeks_p75"] = cycles["gap_p75"].map(
        lambda v: math.ceil(v) if pd.notna(v) else pd.NA
    ).astype("Int16")
    cycles["low_support"] = cycles["n_gaps"] < min_gap_events

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cycles.to_parquet(path, index=False)

    usable = cycles[~cycles["low_support"] & cycles["horizon_weeks"].notna()]
    top50 = cycles.nlargest(50, "n_events")
    diagnostics = {
        "stage": "repurchase_cycles",
        "commodities": len(cycles),
        "with_a_horizon": int(cycles["horizon_weeks"].notna().sum()),
        "low_support": int(cycles["low_support"].sum()),
        "min_gap_events": min_gap_events,
        "horizon_rule": (
            "ceil(median of household median gaps). The per-household view "
            "because the pooled median is dominated by frequent buyers; rounded "
            "up because the estimate is biased short."
        ),
        "horizon_weeks": {
            "min": int(usable["horizon_weeks"].min()) if len(usable) else None,
            "median": int(usable["horizon_weeks"].median()) if len(usable) else None,
            "p90": int(usable["horizon_weeks"].quantile(0.9)) if len(usable) else None,
            "max": int(usable["horizon_weeks"].max()) if len(usable) else None,
        },
        # Task 1.5's headline "3 weeks" is the horizon of the *fastest*
        # commodities, not the median one. Both bases are reported so a global
        # 3-week window is not adopted on the strength of a misread.
        "horizon_weeks_median_top_50": (
            int(top50["horizon_weeks"].dropna().median())
            if top50["horizon_weeks"].notna().any()
            else None
        ),
        "single_week_pairs_share": round(
            float(
                cycles["single_week_pairs"].sum() / cycles["n_pairs"].sum()
            ), 6
        ),
        # Task 1.5 quoted this on the top 50 commodities. Both bases are given,
        # because the long tail is mostly one-off purchases and the
        # all-commodity figure is twice the headline one for that reason alone.
        "single_week_pairs_share_top_50": round(
            float(
                top50["single_week_pairs"].sum() / top50["n_pairs"].sum()
            ), 6
        )
        if len(top50)
        else None,
        "bias": (
            "Every horizon here is a floor. Household-commodity pairs that "
            "bought in exactly one week contribute no gap and are the slowest "
            "buyers, and gaps are right-censored by the 102-week window. "
            "Rounding up is the conservative direction, not a safety margin."
        ),
        "written_to": str(out_path) if out_path is not None else None,
    }
    return cycles, diagnostics


_STAGE_SOURCES = (
    ("ingest", "ingest_report.json"),
    ("clean", "clean_diagnostics.json"),
    ("prices", "prices_diagnostics.json"),
    ("price_index", "price_index_diagnostics.json"),
    ("treatment", "treatment_diagnostics.json"),
    ("features", "feature_diagnostics.json"),
)


def build_quality_report(
    interim_dir: str | Path = "data/interim",
    *,
    household: dict[str, Any] | None = None,
    repurchase: dict[str, Any] | None = None,
    variation: dict[str, Any] | None = None,
    out_path: str | Path | None = "data/interim/quality.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble every exclusion this phase made into one honesty report.

    Reads the diagnostics each stage already wrote rather than recomputing
    anything: a stage that did not record its own exclusions cannot have them
    reconstructed here, and the report says which stages are missing instead of
    quietly showing a shorter list.

    Returns:
        `(report, diagnostics)`. The report is also written to `out_path`.
    """
    interim = Path(interim_dir)
    sources: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    for name, filename in _STAGE_SOURCES:
        path = interim / filename
        if path.exists():
            loaded[name] = json.loads(path.read_text())
            sources[name] = {"file": str(path), "present": True}
        else:
            sources[name] = {
                "file": str(path),
                "present": False,
                "consequence": (
                    f"{name} exclusions are absent from this report; re-run that "
                    f"stage before quoting a total"
                ),
            }

    exclusions: list[dict[str, Any]] = []
    for entry in loaded.get("clean", {}).get("filters", []):
        exclusions.append(
            {
                "stage": "clean",
                "name": entry["name"],
                "action": entry["action"],
                "definition": entry.get("definition"),
                "effect": entry.get("attributed_to_this_stage"),
                "share_of_all": entry.get("share_of_all"),
                "before": entry.get("before"),
                "after": entry.get("after"),
            }
        )
    for entry in loaded.get("prices", {}).get("exclusions", []):
        exclusions.append(
            {
                "stage": "prices",
                "name": entry["name"],
                "action": entry["action"],
                "definition": entry.get("definition"),
                "effect": entry.get("effect"),
                "before": entry.get("before"),
                "after": entry.get("after"),
            }
        )
    exclusions.extend(_late_exclusions(loaded))

    report: dict[str, Any] = {
        "report": "Phase 2 data honesty",
        "sources": sources,
        "actions": {
            "exclude": "rows removed from what flows downstream",
            "flag": "rows retained, marked, and still usable",
            "not_created": (
                "rows that were never materialised; there was nothing to remove"
            ),
        },
        "exclusions": exclusions,
        "scope": _scope_section(loaded),
        "no_margin": _margin_section(loaded),
        "treatment_coverage": _treatment_section(loaded),
        "household_availability": household,
        "repurchase": repurchase,
        "variation": variation,
        "unresolved": _unresolved(loaded, variation),
    }

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")

    diagnostics = {
        "stage": "build_quality_report",
        "sources_present": [k for k, v in sources.items() if v["present"]],
        "sources_missing": [k for k, v in sources.items() if not v["present"]],
        "exclusions_recorded": len(exclusions),
        "written_to": str(out_path) if out_path is not None else None,
    }
    return report, diagnostics


def _late_exclusions(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    """The three filters that were recorded elsewhere but not as exclusions.

    The scope is the largest filter in the whole pipeline — it drops about four
    fifths of transactions — and it was previously visible only under its own
    `scope` key. A reader auditing `exclusions` would not have seen it, which is
    the no-silent-filters rule failing on the biggest filter there is.

    Every number here is read from a diagnostics file that already computed it.
    Nothing is recomputed and no panel is touched.
    """
    entries: list[dict[str, Any]] = []
    prices = loaded.get("prices") or {}
    features = loaded.get("features") or {}

    # A stage may have written a partial diagnostics file. Skipping an entry
    # whose inputs are absent is right; crashing the honesty report because one
    # key is missing is not.
    undefined = prices.get("price_undefined")
    panel_rows = (prices.get("panel") or {}).get("rows")

    if undefined and panel_rows:
        entries.append(
            {
                "stage": "prices",
                "name": "price_undefined",
                # Rows survive; their prices do not. Not an "exclude", but it
                # removes them from every price-based analysis downstream.
                "action": "flag",
                "definition": "units <= 0 or a zero reconstruction base",
                "effect": {
                    "rows": undefined["rows"],
                    "units": undefined["units"],
                    "sales_value": undefined["sales_value"],
                },
                "share_of_all": {"rows_share": undefined["rows_share"]},
                "before": {"rows": panel_rows},
                "after": {"rows": panel_rows},
                "note": (
                    "Retained with null prices rather than dropped. Excluded "
                    "from price-based analysis, not from the panel."
                ),
            }
        )

    coverage = (features.get("scope") or {}).get("coverage")
    grid = features.get("grid")
    totals = prices.get("totals_after")

    if coverage and grid and totals and panel_rows:
        kept_rows = coverage["observed_rows"]

        def _dropped(share_key: str, total_key: str) -> float:
            kept = coverage[share_key] * totals[total_key]
            return round(totals[total_key] - kept, 2)

        entries.append(
            {
                "stage": "features",
                "name": "scope_restriction",
                "action": "exclude",
                "definition": features["scope"]["rule"],
                "effect": {
                    "rows": panel_rows - kept_rows,
                    "units": int(_dropped("units_share", "units")),
                    "sales_value": _dropped("sales_value_share", "sales_value"),
                },
                "share_of_all": {
                    "rows_share": round(1 - kept_rows / panel_rows, 6),
                    "transactions_share": round(
                        1 - coverage["transactions_share"], 6
                    ),
                    "units_share": round(1 - coverage["units_share"], 6),
                    "sales_value_share": round(
                        1 - coverage["sales_value_share"], 6
                    ),
                },
                "before": {"rows": panel_rows},
                "after": {"rows": kept_rows},
                "note": (
                    "The largest filter in the pipeline. What it keeps is in "
                    "the `scope` section; what it drops is here, so the two "
                    "readings cannot diverge."
                ),
            }
        )
        entries.append(
            {
                "stage": "features",
                "name": "carried_pairs_only",
                # These rows were never created, so there is nothing to remove.
                "action": "not_created",
                "definition": (
                    "zero rows are filled only for product-store pairs observed "
                    "at least once inside the scope"
                ),
                "effect": {
                    "rows": grid["full_cross_product_rows"] - grid["rows"],
                    "units": 0,
                    "sales_value": 0.0,
                },
                "before": {"rows": grid["full_cross_product_rows"]},
                "after": {"rows": grid["rows"]},
                "note": (
                    "A pair never observed is a shelf the store does not stock. "
                    "Creating those rows would invent demand observations, so "
                    "they carry no units and no sales by construction."
                ),
            }
        )
    return entries


def _scope_section(loaded: dict[str, Any]) -> dict[str, Any] | None:
    features = loaded.get("features")
    if not features:
        return None
    scope = features.get("scope") or {}
    grid = features.get("grid") or {}
    return {
        "rule": scope.get("rule"),
        "products": scope.get("n_products"),
        "stores": scope.get("n_stores"),
        "weeks": scope.get("n_weeks"),
        "panel_rows": grid.get("rows"),
        "observed_rows": grid.get("observed_rows"),
        "zero_filled_share": grid.get("zero_filled_share"),
        "buys": scope.get("coverage"),
        "costs": (
            "Everything outside the scope is not measured. The coverage shares "
            "above are what the row budget bought; a scope whose coverage is "
            "not recorded is indistinguishable from a silent filter."
        ),
    }


def _margin_section(loaded: dict[str, Any]) -> dict[str, Any] | None:
    ingest = loaded.get("ingest")
    if not ingest:
        return None
    return {
        "has_margin": ingest.get("has_margin"),
        "has_cogs": ingest.get("has_cogs"),
        "reason_code": ingest.get("margin_reason_code"),
        "note": ingest.get("margin_note"),
    }


def _treatment_section(loaded: dict[str, Any]) -> dict[str, Any] | None:
    treatment = loaded.get("treatment")
    if not treatment:
        return None
    return {
        "definition": (treatment.get("definition") or {}).get("treated"),
        "duplicate_rule": (treatment.get("duplicate_rule") or {}).get("rule"),
        "treated_rows": (treatment.get("treated") or {}).get("rows"),
        "absence_assumption": treatment.get("absence_assumption"),
    }


def _unresolved(
    loaded: dict[str, Any], variation: dict[str, Any] | None = None
) -> list[str]:
    """Things a reader of the panel must not assume were handled."""
    items = [
        (
            "Stockouts are unobservable. A zero can be 'nobody wanted it' or "
            "'it was not on the shelf', and the two are indistinguishable here. "
            "Stockouts correlate with promotions, so effects are understated."
        ),
        (
            "The panel cannot tell 'product not stocked by this store' from "
            "'stocked and unsold'. Task 2.6 fills zeros only for product-store "
            "pairs observed at least once, which bounds but does not solve it."
        ),
    ]
    features = loaded.get("features", {})
    if features.get("features", {}).get("holiday_flag", {}).get("populated") is False:
        items.append(
            "The holiday flag is unpopulated: this dataset has no calendar "
            "anchor. week_of_year carries seasonality without naming it."
        )
    treatment = loaded.get("treatment", {})
    outside = treatment.get("absence_assumption", {}).get("rows_outside_envelope")
    if outside:
        items.append(
            f"{outside:,} panel rows sit outside causal_data's coverage. They "
            f"are unobserved, not untreated, and must never be used as controls."
        )
    mismatch = (variation or {}).get("scope_rule_mismatch")
    base = (variation or {}).get("treated_base")
    if mismatch and base:
        items.append(
            f"Scope-rule mismatch, recorded rather than left to be inferred: "
            f"Task 2.6 scopes on BOOL_OR(on_display) OR BOOL_OR(in_mailer) "
            f"while the treatment is display alone, so "
            f"{mismatch['affected_products']} mailer-only products entered the "
            f"panel and can never be treated. The effective treated product "
            f"count is {base['effective_treated_products']}, not "
            f"{base['products_in_panel']}. They are retained deliberately as "
            f"untreated controls, not by oversight. If the row budget binds in "
            f"Phase 4, narrowing the rule to display-only frees "
            f"{mismatch['affected_rows']:,} rows."
        )
    return items


def write_quality_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return out
