"""Transaction cleaning: flags, the product join, and the honesty record.

This stage removes nothing. Every row of `transaction_data` survives into the
returned frame carrying boolean flags that say what is wrong with it, plus one
`usable` column that later stages read. The diagnostics dict then states, for
every flag, how many rows it covers and what share of units and sales value it
carries — before and after, in a declared cascade order.

That shape is deliberate. A filter applied here would be invisible to Phase 5:
free goods are 0.00% of units and would never be missed, but they are a real
promotional cost, and Task 5.1 needs to find them. So this stage marks and
reports; the panel builder in Task 2.3 is where the exclusion is executed, using
`usable`.

**On identifying volume-measured rows.** The plan's Task 2.2 says to use "the
unit-value threshold from docs/data_findings.md". The decision actually recorded
there is narrower and different: settled decision 2 identifies these rows by
`DEPARTMENT IN ('KIOSK-GAS', 'MISC SALES TRAN')`, and Task 1.2 says in terms
that `DEPARTMENT` is the discriminator and unit value is not, because a
unit-value cut also sweeps in several thousand zero-priced grocery giveaways —
which are the free goods Phase 5 must keep. The recorded decision wins. The 0.05
threshold is still computed here, as a cross-check that reports where the two
rules disagree and by how much, so the choice stays visible rather than becoming
a line of code nobody can question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.io import LazyTable, connect

__all__ = [
    "CASCADE",
    "UNIT_VALUE_THRESHOLD",
    "VOLUME_MEASURED_DEPARTMENTS",
    "JoinError",
    "clean_transactions",
    "write_diagnostics",
]

#: Settled decision 2 in docs/data_findings.md. Not a unit-value rule.
VOLUME_MEASURED_DEPARTMENTS: tuple[str, ...] = ("KIOSK-GAS", "MISC SALES TRAN")

#: Task 1.2's unit-value line. Retained only as a cross-check diagnostic; it is
#: not what identifies a volume-measured row.
UNIT_VALUE_THRESHOLD: float = 0.05

#: The order in which exclusions are attributed. A row failing two of these is
#: counted against the first, so the per-filter numbers sum to the total removed
#: instead of double-counting. Changing this order changes the attribution, not
#: the result.
CASCADE: tuple[str, ...] = (
    "nonpositive_quantity",
    "unmatched_product",
    "volume_measured",
)

#: Flags that describe a row without removing it from `usable`.
RETAINED_FLAGS: tuple[str, ...] = ("free_good", "blank_department")

#: Columns of the raw table this stage does not carry forward.
DROPPED_COLUMNS: tuple[str, ...] = ("TRANS_TIME",)

_KEEP = (
    "household_key",
    "BASKET_ID",
    "DAY",
    "WEEK_NO",
    "PRODUCT_ID",
    "STORE_ID",
    "QUANTITY",
    "SALES_VALUE",
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
)


class JoinError(Exception):
    """The product join would change the row count."""


def _totals(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, float]:
    sub = frame if mask is None else frame[mask]
    return {
        "rows": len(sub),
        "units": int(sub["QUANTITY"].sum()),
        "sales_value": round(float(sub["SALES_VALUE"].sum()), 2),
    }


def _shares(part: dict[str, float], whole: dict[str, float]) -> dict[str, float]:
    return {
        f"{k}_share": (round(part[k] / whole[k], 6) if whole[k] else 0.0)
        for k in ("rows", "units", "sales_value")
    }


def _departments_of(frame: pd.DataFrame, mask: pd.Series, limit: int = 5) -> dict[str, int]:
    counts = frame.loc[mask, "DEPARTMENT"].astype("string").value_counts()
    return {str(k): int(v) for k, v in counts.head(limit).items()}


def clean_transactions(
    transactions: LazyTable | pd.DataFrame,
    product: pd.DataFrame,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    out_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag every questionable transaction row and join its category.

    Args:
        transactions: the `LazyTable` handle from `promo.io.load_raw`, or a
            DataFrame with the same columns (used by the tests).
        product: the product table, needing `PRODUCT_ID`, `DEPARTMENT`, and
            `COMMODITY_DESC`.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        out_path: if given, the cleaned frame is also written there as parquet.

    Returns:
        `(cleaned, diagnostics)`. `cleaned` has one row per input row, the six
        flag columns, and `usable`. `diagnostics` records every flag's effect on
        rows, units, and sales value, before and after.

    Raises:
        JoinError: `product` has duplicate `PRODUCT_ID`s, which would fan the
            join out and silently inflate units.
    """
    dup = int(product["PRODUCT_ID"].duplicated().sum())
    if dup:
        raise JoinError(
            f"product has {dup:,} duplicate PRODUCT_ID rows; the join would "
            f"multiply transaction rows and inflate units."
        )

    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(transactions, LazyTable):
            tx_src = transactions.source_sql
            source = {
                "kind": transactions.source_kind,
                "path": str(transactions.path),
                "rows": transactions.rows,
            }
        else:
            con.register("_tx_frame", transactions)
            tx_src = "_tx_frame"
            source = {"kind": "dataframe", "path": None, "rows": len(transactions)}

        con.register("_product_frame", product[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]])

        cols = ", ".join(f"t.{c}" for c in _KEEP)
        depts = ", ".join(f"'{d}'" for d in VOLUME_MEASURED_DEPARTMENTS)
        query = f"""
            SELECT
                {cols},
                p.DEPARTMENT,
                p.COMMODITY_DESC,
                (t.QUANTITY <= 0) AS nonpositive_quantity,
                (t.QUANTITY > 0 AND t.SALES_VALUE = 0) AS free_good,
                (p.PRODUCT_ID IS NULL) AS unmatched_product,
                COALESCE(TRIM(p.DEPARTMENT) = '', FALSE) AS blank_department,
                COALESCE(p.DEPARTMENT IN ({depts}), FALSE) AS volume_measured
            FROM {tx_src} t
            LEFT JOIN _product_frame p ON t.PRODUCT_ID = p.PRODUCT_ID
        """
        cleaned = con.execute(query).df()
    finally:
        if own:
            con.close()

    # A left join against a unique key conserves rows. Checked rather than
    # trusted: a fan-out here would inflate every unit total downstream.
    assert len(cleaned) == source["rows"], (
        f"product join changed the row count: {source['rows']:,} in, "
        f"{len(cleaned):,} out"
    )

    for column in ("DEPARTMENT", "COMMODITY_DESC"):
        cleaned[column] = cleaned[column].astype("category")

    excluded = cleaned[list(CASCADE)].any(axis=1)
    cleaned["usable"] = ~excluded

    diagnostics = _diagnose(cleaned, source)

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(path, index=False)
        diagnostics["written_to"] = str(path)

    return cleaned, diagnostics


def _diagnose(cleaned: pd.DataFrame, source: dict[str, Any]) -> dict[str, Any]:
    """Every flag's effect on rows, units, and sales value, before and after."""
    before_all = _totals(cleaned)
    filters: list[dict[str, Any]] = []

    # Exclusions, attributed in cascade order: a row is charged to the first
    # rule it fails, so the per-filter counts sum to the total removed.
    surviving = pd.Series(True, index=cleaned.index)
    definitions = {
        "nonpositive_quantity": "QUANTITY <= 0",
        "unmatched_product": "PRODUCT_ID absent from product.csv",
        "volume_measured": (
            f"DEPARTMENT IN {VOLUME_MEASURED_DEPARTMENTS} "
            f"(settled decision 2, not a unit-value rule)"
        ),
        "free_good": "QUANTITY > 0 AND SALES_VALUE = 0",
        "blank_department": "DEPARTMENT is whitespace-only",
    }
    reasons = {
        "nonpositive_quantity": (
            "No units bought, so unit value is undefined and the row cannot "
            "contribute to a units panel."
        ),
        "unmatched_product": "No DEPARTMENT or COMMODITY_DESC can be attached.",
        "volume_measured": (
            "Fuel and kiosk lines measured in fractions of a unit. Summing them "
            "with counted groceries measures petrol volume, not demand."
        ),
        "free_good": (
            "Retained: real units at zero revenue. These are a promotional cost "
            "Task 5.1 must price at the regular price, so they must not be lost "
            "here. Read them by this flag, not from the usable subset — a few "
            "sit in an excluded department, as rows_already_excluded_by shows."
        ),
        "blank_department": (
            "Flagged for completeness. On this dataset every one of these rows "
            "also has QUANTITY = 0, so the cascade has already excluded them — "
            "see rows_still_usable before reading this as retention."
        ),
    }

    for name in CASCADE:
        stage_in = _totals(cleaned, surviving)
        hit = surviving & cleaned[name]
        effect = _totals(cleaned, hit)
        surviving = surviving & ~cleaned[name]
        stage_out = _totals(cleaned, surviving)
        entry: dict[str, Any] = {
            "name": name,
            "action": "exclude",
            "definition": definitions[name],
            "reason": reasons[name],
            "flagged_rows_total": int(cleaned[name].sum()),
            "attributed_to_this_stage": effect,
            "share_of_all": _shares(effect, before_all),
            "share_of_stage_input": _shares(effect, stage_in),
            "before": stage_in,
            "after": stage_out,
        }
        filters.append(entry)

    # Flags that keep their rows. `before` equals `after` on purpose: the record
    # shows the retention was a decision, not an oversight. `rows_still_usable`
    # is the part that matters — a flag can be retaining nothing because the
    # cascade already took all of its rows, and that must not read as retention.
    for name in RETAINED_FLAGS:
        flagged = cleaned[name]
        effect = _totals(cleaned, flagged)
        filters.append(
            {
                "name": name,
                "action": "flag",
                "definition": definitions[name],
                "reason": reasons[name],
                "flagged_rows_total": int(flagged.sum()),
                "attributed_to_this_stage": effect,
                "share_of_all": _shares(effect, before_all),
                "share_of_stage_input": _shares(effect, before_all),
                "before": before_all,
                "after": before_all,
                "rows_still_usable": int((flagged & cleaned["usable"]).sum()),
                "rows_already_excluded_by": {
                    rule: int((flagged & cleaned[rule]).sum())
                    for rule in CASCADE
                    if int((flagged & cleaned[rule]).sum())
                },
            }
        )

    usable = _totals(cleaned, cleaned["usable"])
    return {
        "stage": "clean_transactions",
        "source": source,
        "columns_dropped": list(DROPPED_COLUMNS),
        "cascade_order": list(CASCADE),
        "totals_before": before_all,
        "totals_usable": usable,
        "usable_share": _shares(usable, before_all),
        "rows_removed": int(before_all["rows"] - usable["rows"]),
        "filters": filters,
        "product_join": {
            "unmatched_rows": int(cleaned["unmatched_product"].sum()),
            "distinct_departments": int(cleaned["DEPARTMENT"].nunique(dropna=True)),
        },
        "volume_rule_cross_check": _cross_check(cleaned, before_all),
        "notes": [
            (
                "Nothing was dropped. `usable` marks the rows later stages read; "
                "every excluded row is still present and flagged."
            ),
            (
                "Free goods are retained deliberately. They are 0.00% of units "
                "and a real promotional cost, and Task 5.1 reads them from here."
            ),
            (
                "Volume-measured rows are identified by DEPARTMENT per settled "
                "decision 2, not by the unit-value threshold the plan's prose "
                "mentions. See volume_rule_cross_check for what that choice costs."
            ),
        ],
    }


def _cross_check(cleaned: pd.DataFrame, before_all: dict[str, float]) -> dict[str, Any]:
    """Compare the recorded DEPARTMENT rule against the 0.05 unit-value line."""
    counted = cleaned["QUANTITY"] > 0
    unit_value = pd.Series(float("nan"), index=cleaned.index)
    unit_value[counted] = (
        cleaned.loc[counted, "SALES_VALUE"] / cleaned.loc[counted, "QUANTITY"]
    )
    by_threshold = counted & (unit_value < UNIT_VALUE_THRESHOLD)
    by_department = cleaned["volume_measured"]

    only_threshold = by_threshold & ~by_department
    only_department = by_department & ~by_threshold

    return {
        "recorded_rule": f"DEPARTMENT IN {list(VOLUME_MEASURED_DEPARTMENTS)}",
        "compared_against": f"SALES_VALUE / QUANTITY < {UNIT_VALUE_THRESHOLD}",
        "by_department": {
            **_totals(cleaned, by_department),
            **_shares(_totals(cleaned, by_department), before_all),
        },
        "by_threshold": {
            **_totals(cleaned, by_threshold),
            **_shares(_totals(cleaned, by_threshold), before_all),
        },
        "agree": _totals(cleaned, by_threshold & by_department),
        "threshold_only": {
            **_totals(cleaned, only_threshold),
            "departments": _departments_of(cleaned, only_threshold),
            "of_which_free_goods": int((only_threshold & cleaned["free_good"]).sum()),
        },
        "department_only": {
            **_totals(cleaned, only_department),
            "departments": _departments_of(cleaned, only_department),
            "of_which_nonpositive_quantity": int(
                (only_department & cleaned["nonpositive_quantity"]).sum()
            ),
        },
        "reading": (
            "The threshold's extra rows are ordinary groceries priced at zero — "
            "giveaways, which Phase 5 counts as a cost. Excluding them as "
            "volume-measured would delete a cost line. The department rule's "
            "extra rows are kiosk lines the threshold misses. The two rules "
            "differ on rows, barely on units."
        ),
    }


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
