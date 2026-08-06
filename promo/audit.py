"""Where the treatment varies, and therefore what can be compared.

Before any model runs, one question decides whether the rest is possible: is
there an axis along which treated and untreated observations sit side by side?
`variation_axes()` answers it for product, store and week by partitioning the
panel along each and classifying every level as fully treated, fully untreated,
or mixed.

**Mixed mass is what identifies.** A product treated in every week it appears is
a product with no counterfactual of its own; a week in which everything is
treated offers no contemporaneous comparison. Only levels holding both kinds of
observation contribute, and the share of *units* in those levels — not the share
of levels — is what says whether the comparison is made on real demand mass or
on a tail.

**What each axis licenses is different, and the verdict says so.** A mixed
*product* means the product is treated in some product-store-weeks and not
others, so it can be compared against itself. A mixed *week* means treated and
untreated units coexist in that week, so a contemporaneous cross-section is
available. A mixed *store* means the store carries both. These are three
different estimators, not three measurements of one thing, and an axis being
mixed is necessary for its comparison, never sufficient — Task 3.2's overlap
check is what tests whether the comparison is also fair.

**This audit is marginal, and that is coarser than it looks.** A level is mixed
if it holds both kinds of observation *anywhere* — a store counts as mixed when
some product in some week was treated there while something else was not. That
condition is nearly always satisfiable and is a much weaker statement than "for
the same product, in the same week, this store differed from another". Settled
decision 4 rests on the stronger, joint version — `display` varies across stores
within a week for 65.34% of treated products against `mailer`'s 2.28% — and the
marginal classification here cannot separate the two treatments on it: both come
back with essentially every store mixed. Read this module as the necessary
condition and Task 3.2 as the test with teeth. The `limitation` field in the
diagnostics says the same thing to anyone reading the output rather than the
source.

Task 1.4 measured the same shape at entity level on the full panel, weighting
every product equally. This module weights by units and runs on whatever panel
it is given, which is the scoped modelling panel by default: `data_findings.md`
records that the gate should compute on that slice rather than the full panel,
since 80% of stores never appear in `causal_data` at all and would otherwise
pile up as spurious "never treated" mass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.io import connect

__all__ = [
    "AXES",
    "MEANINGFUL_MIXED_SHARE",
    "UnobservedRowsError",
    "variation_axes",
    "write_diagnostics",
]

#: The three axes the panel can be partitioned along, and their key columns.
AXES: dict[str, str] = {
    "product": "PRODUCT_ID",
    "store": "STORE_ID",
    "week": "WEEK_NO",
}

#: Default bar for calling an axis's mixed mass "meaningful". Not a recorded
#: decision — the plan says "meaningful" without defining it — so it is a
#: parameter with a stated default rather than a constant buried in a
#: comparison. A tenth of unit mass is the point below which an estimate rests
#: on a tail of the panel however large the panel is.
MEANINGFUL_MIXED_SHARE: float = 0.10

_CLASSES = ("fully_treated", "fully_untreated", "mixed")


class UnobservedRowsError(Exception):
    """The panel contains rows where treatment was never observed."""


def variation_axes(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    treatment_column: str = "treated",
    meaningful_mixed_share: float = MEANINGFUL_MIXED_SHARE,
    require_observed: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify each axis's levels as fully treated, fully untreated, or mixed.

    Args:
        panel: the modelling panel, as a parquet path or a DataFrame.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        treatment_column: which boolean to treat as the treatment. Parameterised
            so the gate can be re-run under `on_display` and `in_mailer` and show
            why settled decision 4 chose one of them.
        meaningful_mixed_share: unit-mass share above which an axis's mixed
            levels are called usable.
        require_observed: raise if the panel holds rows outside the treatment
            log's coverage. Those rows are unobserved, not untreated (Task 2.5),
            and counting them as untreated manufactures comparison mass.

    Returns:
        `(axes, diagnostics)`. `axes` has one row per (axis, class) with the
        level count, row count, unit mass and their shares.

    Raises:
        UnobservedRowsError: `require_observed` and the panel holds rows where
            treatment was never observed.
        KeyError: `treatment_column` is not in the panel.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        frame = _load(panel, con, treatment_column, require_observed)
    finally:
        if own:
            con.close()

    total_units = float(frame["units"].sum())
    total_rows = len(frame)

    rows: list[dict[str, Any]] = []
    for axis, key in AXES.items():
        # A level is mixed when it holds both kinds of observation. Comparing
        # the treated-row count against 0 and against the level's size is what
        # separates "never" and "always" from "sometimes".
        grouped = frame.groupby(key, observed=True).agg(
            level_rows=("units", "size"),
            level_units=("units", "sum"),
            treated_rows=("_treated", "sum"),
        )
        klass = pd.Series("mixed", index=grouped.index, dtype="string")
        klass = klass.mask(grouped["treated_rows"] == 0, "fully_untreated")
        klass = klass.mask(
            grouped["treated_rows"] == grouped["level_rows"], "fully_treated"
        )
        grouped["class"] = klass

        for name in _CLASSES:
            block = grouped[grouped["class"] == name]
            rows.append(
                {
                    "axis": axis,
                    "key": key,
                    "class": name,
                    "levels": len(block),
                    "levels_share": round(len(block) / len(grouped), 6)
                    if len(grouped)
                    else 0.0,
                    "rows": int(block["level_rows"].sum()),
                    "rows_share": round(
                        float(block["level_rows"].sum()) / total_rows, 6
                    )
                    if total_rows
                    else 0.0,
                    "units": int(block["level_units"].sum()),
                    "units_share": round(
                        float(block["level_units"].sum()) / total_units, 6
                    )
                    if total_units
                    else 0.0,
                }
            )

    axes = pd.DataFrame(rows)
    diagnostics = _diagnose(
        axes,
        frame=frame,
        treatment_column=treatment_column,
        meaningful_mixed_share=meaningful_mixed_share,
        total_units=total_units,
        total_rows=total_rows,
    )
    return axes, diagnostics


def _load(
    panel: str | Path | pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    treatment_column: str,
    require_observed: bool,
) -> pd.DataFrame:
    """Read only the columns the audit needs, and check the panel is auditable."""
    if isinstance(panel, pd.DataFrame):
        frame = panel
    else:
        source = f"read_parquet('{Path(panel).as_posix()}')"
        available = {
            r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        if treatment_column not in available:
            raise KeyError(
                f"{treatment_column!r} is not a column of the panel; found "
                f"{sorted(available)}"
            )
        wanted = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "units", treatment_column]
        if "treatment_observed" in available:
            wanted.append("treatment_observed")
        frame = con.execute(
            f"SELECT {', '.join(wanted)} FROM {source}"
        ).df()

    if treatment_column not in frame.columns:
        raise KeyError(f"{treatment_column!r} is not a column of the panel")

    if require_observed and "treatment_observed" in frame.columns:
        unobserved = int((~frame["treatment_observed"].astype(bool)).sum())
        if unobserved:
            raise UnobservedRowsError(
                f"{unobserved:,} rows have treatment_observed = False. Those are "
                f"unobserved, not untreated, and counting them as untreated "
                f"would manufacture comparison mass. Scope the panel to the "
                f"treatment envelope first, or pass require_observed=False to "
                f"audit anyway."
            )

    frame = frame.copy()
    frame["_treated"] = frame[treatment_column].astype(bool)
    return frame


def _diagnose(
    axes: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    treatment_column: str,
    meaningful_mixed_share: float,
    total_units: float,
    total_rows: int,
) -> dict[str, Any]:
    mixed = axes[axes["class"] == "mixed"].set_index("axis")
    usable = mixed[mixed["units_share"] >= meaningful_mixed_share]

    # `idxmax` on an all-zero column returns the first row, which would name an
    # axis as "best" when nothing is mixed at all. And when two axes tie, the
    # winner would be whichever comes first in AXES — an ordering artefact
    # presented as a finding. Both are reported rather than resolved.
    shares = mixed["units_share"]
    top = float(shares.max()) if len(shares) else 0.0
    leaders = sorted(shares[shares == top].index) if top > 0 else []
    best = leaders[0] if len(leaders) == 1 else None

    # How many products could ever contribute a treated observation. A product
    # that is fully untreated across the whole panel is a control and nothing
    # else, so quoting the scoped product count as if it were the treated count
    # overstates the estimator's base.
    product_rows = axes[axes["axis"] == "product"].set_index("class")
    never_treated = int(product_rows.loc["fully_untreated", "levels"])
    products = int(frame["PRODUCT_ID"].nunique())
    control_only_rows = int(product_rows.loc["fully_untreated", "rows"])
    control_only_units_share = float(
        product_rows.loc["fully_untreated", "units_share"]
    )

    return {
        "stage": "variation_axes",
        "treatment_column": treatment_column,
        "panel": {
            "rows": total_rows,
            "units": int(total_units),
            "treated_rows": int(frame["_treated"].sum()),
            "treated_rows_share": round(float(frame["_treated"].mean()), 6)
            if total_rows
            else 0.0,
            "products": int(frame["PRODUCT_ID"].nunique()),
            "stores": int(frame["STORE_ID"].nunique()),
            "weeks": int(frame["WEEK_NO"].nunique()),
        },
        "treated_base": {
            "products_in_panel": products,
            "effective_treated_products": products - never_treated,
            "control_only_products": never_treated,
            "control_only_rows": control_only_rows,
            "control_only_units_share": round(control_only_units_share, 6),
            "note": (
                f"{products - never_treated} of {products} scoped products can "
                f"ever contribute a treated observation. The other "
                f"{never_treated} are controls and nothing else. Quote the "
                f"effective count, not the scoped count, when describing the "
                f"estimator's base."
            ),
        },
        "mixed_units_share": {
            axis: float(mixed.loc[axis, "units_share"]) for axis in mixed.index
        },
        "threshold": meaningful_mixed_share,
        "usable_axes": sorted(usable.index),
        "best_axis": best,
        "leading_axes": leaders,
        "verdict": (
            f"{best} carries the most mixed unit mass"
            if best is not None
            else (
                "no axis has any mixed mass"
                if not leaders
                else f"{' and '.join(leaders)} tie on mixed unit mass"
            )
        ),
        "reading": (
            "Mixed mass is necessary for a comparison on that axis, never "
            "sufficient. A mixed product can be compared against itself; a "
            "mixed week offers a contemporaneous cross-section; a mixed store "
            "carries both kinds. They are three different estimators. Task 3.2 "
            "is what tests whether the comparison is also fair."
        ),
        "weighting": (
            "Shares are of units, not of levels. An axis can have most of its "
            "levels mixed while the mixing sits on a tail of demand, and the "
            "level share would call that healthy."
        ),
        "limitation": (
            "Marginal, not joint. A level counts as mixed if it holds both "
            "kinds of observation anywhere, so a store is mixed when some "
            "product in some week was treated there while something else was "
            "not. That is far weaker than 'for the same product, in the same "
            "week, this store differed from another', which is what settled "
            "decision 4 rests on. On the real panel display and mailer both "
            "return essentially every store mixed, so this audit cannot tell "
            "them apart and must not be cited as if it could."
        ),
        "scope_rule_mismatch": {
            "what": (
                "Task 2.6 selects the scope from products ever treated by "
                "either mechanic — BOOL_OR(on_display) OR BOOL_OR(in_mailer) — "
                "while the treatment is display alone (settled decision 4). "
                "Products carried only in the mailer therefore enter the scope "
                "and can never contribute a treated observation."
            ),
            "affected_products": never_treated,
            "affected_rows": control_only_rows,
            "affected_units_share": round(control_only_units_share, 6),
            "decision": (
                "Retained deliberately as untreated controls, not an oversight. "
                "They are valid control observations; they are simply never "
                "treated ones. The panel is not rebuilt for this."
            ),
            "if_the_budget_binds": (
                f"Narrowing the scope rule to BOOL_OR(on_display) would free "
                f"{control_only_rows:,} rows for products that can actually "
                f"switch. Worth doing only if the row budget becomes binding in "
                f"Phase 4; it changes Task 2.6's output and requires rebuilding "
                f"the panel."
            ),
        },
    }


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
