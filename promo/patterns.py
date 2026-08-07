"""What kind of promotion works, not which one worked.

Task 7.2. Ranking answers *which campaign* did well. A merchant cannot run that
campaign again — the week has passed — so the question they can act on is *what
kind* of promotion does well, and that means looking across campaigns for
structure.

**The axes and bands are declared here, before any effect is looked at.**
`AXES` is a module constant, and `search_patterns` can only evaluate what is in
it. That ordering is the whole defence: a search that picks its segments after
seeing the numbers will find something in any dataset, and the finding will be
the analyst's, not the data's.

**Every attempted segment is recorded, including the ones that found nothing.**
`diagnostics["register"]` lists all of them with why each did or did not produce
a result — too few campaigns, a missing input, or an effect indistinguishable
from zero. **A pattern search whose denominator of attempted tests is unrecorded
is not a search, it is a story.** Task 7.3 needs that denominator to say whether
a surviving pattern is more than the best of many tries, and `tests_performed`
is the number it will use.

**Pooling aggregates components and divides once.** A segment's break-even
margin is `sum(cost) / sum(revenue)` across its campaigns, never the mean of
per-campaign ratios. Averaging ratios weights a tiny campaign the same as a
large one and is not the number anybody wants.

**`bounded` products are excluded from the depth axis entirely.** Settled
decision 6: their depth is ordinal only, because they are on deal in over 90% of
their priced weeks and the regular price is barely observed. Banding an ordinal
quantity into "20–30%" asserts a cardinal reading the data does not support. The
count removed is reported rather than quietly dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "AXES",
    "DEPTH_BANDS",
    "MIN_CAMPAIGNS_PER_SEGMENT",
    "UndeclaredAxisError",
    "search_patterns",
    "write_diagnostics",
]

#: A segment with fewer campaigns than this is reported but not pooled. Two
#: campaigns cannot distinguish a pattern from a coincidence, and pooling them
#: produces a number that invites exactly that reading.
MIN_CAMPAIGNS_PER_SEGMENT: int = 3

#: Declared before any effect is seen. Depth bands are left-closed.
DEPTH_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-10%", 0.00, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20-30%", 0.20, 0.30),
    ("30-50%", 0.30, 0.50),
    ("50%+", 0.50, np.inf),
)

#: The four axes, their segments, and the campaign column each one reads.
#:
#: `buildable = False` marks a segment declared by the plan that **this dataset
#: cannot supply**. It stays in the register so the denominator of attempted
#: tests is honest: a segment that could not be built is a different kind of
#: null from one that was built and found nothing, and both belong in the count
#: Task 7.3 reads.
AXES: dict[str, dict[str, Any]] = {
    "depth": {
        "why": "does discount depth predict return?",
        "segments": {
            "depth_band": {
                "column": "depth",
                "kind": "band",
                "bands": DEPTH_BANDS,
                "excludes_bounded": True,
            },
        },
    },
    "timing": {
        "why": "does when it ran predict return?",
        "segments": {
            "week_of_year_block": {"column": "week_of_year_block", "kind": "category"},
            "campaign_length_weeks": {"column": "length_weeks", "kind": "category"},
            "position_vs_cycle": {"column": "position_vs_cycle", "kind": "category"},
            "holiday_proximity": {
                "column": None,
                "kind": "category",
                "buildable": False,
                "why_not": (
                    "This dataset carries no calendar dates — only DAY 1-711 "
                    "and WEEK_NO 1-102, with no anchor to a real year — so "
                    "which week contains which holiday is unknowable. Recorded "
                    "in docs/data_findings.md, Task 2.6. Anchoring to a guessed "
                    "year would be an assumption presented as a fact."
                ),
            },
        },
    },
    "product": {
        "why": "does what was promoted predict return?",
        "segments": {
            "department": {"column": "DEPARTMENT", "kind": "category"},
            "commodity": {"column": "COMMODITY_DESC", "kind": "category"},
            "price_tier": {"column": "price_tier", "kind": "category"},
            "cycle_band": {"column": "cycle_band", "kind": "category"},
        },
    },
    "store": {
        "why": "does where it ran predict return?",
        "segments": {
            "traffic_tier": {"column": "traffic_tier", "kind": "category"},
            "stores_carrying_tier": {"column": "stores_carrying_tier", "kind": "category"},
            "treatment_intensity_tier": {
                "column": "treatment_intensity_tier", "kind": "category"
            },
        },
    },
}


class UndeclaredAxisError(Exception):
    """A caller asked for an axis that is not in `AXES`.

    Refused rather than accommodated. Adding an axis after seeing the effects
    is what turns a search into a story, so a new axis is a code change to a
    declared constant, reviewable as one.
    """


def _band_of(value: float, bands: tuple[tuple[str, float, float], ...]) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    for label, low, high in bands:
        if low <= value < high:
            return label
    return None


def _pool(block: pd.DataFrame, columns: dict[str, str]) -> dict[str, Any]:
    """Aggregate the components, then divide once."""
    lift = float(block[columns["lift"]].sum())
    cost = float(block[columns["cost"]].sum())
    revenue = float(block[columns["revenue"]].sum())
    return {
        "campaigns": len(block),
        "products": int(block[columns["products"]].sum())
        if columns["products"] in block
        else None,
        "stores": int(block[columns["stores"]].sum())
        if columns["stores"] in block
        else None,
        "pooled_lift": round(lift, 6),
        "pooled_cost": round(cost, 2),
        "pooled_incremental_revenue": round(revenue, 2),
        # Divided once, at the end. The mean of per-campaign ratios would
        # weight a two-store campaign the same as a hundred-store one.
        "breakeven_margin": round(cost / revenue, 6) if revenue else None,
        "breakeven_undefined": revenue == 0,
    }


def search_patterns(
    campaign_results: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    *,
    axes: tuple[str, ...] | None = None,
    lift_column: str = "lift",
    cost_column: str = "promo_cost",
    revenue_column: str = "incremental_revenue",
    products_column: str = "n_products",
    stores_column: str = "n_stores",
    status_column: str = "price_status",
    min_campaigns: int = MIN_CAMPAIGNS_PER_SEGMENT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pool the evaluated campaigns along the declared axes.

    Args:
        campaign_results: one row per evaluated campaign, carrying the effect
            columns and whatever segment attributes are available. A segment
            whose column is absent is **recorded as not evaluated**, never
            skipped silently.
        panel: unused here and accepted for interface stability — every segment
            attribute this version reads is a campaign-level fact, and deriving
            them inside a search would put attribute construction after the
            effects are visible.
        axes: restrict to these declared axes. Anything not in `AXES` raises.

    Returns:
        `(patterns, diagnostics)`. One row per (axis, segment, value).
        `verdict` is present and null — Task 7.3 owns it, and a column that
        silently disappeared would be harder to notice than one that is empty.

    Raises:
        UndeclaredAxisError: an axis outside `AXES` was requested.
        KeyError: an effect column is missing.
    """
    wanted = tuple(AXES) if axes is None else tuple(axes)
    undeclared = [a for a in wanted if a not in AXES]
    if undeclared:
        raise UndeclaredAxisError(
            f"{undeclared} are not declared in AXES. Adding an axis after the "
            f"effects are visible is what turns a search into a story; declare "
            f"it in the constant first."
        )
    columns = {
        "lift": lift_column, "cost": cost_column, "revenue": revenue_column,
        "products": products_column, "stores": stores_column,
    }
    for role in ("lift", "cost", "revenue"):
        if columns[role] not in campaign_results.columns:
            raise KeyError(f"{columns[role]!r} is not a column of campaign_results")

    rows: list[dict[str, Any]] = []
    register: list[dict[str, Any]] = []
    attempted = 0

    for axis in wanted:
        for name, spec in AXES[axis]["segments"].items():
            entry: dict[str, Any] = {"axis": axis, "segment": name}

            if spec.get("buildable", True) is False:
                entry.update(
                    {"evaluated": False, "reason": "not buildable on this dataset",
                     "detail": spec["why_not"], "values_tested": 0}
                )
                register.append(entry)
                continue

            column = spec["column"]
            if column not in campaign_results.columns:
                entry.update(
                    {"evaluated": False,
                     "reason": f"campaign_results has no {column!r} column",
                     "values_tested": 0}
                )
                register.append(entry)
                continue

            frame = campaign_results
            excluded = 0
            if spec.get("excludes_bounded") and status_column in frame.columns:
                bounded = frame[status_column] == "bounded"
                excluded = int(bounded.sum())
                frame = frame.loc[~bounded]

            if spec["kind"] == "band":
                keys = frame[column].map(lambda v: _band_of(v, spec["bands"]))
            else:
                keys = frame[column]
            grouped = frame.assign(_segment=keys).dropna(subset=["_segment"])

            values, pooled_rows = 0, 0
            for value, block in grouped.groupby("_segment", observed=True):
                values += 1
                attempted += 1
                pooled = _pool(block, columns)
                row = {
                    "axis": axis, "segment": name, "value": value,
                    **pooled,
                    # Task 7.3 owns this. Present and null, because a column
                    # that quietly vanished is harder to notice than an empty one.
                    "verdict": None,
                    "pooled": pooled["campaigns"] >= min_campaigns,
                    "too_few_campaigns": pooled["campaigns"] < min_campaigns,
                }
                rows.append(row)
                pooled_rows += int(row["pooled"])

            entry.update(
                {"evaluated": True, "values_tested": values,
                 "values_pooled": pooled_rows,
                 "bounded_campaigns_excluded": excluded if spec.get("excludes_bounded") else None,
                 "reason": None if values else "no campaign fell into any band"}
            )
            register.append(entry)

    patterns = pd.DataFrame(rows)
    if not patterns.empty:
        patterns = patterns.sort_values(["axis", "segment", "value"]).reset_index(
            drop=True
        )

    evaluated = [e for e in register if e["evaluated"]]
    diagnostics = {
        "stage": "search_patterns",
        "campaigns_in": len(campaign_results),
        "axes_declared": list(AXES),
        "axes_searched": list(wanted),
        "segments_declared": sum(len(AXES[a]["segments"]) for a in AXES),
        "segments_evaluated": len(evaluated),
        "segments_not_evaluated": len(register) - len(evaluated),
        # The denominator Task 7.3 needs.
        "tests_performed": attempted,
        "register": register,
        "min_campaigns_per_segment": min_campaigns,
        "declaration_rule": (
            "AXES is a module constant, fixed before any effect is looked at, "
            "and search_patterns can only evaluate what is in it. A search that "
            "chooses its segments after seeing the numbers finds something in "
            "any dataset."
        ),
        "denominator_rule": (
            "Every attempted segment is in the register, including the ones "
            "that found nothing and the ones that could not be built. A pattern "
            "search whose denominator of attempted tests is unrecorded is not a "
            "search, it is a story. tests_performed is what Task 7.3 divides by."
        ),
        "pooling_rule": (
            "Components are aggregated and divided once: a segment's break-even "
            "margin is sum(cost) / sum(revenue), never the mean of per-campaign "
            "ratios, which would weight a two-store campaign like a "
            "hundred-store one."
        ),
        "verdict_owner": (
            "The verdict column is present and null. Task 7.3 fills it; it is "
            "not computed here, and an empty column is easier to notice than a "
            "missing one."
        ),
    }
    return patterns, diagnostics


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return out
