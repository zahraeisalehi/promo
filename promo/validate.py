"""Where zero actually sits: the placebo distribution, and the gate it owes.

Task 4.5. Everything before this produces a number. This produces the thing the
number has to be compared against.

Run the whole estimator — rollout, residuals, gross and net — over windows where
**nothing happened**. The truth in every one of those windows is zero. Whatever
the estimator returns instead is the null distribution of this comparison, and
the runbook's line is the one that matters: *the band this produces is almost
never at zero*.

**The windows come from ever-treated cells, in weeks they were not treated.**
Not from never-treated cells, which is what this module did first and got
wrong. Task 3.2 separates promoted from unpromoted cells at **AUC 0.70** — they
are different populations, the estimator drifts differently on each, and a null
measured on one is mis-centred as the null for the other. That was not a
theoretical worry: on the five densest commodities the never-treated band's
median disagreed in *sign* with the campaign's own pre-period drift on four of
five, and the disagreement was what exposed it. `POOL_KINDS` keeps the
never-treated pool available so the comparison stays reproducible.

**The band is matched to the campaign, and that is the whole design.** A
placebo drawn on one product-store over three weeks and a campaign measured over
101 product-stores and ten weeks are not the same statistic, and comparing them
would be arithmetic between different units. Every draw therefore takes the same
number of cells, the same campaign length and the same horizon as the campaign
it will be compared against. A band that is not size-matched is narrower than
the truth and turns noise into findings — the exact failure this module exists
to prevent.

**Inside the band means this comparison cannot see the effect.** It never means
the promotion did nothing. The reason code carries that distinction in its
message and there is a test asserting the wording; the same distinction is why
`inside_band` returns evidence rather than a verdict, and why the gate's message
names what would be needed to tell them apart.

**Two things this band is not.**

It is not Task 4.3's `drift_check`, which rolls one window out on the campaign's
*own* cells immediately before it ran. That answers "does this estimator walk on
these cells". This answers "how far does a comparison of this shape walk
anywhere". They are complements: drift is one window and specific, the band is
hundreds and general, and an estimate should clear both.

It is not a confidence interval on the effect. It is the dispersion of the
estimator under a true zero. The quantile band that ships with a lift is the
model's own predictive uncertainty and is, as Task 4.4 recorded, uninformative
on this panel — coverage 1.00 against a nominal 0.80. The placebo band is the
one that carries information about whether an estimate is distinguishable from
nothing, which is why the gate keys on this and not on that.

**The band is measured in-sample and is therefore optimistic.** The baseline
trains on every `treated == 0` row, which includes the untreated weeks of
ever-treated cells. The model has seen them. A band drawn on cells the model has never seen would be
wider, and the true dispersion of the estimator on a genuinely new campaign is
wider still. Recorded in the diagnostics rather than corrected, because
correcting it would need a held-out fit and that is a different piece of work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.baseline import (
    ESTIMATION_WINDOW,
    BaselineModel,
    add_price_history,
    rollout,
)
from promo.io import connect
from promo.lift import LiftCampaign, resolve_horizon

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POOL",
    "DEFAULT_WINDOWS",
    "MIN_WINDOWS",
    "POOL_KINDS",
    "InsufficientPlaceboError",
    "PlaceboPool",
    "band_for_campaign",
    "inside_band",
    "never_treated_cells",
    "placebo_band",
    "placebo_pool",
    "summarise_band",
    "write_band",
]

#: Which population the placebo windows are drawn from.
#:
#: - ``ever_treated`` — cells that were promoted at some point, sampled in
#:   windows where they were **not**. This is the default, and the reason is
#:   Task 3.2: a classifier separates promoted from unpromoted cells at
#:   **AUC 0.70**, so they are different populations. The estimator drifts
#:   differently on each, and a null measured on one is mis-centred as the null
#:   for the other. Measured directly: on the five densest commodities the
#:   never-treated band's median disagreed in *sign* with the campaign's own
#:   pre-period drift on **four of five**.
#: - ``never_treated`` — cells never promoted anywhere. Retained so the two can
#:   be compared and the claim above stays checkable, not because it is right.
POOL_KINDS: tuple[str, ...] = ("ever_treated", "never_treated")

DEFAULT_POOL: str = "ever_treated"

#: The plan's floor. Fewer draws than this and the tails of the band are being
#: read off a handful of observations.
MIN_WINDOWS: int = 300

DEFAULT_WINDOWS: int = 300

#: Two-sided. The band is the central 90% of the null, so an estimate inside it
#: is one this comparison cannot separate from a week when nothing happened.
DEFAULT_ALPHA: float = 0.10

_KEY = ("PRODUCT_ID", "STORE_ID", "WEEK_NO")


class InsufficientPlaceboError(Exception):
    """The panel cannot supply the placebo windows that were asked for.

    Raised rather than returning a band from whatever was available. A band
    computed on forty windows and reported as though it were three hundred is a
    narrower band than the data supports, and every estimate compared against it
    looks more significant than it is.
    """


@dataclass
class PlaceboPool:
    """The rows a placebo draws from, and when each cell is drawable.

    For a never-treated pool every cell is drawable in every week: nothing ever
    happened to it. For an ever-treated pool the cell is only drawable in
    windows that are clean, which `eligible()` decides.
    """

    kind: str
    rows: pd.DataFrame
    pairs: pd.DataFrame
    #: `(n_pairs, n_weeks)` bool, True where that cell was treated. Empty for a
    #: never-treated pool.
    treated_by_week: np.ndarray
    week_min: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def eligible(self, start: int, span: int, guard: int) -> np.ndarray:
        """Cells whose window at `start` carries no promotion of their own.

        Two exclusions, and the second is the one that is easy to forget.

        The window itself must be untreated, or the "placebo" contains a real
        effect. And the `guard` weeks *before* it must be untreated too: a cell
        promoted the week before is paying that promotion back inside the
        window, so its truth is not zero either. `guard` is the horizon, the
        same number of weeks the campaign's own tail is measured over.

        History before the guard may contain promotions, and is left alone on
        purpose — a real campaign's history does too, so excluding it would
        make the placebo cleaner than the thing it is a null for.
        """
        if self.kind == "never_treated":
            return np.ones(len(self.pairs), dtype=bool)
        lo = max(0, start - guard - self.week_min)
        hi = min(self.treated_by_week.shape[1], start + span - self.week_min)
        if hi <= lo:
            return np.ones(len(self.pairs), dtype=bool)
        return ~self.treated_by_week[:, lo:hi].any(axis=1)


def placebo_pool(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    kind: str = DEFAULT_POOL,
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
) -> PlaceboPool:
    """Build the population the placebo windows are drawn from.

    See `POOL_KINDS` for why `ever_treated` is the default. In short: the
    campaign's cells are ever-treated cells, so the null has to be measured on
    ever-treated cells. A never-treated pool is a different population and its
    band is centred somewhere the campaign's own null is not.

    Raises:
        ValueError: `kind` is not one of `POOL_KINDS`.
    """
    if kind not in POOL_KINDS:
        raise ValueError(f"kind must be one of {list(POOL_KINDS)}, got {kind!r}")

    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(panel, pd.DataFrame):
            con.register("_placebo_panel", panel)
            source = "_placebo_panel"
        else:
            source = f"read_parquet('{Path(panel).as_posix()}')"

        window = (
            f"WHERE p.WEEK_NO BETWEEN {week_range[0]} AND {week_range[1]}"
            if week_range is not None
            else ""
        )
        wanted = 1 if kind == "ever_treated" else 0
        rows = con.execute(
            f"""
            WITH cell AS (
                SELECT PRODUCT_ID, STORE_ID, MAX(treated::INT) AS ever_treated
                FROM {source} GROUP BY 1, 2
            )
            SELECT p.* FROM {source} p
            JOIN cell c USING (PRODUCT_ID, STORE_ID)
            {window}
            {"AND" if window else "WHERE"} c.ever_treated = {wanted}
            ORDER BY p.PRODUCT_ID, p.STORE_ID, p.WEEK_NO
            """
        ).df()
        totals = con.execute(
            f"""
            WITH cell AS (
                SELECT PRODUCT_ID, STORE_ID, MAX(treated::INT) AS ever_treated
                FROM {source} GROUP BY 1, 2
            )
            SELECT ever_treated, COUNT(*) AS pairs FROM cell GROUP BY 1
            """
        ).df()
        rows = add_price_history(rows, con=con)
    finally:
        if isinstance(panel, pd.DataFrame):
            con.unregister("_placebo_panel")
        if own:
            con.close()

    pairs = (
        rows[["PRODUCT_ID", "STORE_ID"]].drop_duplicates().reset_index(drop=True)
    )
    week_min = int(rows["WEEK_NO"].min())
    week_max = int(rows["WEEK_NO"].max())

    if kind == "ever_treated":
        position = {
            (p, s): i
            for i, (p, s) in enumerate(
                zip(pairs["PRODUCT_ID"], pairs["STORE_ID"], strict=True)
            )
        }
        matrix = np.zeros((len(pairs), week_max - week_min + 1), dtype=bool)
        treated = rows.loc[rows["treated"], ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]]
        matrix[
            [position[(p, s)] for p, s in
             zip(treated["PRODUCT_ID"], treated["STORE_ID"], strict=True)],
            treated["WEEK_NO"].to_numpy() - week_min,
        ] = True
    else:
        matrix = np.zeros((len(pairs), 0), dtype=bool)

    by_flag = dict(zip(totals["ever_treated"], totals["pairs"], strict=True))
    diagnostics = {
        "kind": kind,
        "rows": len(rows),
        "pairs": len(pairs),
        "products": int(rows["PRODUCT_ID"].nunique()),
        "stores": int(rows["STORE_ID"].nunique()),
        "weeks": [week_min, week_max],
        "panel_pairs_never_treated": int(by_flag.get(0, 0)),
        "panel_pairs_ever_treated": int(by_flag.get(1, 0)),
        "mean_units": round(float(rows["units"].mean()), 6),
        "treated_cell_weeks": int(matrix.sum()) if matrix.size else 0,
        "why": (
            "The campaign's cells are ever-treated cells. Task 3.2 separates "
            "promoted from unpromoted cells at AUC 0.70, so a null measured on "
            "never-treated cells is measured on a different population — and "
            "on the five densest commodities its median disagreed in sign with "
            "the campaign's own pre-period drift on four of five."
            if kind == "ever_treated"
            else "Never treated anywhere in the panel. Retained for comparison "
            "against the ever-treated pool, not because it is the right null: "
            "see POOL_KINDS."
        ),
    }
    return PlaceboPool(
        kind=kind,
        rows=rows,
        pairs=pairs,
        treated_by_week=matrix,
        week_min=week_min,
        diagnostics=diagnostics,
    )


def never_treated_cells(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every panel row for product-stores that were never promoted, anywhere.

    **Never treated in the whole panel**, not merely untreated in the window
    being drawn. A cell promoted in week 40 is still paying that promotion back
    in week 45, so a "placebo" window sitting on top of someone else's payback
    period measures a real effect and calls it noise — which would widen the
    band and hide genuine findings behind it.

    Returns the rows with `price_rel_category_lag` derived, since the baseline
    needs it and deriving it per draw would recompute the same window hundreds
    of times.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(panel, pd.DataFrame):
            con.register("_placebo_panel", panel)
            source = "_placebo_panel"
        else:
            source = f"read_parquet('{Path(panel).as_posix()}')"

        where = ""
        if week_range is not None:
            where = f"WHERE p.WEEK_NO BETWEEN {week_range[0]} AND {week_range[1]}"
        cells = con.execute(
            f"""
            WITH cell AS (
                SELECT PRODUCT_ID, STORE_ID, MAX(treated::INT) AS ever_treated
                FROM {source} GROUP BY 1, 2
            )
            SELECT p.* FROM {source} p
            JOIN cell c USING (PRODUCT_ID, STORE_ID)
            {where}
            {"AND" if where else "WHERE"} c.ever_treated = 0
            ORDER BY p.PRODUCT_ID, p.STORE_ID, p.WEEK_NO
            """
        ).df()
        totals = con.execute(
            f"""
            WITH cell AS (
                SELECT PRODUCT_ID, STORE_ID, MAX(treated::INT) AS ever_treated
                FROM {source} GROUP BY 1, 2
            )
            SELECT ever_treated, COUNT(*) AS pairs FROM cell GROUP BY 1
            """
        ).df()
        cells = add_price_history(cells, con=con)
    finally:
        if isinstance(panel, pd.DataFrame):
            con.unregister("_placebo_panel")
        if own:
            con.close()

    pairs = cells[["PRODUCT_ID", "STORE_ID"]].drop_duplicates()
    by_flag = dict(zip(totals["ever_treated"], totals["pairs"], strict=True))
    diagnostics = {
        "rows": len(cells),
        "pairs": len(pairs),
        "products": int(cells["PRODUCT_ID"].nunique()),
        "stores": int(cells["STORE_ID"].nunique()),
        "weeks": [int(cells["WEEK_NO"].min()), int(cells["WEEK_NO"].max())],
        "panel_pairs_never_treated": int(by_flag.get(0, 0)),
        "panel_pairs_ever_treated": int(by_flag.get(1, 0)),
        "mean_units": round(float(cells["units"].mean()), 6),
        "rule": (
            "Never treated anywhere in the panel, not merely untreated in the "
            "drawn window. A cell promoted earlier is still paying it back "
            "later, and a placebo window sitting on that payback would measure "
            "a real effect and call it noise."
        ),
    }
    return cells, diagnostics


def placebo_band(
    model: BaselineModel | str | Path,
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    n_cells: int,
    campaign_length: int,
    horizon_weeks: int,
    n_windows: int = DEFAULT_WINDOWS,
    con: duckdb.DuckDBPyConnection | None = None,
    pool: PlaceboPool | str = DEFAULT_POOL,
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
    start_range: tuple[int, int] | None = None,
    min_history_weeks: int = 13,
    alpha: float = DEFAULT_ALPHA,
    outcome: str = "units",
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the estimator over `n_windows` windows where nothing happened.

    Each draw samples `n_cells` never-treated product-stores and a window start,
    then runs exactly the machinery a real campaign runs: recursive rollout from
    the pre-window history, residuals over the campaign weeks, residuals over
    the horizon-length tail. The truth is zero in every draw.

    Args:
        model: the fitted baseline, or a directory to load one from.
        panel: the modelling panel. Ignored when `cells` is supplied.
        n_cells: product-stores per draw. **Match the campaign**, or the band
            describes a different statistic — see the module docstring.
        campaign_length: promoted weeks per draw. Match the campaign.
        horizon_weeks: post-window weeks per draw. Match the campaign.
        n_windows: how many draws. Below `MIN_WINDOWS` this raises.
        cells: a pre-built never-treated frame, to avoid rebuilding it per
            campaign. `never_treated_cells()` produces it.
        pool: a built `PlaceboPool`, or a key of `POOL_KINDS` to build one.
            Defaults to `ever_treated` — the campaign's cells are ever-treated
            cells, so the null must be measured on ever-treated cells.
        week_range: restrict window starts to the estimation window.
        start_range: inclusive `(first, last)` week the window may start in.
            Defaults to every week the history allows. Narrowing it to the
            campaign's own pre-period makes the null *period-matched*: the
            estimator's drift can be a property of the weeks rather than of the
            cells, and a band averaged over the panel would not see that.
        min_history_weeks: history a draw needs before its window, to seed the
            lags. Draws starting earlier than this are not offered.
        alpha: two-sided. The band spans `alpha/2` to `1 - alpha/2`.
        seed: explicit, per the project's randomness rule.

    Returns:
        `(draws, diagnostics)`. One row per draw with `gross`, `post`, `net`,
        the counterfactual it was measured against, and the normalised
        `gross_share`. The band itself is in `diagnostics["band"]`.

    Raises:
        InsufficientPlaceboError: too few never-treated cells, too few weeks to
            place a window, or `n_windows` below `MIN_WINDOWS`.
    """
    if n_windows < MIN_WINDOWS:
        raise InsufficientPlaceboError(
            f"n_windows={n_windows} is below the {MIN_WINDOWS}-window floor. "
            f"A band read off fewer draws has tails estimated from a handful of "
            f"observations, and every estimate compared against it looks more "
            f"significant than it is."
        )
    if not isinstance(model, BaselineModel):
        model = BaselineModel.load(model)

    if isinstance(pool, str):
        pool = placebo_pool(panel, pool, con=con, week_range=week_range)
    cells, cells_diag = pool.rows, dict(pool.diagnostics)
    pairs = pool.pairs
    if len(pairs) < n_cells:
        raise InsufficientPlaceboError(
            f"the campaign spans {n_cells} product-stores but the "
            f"{pool.kind} pool holds only {len(pairs)}, so a size-matched draw "
            f"is impossible. A band drawn on fewer cells is narrower than the "
            f"statistic it would be compared against."
        )

    first_week = int(cells["WEEK_NO"].min())
    last_week = int(cells["WEEK_NO"].max())
    span = campaign_length + horizon_weeks
    earliest = first_week + min_history_weeks
    latest = last_week - span + 1
    if latest < earliest:
        raise InsufficientPlaceboError(
            f"a {span}-week window needs {min_history_weeks} weeks of history "
            f"before it, and the never-treated rows span weeks {first_week}-"
            f"{last_week}. There is no valid start."
        )

    # Which starts can seat a full draw, worked out once. Eligibility depends
    # on the window, so this cannot be folded into the cell sample — and
    # skipping starved starts inside the loop would quietly shrink the band's
    # sample size *and* bias the start distribution without saying so.
    if start_range is not None:
        earliest = max(earliest, start_range[0])
        latest = min(latest, start_range[1])
        if latest < earliest:
            raise InsufficientPlaceboError(
                f"start_range {start_range} leaves no week that also has "
                f"{min_history_weeks} weeks of history and room for a "
                f"{span}-week window inside weeks {first_week}-{last_week}."
            )
    candidates = range(earliest, latest + 1)
    eligible_at = {
        start: np.flatnonzero(pool.eligible(start, span, horizon_weeks))
        for start in candidates
    }
    usable = [s for s, cells_at in eligible_at.items() if len(cells_at) >= n_cells]
    if not usable:
        best = max((len(c) for c in eligible_at.values()), default=0)
        raise InsufficientPlaceboError(
            f"no week can seat a {n_cells}-cell draw: the best start offers "
            f"{best} cells whose {span}-week window is clean and clear of an "
            f"earlier promotion's payback. The {pool.kind} pool is too "
            f"saturated for a campaign this wide."
        )

    rng = np.random.default_rng(seed)
    indexed = cells.set_index(["PRODUCT_ID", "STORE_ID"]).sort_index()

    rows: list[dict[str, Any]] = []
    eligible_counts = [len(eligible_at[s]) for s in usable]
    for draw in range(n_windows):
        start = int(usable[rng.integers(len(usable))])
        available = eligible_at[start]
        take = pairs.iloc[rng.choice(available, size=n_cells, replace=False)]
        block = indexed.loc[
            list(zip(take["PRODUCT_ID"], take["STORE_ID"], strict=True))
        ].reset_index()

        history = block[block["WEEK_NO"] < start]
        window = block[block["WEEK_NO"].between(start, start + span - 1)]
        if history.empty or window.empty:
            continue
        path, _ = rollout(model, history, window, outcome=outcome)
        joined = window[[*_KEY, outcome]].merge(
            path[[*_KEY, "counterfactual_units"]], on=list(_KEY), validate="one_to_one"
        )
        in_campaign = joined["WEEK_NO"] < start + campaign_length
        gross = float(
            joined.loc[in_campaign, outcome].sum()
            - joined.loc[in_campaign, "counterfactual_units"].sum()
        )
        post = float(
            joined.loc[~in_campaign, outcome].sum()
            - joined.loc[~in_campaign, "counterfactual_units"].sum()
        )
        counterfactual = float(
            joined.loc[in_campaign, "counterfactual_units"].sum()
        )
        rows.append(
            {
                "draw": draw,
                "week_start": start,
                "cells": n_cells,
                "gross": gross,
                "post": post,
                # Added, never netted — the same rule as the lift itself.
                "net": gross + post,
                "counterfactual_units": counterfactual,
                "observed_units": float(joined.loc[in_campaign, outcome].sum()),
                "gross_share": gross / counterfactual if counterfactual else np.nan,
            }
        )

    draws = pd.DataFrame(rows)
    if len(draws) < MIN_WINDOWS:
        raise InsufficientPlaceboError(
            f"only {len(draws)} of {n_windows} draws produced a window, which "
            f"is below the {MIN_WINDOWS} floor."
        )

    diagnostics = {
        "stage": "placebo_band",
        "windows": len(draws),
        "pool_kind": pool.kind,
        "eligibility": {
            "why": (
                "An ever-treated cell is drawable only where its own window is "
                "clean: no promotion inside it, and none in the horizon-length "
                "run-up either, since that promotion's payback would land "
                "inside. History before the run-up may contain promotions and "
                "is left alone — a real campaign's history does too."
            ),
            "guard_weeks": horizon_weeks,
            "eligible_cells_per_usable_start": {
                "min": int(min(eligible_counts)),
                "median": float(np.median(eligible_counts)),
                "max": int(max(eligible_counts)),
            },
            "usable_starts": len(usable),
            "candidate_starts": len(eligible_at),
            "start_range": list(start_range) if start_range else None,
            "period_matched": start_range is not None,
            "starts_note": (
                "Starts that cannot seat a full draw are excluded up front, so "
                "every draw is a real one and the window floor means what it "
                "says. The cost is that starts are uniform over the *usable* "
                "weeks, not over all weeks — where promotions cluster in time, "
                "the placebo cannot sample there."
            ),
        },
        "shape": {
            "cells_per_draw": n_cells,
            "campaign_length_weeks": campaign_length,
            "horizon_weeks": horizon_weeks,
            "window_length_weeks": span,
            "week_starts": [int(draws["week_start"].min()), int(draws["week_start"].max())],
            "why_matched": (
                "Every draw has the campaign's shape. A placebo over a "
                "different number of cells or a different window length is a "
                "different statistic, and a band built from it would be "
                "narrower than the truth — which turns noise into findings."
            ),
        },
        "pool": cells_diag,
        "band": summarise_band(draws, alpha=alpha),
        "model": {
            "target": model.target,
            "features": len(model.features),
            "train_window": list(model.train_window),
        },
        "in_sample_caveat": (
            "The baseline trains on every treated == 0 row, which includes "
            "these never-treated cells, so the model has seen them. This band "
            "is therefore optimistic: one drawn on unseen cells would be wider, "
            "and the dispersion on a genuinely new campaign wider still. "
            "Recorded rather than corrected — correcting it needs a held-out "
            "fit."
        ),
        "not_a_confidence_interval": (
            "This is the dispersion of the estimator under a true zero, not an "
            "interval on the effect. The quantile band that ships with a lift "
            "is the model's own predictive uncertainty and Task 4.4 recorded it "
            "as uninformative on this panel (coverage 1.00 against a nominal "
            "0.80). The placebo band is the one that says whether an estimate "
            "is distinguishable from nothing happening."
        ),
        "seed": seed,
    }
    return draws, diagnostics


def summarise_band(
    draws: pd.DataFrame, *, alpha: float = DEFAULT_ALPHA, column: str = "gross"
) -> dict[str, Any]:
    """The band itself: its edges, its centre, and where zero sits in it."""
    values = draws[column].to_numpy(dtype="float64")
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    median = float(np.median(values))
    return {
        "column": column,
        "alpha": alpha,
        "coverage": round(1 - alpha, 4),
        "low": round(float(low), 6),
        "high": round(float(high), 6),
        "median": round(median, 6),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)), 6),
        "width": round(float(high - low), 6),
        "n": len(values),
        # The runbook's line, measured rather than asserted: the centre of the
        # null is almost never at zero, and an estimator whose placebo median
        # sits away from zero has a bias, not just a spread.
        "zero_inside": bool(low <= 0.0 <= high),
        "median_is_zero": bool(abs(median) < 1e-9),
        "why_the_median_matters": (
            "The centre of this band is where the estimator puts zero. It is "
            "almost never at zero itself; the distance is the estimator's bias "
            "on windows where nothing happened, and it is the same quantity "
            "Task 4.3's drift check reports for one window."
        ),
    }


def inside_band(
    estimate: float,
    draws: pd.DataFrame | dict[str, Any],
    *,
    alpha: float = DEFAULT_ALPHA,
    column: str = "gross",
) -> tuple[bool, dict[str, Any]]:
    """Does this estimate fall inside the band?

    The detection behind `PLACEBO_OVERLAP`. Returns the verdict and the evidence
    a sceptic would ask for — where the band sits, where the estimate sits in
    it, and how many placebo windows were at least as extreme.

    Args:
        estimate: the measured quantity, on the same scale the band was built
            on. Passing a gross against a band of nets, or a per-cell figure
            against a size-matched band, compares two different things.
        draws: the frame from `placebo_band`, or an already-summarised band.
        alpha: two-sided, as in `placebo_band`.

    Returns:
        `(inside, evidence)`. `inside` True means **this comparison cannot see
        the effect** — never that the promotion had none.
    """
    if isinstance(draws, pd.DataFrame):
        band = summarise_band(draws, alpha=alpha, column=column)
        values = draws[column].to_numpy(dtype="float64")
    else:
        band = draws
        values = None

    inside = bool(band["low"] <= estimate <= band["high"])
    evidence: dict[str, Any] = {
        "estimate": float(estimate),
        "band_low": band["low"],
        "band_high": band["high"],
        "band_median": band["median"],
        "band_width": band["width"],
        "windows": band["n"],
        "coverage": band["coverage"],
        "inside": inside,
        "meaning": (
            "Inside the band means this comparison cannot separate the "
            "promotion from ordinary week-to-week movement. It is a statement "
            "about what the comparison can see, never evidence that the "
            "promotion did nothing."
            if inside
            else "Outside the band means the estimate is larger than this "
            "comparison produces on weeks when nothing happened. That is "
            "necessary for the estimate to mean anything, and not sufficient: "
            "the band is in-sample and therefore optimistic."
        ),
    }
    if values is not None:
        # Two-sided empirical p, from the placebo distribution rather than from
        # an assumed centre: the null here is not centred at zero.
        at_or_below = float(np.mean(values <= estimate))
        at_or_above = float(np.mean(values >= estimate))
        evidence["p_value"] = round(min(1.0, 2 * min(at_or_below, at_or_above)), 6)
        evidence["percentile"] = round(
            float(np.mean(values < estimate)) * 100.0, 4
        )
        evidence["more_extreme_windows"] = int(
            np.sum(np.abs(values - band["median"]) >= abs(estimate - band["median"]))
        )
    return inside, evidence


def band_for_campaign(
    campaign: LiftCampaign,
    model: BaselineModel | str | Path = "data/interim/baseline",
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    cycles: str | Path | pd.DataFrame = "data/interim/repurchase_cycles.parquet",
    *,
    n_cells: int,
    con: duckdb.DuckDBPyConnection | None = None,
    n_windows: int = DEFAULT_WINDOWS,
    pool: PlaceboPool | str = DEFAULT_POOL,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A band shaped like one campaign: its cells, its length, its horizon.

    `n_cells` is required rather than derived, because deriving it would mean
    re-running `campaign_cells` here and the caller has almost always just done
    that — `estimate_lift`'s diagnostics carry it as `cells.pairs`.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        horizon_weeks, horizon_diag = resolve_horizon(campaign, cycles, con=con)
        if horizon_weeks is None:
            raise InsufficientPlaceboError(
                f"campaign {campaign.name!r} has no horizon, so a matched "
                f"window has no length. Resolve the horizon first."
            )
        first, last = campaign.weeks
        draws, diagnostics = placebo_band(
            model,
            panel,
            n_cells=n_cells,
            campaign_length=last - first + 1,
            horizon_weeks=horizon_weeks,
            n_windows=n_windows,
            con=con,
            pool=pool,
            alpha=alpha,
            seed=seed,
        )
    finally:
        if own:
            con.close()
    diagnostics["campaign"] = campaign.model_dump()
    diagnostics["horizon"] = horizon_diag
    return draws, diagnostics


def write_band(
    draws: pd.DataFrame,
    diagnostics: dict[str, Any],
    directory: str | Path = "data/interim",
    stem: str = "placebo_band",
) -> dict[str, Path]:
    """Write the draws as parquet and the band as JSON. Returns both paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / f"{stem}.parquet"
    js = out / f"{stem}.json"
    draws.to_parquet(parquet, index=False)
    js.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return {"draws": parquet, "band": js}
