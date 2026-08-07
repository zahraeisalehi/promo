"""Incremental units for a campaign, over a window that outlasts the cycle.

Task 4.3. The counterfactual comes from Task 4.1's model and Task 4.2's
recursive rollout; this module turns the residuals into the four numbers a
campaign report states, and it fixes where the window closes.

**The window is the argument.** Measure a promotion over its promoted weeks
alone and you bank the peak: the household that bought two weeks' worth on deal
does not come back next week, and that trough lands outside the window and never
gets counted against the campaign. Task 1.5 measured the cycle per commodity for
exactly this reason, and the rule it fixed is that the measurement window
extends past campaign end by at least that cycle. So:

- **gross incremental** — residuals summed over the promoted weeks. The peak.
- **post-window residual** — residuals summed over the cycle-length tail. The
  trough, and usually negative.
- **net incremental** — `gross + post`. **Added, never netted**: the tail is
  already signed, and subtracting a negative would report the trough as a
  second helping of lift.
- **retention ratio** — `net / gross`. What survived the payback period, as a
  share of what the promoted weeks appeared to deliver.

Three things this module refuses to do quietly.

**It will not hide a short window.** A caller may pass `horizon_weeks`; if it is
shorter than the commodity's cycle the numbers are still computed, and the
diagnostics carry `HORIZON_TOO_SHORT` with the shortfall. The Phase 3 gate is
what refuses — this is the same condition, named identically, so an estimate and
its verdict cannot disagree.

**It will not pretend a truncated tail is a tail.** This panel ends at week 101.
A campaign ending at week 99 has two weeks of post-window where its commodity
needs six, and a campaign ending at 101 has none at all. `net` is then not net,
it is gross wearing net's name — so the truncation is recorded, and when no post
week exists at all the retention ratio comes back `None` rather than the 1.0 the
arithmetic would produce.

**It will not report a ratio of two estimates as a point.** The gross and net
figures carry an interval from the baseline's quantile paths, and the retention
ratio is computed once per path rather than averaged across them. That interval
is the *model's own* uncertainty about the counterfactual. It is **not** the
placebo band — Task 4.5 owns that, and an estimate is only defensible against
both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.baseline import BaselineModel, add_price_history, rollout
from promo.gates import CampaignSpec
from promo.io import connect

__all__ = [
    "AGGREGATION_LEVELS",
    "DEFAULT_QUANTILES",
    "LiftCampaign",
    "NoCellsError",
    "UnknownLevelError",
    "aggregate_residuals",
    "campaign_cells",
    "estimate_lift",
    "resolve_horizon",
    "write_diagnostics",
]

#: How per-cell-week residuals are grouped into reported estimates.
#:
#: The residuals themselves are always at product-store-week — that is the only
#: grain the baseline predicts at, and aggregating the *outcome* before
#: estimating would need a model fitted on the aggregate series. What the level
#: changes is how many cells are pooled into one number, and therefore what
#: placebo band that number has to clear.
#:
#: **The campaign total is identical at every level.** Grouping partitions the
#: same residuals; it does not change their sum. What changes is the
#: signal-to-noise of each reported unit, because a placebo band is matched to
#: the cell count of the unit it is compared against.
AGGREGATION_LEVELS: dict[str, tuple[str, ...]] = {
    "campaign": (),
    "commodity": ("COMMODITY_DESC",),
    "commodity_store": ("COMMODITY_DESC", "STORE_ID"),
    "cell": ("PRODUCT_ID", "STORE_ID"),
}

#: The counterfactual band carried through to the lift. The baseline fits these
#: as independent quantile models, so this is its own predictive uncertainty and
#: not a placebo comparison.
DEFAULT_QUANTILES: tuple[float, float] = (0.1, 0.9)

_KEY = ("PRODUCT_ID", "STORE_ID", "WEEK_NO")


class UnknownLevelError(Exception):
    """The requested aggregation level is not one this module defines."""


class NoCellsError(Exception):
    """The campaign matched no treated product-store in its weeks.

    Raised rather than returning zero lift. Zero incremental units is a finding;
    "this campaign does not appear in the panel" is a different statement, and
    reporting the second as the first is how a scoping mistake becomes a result.
    """


class LiftCampaign(CampaignSpec):
    """A campaign, plus the cells and weeks it actually ran in.

    `CampaignSpec` is deliberately minimal — it carries what the Phase 3 gates
    need to render a verdict. Measuring a lift needs more: which weeks were
    promoted, and optionally which products and stores to restrict to. This
    extends rather than replaces it so a campaign has one vocabulary across the
    audit and the estimate.

    Attributes:
        weeks: inclusive `(first, last)` promoted week.
        products: restrict to these products. Defaults to `product` if set,
            else every product of `commodity`.
        stores: restrict to these stores. Defaults to every store the campaign
            ran in.
    """

    weeks: tuple[int, int]
    products: tuple[int, ...] | None = None
    stores: tuple[int, ...] | None = None

    @property
    def product_ids(self) -> tuple[int, ...] | None:
        if self.products:
            return tuple(self.products)
        return (self.product,) if self.product is not None else None


def resolve_horizon(
    campaign: LiftCampaign,
    cycles: str | Path | pd.DataFrame = "data/interim/repurchase_cycles.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    required_column: str = "horizon_weeks",
) -> tuple[int | None, dict[str, Any]]:
    """How many weeks the window stays open after the campaign ends.

    Defaults to the commodity's repurchase cycle, which is the rule Task 1.5
    fixed. A caller-supplied `horizon_weeks` is honoured and checked against it,
    using the same column and the same code name as `promo.audit.horizon_check`
    so the estimate and the gate cannot disagree about what "too short" means.

    Returns:
        `(weeks, diagnostics)`. `weeks` is None only when neither the campaign
        nor the cycles table supplies one.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        table = (
            cycles
            if isinstance(cycles, pd.DataFrame)
            else con.execute(
                f"SELECT * FROM read_parquet('{Path(cycles).as_posix()}')"
            ).df()
        )
    finally:
        if own:
            con.close()

    required: int | None = None
    low_support = False
    if campaign.commodity is not None and "COMMODITY_DESC" in table.columns:
        match = table.loc[table["COMMODITY_DESC"] == campaign.commodity]
        if len(match):
            value = match.iloc[0][required_column]
            required = None if pd.isna(value) else int(value)
            low_support = bool(match.iloc[0].get("low_support", False))

    requested = campaign.horizon_weeks
    weeks = requested if requested is not None else required

    if required is None:
        status = "UNKNOWN_CYCLE"
    elif requested is not None and requested < required:
        status = "HORIZON_TOO_SHORT"
    else:
        status = "OK"

    diagnostics = {
        "commodity": campaign.commodity,
        "requested_weeks": requested,
        "required_weeks": required,
        "horizon_weeks": weeks,
        "source": "campaign" if requested is not None else "repurchase cycle",
        "required_column": required_column,
        "low_support": low_support,
        "status": status,
        "rule": (
            "The window extends past campaign end by at least the commodity's "
            "repurchase cycle. A shorter one banks the promotional peak and "
            "closes before the trough that follows it."
        ),
    }
    if status == "HORIZON_TOO_SHORT":
        diagnostics["shortfall_weeks"] = int(required - requested)  # type: ignore[operator]
        diagnostics["consequence"] = (
            "The net figure below is biased upward: part of the payback period "
            "falls outside the window. This is the Phase 3 refusal "
            "HORIZON_TOO_SHORT, reported here rather than raised — the gate is "
            "what stops the pipeline."
        )
    if status == "UNKNOWN_CYCLE":
        diagnostics["consequence"] = (
            "No cycle is recorded for this commodity, so the window has not "
            "been shown to be long enough. Absence of a check is not a pass."
        )
    return weeks, diagnostics


def campaign_cells(
    campaign: LiftCampaign,
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    horizon_weeks: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every panel row for the product-stores this campaign ran in.

    A product-store is in the campaign if it was **treated at least once** in
    the promoted weeks. Once it is in, every week of the measurement window
    counts for it, including promoted weeks in which that particular store did
    not run the display: the campaign's effect on a store is not confined to the
    weeks its own display was up, and the payback period certainly is not.

    Returns rows from the start of the panel to the end of the measurement
    window — the earlier weeks are the history the rollout is seeded from.
    """
    first, last = campaign.weeks
    window_end = last + horizon_weeks

    filters = ["treated", f"WEEK_NO BETWEEN {int(first)} AND {int(last)}"]
    products = campaign.product_ids
    if products:
        filters.append(f"PRODUCT_ID IN ({','.join(str(int(p)) for p in products)})")
    if campaign.stores:
        filters.append(f"STORE_ID IN ({','.join(str(int(s)) for s in campaign.stores)})")
    if not products and campaign.commodity is not None:
        filters.append(f"COMMODITY_DESC = '{campaign.commodity.replace(chr(39), chr(39) * 2)}'")

    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(panel, pd.DataFrame):
            con.register("_lift_panel", panel)
            source = "_lift_panel"
        else:
            source = f"read_parquet('{Path(panel).as_posix()}')"

        ran = con.execute(
            f"SELECT DISTINCT PRODUCT_ID, STORE_ID FROM {source} "
            f"WHERE {' AND '.join(filters)}"
        ).df()
        if ran.empty:
            raise NoCellsError(
                f"campaign {campaign.name!r} matched no treated product-store in "
                f"weeks {first}-{last}. That is a scoping result, not a lift of "
                f"zero, so it is raised rather than returned."
            )
        con.register("_lift_ran", ran)
        cells = con.execute(
            f"SELECT p.* FROM {source} p SEMI JOIN _lift_ran r "
            f"  ON p.PRODUCT_ID = r.PRODUCT_ID AND p.STORE_ID = r.STORE_ID "
            f"WHERE p.WEEK_NO <= {int(window_end)} "
            f"ORDER BY p.PRODUCT_ID, p.STORE_ID, p.WEEK_NO"
        ).df()
        cells = add_price_history(cells, con=con)
        panel_last_week = int(
            con.execute(f"SELECT MAX(WEEK_NO) FROM {source}").fetchone()[0]
        )
    finally:
        con.unregister("_lift_ran")
        if isinstance(panel, pd.DataFrame):
            con.unregister("_lift_panel")
        if own:
            con.close()

    promoted = cells["WEEK_NO"].between(first, last)
    available_post = int(min(panel_last_week, window_end) - last)
    diagnostics = {
        "pairs": len(ran),
        "rows": len(cells),
        "weeks": {
            "campaign": [int(first), int(last)],
            "post_window": [int(last + 1), int(window_end)],
            "measurement_window": [int(first), int(window_end)],
        },
        "history_weeks": int(first - int(cells["WEEK_NO"].min())),
        "membership_rule": (
            "A product-store is in the campaign if it was treated at least once "
            "in the promoted weeks. Every week of the measurement window then "
            "counts for it, including promoted weeks its own display was down — "
            "carryover and payback are not confined to the weeks the display "
            "was up."
        ),
        "treated_pair_weeks": int(cells.loc[promoted, "treated"].sum()),
        "promoted_pair_weeks": int(promoted.sum()),
        "treated_share_of_campaign_weeks": (
            round(float(cells.loc[promoted, "treated"].mean()), 6)
            if promoted.any()
            else None
        ),
        "post_window": {
            "required_weeks": int(horizon_weeks),
            "available_weeks": max(0, available_post),
            "truncated": available_post < horizon_weeks,
            "panel_last_week": panel_last_week,
        },
    }
    if diagnostics["post_window"]["truncated"]:
        diagnostics["post_window"]["consequence"] = (
            f"The panel ends at week {panel_last_week}, so only "
            f"{max(0, available_post)} of the {int(horizon_weeks)} post-window "
            f"weeks exist. The net figure is missing part of the payback period "
            f"and is biased upward by however much of the trough falls outside "
            f"the data."
        )
    return cells, diagnostics


def drift_check(
    model: BaselineModel,
    cells: pd.DataFrame,
    campaign_start: int,
    window_weeks: int,
    campaign_length: int,
    *,
    outcome: str = "units",
    min_history_weeks: int = 4,
) -> dict[str, Any]:
    """Roll the same window out over the weeks *before* the campaign.

    Nothing happened there, so the residual should be zero. Whatever it is
    instead is this estimator's drift on this campaign, at this horizon, over
    these cells — and it is the number the gross and net figures have to be read
    against. A recursive rollout compounds: each step's features are built from
    earlier predictions, so an error that is invisible at one week is not at
    seven.

    This is **not** the placebo band. It is one window, before one campaign, and
    it says how far this estimator walks — not whether the estimate is
    distinguishable from noise. Task 4.5 owns that, over at least 300 windows,
    and a lift is defensible only against both.
    """
    first = campaign_start - window_weeks
    if first - int(cells["WEEK_NO"].min()) < min_history_weeks:
        return {
            "ran": False,
            "why": (
                f"a {window_weeks}-week drift window before week "
                f"{campaign_start} leaves under {min_history_weeks} weeks of "
                f"history to seed the rollout"
            ),
        }

    history = cells[cells["WEEK_NO"] < first]
    window = cells[cells["WEEK_NO"].between(first, campaign_start - 1)]
    path, _ = rollout(model, history, window, outcome=outcome)

    joined = window[[*_KEY, outcome, "treated"]].merge(
        path[[*_KEY, "counterfactual_units"]], on=list(_KEY), validate="one_to_one"
    )
    observed = float(joined[outcome].sum())
    counterfactual = float(joined["counterfactual_units"].sum())
    residual = observed - counterfactual
    treated_weeks = int(joined["treated"].sum())

    # Split at the same step the campaign window splits at, so gross — which
    # spans the promoted weeks only — is compared against drift accumulated
    # over the same number of recursion steps. Drift grows with depth, so
    # comparing a four-week gross to a ten-week drift figure would overstate it.
    head = joined[joined["WEEK_NO"] < first + campaign_length]
    head_residual = float(
        head[outcome].sum() - head["counterfactual_units"].sum()
    )

    return {
        "ran": True,
        "weeks": [int(first), int(campaign_start - 1)],
        "observed_units": round(observed, 6),
        "counterfactual_units": round(counterfactual, 6),
        "residual_units": round(residual, 6),
        "residual_share": (
            round(residual / counterfactual, 6) if counterfactual else None
        ),
        "campaign_length_weeks": int(campaign_length),
        "residual_units_first_weeks": round(head_residual, 6),
        "treated_pair_weeks": treated_weeks,
        "clean": treated_weeks == 0,
        "why": (
            "The same rollout, the same length, over the weeks before the "
            "campaign. Nothing happened there, so this residual is the "
            "estimator's own drift and the campaign's figures should be read "
            "against it."
        ),
        "not_the_placebo_band": (
            "One window before one campaign. It measures how far this "
            "estimator walks, not whether the estimate is distinguishable from "
            "nothing happening — Task 4.5 owns that, over at least 300 windows."
        ),
        **(
            {}
            if treated_weeks == 0
            else {
                "contaminated": (
                    f"{treated_weeks} of the {len(joined)} pair-weeks in this "
                    f"window were themselves treated, so the residual mixes "
                    f"drift with an earlier promotion's effect. Read it as an "
                    f"upper bound on drift, not as drift."
                )
            }
        ),
    }


def estimate_lift(
    campaign: LiftCampaign,
    model: BaselineModel | str | Path = "data/interim/baseline",
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    cycles: str | Path | pd.DataFrame = "data/interim/repurchase_cycles.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    quantiles: tuple[float, float] | None = DEFAULT_QUANTILES,
    check_drift: bool = True,
    level: str = "campaign",
    outcome: str = "units",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Incremental units for one campaign, over a window that outlasts the cycle.

    Args:
        campaign: what ran, where, and when.
        model: a fitted `BaselineModel`, or a directory to load one from.
        panel: the Task 2.6 modelling panel.
        cycles: the Task 1.5 repurchase-cycle table, for the horizon.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        quantiles: `(low, high)` counterfactual paths for the interval, or None
            for a point estimate only. Each is its own recursion.
        level: how to group the residuals into reported units — a key of
            `AGGREGATION_LEVELS`. The campaign total is the same at every
            level; what changes is how many cells back each reported number.
            The breakdown lands in `diagnostics["by_level"]`.
        outcome: the observed units column.

    Returns:
        `(residuals, diagnostics)`. `residuals` is one row per product-store-week
        of the measurement window with the observed units, the counterfactual
        and its band, the residual, and whether the week is `campaign` or
        `post`. Phase 5 needs the per-cell incremental units to value them at
        the promoted price, which is why the frame and not a scalar is the
        return value; the campaign totals are in `diagnostics["lift"]`.

    Raises:
        NoCellsError: the campaign matched no treated product-store.
        ValueError: no horizon could be resolved, so the window has no end.
    """
    if level not in AGGREGATION_LEVELS:
        raise UnknownLevelError(
            f"level must be one of {sorted(AGGREGATION_LEVELS)}, got {level!r}"
        )
    own = con is None
    con = connect() if con is None else con
    try:
        if not isinstance(model, BaselineModel):
            model = BaselineModel.load(model)

        horizon_weeks, horizon_diag = resolve_horizon(campaign, cycles, con=con)
        if horizon_weeks is None:
            raise ValueError(
                f"campaign {campaign.name!r} has no horizon: it supplies none "
                f"and no repurchase cycle is recorded for commodity "
                f"{campaign.commodity!r}. The window would have no end, and "
                f"defaulting it to zero would silently measure the peak alone."
            )
        cells, cells_diag = campaign_cells(
            campaign, panel, horizon_weeks=horizon_weeks, con=con
        )

        first, last = campaign.weeks
        history = cells[cells["WEEK_NO"] < first]
        window = cells[cells["WEEK_NO"] >= first]
        path, rollout_diag = rollout(model, history, window, outcome=outcome)

        paths = {"counterfactual_units": path}
        if quantiles is not None:
            for name, alpha in (
                ("counterfactual_low", min(quantiles)),
                ("counterfactual_high", max(quantiles)),
            ):
                paths[name], _ = rollout(
                    model, history, window, quantile=alpha, outcome=outcome
                )

        drift = (
            drift_check(
                model,
                cells,
                first,
                len(window["WEEK_NO"].unique()),
                last - first + 1,
                outcome=outcome,
            )
            if check_drift
            else {"ran": False, "why": "check_drift=False"}
        )
    finally:
        if own:
            con.close()

    carried = [c for c in ("COMMODITY_DESC",) if c in window.columns]
    residuals = window[[*_KEY, *carried, outcome, "treated"]].rename(
        columns={outcome: "observed_units"}
    )
    for name, frame in paths.items():
        residuals = residuals.merge(
            frame[[*_KEY, "counterfactual_units"]].rename(
                columns={"counterfactual_units": name}
            ),
            on=list(_KEY),
            validate="one_to_one",
        )
    residuals["residual"] = (
        residuals["observed_units"] - residuals["counterfactual_units"]
    )
    if quantiles is not None:
        residuals["residual_low"] = (
            residuals["observed_units"] - residuals["counterfactual_high"]
        )
        residuals["residual_high"] = (
            residuals["observed_units"] - residuals["counterfactual_low"]
        )
    residuals["phase"] = np.where(
        residuals["WEEK_NO"].between(first, last), "campaign", "post"
    )
    residuals = residuals.sort_values(list(_KEY)).reset_index(drop=True)

    available_post = cells_diag["post_window"]["available_weeks"]
    lift = _aggregate(residuals, quantiles, available_post)
    units, level_diag = aggregate_residuals(
        residuals, level, available_post_weeks=available_post
    )
    level_diag["units"] = units.to_dict(orient="records")
    _compare_drift_to_gross(
        drift, lift["gross_incremental"], lift["net_incremental"]
    )

    diagnostics = {
        "stage": "estimate_lift",
        "campaign": campaign.model_dump(),
        "lift": lift,
        "by_level": level_diag,
        "horizon": horizon_diag,
        "cells": cells_diag,
        "drift_check": drift,
        "counterfactual": {
            "model_features": list(model.features),
            "train_window": list(model.train_window),
            "quantiles": list(quantiles) if quantiles else None,
            "rollout": rollout_diag,
        },
        "definitions": {
            "gross_incremental": (
                "residuals summed over the promoted weeks — the peak"
            ),
            "post_window_residual": (
                "residuals summed over the cycle-length tail — the trough, "
                "usually negative"
            ),
            "net_incremental": (
                "gross + post. Added, never netted: the tail is already signed, "
                "and subtracting a negative would report the trough as more lift."
            ),
            "retention_ratio": (
                "net / gross — what survived the payback period as a share of "
                "what the promoted weeks appeared to deliver. Computed once per "
                "counterfactual path, never averaged across paths."
            ),
        },
        "interval_note": (
            "The interval comes from the baseline's own quantile paths: it is "
            "the model's uncertainty about the counterfactual, not a placebo "
            "comparison. Task 4.5 owns the band that says whether this estimate "
            "is distinguishable from nothing happening, and no lift is "
            "defensible without both."
        ),
    }
    return residuals, diagnostics


def _aggregate(
    residuals: pd.DataFrame,
    quantiles: tuple[float, float] | None,
    available_post_weeks: int,
) -> dict[str, Any]:
    """Gross, post, net and retention — per path, then once each."""
    columns = {"point": "residual"}
    if quantiles is not None:
        columns["low"] = "residual_low"
        columns["high"] = "residual_high"

    campaign_rows = residuals["phase"] == "campaign"
    post_rows = ~campaign_rows

    by_path: dict[str, dict[str, float | None]] = {}
    for path, column in columns.items():
        gross = float(residuals.loc[campaign_rows, column].sum())
        post = float(residuals.loc[post_rows, column].sum())
        by_path[path] = {
            "gross_incremental": round(gross, 6),
            "post_window_residual": round(post, 6),
            # Added. Never subtracted — see the module docstring.
            "net_incremental": round(gross + post, 6),
            "retention_ratio": _retention(gross, gross + post, available_post_weeks),
        }

    point = by_path["point"]
    lift: dict[str, Any] = {
        **point,
        "observed_units": {
            "campaign": round(
                float(residuals.loc[campaign_rows, "observed_units"].sum()), 6
            ),
            "post": round(float(residuals.loc[post_rows, "observed_units"].sum()), 6),
        },
        "counterfactual_units": {
            "campaign": round(
                float(residuals.loc[campaign_rows, "counterfactual_units"].sum()), 6
            ),
            "post": round(
                float(residuals.loc[post_rows, "counterfactual_units"].sum()), 6
            ),
        },
        "by_path": by_path,
    }

    if quantiles is not None:
        for field in ("gross_incremental", "post_window_residual", "net_incremental"):
            lift[f"{field}_interval"] = [
                by_path["low"][field],
                by_path["high"][field],
            ]
        ratios = [
            by_path[p]["retention_ratio"] for p in ("low", "point", "high")
        ]
        present = [r for r in ratios if r is not None]
        # The envelope of the paths, not a distribution: two quantile paths give
        # two ratios, and calling that an interval is the most that can be
        # claimed. Task 5.2's bootstrap is what produces a real one.
        lift["retention_ratio_interval"] = (
            [min(present), max(present)] if len(present) > 1 else None
        )
        lift["retention_ratio_interval_note"] = (
            "The envelope of the counterfactual paths, not a sampling "
            "distribution. Each ratio divides once, after its own components "
            "are summed; ratios are never averaged."
        )

    if point["retention_ratio"] is None:
        lift["retention_ratio_absent"] = _retention_reason(
            point["gross_incremental"], available_post_weeks
        )
    return lift


def aggregate_residuals(
    residuals: pd.DataFrame,
    level: str = "campaign",
    *,
    available_post_weeks: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group per-cell-week residuals into the units a level reports.

    One row per unit with its cell count, its gross, post and net, and the mean
    units per cell that says how thin the unit is. The cell count is the number
    the unit's placebo band has to be matched to.

    Raises:
        UnknownLevelError: `level` is not in `AGGREGATION_LEVELS`.
        KeyError: the residuals lack a column the level groups by.
    """
    if level not in AGGREGATION_LEVELS:
        raise UnknownLevelError(
            f"level must be one of {sorted(AGGREGATION_LEVELS)}, got {level!r}"
        )
    keys = list(AGGREGATION_LEVELS[level])
    missing = [k for k in keys if k not in residuals.columns]
    if missing:
        raise KeyError(
            f"level {level!r} groups by {missing}, which the residuals do not "
            f"carry. Was the panel projected before it reached estimate_lift?"
        )

    campaign_rows = residuals["phase"] == "campaign"
    frame = residuals.assign(
        _gross=residuals["residual"].where(campaign_rows, 0.0),
        _post=residuals["residual"].where(~campaign_rows, 0.0),
        _campaign_units=residuals["observed_units"].where(campaign_rows, 0.0),
        _campaign_cf=residuals["counterfactual_units"].where(campaign_rows, 0.0),
    )
    grouped = frame.groupby(keys, observed=True) if keys else frame.groupby(
        lambda _: "campaign"
    )
    units = grouped.agg(
        cell_weeks=("residual", "size"),
        gross=("_gross", "sum"),
        post=("_post", "sum"),
        observed_units=("_campaign_units", "sum"),
        counterfactual_units=("_campaign_cf", "sum"),
    )
    cells = (
        frame.drop_duplicates(["PRODUCT_ID", "STORE_ID"])
        .groupby(keys, observed=True)
        .size()
        if keys
        else pd.Series(
            {"campaign": frame.drop_duplicates(["PRODUCT_ID", "STORE_ID"]).shape[0]}
        )
    )
    units["cells"] = cells
    units["net"] = units["gross"] + units["post"]
    units["mean_units_per_cell"] = (
        units["observed_units"] / units["cell_weeks"]
    )
    units["retention_ratio"] = [
        _retention(g, n, available_post_weeks)
        for g, n in zip(units["gross"], units["net"], strict=True)
    ]
    units = units.reset_index()
    if not keys:
        units = units.drop(columns=[c for c in ("index", "level_0") if c in units])

    diagnostics = {
        "level": level,
        "keys": keys,
        "n_units": len(units),
        "cells_total": int(units["cells"].sum()),
        "cells_per_unit": {
            "min": int(units["cells"].min()),
            "median": float(units["cells"].median()),
            "max": int(units["cells"].max()),
        },
        "gross_total": round(float(units["gross"].sum()), 6),
        "net_total": round(float(units["net"].sum()), 6),
        "invariant": (
            "The totals above are identical at every level: grouping "
            "partitions the same residuals and cannot change their sum. What "
            "the level changes is how many cells back each reported number, "
            "and therefore which placebo band it has to clear."
        ),
    }
    return units, diagnostics


def _compare_drift_to_gross(
    drift: dict[str, Any], gross: float, net: float
) -> None:
    """State the comparison the two blocks invite, rather than leaving it.

    An estimate smaller than its own estimator's drift on weeks where nothing
    happened is not a small effect — it is a number the method cannot resolve.
    A reader who has to divide two figures in different sections of a
    diagnostics file to notice that will sometimes not notice.

    Each figure is compared against drift over the same number of recursion
    steps: gross against the drift window's opening weeks, net against all of
    it. Drift grows with depth, so a mismatched comparison would flatter one
    and damn the other.
    """
    if not drift.get("ran"):
        return
    over_campaign = drift["residual_units_first_weeks"]
    over_window = drift["residual_units"]
    exceeds_gross = abs(over_campaign) >= abs(gross)
    exceeds_net = abs(over_window) >= abs(net)

    drift["gross_incremental"] = gross
    drift["net_incremental"] = net
    drift["exceeds_gross"] = exceeds_gross
    drift["exceeds_net"] = exceeds_net
    drift["reading"] = (
        f"Over the promoted weeks' worth of steps the estimator produces "
        f"{over_campaign:,.2f} units where nothing happened, against a gross of "
        f"{gross:,.2f}; over the full window, {over_window:,.2f} against a net "
        f"of {net:,.2f}. "
        + (
            "The drift is at least as large as the estimate, so this campaign's "
            "lift is not resolvable by this method at this horizon — the number "
            "exists but should not be acted on. Task 4.5's placebo band is what "
            "turns that judgement into a refusal."
            if exceeds_gross or exceeds_net
            else "The drift is smaller than the estimate, which is necessary "
            "for the estimate to mean anything and not sufficient — Task 4.5's "
            "placebo band is the test that decides."
        )
    )


def _retention(
    gross: float, net: float, available_post_weeks: int
) -> float | None:
    """`net / gross`, or None when that number would mislead.

    Two refusals. With no post-window week in the data the ratio is exactly 1.0
    by construction, which reads as "the campaign retained everything" when it
    means "the payback period was never observed". And with gross at or below
    zero there is nothing for a retention share to be a share *of* — the
    arithmetic still produces a number, and it is not one anybody should act on.
    """
    if available_post_weeks <= 0 or gross <= 0:
        return None
    return round(net / gross, 6)


def _retention_reason(gross: float, available_post_weeks: int) -> str:
    if available_post_weeks <= 0:
        return (
            "No post-window week exists in the panel, so net equals gross and "
            "the ratio would be exactly 1.0 — 'retained everything' when the "
            "truth is 'the payback period was never observed'."
        )
    return (
        f"Gross incremental is {gross:,.2f}, which is not positive. A retention "
        f"share of a non-positive base is arithmetic without a meaning: the "
        f"campaign has no measured peak for a tail to erode."
    )


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return out
