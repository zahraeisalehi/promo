"""Ranking without the winner's curse, and what a holdout would cost to see.

Task 7.1. Three things a merchant asks after the measurement: which campaigns
were best, how deep a discount is worth running, and how big a test would have
to be to answer either question next time.

**The top of a raw ranking is a biased estimate, always.** Pick the highest of
many noisy numbers and you have selected partly on noise, so the winner's true
value is lower than its estimate — the winner's curse. James-Stein shrinkage
pulls every estimate towards the precision-weighted grand mean by an amount set
by how much of the spread is real, and the rule that makes it honest is that
**the shrunk value is what gets published**. Ranking on shrunk estimates and
then reporting the raw one for the winner reintroduces the whole bias, and the
recommendation underperforms its forecast by construction.

**A response curve says where to stop, not how much to run.** Fitting lift
against depth per commodity and finding where marginal return crosses zero
gives the depth beyond which another point of discount costs more than it
returns. It is a within-sample fit on observational data, so it describes the
depths that *were* run, not the ones that were not — extrapolating past the
observed range is the obvious way to misuse it and the curve records its own
support so a caller can see the edge.

**The MDE calculator is the closing argument.** A 5% holdout feels cheap
because its cost is counted in forgone promoted sales. Its real price is in the
effects it can no longer detect: at a fixed cluster count the detectable effect
is minimised at a balanced split, and a 5% holdout needs roughly 2.3x the
effect a 50/50 split would find. That is the trade a merchant should be shown
before they choose, not after.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_POWER",
    "DEFAULT_SIGNIFICANCE",
    "james_stein_shrink",
    "mde",
    "mde_grid",
    "rank_campaigns",
    "response_curve",
    "write_diagnostics",
]

#: Two-sided 5% and 80% power — the conventional pair, named so a caller
#: changing them has to say so.
DEFAULT_SIGNIFICANCE: float = 1.96
DEFAULT_POWER: float = 0.84

#: Below this the shrinkage weight is treated as zero: the between-campaign
#: variance is indistinguishable from sampling noise, so every estimate
#: collapses onto the grand mean and no ranking is defensible.
MIN_TAU2: float = 1e-9


def james_stein_shrink(
    estimates: np.ndarray | pd.Series,
    standard_errors: np.ndarray | pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pull estimates towards the precision-weighted grand mean.

    `tau2` is the between-campaign variance left after removing what sampling
    noise alone would produce. When it comes back at the floor, the campaigns
    are indistinguishable from one another and every shrunk value equals the
    grand mean — which is a finding, not a failure, and the diagnostics say so
    rather than returning a ranking that is really an ordering of noise.

    Returns:
        `(shrunk, diagnostics)`.
    """
    est = np.asarray(estimates, dtype="float64")
    se = np.asarray(standard_errors, dtype="float64")
    if est.shape != se.shape:
        raise ValueError(
            f"estimates and standard_errors differ in shape: {est.shape} vs "
            f"{se.shape}"
        )
    if est.size == 0:
        raise ValueError("nothing to shrink")
    if np.any(se <= 0):
        raise ValueError(
            "every standard error must be positive; a zero would claim an "
            "estimate with no uncertainty and take infinite weight"
        )

    weights = 1.0 / se**2
    grand = float(np.average(est, weights=weights))
    # What is left of the spread once sampling noise is accounted for. Negative
    # means the observed spread is smaller than noise alone predicts, so there
    # is no real between-campaign variation to preserve.
    raw_tau2 = float(np.var(est) - np.mean(se**2))
    tau2 = max(raw_tau2, MIN_TAU2)
    shrink_weight = tau2 / (tau2 + se**2)
    shrunk = grand + shrink_weight * (est - grand)

    collapsed = raw_tau2 <= MIN_TAU2
    diagnostics = {
        "n": int(est.size),
        "grand_mean": round(grand, 6),
        "tau2": round(tau2, 9),
        "tau2_raw": round(raw_tau2, 9),
        "mean_shrink_weight": round(float(shrink_weight.mean()), 6),
        "collapsed_to_grand_mean": bool(collapsed),
        "why_shrink": (
            "The top of a raw ranking is selected partly on noise, so the "
            "winner's true value is below its estimate. Shrinking towards the "
            "grand mean by how much of the spread is real removes most of that."
        ),
        "collapse_meaning": (
            "Between-campaign variance is indistinguishable from sampling "
            "noise, so every shrunk value is the grand mean. The campaigns "
            "cannot be ranked apart — that is a finding about this evidence, "
            "not a defect in the method."
            if collapsed
            else None
        ),
    }
    return shrunk, diagnostics


def rank_campaigns(
    campaigns: pd.DataFrame,
    *,
    estimate_column: str = "lift",
    se_column: str = "se",
    name_column: str = "campaign",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank on shrunk estimates and publish the shrunk value as the expectation.

    The raw estimate is kept in the output because hiding it would be its own
    dishonesty — but `expectation` is the shrunk number, and the ranking is on
    that. A caller that ranks on `shrunk` and then reports `lift` for the
    winner has reintroduced the whole bias.
    """
    for column in (estimate_column, se_column):
        if column not in campaigns.columns:
            raise KeyError(f"{column!r} is not a column of campaigns")

    shrunk, shrink_diag = james_stein_shrink(
        campaigns[estimate_column], campaigns[se_column]
    )
    ranked = campaigns.copy()
    ranked["shrunk"] = shrunk
    #: The number to report. Named so that publishing the wrong one is a
    #: visible choice rather than an oversight.
    ranked["expectation"] = shrunk
    ranked["shrinkage"] = ranked[estimate_column] - ranked["shrunk"]
    ranked = ranked.sort_values("shrunk", ascending=False).reset_index(drop=True)
    # When the shrinkage collapsed, every shrunk value is the grand mean and
    # any order between them is floating-point dust. Ranking them 1..n would
    # dress that dust as a finding, so they are all tied at 1 instead.
    collapsed = shrink_diag["collapsed_to_grand_mean"]
    ranked["rank"] = 1 if collapsed else np.arange(1, len(ranked) + 1)
    ranked["rank_meaningful"] = not collapsed

    raw_order = campaigns.sort_values(estimate_column, ascending=False)
    raw_top = (
        raw_order.iloc[0][name_column] if name_column in campaigns.columns else None
    )
    shrunk_top = (
        ranked.iloc[0][name_column] if name_column in ranked.columns else None
    )

    diagnostics = {
        "stage": "rank_campaigns",
        "campaigns": len(ranked),
        "shrinkage": shrink_diag,
        "ranked_on": "shrunk",
        "published_as_expectation": "shrunk",
        "raw_top": raw_top,
        "shrunk_top": shrunk_top,
        "top_changed": bool(raw_top != shrunk_top) and not collapsed,
        "ranking_meaningful": not collapsed,
        "why_not_rankable": (
            "Between-campaign variance is indistinguishable from sampling "
            "noise, so every shrunk estimate is the grand mean and every "
            "campaign is tied at rank 1. Any order between them would be "
            "floating-point dust presented as a recommendation. More "
            "campaigns, or tighter standard errors, is what would separate "
            "them."
            if collapsed
            else None
        ),
        "largest_shrinkage": round(float(ranked["shrinkage"].abs().max()), 6),
        "publish_rule": (
            "Rank on the shrunk estimate and publish the shrunk value. "
            "Reporting the raw estimate for the campaign the shrunk ranking "
            "chose puts the winner's curse straight back in, and the "
            "recommendation then underperforms its own forecast on average."
        ),
    }
    return ranked, diagnostics


def response_curve(
    observations: pd.DataFrame,
    *,
    depth_column: str = "depth",
    lift_column: str = "lift",
    group_column: str = "COMMODITY_DESC",
    min_points: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lift against depth per commodity, and where marginal return crosses zero.

    A quadratic `lift = a + b*depth + c*depth^2` is fitted per group. Marginal
    return is `b + 2c*depth`, so it crosses zero at `-b / (2c)` — the depth past
    which another point of discount returns less than the last one.

    **The turning point is only meaningful inside the observed range.** A
    concave fit always has one, and if it lies beyond the depths actually run
    it is an extrapolation of a curve nobody tested there. Each row carries
    `within_support`, and a caller reading the crossing without it is reading a
    number the data does not contain.

    Groups with fewer than `min_points` distinct depths are returned with a null
    crossing and a stated reason: three points determine a parabola exactly, so
    a turning point from three is arithmetic rather than evidence.
    """
    for column in (depth_column, lift_column, group_column):
        if column not in observations.columns:
            raise KeyError(f"{column!r} is not a column of observations")

    rows: list[dict[str, Any]] = []
    for group, block in observations.groupby(group_column, observed=True):
        depths = block[depth_column].to_numpy(dtype="float64")
        lifts = block[lift_column].to_numpy(dtype="float64")
        distinct = int(np.unique(depths).size)
        low, high = float(depths.min()), float(depths.max())

        row: dict[str, Any] = {
            group_column: group,
            "n_campaigns": len(block),
            "distinct_depths": distinct,
            "depth_min": low,
            "depth_max": high,
            "a": None, "b": None, "c": None,
            "peak_depth": None,
            "within_support": None,
            "concave": None,
            "reason": None,
        }
        if distinct < min_points:
            row["reason"] = (
                f"only {distinct} distinct depths; three determine a parabola "
                f"exactly, so a turning point below {min_points} is arithmetic "
                f"rather than evidence"
            )
            rows.append(row)
            continue

        c, b, a = np.polyfit(depths, lifts, 2)
        row.update({"a": float(a), "b": float(b), "c": float(c)})
        if abs(c) < 1e-12:
            row["reason"] = "the fit is linear; marginal return never crosses zero"
            rows.append(row)
            continue

        peak = float(-b / (2.0 * c))
        row["peak_depth"] = peak
        row["concave"] = bool(c < 0)
        row["within_support"] = bool(low <= peak <= high)
        if not row["within_support"]:
            row["reason"] = (
                f"the crossing at {peak:.3f} lies outside the depths actually "
                f"run ({low:.3f} to {high:.3f}), so it is an extrapolation"
            )
        elif not row["concave"]:
            row["reason"] = (
                "the fit is convex, so this is a minimum rather than a point "
                "of diminishing return"
            )
        rows.append(row)

    curve = pd.DataFrame(rows)
    usable = curve[
        curve["within_support"].fillna(False) & curve["concave"].fillna(False)
    ]
    diagnostics = {
        "stage": "response_curve",
        "groups": len(curve),
        "groups_with_usable_crossing": len(usable),
        "form": "lift = a + b*depth + c*depth^2, marginal return b + 2c*depth",
        "crossing": "-b / (2c)",
        "min_points": min_points,
        "observational": (
            "This is a within-sample fit on depths a retailer chose to run, not "
            "an experiment. It describes the depths that were tried; it says "
            "nothing about the ones that were not, and the depths that were "
            "tried were chosen for reasons that may themselves relate to lift."
        ),
        "support_rule": (
            "A concave fit always has a turning point. Outside the observed "
            "depth range it is an extrapolation of a curve nobody tested "
            "there, so within_support must be read alongside peak_depth."
        ),
    }
    return curve, diagnostics


def mde(
    sigma: float,
    clusters: int,
    cluster_size: int,
    icc: float,
    holdout_fraction: float,
    *,
    significance: float = DEFAULT_SIGNIFICANCE,
    power: float = DEFAULT_POWER,
) -> float:
    """The smallest effect a design of this shape could detect.

    `deff = 1 + (n - 1) * icc` is the design effect: units inside a cluster are
    correlated, so a cluster of 50 stores is worth far less than 50 independent
    ones. Ignoring it is the standard way to promise a test more power than it
    has.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            f"holdout_fraction must be strictly between 0 and 1, got "
            f"{holdout_fraction}. A design with no holdout, or no treatment, "
            f"detects nothing at any size."
        )
    if clusters < 1 or cluster_size < 1:
        raise ValueError("clusters and cluster_size must both be at least 1")
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must be between 0 and 1, got {icc}")

    deff = 1.0 + (cluster_size - 1) * icc
    denominator = (
        clusters * cluster_size * holdout_fraction * (1.0 - holdout_fraction)
    )
    return float((significance + power) * sigma * np.sqrt(deff / denominator))


def mde_grid(
    sigma: float,
    clusters: int,
    cluster_size: int,
    icc: float,
    *,
    holdout_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    significance: float = DEFAULT_SIGNIFICANCE,
    power: float = DEFAULT_POWER,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """What each holdout size costs in effects it can no longer see.

    The point of the table is the ratio column. A 5% holdout feels cheap
    because its cost is counted in forgone promoted sales; its real price is
    that it needs roughly 2.3x the effect a balanced split would detect. Show
    the merchant the trade before they choose.
    """
    rows = []
    for fraction in holdout_fractions:
        rows.append(
            {
                "holdout_fraction": fraction,
                "mde": mde(
                    sigma, clusters, cluster_size, icc, fraction,
                    significance=significance, power=power,
                ),
            }
        )
    table = pd.DataFrame(rows)
    balanced = mde(
        sigma, clusters, cluster_size, icc, 0.5,
        significance=significance, power=power,
    )
    table["vs_balanced"] = table["mde"] / balanced

    deff = 1.0 + (cluster_size - 1) * icc
    diagnostics = {
        "stage": "mde_grid",
        "sigma": sigma,
        "clusters": clusters,
        "cluster_size": cluster_size,
        "icc": icc,
        "design_effect": round(float(deff), 6),
        "effective_sample": round(float(clusters * cluster_size / deff), 2),
        "balanced_mde": round(float(balanced), 6),
        "cheapest_to_detect": float(
            table.loc[table["mde"].idxmin(), "holdout_fraction"]
        ),
        "design_effect_note": (
            f"Units inside a cluster are correlated, so {clusters * cluster_size:,} "
            f"units behave like {clusters * cluster_size / deff:,.0f} independent "
            f"ones at an ICC of {icc}. Ignoring the design effect promises a "
            f"test more power than it has."
        ),
        "the_trade": (
            "A small holdout is cheap in forgone promoted sales and expensive "
            "in effects it can no longer see. The detectable effect is "
            "minimised at a balanced split and rises steeply as the holdout "
            "shrinks — the merchant should be shown that before choosing, not "
            "after the test fails to find anything."
        ),
    }
    return table, diagnostics


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2, default=str) + "\n")
    return out
