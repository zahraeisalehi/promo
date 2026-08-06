"""The refusal engine.

A gate turns a silent failure into a named, explainable refusal. This module
holds the vocabulary — one reason code per condition, each with a severity and a
deterministic message — and the runner that assembles them into a verdict.

**Refusal is a product state, not an error.** Nothing here raises on a data
condition. A gate that fires returns a `GateResult` the caller can render, and
`run_audit()` stops the pipeline rather than letting a stage produce a number it
cannot defend. A complete-looking output with a hidden failure is the thing this
project exists to argue against.

**Every message is written for a category manager, not a modeller.** Each names
what is missing and what it would take to fix, never blames the caller, never
uses the word error, and never says the promotion had no effect when what is
meant is that this comparison cannot see one. `promo/narrate.py` will rewrite
them for display in Phase 8; these are the deterministic fallback, and the
fallback has to be correct on its own.

Detection lives in the stage modules — `promo/audit.py` computes the evidence
and this module maps it to a verdict — so the numbers a sceptic would ask for
are produced where they are cheapest to compute and carried in `detail`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

__all__ = [
    "REASON_CODES",
    "CampaignSpec",
    "GateResult",
    "ReasonSpec",
    "UnknownReasonCode",
    "message_for",
    "run_audit",
    "write_audit",
]

Status = Literal["pass", "bounded", "refuse"]


class UnknownReasonCode(Exception):
    """A reason code was used that this module does not define."""


class ReasonSpec(BaseModel):
    """What a reason code means, how severe it is, and how it reads."""

    code: str
    severity: Status
    trigger: str
    template: str
    evidence_keys: tuple[str, ...]
    why_this_severity: str


class GateResult(BaseModel):
    """One gate's verdict. The unit the whole pipeline speaks in."""

    gate: str
    status: Status
    reason_code: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    message: str

    @property
    def refuses(self) -> bool:
        return self.status == "refuse"


class CampaignSpec(BaseModel):
    """The minimum a campaign must state to be audited.

    Deliberately small: only what the Phase 3 gates need. Campaigns are supplied
    to the gate rather than derived from the panel — Phase 3's objective is a
    verdict "for any proposed campaign" — so what else a campaign carries is the
    caller's business.

    Every field but `name` is optional, and a gate whose input is missing
    returns `pass` with a stated reason rather than being silently skipped.
    """

    name: str
    commodity: str | None = None
    product: int | None = None
    depth: float | None = None
    horizon_weeks: int | None = None
    margin: float | None = None


# --------------------------------------------------------------------------
# The vocabulary.
# --------------------------------------------------------------------------

_SPECS: tuple[ReasonSpec, ...] = (
    ReasonSpec(
        code="NO_VARIATION",
        severity="refuse",
        trigger="no axis has usable mixed mass",
        evidence_keys=("best_axis", "mixed_units_share", "threshold"),
        template=(
            "There is no group of comparable weeks to measure against: the best "
            "axis is {best_axis}, where only {mixed_units_share:.1%} of units "
            "sit in groups holding both promoted and unpromoted sales, against "
            "a {threshold:.0%} bar. Measuring this needs weeks where the same "
            "products ran in some stores and not others."
        ),
        why_this_severity=(
            "Without mixed mass there is no comparison to make, so no estimate "
            "exists to weaken."
        ),
    ),
    ReasonSpec(
        code="NO_OVERLAP",
        severity="refuse",
        trigger="treated and untreated are separable and extremes are common",
        evidence_keys=("auc", "outside_share"),
        template=(
            "Promoted and unpromoted weeks are too unalike to compare: a model "
            "tells them apart with an accuracy of {auc:.2f}, and "
            "{outside_share:.1%} of rows have almost no counterpart on the "
            "other side. A like-for-like comparison needs unpromoted weeks that "
            "resemble the promoted ones."
        ),
        why_this_severity=(
            "A comparison between populations with no common support is not a "
            "weaker estimate, it is a different quantity."
        ),
    ),
    ReasonSpec(
        code="LEAKED_FEATURE",
        severity="refuse",
        trigger="high separability driven by one covariate",
        evidence_keys=("feature", "gain_share", "auc"),
        template=(
            "A control variable is carrying the promotion itself: {feature} "
            "accounts for {gain_share:.0%} of how the model separates promoted "
            "from unpromoted weeks, at an accuracy of {auc:.2f}. That column "
            "needs removing or rebuilding before the counterfactual can be "
            "trusted."
        ),
        why_this_severity=(
            "This is a defect in the feature set rather than a limit of the "
            "data, and every downstream number inherits it."
        ),
    ),
    ReasonSpec(
        code="DEPTH_BOUNDED",
        severity="bounded",
        trigger="product on deal above the threshold share of its priced weeks",
        evidence_keys=("product", "deal_share", "threshold"),
        template=(
            "Product {product} is on deal in {deal_share:.0%} of the weeks it "
            "was priced, above the {threshold:.0%} mark, so its ordinary shelf "
            "price is barely observed. Its promotions can be ranked against "
            "each other but not priced against a normal week; a stretch of "
            "undiscounted weeks would restore the full figure."
        ),
        why_this_severity=(
            "A weaker quantity survives: depth is ordinal for this product, so "
            "its deals remain comparable with one another."
        ),
    ),
    ReasonSpec(
        code="INSUFFICIENT_SUPPORT",
        severity="refuse",
        trigger="fewer priced weeks than the minimum",
        evidence_keys=("product", "weeks_priced", "minimum_weeks"),
        template=(
            "Product {product} appears in only {weeks_priced} priced weeks, "
            "fewer than the {minimum_weeks} needed before anything can be said "
            "about its discounting in either direction. This is a limit of how "
            "much of this product we can see, not a finding about the "
            "promotion — more weeks of history would settle it."
        ),
        why_this_severity=(
            "Nothing can be said in either direction, which is a different "
            "sentence from DEPTH_BOUNDED's 'depth is ordinal only'. A category "
            "manager acts on the two differently: this one implies no action on "
            "the promotion at all."
        ),
    ),
    ReasonSpec(
        code="KAPPA_IMPOSSIBLE",
        severity="refuse",
        trigger="required incremental share of one or more",
        evidence_keys=("depth", "margin", "kappa"),
        template=(
            "At a {margin:.0%} gross margin, a {depth:.1%} discount needs "
            "{kappa:.0%} of the units sold on promotion to be sales that would "
            "not otherwise have happened — more than were sold at all. This "
            "promotion cannot cover its own discount at that margin at any "
            "volume; a shallower discount or a higher margin is what would "
            "change it."
        ),
        why_this_severity=(
            "The profitability question is settled by arithmetic before any "
            "model runs, so spending the estimate is waste. The lift may still "
            "be real and the runner's partial results keep whatever was "
            "computed before the stop."
        ),
    ),
    ReasonSpec(
        code="NO_MARGIN",
        severity="bounded",
        trigger="no margin supplied and none derivable",
        evidence_keys=("minimum_viable_margin",),
        template=(
            "No gross margin was supplied and this dataset carries no cost of "
            "goods, so profit and return cannot be computed and none will be "
            "assumed. What is available instead: this promotion needed a margin "
            "of at least {minimum_viable_margin:.1%} to cover its own discount, "
            "and the sensitivity table gives the answer at every margin from "
            "10% to 50%."
        ),
        why_this_severity=(
            "A weaker quantity survives and is arguably the more useful one: "
            "the break-even margin and the sweep need no margin at all."
        ),
    ),
    ReasonSpec(
        code="PLACEBO_OVERLAP",
        severity="refuse",
        trigger="estimate inside the placebo band",
        evidence_keys=("estimate", "band_low", "band_high"),
        template=(
            "The measured change of {estimate:,.0f} units sits inside the range "
            "of {band_low:,.0f} to {band_high:,.0f} that this comparison "
            "produces on weeks when nothing happened. The promotion cannot be "
            "separated from ordinary week-to-week movement here, and telling "
            "them apart would need a larger promotion, more weeks of it, or a "
            "closer set of comparison weeks. That is a statement about what "
            "this comparison can see, not evidence that the promotion did "
            "nothing."
        ),
        why_this_severity=(
            "Reporting a number indistinguishable from the null distribution "
            "would present noise as a finding."
        ),
    ),
    ReasonSpec(
        code="OVERLAPPING_TREATMENTS",
        severity="bounded",
        trigger="a second mechanic runs alongside the treatment",
        evidence_keys=("collision_share", "contaminated_share"),
        template=(
            "A mailer ran alongside the display on {collision_share:.0%} of the "
            "promoted weeks, and on {contaminated_share:.0%} of the comparison "
            "weeks. What can be measured here is the combined effect of the "
            "display and the mailer together; telling them apart needs weeks "
            "where only one of them ran."
        ),
        why_this_severity=(
            "The joint effect is a real and reportable quantity. It is simply a "
            "different one from the display effect, and must be labelled as "
            "joint wherever it appears."
        ),
    ),
    ReasonSpec(
        code="ROI_UNBOUNDED",
        severity="bounded",
        trigger="denominator interval spans zero",
        evidence_keys=("low", "high"),
        template=(
            "The incremental figure ranges from {low:,.0f} to {high:,.0f}, which "
            "crosses zero, so dividing by it gives a return without bounds. "
            "Incremental units and the cost of the promotion are still "
            "reported; a return figure needs a lift range that stays on one "
            "side of zero."
        ),
        why_this_severity=(
            "Every component is available and only the ratio is undefined. "
            "Reporting the components is more honest than reporting a ratio "
            "whose bounds are infinite."
        ),
    ),
    ReasonSpec(
        code="HORIZON_TOO_SHORT",
        severity="refuse",
        trigger="measurement window shorter than the repurchase cycle",
        evidence_keys=("commodity", "horizon_weeks", "required_weeks"),
        template=(
            "The measurement window stays open {horizon_weeks} weeks after the "
            "promotion ends, but shoppers rebuy {commodity} about every "
            "{required_weeks} weeks. A window this short counts the promotional "
            "peak and closes before the quiet period that follows it, which "
            "flatters the result. Extending it to at least {required_weeks} "
            "weeks past the end would fix it."
        ),
        why_this_severity=(
            "The number a short window produces is biased upward rather than "
            "merely imprecise, so it is wrong and not weak."
        ),
    ),
)

#: Every reason code this pipeline can emit, keyed by code.
REASON_CODES: dict[str, ReasonSpec] = {spec.code: spec for spec in _SPECS}


def message_for(code: str, **evidence: Any) -> str:
    """Render a reason code's deterministic message.

    Raises:
        UnknownReasonCode: the code is not defined here.
        KeyError: the evidence is missing a key the template needs. A template
            that cannot render is a defect, not a data condition, so this one
            does raise.
    """
    if code not in REASON_CODES:
        raise UnknownReasonCode(
            f"{code!r} is not a defined reason code; known codes are "
            f"{sorted(REASON_CODES)}"
        )
    spec = REASON_CODES[code]
    missing = [k for k in spec.evidence_keys if k not in evidence]
    if missing:
        raise KeyError(
            f"{code} needs evidence {missing} to render its message"
        )
    return spec.template.format(**evidence)


def gate_result(
    gate: str, code: str, detail: dict[str, Any], **evidence: Any
) -> GateResult:
    """Build a `GateResult` for a fired gate, at the code's own severity."""
    spec = REASON_CODES.get(code)
    if spec is None:
        raise UnknownReasonCode(f"{code!r} is not a defined reason code")
    return GateResult(
        gate=gate,
        status=spec.severity,
        reason_code=code,
        detail=detail,
        message=message_for(code, **evidence),
    )


def passing(gate: str, message: str, detail: dict[str, Any] | None = None) -> GateResult:
    """A gate that did not fire, recorded for the audit trail."""
    return GateResult(
        gate=gate, status="pass", reason_code=None, detail=detail or {}, message=message
    )


# --------------------------------------------------------------------------
# The runner.
# --------------------------------------------------------------------------

#: Gates run cheapest first, so a campaign that fails on arithmetic never pays
#: for a model fit. `overlap` is last because it costs about two and a half
#: minutes on the full panel and everything before it costs seconds.
GATE_ORDER: tuple[str, ...] = (
    "break_even",
    "horizon",
    "price_status",
    "variation",
    "collisions",
    "overlap",
)


def run_audit(
    campaign: CampaignSpec,
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    cycles: str | Path | pd.DataFrame = "data/interim/repurchase_cycles.parquet",
    statuses: str | Path | pd.DataFrame = "data/interim/prices.parquet",
    stop_on_refuse: bool = True,
    run_overlap: bool = True,
    **audit_kwargs: Any,
) -> tuple[list[GateResult], dict[str, Any]]:
    """Run the Phase 3 gates and return a verdict.

    Stops at the first `refuse` and returns the results computed so far.
    Partial output plus a stated reason is the product; a complete-looking
    output with a hidden failure is not.

    Args:
        campaign: what is being audited. Gates whose inputs it does not carry
            return `pass` with a stated reason rather than being skipped
            silently.
        panel: the modelling panel.
        cycles: the Task 2.7 repurchase-cycle table.
        statuses: the per-product price-status table. Defaults to
            `prices.parquet` rather than the panel: `panel.parquet` keeps
            `price_status` but Task 2.6's projection dropped `deal_share` and
            `n_weeks_priced`, which are the numbers the messages quote.
        stop_on_refuse: the default. Set False to collect every gate's verdict
            for diagnosis — a campaign can fail more than one and the
            short-circuit hides the rest.
        run_overlap: set False to skip the expensive model fit.
        **audit_kwargs: forwarded to the underlying `promo.audit` checks.

    Returns:
        `(results, diagnostics)`.
    """
    from promo import audit as audit_module

    results: list[GateResult] = []
    ran: list[str] = []
    skipped: list[str] = []
    stopped_at: str | None = None

    checks = {
        "break_even": lambda: _gate_break_even(campaign, audit_module),
        "horizon": lambda: _gate_horizon(campaign, cycles, audit_module),
        "price_status": lambda: _gate_price_status(
            campaign, statuses, audit_module
        ),
        "variation": lambda: _gate_variation(panel, audit_module, audit_kwargs),
        "collisions": lambda: _gate_collisions(panel, audit_module, audit_kwargs),
        "overlap": lambda: _gate_overlap(panel, audit_module, audit_kwargs),
    }

    for name in GATE_ORDER:
        if stopped_at is not None:
            skipped.append(name)
            continue
        if name == "overlap" and not run_overlap:
            skipped.append(name)
            continue
        result = checks[name]()
        results.append(result)
        ran.append(name)
        if result.refuses and stop_on_refuse:
            stopped_at = name

    refusals = [r for r in results if r.status == "refuse"]
    bounded = [r for r in results if r.status == "bounded"]
    verdict = (
        "not identified" if refusals else ("bounded" if bounded else "measurable")
    )
    return results, {
        "stage": "run_audit",
        "campaign": campaign.model_dump(),
        "verdict": verdict,
        "gate_order": list(GATE_ORDER),
        "gates_run": ran,
        "gates_skipped": skipped,
        "stopped_at": stopped_at,
        "refusals": [r.reason_code for r in refusals],
        "bounded": [r.reason_code for r in bounded],
        "short_circuit": (
            "Gates run cheapest first so a campaign that fails on arithmetic "
            "never pays for a model fit. Everything after the stop is skipped, "
            "so a campaign may have more problems than the list shows — pass "
            "stop_on_refuse=False to see all of them."
        ),
    }


def _gate_break_even(campaign: CampaignSpec, audit_module: Any) -> GateResult:
    if campaign.depth is None:
        return passing(
            "break_even",
            "No discount depth was supplied, so the break-even arithmetic was "
            "not run.",
        )
    if campaign.margin is None:
        _, sweep_diag = audit_module.margin_sweep(campaign.depth)
        return gate_result(
            "break_even",
            "NO_MARGIN",
            detail=sweep_diag,
            minimum_viable_margin=sweep_diag["minimum_viable_margin"],
        )
    result = audit_module.kappa_star(campaign.depth, campaign.margin)
    if result.reason_code == "KAPPA_IMPOSSIBLE":
        return gate_result(
            "break_even",
            "KAPPA_IMPOSSIBLE",
            detail={"depth": result.depth, "margin": result.margin,
                    "kappa": result.kappa},
            depth=result.depth,
            margin=result.margin,
            kappa=result.kappa,
        )
    return passing(
        "break_even",
        f"At a {result.margin:.0%} margin this promotion breaks even if "
        f"{result.kappa:.0%} of its units are incremental.",
        {"kappa": result.kappa},
    )


def _gate_horizon(
    campaign: CampaignSpec, cycles: Any, audit_module: Any
) -> GateResult:
    if campaign.commodity is None or campaign.horizon_weeks is None:
        return passing(
            "horizon",
            "No commodity or measurement horizon was supplied, so the window "
            "was not checked against the repurchase cycle.",
        )
    frame = pd.DataFrame(
        {
            "COMMODITY_DESC": [campaign.commodity],
            "horizon_weeks": [campaign.horizon_weeks],
        }
    )
    checked, diag = audit_module.horizon_check(frame, cycles)
    row = checked.iloc[0]
    if row["status"] == "HORIZON_TOO_SHORT":
        return gate_result(
            "horizon",
            "HORIZON_TOO_SHORT",
            detail=diag,
            commodity=campaign.commodity,
            horizon_weeks=campaign.horizon_weeks,
            required_weeks=int(row["required_weeks"]),
        )
    if row["status"] == "UNKNOWN_CYCLE":
        return passing(
            "horizon",
            f"No repurchase cycle is recorded for {campaign.commodity}, so the "
            f"window could not be checked. It has not been shown to be long "
            f"enough.",
            diag,
        )
    return passing(
        "horizon",
        f"The window stays open {campaign.horizon_weeks} weeks past the end, "
        f"clearing {campaign.commodity}'s cycle of "
        f"{int(row['required_weeks'])} weeks.",
        diag,
    )


def _gate_price_status(
    campaign: CampaignSpec, statuses: Any, audit_module: Any
) -> GateResult:
    """Is this product's depth cardinal, ordinal only, or unsayable?

    The two codes here are deliberately not one code. `DEPTH_BOUNDED` is a
    pricing fact — this product is almost always on deal, so its discounts rank
    against each other but not against a shelf price — and a weaker quantity
    survives. `INSUFFICIENT_SUPPORT` is a coverage fact — we have not seen the
    product enough to say anything in either direction — and nothing survives.
    A category manager acts differently on each, which is why settled decision 6
    split them.
    """
    if campaign.product is None:
        _, diag = audit_module.price_status_check(statuses, None)
        counts = diag["status_counts"]
        return passing(
            "price_status",
            (
                "No product was named, so this campaign's depth status was not "
                "judged. Across the price table "
                f"{counts.get('identified', 0):,} products have a usable depth, "
                f"{counts.get('bounded', 0):,} are on deal too often to price "
                f"against a normal week, and "
                f"{counts.get('insufficient_support', 0):,} are seen too "
                "rarely to say. A commodity spans products with different "
                "statuses, so naming the product is what makes a verdict "
                "possible."
            ),
            diag,
        )

    _, diag = audit_module.price_status_check(statuses, campaign.product)
    if not diag.get("found"):
        return passing(
            "price_status",
            (
                f"Product {campaign.product} does not appear in the price "
                f"table, so its discount depth could not be judged. It has not "
                f"been shown to be usable."
            ),
            diag,
        )

    status = diag["status"]
    if status == "bounded":
        return gate_result(
            "price_status",
            "DEPTH_BOUNDED",
            detail=diag,
            product=diag["product"],
            deal_share=diag["deal_share"],
            threshold=diag["bounded_threshold"],
        )
    if status == "insufficient_support":
        return gate_result(
            "price_status",
            "INSUFFICIENT_SUPPORT",
            detail=diag,
            product=diag["product"],
            weeks_priced=diag["weeks_priced"],
            minimum_weeks=diag["min_priced_weeks"],
        )
    return passing(
        "price_status",
        (
            f"Product {diag['product']} was priced in {diag['weeks_priced']} "
            f"weeks and on deal in {diag['deal_share']:.0%} of them, so its "
            f"discount depth is a usable number."
        ),
        diag,
    )


def _gate_variation(panel: Any, audit_module: Any, kwargs: dict) -> GateResult:
    _, diag = audit_module.variation_axes(
        panel, **{k: v for k, v in kwargs.items() if k in {"treatment_column"}}
    )
    if not diag["usable_axes"]:
        best = diag["best_axis"] or (diag["leading_axes"] or ["none"])[0]
        return gate_result(
            "variation",
            "NO_VARIATION",
            detail=diag,
            best_axis=best,
            mixed_units_share=max(diag["mixed_units_share"].values(), default=0.0),
            threshold=diag["threshold"],
        )
    return passing(
        "variation",
        f"Comparable weeks exist on {', '.join(diag['usable_axes'])}; "
        f"{diag['best_axis'] or 'several axes'} carries the most.",
        diag,
    )


def _gate_collisions(panel: Any, audit_module: Any, kwargs: dict) -> GateResult:
    _, diag = audit_module.collisions(
        panel, **{k: v for k, v in kwargs.items() if k in {"treatment_column"}}
    )
    if diag["status"] == "OVERLAPPING_TREATMENTS":
        return gate_result(
            "collisions",
            "OVERLAPPING_TREATMENTS",
            detail=diag,
            collision_share=diag["collision"]["share_of_treated"],
            contaminated_share=diag["contaminated_controls"]["share_of_controls"],
        )
    return passing(
        "collisions",
        "The display ran on its own: no meaningful share of promoted or "
        "comparison weeks also carried a mailer.",
        diag,
    )


def _gate_overlap(panel: Any, audit_module: Any, kwargs: dict) -> GateResult:
    allowed = {"treatment_column", "n_folds", "n_estimators", "cv", "seed"}
    _, diag = audit_module.overlap(
        panel, **{k: v for k, v in kwargs.items() if k in allowed}
    )
    if diag["diagnosis"] == "LEAKAGE_SUSPECTED":
        top = diag["top_features"][0]
        return gate_result(
            "overlap",
            "LEAKED_FEATURE",
            detail=diag,
            feature=top["feature"],
            gain_share=top["gain_share"],
            auc=diag["auc"],
        )
    if diag["diagnosis"] == "NON_OVERLAP_SUSPECTED":
        return gate_result(
            "overlap",
            "NO_OVERLAP",
            detail=diag,
            auc=diag["auc"],
            outside_share=diag["propensity_extremes"]["outside_share"],
        )
    return passing(
        "overlap",
        f"Promoted and unpromoted weeks are comparable: a model separates them "
        f"at only {diag['auc']:.2f}, and "
        f"{diag['propensity_extremes']['outside_share']:.1%} of rows sit "
        f"without a counterpart.",
        diag,
    )


def write_audit(
    results: list[GateResult], diagnostics: dict[str, Any], path: str | Path
) -> Path:
    """Write an audit verdict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **diagnostics,
        "results": [r.model_dump() for r in results],
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out
