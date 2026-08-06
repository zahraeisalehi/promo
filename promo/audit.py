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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.features import CONTEMPORANEOUS_FEATURES, LAGGED_FEATURES
from promo.io import connect

__all__ = [
    "AXES",
    "DEFAULT_COVARIATES",
    "LEAKAGE_AUC",
    "MARGIN_GRID",
    "MAX_COLLISION_SHARE",
    "MEANINGFUL_MIXED_SHARE",
    "PROPENSITY_HIGH",
    "PROPENSITY_LOW",
    "KappaResult",
    "UnobservedRowsError",
    "collisions",
    "horizon_check",
    "kappa_star",
    "margin_sweep",
    "overlap",
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

#: The covariates the propensity model sees by default: exactly the Phase 2.6
#: feature set. Identifiers are deliberately absent — a tree given PRODUCT_ID
#: memorises which products get displayed and returns a near-perfect AUC that
#: says nothing about confounding. Pass them explicitly if that is the question.
DEFAULT_COVARIATES: tuple[str, ...] = (*LAGGED_FEATURES, *CONTEMPORANEOUS_FEATURES)

#: Propensity outside these bounds means a row has almost no counterpart of the
#: other kind. The plan names both figures.
PROPENSITY_LOW: float = 0.02
PROPENSITY_HIGH: float = 0.98

#: Above this cross-validated AUC, treated and untreated are so separable that
#: leakage is the first explanation to rule out, not the last.
LEAKAGE_AUC: float = 0.95

#: The assumed-margin grid: 10% to 50% in 5-point steps, nine points. Written
#: as literals rather than built by arithmetic so no float drift can put 0.35
#: at 0.35000000000000003 and make two tables disagree on a key.
#:
#: Task 5.2 is the authority on this grid. It lives here because audit.py is
#: what exists; `promo/accounting.py` must import it rather than restate it, or
#: the gate's sweep and the accounting table can silently diverge.
MARGIN_GRID: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
)

#: Share of treated rows (or of controls) that may carry a second promotional
#: mechanic before the estimand stops being a single-treatment effect. Not a
#: recorded decision — the plan asks for the check, not a bar — so it is a
#: parameter with a stated default. Below a twentieth, contamination moves the
#: estimate by a few percent even if the second mechanic were as strong as the
#: first; above it, the label on the estimate is wrong.
MAX_COLLISION_SHARE: float = 0.05


class UnobservedRowsError(Exception):
    """The panel contains rows where treatment was never observed."""


@dataclass(frozen=True)
class KappaResult:
    """The required incremental share, or why there is not one.

    `kappa` is None exactly when `margin` is None: the dataset has no COGS, so
    a break-even share cannot be computed without an assumption the user
    supplies. That is a refusal with a reason code, not a missing value.
    """

    depth: float
    margin: float | None
    kappa: float | None
    feasible: bool | None
    reason_code: str | None
    detail: str


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


def overlap(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    treatment_column: str = "treated",
    covariates: tuple[str, ...] | list[str] | None = None,
    n_folds: int = 5,
    cv: str = "group",
    n_estimators: int = 200,
    seed: int = 0,
    low: float = PROPENSITY_LOW,
    high: float = PROPENSITY_HIGH,
    leakage_auc: float = LEAKAGE_AUC,
    require_observed: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Can a model tell treated rows from untreated ones, and if so, why?

    Fits a gradient-boosted classifier for `treated` from the covariates,
    cross-validated, and reports the out-of-fold AUC, the share of rows whose
    propensity sits outside `[low, high]`, and every feature's importance.

    **A high AUC is ambiguous, which is the point of the importance table.** It
    can mean genuine non-overlap — treated and untreated units really are
    different populations, and no counterfactual exists for some of them — or it
    can mean a covariate encodes the treatment. Those need opposite responses:
    the first is a refusal, the second is a bug. One feature carrying most of
    the gain is the signature of the second, so the top five are reported
    alongside the AUC rather than in a separate place.

    **Folds are grouped by product-store by default.** Adjacent weeks of the same
    product-store are near-duplicates — they share lagged units almost exactly —
    so a random split puts near-copies of a row on both sides and inflates the
    AUC. Grouping keeps a product-store entirely within one fold. `cv="random"`
    is available and the gap between the two is itself diagnostic.

    Args:
        panel: the modelling panel, as a parquet path or a DataFrame.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        treatment_column: the boolean being predicted.
        covariates: columns the model may see. Defaults to `DEFAULT_COVARIATES`.
        n_folds: cross-validation folds.
        cv: `"group"` (by product-store) or `"random"` (stratified).
        n_estimators: boosting rounds per fold. Lower trades precision for
            time; the default takes about two and a half minutes on the full
            panel.
        seed: passed to the splitter and the model; no global seeding.
        low, high: propensity bounds outside which a row has no counterpart.
        leakage_auc: AUC above which leakage is flagged as the first suspect.
        require_observed: as in `variation_axes`.

    Returns:
        `(importances, diagnostics)`. `importances` is every covariate ranked by
        mean gain across folds.

    Raises:
        UnobservedRowsError: as in `variation_axes`.
        KeyError: a requested covariate is not in the panel.
        ValueError: `cv` is not one of the two schemes, or the treatment does
            not vary at all, which makes a classifier meaningless.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if cv not in {"group", "random"}:
        raise ValueError(f"cv must be 'group' or 'random', got {cv!r}")
    names = list(DEFAULT_COVARIATES if covariates is None else covariates)

    own = con is None
    con = connect() if con is None else con
    try:
        frame = _load_for_overlap(
            panel, con, treatment_column, names, require_observed
        )
    finally:
        if own:
            con.close()

    y = frame["_treated"].to_numpy()
    if y.all() or not y.any():
        raise ValueError(
            "the treatment does not vary in this panel, so a classifier "
            "separating treated from untreated is meaningless"
        )

    features = frame[names]
    groups = (
        frame["PRODUCT_ID"].astype(str) + "_" + frame["STORE_ID"].astype(str)
        if cv == "group"
        else None
    )
    splitter = (
        GroupKFold(n_splits=n_folds)
        if cv == "group"
        else StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    )
    splits = list(
        splitter.split(features, y, groups)
        if cv == "group"
        else splitter.split(features, y)
    )

    propensity = np.full(len(frame), np.nan)
    gains = np.zeros(len(names))
    fold_auc: list[float] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = lgb.LGBMClassifier(
            # CLAUDE.md: max_bin 63, num_leaves no greater than 63.
            max_bin=63,
            num_leaves=31,
            n_estimators=n_estimators,
            learning_rate=0.05,
            random_state=seed + fold,
            n_jobs=2,
            verbose=-1,
        )
        model.fit(features.iloc[train_idx], y[train_idx])
        propensity[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        gains += model.booster_.feature_importance(importance_type="gain")
        fold_auc.append(
            float(roc_auc_score(y[test_idx], propensity[test_idx]))
            if len(set(y[test_idx])) > 1
            else float("nan")
        )

    importances = (
        pd.DataFrame({"feature": names, "gain": gains})
        .assign(gain_share=lambda d: d["gain"] / d["gain"].sum())
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    diagnostics = _diagnose_overlap(
        frame=frame,
        y=y,
        propensity=propensity,
        fold_auc=fold_auc,
        importances=importances,
        names=names,
        treatment_column=treatment_column,
        cv=cv,
        n_folds=n_folds,
        n_estimators=n_estimators,
        seed=seed,
        low=low,
        high=high,
        leakage_auc=leakage_auc,
        auc=float(roc_auc_score(y, propensity)),
    )
    return importances, diagnostics


def _load_for_overlap(
    panel: str | Path | pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    treatment_column: str,
    names: list[str],
    require_observed: bool,
) -> pd.DataFrame:
    wanted = ["PRODUCT_ID", "STORE_ID", "WEEK_NO", treatment_column, *names]
    if isinstance(panel, pd.DataFrame):
        missing = [c for c in wanted if c not in panel.columns]
        if missing:
            raise KeyError(f"not columns of the panel: {missing}")
        frame = panel[
            [*wanted, *(["treatment_observed"] if "treatment_observed" in panel else [])]
        ].copy()
    else:
        source = f"read_parquet('{Path(panel).as_posix()}')"
        available = {
            r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
        }
        missing = [c for c in wanted if c not in available]
        if missing:
            raise KeyError(f"not columns of the panel: {missing}")
        if "treatment_observed" in available:
            wanted.append("treatment_observed")
        frame = con.execute(f"SELECT {', '.join(wanted)} FROM {source}").df()

    if require_observed and "treatment_observed" in frame.columns:
        unobserved = int((~frame["treatment_observed"].astype(bool)).sum())
        if unobserved:
            raise UnobservedRowsError(
                f"{unobserved:,} rows have treatment_observed = False. Fitting a "
                f"propensity model on them treats unobserved as untreated."
            )

    frame["_treated"] = frame[treatment_column].astype(bool)
    return frame


def _diagnose_overlap(
    *,
    frame: pd.DataFrame,
    y: np.ndarray,
    propensity: np.ndarray,
    fold_auc: list[float],
    importances: pd.DataFrame,
    names: list[str],
    treatment_column: str,
    cv: str,
    n_folds: int,
    n_estimators: int,
    seed: int,
    low: float,
    high: float,
    leakage_auc: float,
    auc: float,
) -> dict[str, Any]:
    below = propensity < low
    above = propensity > high
    extreme = below | above
    top = importances.head(5)
    top_share = float(importances.iloc[0]["gain_share"]) if len(importances) else 0.0

    # A high AUC has two incompatible explanations and they need opposite
    # responses, so the diagnosis names which one the evidence points to rather
    # than reporting a number and leaving the reader to guess.
    if auc >= leakage_auc:
        diagnosis = (
            "LEAKAGE_SUSPECTED" if top_share >= 0.5 else "NON_OVERLAP_SUSPECTED"
        )
    elif auc >= 0.7:
        diagnosis = "SEPARABLE_BUT_PLAUSIBLE"
    else:
        diagnosis = "WELL_OVERLAPPED"

    constant = [c for c in names if frame[c].nunique(dropna=False) <= 1]

    return {
        "stage": "overlap",
        "treatment_column": treatment_column,
        "model": {
            "estimator": "LGBMClassifier",
            "max_bin": 63,
            "num_leaves": 31,
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "seed": seed,
        },
        "cv": {
            "scheme": cv,
            "folds": n_folds,
            "grouped_by": "PRODUCT_ID x STORE_ID" if cv == "group" else None,
            "why": (
                "Adjacent weeks of one product-store share their lagged units "
                "almost exactly. A random split puts near-copies on both sides "
                "and inflates the AUC; grouping keeps a product-store whole."
            ),
        },
        "rows": len(frame),
        "treated_rows": int(y.sum()),
        "treated_share": round(float(y.mean()), 6),
        "auc": round(auc, 6),
        "auc_by_fold": [round(a, 6) for a in fold_auc],
        "propensity_extremes": {
            "low": low,
            "high": high,
            "below_low": int(below.sum()),
            "below_low_share": round(float(below.mean()), 6),
            "above_high": int(above.sum()),
            "above_high_share": round(float(above.mean()), 6),
            "outside_share": round(float(extreme.mean()), 6),
            # Which side loses its counterpart matters: a treated row with no
            # untreated match cannot be estimated, an untreated row with no
            # treated match is simply an unused control.
            "treated_below_low": int((below & y).sum()),
            "untreated_above_high": int((above & ~y).sum()),
            "caveat": (
                f"These shares depend on the model as well as on the data. "
                f"{n_estimators} rounds at learning rate 0.05 shrink the logit "
                f"towards the base rate, so a longer or less regularised fit "
                f"would push more mass past both bounds even on identical "
                f"data. Read the count as a lower bound on how much of the "
                f"panel lacks a counterpart, and compare it across runs only "
                f"at fixed hyperparameters."
            ),
        },
        "covariates": names,
        "constant_covariates": constant,
        "top_features": [
            {"feature": r["feature"], "gain_share": round(float(r["gain_share"]), 6)}
            for _, r in top.iterrows()
        ],
        "top_feature_gain_share": round(top_share, 6),
        "diagnosis": diagnosis,
        "reading": (
            "A high AUC is ambiguous. Genuine non-overlap means treated and "
            "untreated are different populations and some rows have no "
            "counterfactual — a refusal. A leaked covariate means a feature "
            "encodes the treatment — a bug. One feature carrying most of the "
            "gain points to the second; gain spread across many points to the "
            "first."
        ),
        "identifiers_excluded": (
            "PRODUCT_ID, STORE_ID and WEEK_NO are not covariates by default. A "
            "tree given them memorises which products are displayed and returns "
            "a near-perfect AUC that says nothing about confounding."
        ),
    }


def collisions(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    treatment_column: str = "treated",
    secondary_column: str = "in_mailer",
    max_collision_share: float = MAX_COLLISION_SHARE,
    require_observed: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Where a second promotional mechanic fires alongside the treatment.

    Settled decision 4 makes `display` the treatment and keeps `mailer` as a
    covariate. That is a statement about identification, not about the world:
    the mailer still ran. Two consequences follow and this check measures both,
    because they break the estimate in different directions.

    **Treated rows carrying a mailer.** For those rows the measured effect is
    display *and* mailer jointly. Calling it a display effect overstates what
    display alone does, by however much the mailer contributed.

    **Untreated rows carrying a mailer.** These are worse and easier to miss.
    They sit in the control group and in the Phase 4 baseline's training set,
    yet they were promoted. A contaminated control has its units lifted, so the
    counterfactual is fitted too high and the estimated lift is biased
    *towards zero*.

    The second is the reason this check reports all four cells rather than the
    single collision count the plan asks for: a clean treated group against a
    promoted control group is not a clean comparison.

    Returns:
        `(cells, diagnostics)`. `cells` has one row per (treated, secondary)
        combination with rows, units and their shares.

    Raises:
        UnobservedRowsError: as in `variation_axes`.
        KeyError: either column is missing from the panel.
    """
    own = con is None
    con = connect() if con is None else con
    try:
        # `units` rides along as a requested column rather than a covariate:
        # this check fits nothing, it counts.
        frame = _load_for_overlap(
            panel, con, treatment_column, [secondary_column, "units"],
            require_observed,
        )
    finally:
        if own:
            con.close()

    frame["_secondary"] = frame[secondary_column].astype(bool)
    treated, secondary = frame["_treated"], frame["_secondary"]
    total_rows, total_units = len(frame), float(frame["units"].sum())

    cells = []
    for is_treated, is_secondary, label in (
        (True, True, "treated_with_secondary"),
        (True, False, "treated_clean"),
        (False, True, "control_with_secondary"),
        (False, False, "control_clean"),
    ):
        mask = (treated == is_treated) & (secondary == is_secondary)
        cells.append(
            {
                "cell": label,
                "treated": is_treated,
                "secondary": is_secondary,
                "rows": int(mask.sum()),
                "rows_share": round(float(mask.mean()), 6),
                "units": int(frame.loc[mask, "units"].sum()),
                "units_share": round(
                    float(frame.loc[mask, "units"].sum()) / total_units, 6
                )
                if total_units
                else 0.0,
            }
        )
    cells = pd.DataFrame(cells)

    n_treated = int(treated.sum())
    n_control = total_rows - n_treated
    collision = int((treated & secondary).sum())
    contaminated = int((~treated & secondary).sum())
    collision_share = collision / n_treated if n_treated else 0.0
    contaminated_share = contaminated / n_control if n_control else 0.0

    diagnostics = {
        "stage": "collisions",
        "treatment_column": treatment_column,
        "secondary_column": secondary_column,
        "rows": total_rows,
        "treated_rows": n_treated,
        "control_rows": n_control,
        "collision": {
            "rows": collision,
            "share_of_treated": round(collision_share, 6),
            "units": int(frame.loc[treated & secondary, "units"].sum()),
            "meaning": (
                f"On these rows the estimand is the joint effect of "
                f"{treatment_column} and {secondary_column}, not of "
                f"{treatment_column} alone."
            ),
        },
        "contaminated_controls": {
            "rows": contaminated,
            "share_of_controls": round(contaminated_share, 6),
            "units": int(frame.loc[~treated & secondary, "units"].sum()),
            "meaning": (
                f"These rows are untreated by the {treatment_column} definition "
                f"but were promoted through {secondary_column}. They enter the "
                f"Phase 4 baseline as controls, lifting the counterfactual and "
                f"biasing the measured effect towards zero."
            ),
        },
        "threshold": max_collision_share,
        "status": (
            "OVERLAPPING_TREATMENTS"
            if max(collision_share, contaminated_share) > max_collision_share
            else "SEPARABLE"
        ),
        "reading": (
            "Both figures matter and they push opposite ways. Collisions "
            "inflate the effect by crediting display with the mailer's work; "
            "contaminated controls deflate it by raising the baseline. They do "
            "not cancel, because they act on different rows."
        ),
        "remedies": [
            (
                f"Restrict to rows where {secondary_column} is false, which "
                f"estimates a clean {treatment_column} effect on a smaller "
                f"panel."
            ),
            (
                f"Keep {secondary_column} as a covariate so the model can "
                f"separate them, which is what settled decision 4 assumes and "
                f"what the Phase 2.6 feature set does not currently include."
            ),
            "Report the joint effect and label it as joint.",
        ],
    }
    return cells, diagnostics


def horizon_check(
    campaigns: pd.DataFrame,
    cycles: str | Path | pd.DataFrame = "data/interim/repurchase_cycles.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    commodity_column: str = "COMMODITY_DESC",
    horizon_column: str = "horizon_weeks",
    required_column: str = "horizon_weeks",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Is each campaign's measurement window long enough for its commodity?

    A window that closes before the category's repurchase cycle has run banks
    the promotional peak and never sees the trough that follows it. Task 1.5
    fixed the rule: the measurement window extends past campaign end by at
    least the commodity's cycle.

    Campaigns are **supplied**, not derived. Phase 3's job is to give a verdict
    on a proposed campaign, so what constitutes one belongs to the caller; this
    check takes whatever it is given and compares it against the cycles Task 2.7
    recorded.

    Args:
        campaigns: one row per campaign, carrying `commodity_column` and
            `horizon_column` — the number of weeks the window stays open
            **after campaign end**.
        cycles: the Task 2.7 repurchase-cycle table.
        commodity_column: the campaign column naming the commodity.
        horizon_column: the campaign column holding the proposed horizon.
        required_column: the cycles column holding the required weeks.

    Returns:
        `(checked, diagnostics)`. `checked` is `campaigns` with `required_weeks`,
        `shortfall_weeks`, `low_support` and `status` added.

    Raises:
        KeyError: a required column is missing from either frame.
    """
    for column in (commodity_column, horizon_column):
        if column not in campaigns.columns:
            raise KeyError(f"{column!r} is not a column of campaigns")

    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(cycles, pd.DataFrame):
            table = cycles
        else:
            table = con.execute(
                f"SELECT * FROM read_parquet('{Path(cycles).as_posix()}')"
            ).df()
    finally:
        if own:
            con.close()

    for column in (commodity_column, required_column):
        if column not in table.columns:
            raise KeyError(f"{column!r} is not a column of the cycles table")

    keep = [commodity_column, required_column]
    if "low_support" in table.columns:
        keep.append("low_support")
    lookup = table[keep].rename(columns={required_column: "required_weeks"})

    checked = campaigns.merge(
        lookup, on=commodity_column, how="left", validate="many_to_one"
    )
    if "low_support" not in checked.columns:
        checked["low_support"] = False
    checked["low_support"] = checked["low_support"].fillna(False).astype(bool)

    required = pd.to_numeric(checked["required_weeks"], errors="coerce")
    proposed = pd.to_numeric(checked[horizon_column], errors="coerce")
    checked["shortfall_weeks"] = (required - proposed).where(required > proposed)

    # A commodity with no recorded cycle cannot be checked. That is not a pass:
    # an unknown requirement is reported as unknown so it cannot be read as
    # "long enough".
    status = pd.Series("OK", index=checked.index, dtype="string")
    status = status.mask(required > proposed, "HORIZON_TOO_SHORT")
    status = status.mask(required.isna() | proposed.isna(), "UNKNOWN_CYCLE")
    checked["status"] = status.astype("string")

    counts = checked["status"].value_counts()
    too_short = checked[checked["status"] == "HORIZON_TOO_SHORT"]
    diagnostics = {
        "stage": "horizon_check",
        "campaigns": len(checked),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "too_short": len(too_short),
        "unknown_cycle": int((checked["status"] == "UNKNOWN_CYCLE").sum()),
        "on_a_low_support_cycle": int(
            (checked["low_support"] & (checked["status"] != "UNKNOWN_CYCLE")).sum()
        ),
        "max_shortfall_weeks": (
            int(too_short["shortfall_weeks"].max()) if len(too_short) else 0
        ),
        "rule": (
            "The measurement window must extend past campaign end by at least "
            "the commodity's repurchase cycle, or the estimate banks the peak "
            "and never sees the trough."
        ),
        "floors_not_estimates": (
            "Task 2.7 established every recorded cycle is a floor: pairs buying "
            "in exactly one week contribute no gap and are the slowest buyers, "
            "and gaps are right-censored by the 102-week window. Clearing the "
            "requirement is therefore necessary and not sufficient."
        ),
        "unknown_is_not_a_pass": (
            "A commodity with no recorded cycle returns UNKNOWN_CYCLE, never "
            "OK. low_support cycles are checked but flagged: a median over a "
            "handful of gaps is a weak requirement, not a safe one."
        ),
    }
    return checked, diagnostics


def kappa_star(depth: float, margin: float | None) -> KappaResult:
    """The incremental share a promotion needs to break even.

    `kappa_star = depth / margin`. The derivation is short and the result is
    cleaner than it looks, which is worth showing because the cancellation is
    not obvious. For promoted units `Q` at regular price `p`, depth `d` and
    gross margin rate `m`:

    - a promoted unit earns `p(1-d) - p(1-m) = p(m-d)`;
    - an incremental share `k` earns that on `kQ` units;
    - the `(1-k)Q` units that would have sold anyway each lose `p·d`;
    - break-even is `k·p(m-d) = (1-k)·p·d`, and both the price and the `kd`
      terms cancel, leaving `k·m = d`.

    So the required incremental share is depth over margin, exactly, with no
    dependence on price or volume.

    **`kappa_star > 1` is arithmetically impossible**, not merely unlikely: it
    asks for more incremental units than were sold. Since `k <= 1` iff
    `m >= d`, the depth *is* the minimum margin at which the promotion can
    break even at all — which is Task 5.2's `m_star` in the case where
    promotional cost is subsidy only.

    Args:
        depth: discount depth in `[0, 1]`, from Task 2.3.
        margin: gross margin rate in `(0, 1]`. **Required and user-supplied.**
            This dataset has no COGS — `promo/io.py` establishes it by searching
            all 46 column names — so `None` is the honest default and returns
            no number.

    Returns:
        A `KappaResult`. When `margin is None`, `kappa` is None and
        `reason_code` is `NO_MARGIN`.

    Raises:
        ValueError: depth outside `[0, 1]`, or a supplied margin outside
            `(0, 1]`.
    """
    if math.isnan(depth) or not (0.0 <= depth <= 1.0):
        raise ValueError(f"depth must be in [0, 1], got {depth!r}")

    if margin is None:
        return KappaResult(
            depth=depth,
            margin=None,
            kappa=None,
            feasible=None,
            reason_code="NO_MARGIN",
            detail=(
                "No margin was supplied and none can be derived: this dataset "
                "has no COGS column. The margin sweep is the answer instead — "
                "it needs no margin because it varies one."
            ),
        )

    if math.isnan(margin) or not (0.0 < margin <= 1.0):
        raise ValueError(f"margin must be in (0, 1], got {margin!r}")

    kappa = depth / margin
    feasible = kappa <= 1.0
    return KappaResult(
        depth=depth,
        margin=margin,
        kappa=kappa,
        feasible=feasible,
        reason_code=None if feasible else "KAPPA_IMPOSSIBLE",
        detail=(
            f"{kappa:.1%} of promoted units must be incremental to break even "
            f"at a {margin:.0%} margin."
            if feasible
            else (
                f"Breaking even needs {kappa:.1%} of promoted units to be "
                f"incremental, which exceeds 100%. At a {margin:.0%} margin a "
                f"{depth:.1%} discount cannot pay for itself at any volume."
            )
        ),
    )


def margin_sweep(
    depth: float,
    margins: tuple[float, ...] | list[float] = MARGIN_GRID,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Required incremental share across the assumed-margin grid.

    This is the honest MVP 03 answer on a dataset with no COGS. It needs no
    margin because it varies one: a merchant who knows theirs reads their own
    row off the table.

    Returns:
        `(sweep, diagnostics)`. `sweep` has one row per margin with the required
        incremental share and whether it is achievable.

    Raises:
        ValueError: depth outside `[0, 1]`, or any margin outside `(0, 1]`.
    """
    rows = []
    for margin in margins:
        result = kappa_star(depth, margin)
        rows.append(
            {
                "margin": margin,
                "kappa_star": result.kappa,
                "feasible": result.feasible,
                "reason_code": result.reason_code,
            }
        )
    sweep = pd.DataFrame(rows)

    feasible = sweep[sweep["feasible"]]
    # k <= 1 iff m >= d, so the depth itself is the minimum margin at which the
    # promotion can break even. Reported as a number rather than left implicit
    # in the table.
    minimum_margin = depth
    return sweep, {
        "stage": "margin_sweep",
        "depth": depth,
        "grid": list(margins),
        "grid_source": (
            "Task 5.2 is the authority on this grid: 10% to 50% in 5-point "
            "steps. MARGIN_GRID is the single definition; promo/accounting.py "
            "must import it rather than restate it, or the gate's sweep and the "
            "accounting table can silently disagree."
        ),
        "minimum_viable_margin": round(minimum_margin, 6),
        "feasible_margins": [round(float(m), 6) for m in feasible["margin"]],
        "infeasible_margins": [
            round(float(m), 6) for m in sweep.loc[~sweep["feasible"], "margin"]
        ],
        "feasible_at_any_grid_margin": bool(len(feasible)),
        "headline": (
            f"A {depth:.1%} discount needs a gross margin of at least "
            f"{minimum_margin:.1%} before it can break even at any volume."
        ),
        "reason_code": None if len(feasible) else "KAPPA_IMPOSSIBLE",
        "reading": (
            "Read the row for the margin you actually run, not the table's "
            "average. The pipeline cannot know your margin and does not guess: "
            "no margin is imputed anywhere."
        ),
        "identity": (
            "kappa_star(m) = m_star / m, where m_star is Task 5.2's break-even "
            "margin. Here m_star is the depth, because this ratio counts the "
            "discount subsidy only. Task 5.1's free goods are a second cost "
            "component and raise m_star above the depth, so this sweep is "
            "optimistic wherever free goods were given away."
        ),
    }


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
