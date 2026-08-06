"""The counterfactual baseline: what a product-store-week would have sold.

Task 4.1. This module fits the model that answers *what would this row have done
without the display*, and nothing else. It does not roll a counterfactual
forward over a window (Task 4.2) and it does not turn residuals into a lift
(Task 4.3).

Four properties of the fit are load-bearing.

**No treated row reaches the fit.** `.claude/rules/causal-inference.md` requires
that a caller handing over a frame containing treated rows gets a
`TreatedRowsError`, not a quietly filtered frame — a silent filter hides the
caller's bug and the model still returns a number. When this module builds the
training frame itself from a parquet path, the selection is explicit
(`control_rows()`), and it records rows, units, and sales value before and after
each exclusion.

**`in_mailer` is a covariate, and that is settled decision 8.** 13.07% of the
control rows in the estimation window carry a mailer. They were promoted; if the
model is told nothing about it, it learns an inflated normal, the counterfactual
is fitted too high, and **every measured display effect is biased towards zero**.
The no-promotion-features rule exists to stop the counterfactual seeing *the
promotion being measured*; `in_mailer` is a different mechanic, is not derived
from `display`, is known before the display decision, and is not a
post-treatment aggregate. Conditioning on a concurrent treatment is what avoids
omitted-variable bias. Depth, `on_deal`, `price_status` and the discount columns
stay out — `FORBIDDEN_FEATURES` enforces that, so the exception is exactly one
flag wide.

**The three mediator features are out.** `n_stores_carrying`,
`category_units_ex_focal` and `store_traffic` are measured in week *w* and can
be moved by the promotion itself, so conditioning on them absorbs part of the
effect being measured — Task 2.6 flagged them and Task 3.2 deferred the
question. Task 4.4 settled it by recovery: dropping them roughly halves the
bias at a true effect of zero. See settled decision 13 for why the first
measurement said the opposite. `fit_baseline(features=BASELINE_FEATURES +
MEDIATOR_FEATURES)` puts them back.

**The outcome is fitted directly, not through a log.** The obvious target for a
count is `log1p(units)` inverted with `expm1`, and it is wrong on a panel with
mass at zero: `expm1` of a conditional mean in log space lands nearer the
conditional *median*, understating the counterfactual, and the rollout feeds
that understatement back until it compounds. Task 4.4 measured a 17% one-step
shortfall at 10% no-sale weeks, decaying to 55% by the seventh step, and a true
effect of zero coming back as +0.28 of counterfactual units. A Poisson
objective on raw units predicts the conditional mean directly — nothing to
invert, nothing to correct — and brings that to +0.017. See `TARGETS` for the
full comparison and for the two rejected alternatives, both kept runnable.

**A feature's missingness may not be the outcome either.** Phase 2.6 asserts
that no lag uses the current week's outcome, but it checks lag *values*, not
column *availability*. `price_rel_category` is null on exactly the 2,046,518
control rows in the estimation window where `units == 0` — the ratio exists only
where a sale set a price — and it took 94.3% of the gain in the first fit of
this module before that was caught. The model was reading the answer off the
null. The raw column is in `FORBIDDEN_FEATURES`, `add_price_history()` supplies
`price_rel_category_lag` in its place, and `missingness_coupling()` measures
every feature so the next column with this shape raises `OutcomeLeakError`
rather than quietly topping the importance table.

**What the covariate does not fix is reported, not papered over.** 39.49% of
in-window treated rows carry both mechanics. For those, holding `in_mailer`
fixed at True holds an interaction fixed rather than removing it, so they
identify a *joint* display-and-mailer effect. `mechanic_strata()` labels every
treated row so Tasks 4.3 and 4.5 can report the two groups separately; pooling
them into one "display effect" is the error this exists to prevent.

**This module returns a model, not a DataFrame.** Every other stage function
returns `(DataFrame, dict)`; the artefact of a fit is a fitted model, and
pretending otherwise would mean stuffing four boosters into a diagnostics dict.
`fit_baseline()` returns `(BaselineModel, dict)`, and `BaselineModel.importances()`
supplies the DataFrame view. `promo/io.py` records its own deviation from the
convention the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from promo.features import CONTEMPORANEOUS_FEATURES, LAGGED_FEATURES
from promo.io import connect

__all__ = [
    "BASELINE_FEATURES",
    "DEFAULT_TARGET",
    "DERIVED_FROM",
    "ESTIMATION_WINDOW",
    "FORBIDDEN_FEATURES",
    "IDENTITY_FEATURES",
    "MEDIATOR_FEATURES",
    "MISSINGNESS_LEAK_THRESHOLD",
    "QUANTILES",
    "RECURSIVE_FEATURES",
    "TARGETS",
    "TWEEDIE_VARIANCE_POWER",
    "BaselineModel",
    "ForbiddenFeatureError",
    "NonContiguousWeeksError",
    "OutcomeLeakError",
    "TreatedRowsError",
    "add_price_history",
    "control_rows",
    "fit_baseline",
    "mechanic_strata",
    "missingness_coupling",
    "rollout",
    "write_diagnostics",
]

#: The three week-*w* features the promotion can itself move — settled decision
#: 13. Excluded from `BASELINE_FEATURES`; `fit_baseline(features=...)` can put
#: them back, which is how the two-by-two that settled it stays runnable.
MEDIATOR_FEATURES: tuple[str, ...] = (
    "n_stores_carrying",
    "category_units_ex_focal",
    "store_traffic",
)

#: The Phase 2.6 feature set plus `in_mailer` (settled decision 8), with
#: `price_rel_category` replaced by its carried-forward form (decision 9) and
#: the mediator block removed (decision 13) — twelve columns. The Phase 2 names
#: are imported rather than retyped so a change there cannot silently leave the
#: baseline fitting on a stale list.
BASELINE_FEATURES: tuple[str, ...] = (
    *LAGGED_FEATURES,
    *(
        "price_rel_category_lag" if c == "price_rel_category" else c
        for c in CONTEMPORANEOUS_FEATURES
        if c not in MEDIATOR_FEATURES
    ),
    "in_mailer",
)

#: Features this module derives from a panel column, rather than reading. The
#: only entry exists because `price_rel_category` is null on exactly the rows
#: where `units == 0` — see `add_price_history` and `FORBIDDEN_FEATURES`.
DERIVED_FROM: dict[str, str] = {"price_rel_category_lag": "price_rel_category"}

#: Appended when `include_identity=True`, as LightGBM categoricals. Off by
#: default: the lags already carry the level, and a fit keyed on 300 particular
#: product IDs does not transfer to a product outside the scope.
IDENTITY_FEATURES: tuple[str, ...] = ("PRODUCT_ID", "STORE_ID", "COMMODITY_DESC")

#: Columns that may never be features. `on_display`/`treated` are the treatment
#: itself; `depth`, `price_status` and `paid_price` are derived from the deal;
#: `units` and `sales_value` are the outcome. Decision 8 admits `in_mailer` and
#: nothing else, and this is where that is enforced rather than remembered.
FORBIDDEN_FEATURES: tuple[str, ...] = (
    "on_display",
    "treated",
    "display_code",
    "depth",
    "price_status",
    "paid_price",
    "on_deal",
    "units",
    "sales_value",
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
    # Measured on the real panel, weeks 18-101, control rows: price_rel_category
    # is null on 2,046,518 rows and units == 0 on exactly those same rows, with
    # no exception in either direction. Its *missingness pattern is the
    # outcome*, so a model given it reads the answer instead of predicting it —
    # it took 94.3% of the gain before this was caught. Use
    # `price_rel_category_lag`, which carries the last price observed strictly
    # before week w.
    "price_rel_category",
    "regular_price",
    "real_regular_price",
)

#: Above this, a feature's nullity and `units == 0` determine each other closely
#: enough to call it outcome leakage rather than coincidence. Applied in both
#: directions, so a merely rare null cannot trip it.
MISSINGNESS_LEAK_THRESHOLD: float = 0.99

#: Quantile variants, per the task prompt. Fitted independently of each other
#: and of the mean model.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

#: Settled decision 5. Weeks 9-17 of the panel are the recruitment ramp, where a
#: level comparison compares panel size rather than demand.
ESTIMATION_WINDOW: tuple[int, int] = (18, 101)

#: Boolean-valued features LightGBM must see as 0/1 rather than object.
_BOOL_FEATURES: tuple[str, ...] = ("in_mailer", "is_holiday_week")

_KEY = ("PRODUCT_ID", "STORE_ID", "WEEK_NO")


class TreatedRowsError(Exception):
    """Treated rows reached the training frame.

    Raised, never filtered around. A caller that passes promoted rows to the
    baseline has a bug; filtering them out here would fit a correct model on a
    frame the caller does not think it supplied, and the bug would surface as a
    wrong number somewhere else.
    """


class ForbiddenFeatureError(Exception):
    """A requested feature is derived from the treatment or from the outcome."""


class OutcomeLeakError(Exception):
    """A feature's missingness pattern determines the outcome.

    Raised rather than repaired. A column that is null on exactly the rows where
    `units == 0` hands the model the answer through its own absence, and no
    amount of imputation after the fact makes the resulting fit a counterfactual.
    """


#: The window that defines `price_rel_category_lag`: the most recent
#: `price_rel_category` observed **strictly before** week w, within the
#: product-store. Written once and used for both the parquet and the DataFrame
#: path, so there is a single definition of the feature.
_PRICE_HISTORY_SQL = """
SELECT *,
       LAST_VALUE(price_rel_category IGNORE NULLS) OVER (
           PARTITION BY PRODUCT_ID, STORE_ID ORDER BY WEEK_NO
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
       ) AS price_rel_category_lag
FROM {src}
"""


def add_price_history(
    frame: pd.DataFrame, *, con: duckdb.DuckDBPyConnection | None = None
) -> pd.DataFrame:
    """Add `price_rel_category_lag`: the last price ratio seen before week w.

    `price_rel_category` itself may never be a feature. On the real panel it is
    null on 2,046,518 control rows in the estimation window and `units == 0` on
    exactly those rows — the column is observed if and only if the product sold,
    so its missingness *is* the outcome. Phase 2.6 left it null deliberately and
    said a filled version, if Phase 4 wanted one, should be a separate named
    column. This is that column.

    Carrying the last observed value forward uses only weeks strictly before w,
    which puts it in the same class as `units_lag_1`. It is still a *regular*
    price ratio, never a paid price, so it does not encode the deal.

    **Apply this to the whole panel before any row filter.** Run on a
    control-only frame, "the previous week" silently means "the previous
    unpromoted week", and the feature changes meaning depending on what the
    caller happened to filter first.
    """
    if "price_rel_category" not in frame.columns:
        raise KeyError(
            "price_rel_category is not a column of the frame, so "
            "price_rel_category_lag cannot be derived from it"
        )
    own = con is None
    con = connect() if con is None else con
    try:
        con.register("_price_history_frame", frame)
        return con.execute(
            _PRICE_HISTORY_SQL.format(src="_price_history_frame")
            + " ORDER BY PRODUCT_ID, STORE_ID, WEEK_NO"
        ).df()
    finally:
        con.unregister("_price_history_frame")
        if own:
            con.close()


#: Features the rollout recomputes at every step from the counterfactual series
#: rather than reading. Exactly the Phase 2.6 lagged block: every one of them is
#: a function of past units, and inside a counterfactual window "past units"
#: means the counterfactual's own path.
RECURSIVE_FEATURES: tuple[str, ...] = LAGGED_FEATURES


#: Features the rollout holds at their last pre-window value **even when the
#: caller supplies them in `exog_weeks`**. `price_rel_category_lag` is here
#: because its in-window value depends on whether the product sold, and whether
#: it sold is what the counterfactual exists to say — settled decision 9. The
#: natural way to call `rollout` is to hand it a slice of the panel, which
#: carries the observed column; without this the risky path would be the
#: default and nothing would say so. Pass `carry=()` to opt out deliberately.
CARRIED_BY_DEFAULT: tuple[str, ...] = ("price_rel_category_lag",)


class NonContiguousWeeksError(Exception):
    """The rollout weeks do not continue the history week by week.

    Raised rather than patched. A lag is defined by arithmetic on `WEEK_NO`, so
    a gap silently turns `units_lag_1` into "the last week we happened to have",
    which is a different variable in every row — the same failure Phase 2.6
    built explicit zeros to avoid.
    """


def missingness_coupling(
    frame: pd.DataFrame,
    features: tuple[str, ...] | list[str],
    *,
    outcome: str = "units",
    threshold: float = MISSINGNESS_LEAK_THRESHOLD,
) -> tuple[dict[str, Any], list[str]]:
    """How closely each feature's nullity and a zero outcome imply each other.

    Two conditional probabilities per feature: P(outcome == 0 | feature null)
    and P(feature null | outcome == 0). A feature that scores high on **both** is
    observed if and only if the row sold, which makes its absence a readout of
    the outcome rather than a gap in a covariate.

    Returns:
        `(measurements, leaking)` — every feature that has nulls, and the names
        that breach the threshold in both directions.
    """
    zero = (frame[outcome] == 0).to_numpy()
    n_zero = int(zero.sum())
    measurements: dict[str, Any] = {}
    leaking: list[str] = []
    for column in features:
        null = frame[column].isna().to_numpy()
        n_null = int(null.sum())
        if not n_null:
            continue
        both = int((null & zero).sum())
        zero_given_null = both / n_null
        null_given_zero = both / n_zero if n_zero else 0.0
        measurements[column] = {
            "null_rows": n_null,
            "null_share": round(float(n_null / len(frame)), 6),
            "p_zero_given_null": round(float(zero_given_null), 6),
            "p_null_given_zero": round(float(null_given_zero), 6),
        }
        if zero_given_null >= threshold and null_given_zero >= threshold:
            leaking.append(column)
    return measurements, leaking


#: What the mean model is fitted on, and therefore how its output becomes units.
#:
#: - `log1p` — regression on `log1p(units)`, inverted with `expm1`. **Biased low
#:   on a zero-inflated outcome**: `expm1` of a conditional mean in log space is
#:   nearer the conditional median than the conditional mean, and the gap grows
#:   with the mass at zero. Task 4.4 measured it at 17% one-step on a panel with
#:   10% no-sale weeks, compounding to 55% by the seventh rollout step.
#: - `log1p_smearing` — the same fit, inverted with Duan's smearing factor
#:   estimated from the training residuals, which targets the conditional mean.
#: - `tweedie`, `poisson` — fitted on raw units with a log link. The booster's
#:   output *is* the conditional mean, so there is no retransformation to correct
#:   and zero-inflation is handled by the likelihood rather than around it.
#:
#: **`poisson` is the default**, on the Task 4.4 measurement rather than on
#: taste. Recovering a true effect of zero, bias as a share of counterfactual
#: units, three seeds per cell:
#:
#: | no-sale rate |  0% |  5% |   10% |   20% |
#: |--------------|----:|----:|------:|------:|
#: | log1p        |-.029|.122 | **.277** | **.444** |
#: | log1p_smearing|-.052|-.025| .041  | .099  |
#: | tweedie      |-.058|-.021| .043  | .085  |
#: | poisson      |-.050|-.022| **.017** | **.058** |
#:
#: Both fixes work; the one that never transforms wins at the sparsity that
#: matters, and the real panel is 87% zero rows. `log1p` is kept so the
#: comparison stays reproducible and so the defect it carries stays visible.
TARGETS: tuple[str, ...] = ("log1p", "log1p_smearing", "tweedie", "poisson")

#: The target the pipeline fits with unless a caller says otherwise.
DEFAULT_TARGET: str = "poisson"

#: Targets whose booster output is already on the units scale.
_DIRECT_TARGETS: frozenset[str] = frozenset({"tweedie", "poisson"})

#: Compound Poisson-gamma. Between 1 and 2: a point mass at zero with a
#: continuous positive part, which is the shape of a product-store-week.
TWEEDIE_VARIANCE_POWER: float = 1.3


def _quantile_key(alpha: float) -> str:
    return f"q{round(alpha * 100):02d}"


@dataclass
class BaselineModel:
    """The fitted counterfactual: a mean model plus its quantile variants.

    `boosters` is keyed `"mean"` and `"q10"`/`"q50"`/`"q90"`. The quantile models
    are separate fits, not a decomposition of the mean model, so their
    predictions can cross; `fit_baseline` measures how often and records it
    rather than sorting the crossings away.
    """

    boosters: dict[str, Any]
    features: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    train_window: tuple[int, int] = ESTIMATION_WINDOW
    n_train_rows: int = 0
    seed: int = 0
    #: Categories seen at fit time, per categorical feature, so `load()` can put
    #: the same encoding back. Empty when `include_identity=False`.
    categories: dict[str, list[Any]] = field(default_factory=dict)
    #: One of `TARGETS`. Decides how the booster's output becomes units.
    target: str = "log1p"
    #: Duan's smearing factor, `mean(exp(training residual))` in log space. Used
    #: only by `log1p_smearing`, and only for the mean — see `predict`.
    smearing_factor: float = 1.0

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The model matrix for `frame`, in the order the boosters were fitted.

        Raises:
            KeyError: a feature is absent. Named rather than dropped: a rollout
                that quietly loses a column still returns a plausible number.
        """
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(
                f"not columns of the frame: {missing}. The baseline was fitted "
                f"on {list(self.features)}."
            )
        return _prepare(frame[list(self.features)], self.categorical, self.categories)

    def predict_raw(
        self, frame: pd.DataFrame, quantile: float | None = None
    ) -> np.ndarray:
        """The booster's own output, before any inversion.

        `log1p(units)` under the log1p targets; units under `tweedie` and
        `poisson`, whose log link is applied inside LightGBM.
        """
        booster = self.boosters[
            "mean" if quantile is None else _quantile_key(quantile)
        ]
        return np.asarray(booster.predict(self.design(frame)), dtype="float64")

    def predict(
        self, frame: pd.DataFrame, quantile: float | None = None
    ) -> np.ndarray:
        """Counterfactual units, floored at zero.

        How the booster's output becomes units depends on `target`:

        - `tweedie` / `poisson` — it already is the conditional mean.
        - `log1p` — `expm1`, which returns something closer to the conditional
          median than the mean once the outcome has mass at zero.
        - `log1p_smearing` — `exp(raw) * S - 1` with Duan's smearing factor `S`,
          which targets the conditional mean. **The correction applies to the
          mean only.** Quantiles are equivariant under a monotone transform, so
          `expm1` of a log-scale quantile already *is* that quantile of units;
          multiplying it by `S` would inflate the band for no reason.

        The floor is the only repair in this module and it is a range
        constraint, not an imputation: a negative unit count is not a quantity.
        """
        raw = self.predict_raw(frame, quantile)
        if self.target in _DIRECT_TARGETS:
            units = raw
        elif self.target == "log1p_smearing" and quantile is None:
            units = np.exp(raw) * self.smearing_factor - 1.0
        else:
            units = np.expm1(raw)
        return np.clip(units, 0.0, None)

    def importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """Every feature's importance in the mean model, ranked."""
        booster = self.boosters["mean"]
        gain = np.asarray(
            booster.feature_importance(importance_type=importance_type),
            dtype="float64",
        )
        total = gain.sum()
        return (
            pd.DataFrame({"feature": list(self.features), importance_type: gain})
            .assign(
                gain_share=lambda d: d[importance_type] / total if total else 0.0
            )
            .sort_values(importance_type, ascending=False)
            .reset_index(drop=True)
        )

    def save(self, out_dir: str | Path) -> Path:
        """Write each booster as LightGBM text plus a metadata JSON."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, booster in self.boosters.items():
            booster.save_model(str(path / f"{name}.txt"))
        meta = {
            "features": list(self.features),
            "categorical": list(self.categorical),
            "categories": {k: list(v) for k, v in self.categories.items()},
            "params": self.params,
            "train_window": list(self.train_window),
            "n_train_rows": self.n_train_rows,
            "seed": self.seed,
            "boosters": sorted(self.boosters),
            "target": self.target,
            "smearing_factor": self.smearing_factor,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> BaselineModel:
        """Read back what `save()` wrote."""
        import lightgbm as lgb

        path = Path(out_dir)
        meta = json.loads((path / "meta.json").read_text())
        boosters = {
            name: lgb.Booster(model_file=str(path / f"{name}.txt"))
            for name in meta["boosters"]
        }
        return cls(
            boosters=boosters,
            features=tuple(meta["features"]),
            categorical=tuple(meta["categorical"]),
            params=meta["params"],
            train_window=tuple(meta["train_window"]),  # type: ignore[arg-type]
            n_train_rows=meta["n_train_rows"],
            seed=meta["seed"],
            categories={k: list(v) for k, v in meta.get("categories", {}).items()},
            target=meta.get("target", "log1p"),
            smearing_factor=meta.get("smearing_factor", 1.0),
        )


def _prepare(
    frame: pd.DataFrame,
    categorical: tuple[str, ...],
    categories: dict[str, list[Any]],
) -> pd.DataFrame:
    """Cast a feature frame to the dtypes LightGBM was fitted on.

    float32 per CLAUDE.md, booleans to int8, and categoricals pinned to the
    categories seen at fit time so an unseen product at predict time becomes a
    null the model can split on rather than a silently renumbered code.
    """
    out = frame.copy()
    for column in out.columns:
        if column in categorical:
            out[column] = pd.Categorical(
                out[column], categories=categories.get(column)
            )
        elif column in _BOOL_FEATURES:
            out[column] = out[column].fillna(False).astype("int8")
        else:
            out[column] = out[column].astype("float32")
    return out


def _resolve_features(
    features: tuple[str, ...] | list[str] | None, include_identity: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = tuple(BASELINE_FEATURES if features is None else features)
    identity = tuple(IDENTITY_FEATURES) if include_identity else ()
    if include_identity:
        names = (*names, *(c for c in identity if c not in names))

    banned = [c for c in names if c in FORBIDDEN_FEATURES]
    if banned:
        raise ForbiddenFeatureError(
            f"{banned} may not be baseline features: they are the treatment "
            f"itself, derived from the deal, or the outcome. Settled decision 8 "
            f"admits in_mailer and nothing else."
        )
    return names, identity


def _require_untreated(frame: pd.DataFrame, treatment_column: str) -> int:
    """Raise if any treated row is present. Never filters."""
    if treatment_column not in frame.columns:
        raise KeyError(
            f"{treatment_column!r} is not a column of the frame, so it cannot be "
            f"shown that no treated row reached the fit. Pass the column."
        )
    treated = frame[treatment_column].astype(bool)
    n = int(treated.sum())
    if n:
        raise TreatedRowsError(
            f"{n:,} of {len(frame):,} rows have {treatment_column} = True. The "
            f"baseline trains on control rows only and will not filter them out "
            f"for you — a silent filter would hide the caller bug that produced "
            f"this frame."
        )
    return n


def control_rows(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    features: tuple[str, ...] | list[str] = BASELINE_FEATURES,
    treatment_column: str = "treated",
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The training frame, with both exclusions recorded.

    Two rows are dropped from the panel here and each is counted: treated rows,
    which the baseline may never see, and rows outside the estimation window,
    which settled decision 5 puts at weeks 18-101 because the household panel is
    still recruiting before week 18 and a level comparison against those weeks
    compares panel size rather than demand.

    Args:
        panel: the Task 2.6 modelling panel, as a path or a DataFrame. **A
            DataFrame is never filtered** — it is checked, and a treated row
            raises. Only a path lets this function do the selection.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        features: columns the fit will need, so only those are read.
        treatment_column: the boolean that must be False on every training row.
        week_range: inclusive `(first, last)` week, or None for the whole panel.

    Returns:
        `(frame, diagnostics)`. The diagnostics carry rows, units, and sales
        value before and after each exclusion.

    Raises:
        TreatedRowsError: `panel` is a DataFrame containing treated rows.
        KeyError: a requested column is absent.
    """
    derived = [c for c in features if c in DERIVED_FROM]
    wanted = [*_KEY, "units", "sales_value", treatment_column, "in_mailer"]
    wanted += [c for c in features if c not in wanted and c not in DERIVED_FROM]
    # The source column of a derived feature is read but never fitted on.
    wanted += [
        DERIVED_FROM[c] for c in derived if DERIVED_FROM[c] not in wanted
    ]

    if isinstance(panel, pd.DataFrame):
        missing = [c for c in wanted if c not in panel.columns]
        if missing:
            raise KeyError(f"not columns of the panel: {missing}")
        source = panel[wanted].copy()
        totals = _totals(source)
        # Derived before any filter: the window that defines
        # price_rel_category_lag must see every week, including the promoted
        # ones, or "the previous week" quietly becomes "the previous
        # unpromoted week".
        if derived:
            source = add_price_history(source, con=con)
        # Checked, not filtered: see the module docstring.
        _require_untreated(source, treatment_column)
        after_treated = _totals(source)
        if week_range is not None:
            inside = source["WEEK_NO"].between(week_range[0], week_range[1])
            source = source.loc[inside].reset_index(drop=True)
        frame = source
        selection = "caller-supplied DataFrame, checked and never filtered"
    else:
        own = con is None
        con = connect() if con is None else con
        try:
            src = f"read_parquet('{Path(panel).as_posix()}')"
            available = {
                r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
            }
            missing = [c for c in wanted if c not in available]
            if missing:
                raise KeyError(f"not columns of the panel: {missing}")
            cols = ", ".join(f'"{c}"' for c in wanted)
            totals = _sql_totals(con, src, "TRUE", treatment_column)
            after_treated = _sql_totals(
                con, src, f"NOT {treatment_column}", treatment_column
            )
            where = f"NOT {treatment_column}"
            if week_range is not None:
                where += f" AND WEEK_NO BETWEEN {week_range[0]} AND {week_range[1]}"
            # The window runs in the inner query, over the whole panel, so the
            # row filter cannot change what "the previous week" means.
            inner = (
                _PRICE_HISTORY_SQL.format(src=f"(SELECT {cols} FROM {src})")
                if derived
                else f"SELECT {cols} FROM {src}"
            )
            select = ", ".join(f'"{c}"' for c in [*wanted, *derived])
            frame = con.execute(
                f"SELECT {select} FROM ({inner}) WHERE {where} "
                f"ORDER BY PRODUCT_ID, STORE_ID, WEEK_NO"
            ).df()
        finally:
            if own:
                con.close()
        selection = f"WHERE NOT {treatment_column}, in DuckDB"
        # Belt and braces: the SQL selected control rows, and this proves it.
        _require_untreated(frame, treatment_column)

    after_window = _totals(frame)
    mailer = frame["in_mailer"].astype(bool)

    diagnostics = {
        "stage": "control_rows",
        "selection": selection,
        "treatment_column": treatment_column,
        "derived_features": {
            c: {
                "from": DERIVED_FROM[c],
                "how": (
                    "last value observed strictly before week w, within "
                    "product-store, computed over the whole panel before any "
                    "row filter"
                ),
                "null_share": round(float(frame[c].isna().mean()), 6),
            }
            for c in derived
        },
        "week_range": list(week_range) if week_range is not None else None,
        "exclusions": [
            {
                "name": "treated rows",
                "why": (
                    "The baseline estimates what a row would have done without "
                    "the display, so a promoted row cannot be in its training "
                    "set. A caller-supplied frame is never filtered — it raises."
                ),
                "before": totals,
                "after": after_treated,
                "removed": _diff(totals, after_treated),
            },
            {
                "name": "weeks outside the estimation window",
                "why": (
                    "Settled decision 5: weeks 18-101. Before week 18 the "
                    "household panel is still recruiting, so a level comparison "
                    "against those weeks compares panel size, not demand. "
                    "causal_data records no treatment after week 101."
                ),
                "before": after_treated,
                "after": after_window,
                "removed": _diff(after_treated, after_window),
            },
        ],
        "training_frame": {
            **after_window,
            "zero_unit_rows": int((frame["units"] == 0).sum()),
            "zero_unit_share": _share((frame["units"] == 0).sum(), len(frame)),
            "why_zeros_stay": (
                "These are the explicit demand zeros Phase 2.6 built inside the "
                "scope. Dropping them would refit the model on 'weeks the "
                "product happened to sell', which is a different variable."
            ),
            "mailer_rows": int(mailer.sum()),
            "mailer_share": _share(mailer.sum(), len(frame)),
            "weeks": [int(frame["WEEK_NO"].min()), int(frame["WEEK_NO"].max())]
            if len(frame)
            else None,
            "product_store_pairs": int(
                frame[["PRODUCT_ID", "STORE_ID"]].drop_duplicates().shape[0]
            ),
        },
    }
    return frame, diagnostics


def _totals(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": len(frame),
        "units": float(frame["units"].sum()),
        "sales_value": round(float(frame["sales_value"].sum()), 2),
    }


def _sql_totals(
    con: duckdb.DuckDBPyConnection, src: str, where: str, treatment_column: str
) -> dict[str, float | int]:
    row = con.execute(
        f"SELECT COUNT(*) AS rows, SUM(units) AS units, "
        f"SUM(sales_value) AS sales_value FROM {src} WHERE {where}"
    ).fetchone()
    return {
        "rows": int(row[0]),
        "units": float(row[1] or 0.0),
        "sales_value": round(float(row[2] or 0.0), 2),
    }


def _diff(
    before: dict[str, float | int], after: dict[str, float | int]
) -> dict[str, float | int]:
    return {
        "rows": int(before["rows"] - after["rows"]),
        "units": round(float(before["units"] - after["units"]), 4),
        "sales_value": round(float(before["sales_value"] - after["sales_value"]), 2),
        "rows_share": _share(before["rows"] - after["rows"], before["rows"]),
        "units_share": _share(before["units"] - after["units"], before["units"]),
        "sales_value_share": _share(
            before["sales_value"] - after["sales_value"], before["sales_value"]
        ),
    }


def _share(numerator: float, denominator: float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def mechanic_strata(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    treatment_column: str = "treated",
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Label every treated row `display_only` or `display_and_mailer`.

    Adding `in_mailer` as a covariate fixes the *control* group. It does not
    purify the treated group: on a row carrying both mechanics, holding
    `in_mailer` fixed at True holds an interaction fixed rather than removing
    it, so the row identifies a joint display-and-mailer effect. Decision 8
    requires the two strata be reported separately and never pooled into one
    "display effect"; this is the column that makes that possible.

    Returns:
        `(strata, diagnostics)`. `strata` is one row per treated key with a
        `stratum` label.
    """
    wanted = [*_KEY, treatment_column, "in_mailer", "units", "sales_value"]
    if isinstance(panel, pd.DataFrame):
        missing = [c for c in wanted if c not in panel.columns]
        if missing:
            raise KeyError(f"not columns of the panel: {missing}")
        frame = panel[wanted].copy()
    else:
        own = con is None
        con = connect() if con is None else con
        try:
            src = f"read_parquet('{Path(panel).as_posix()}')"
            cols = ", ".join(f'"{c}"' for c in wanted)
            frame = con.execute(f"SELECT {cols} FROM {src}").df()
        finally:
            if own:
                con.close()

    if week_range is not None:
        frame = frame.loc[frame["WEEK_NO"].between(*week_range)]
    treated = frame.loc[frame[treatment_column].astype(bool)].copy()
    both = treated["in_mailer"].astype(bool)
    treated["stratum"] = np.where(both, "display_and_mailer", "display_only")
    strata = treated[[*_KEY, "stratum", "units", "sales_value"]].reset_index(drop=True)

    counts = strata.groupby("stratum", observed=True).agg(
        rows=("stratum", "size"),
        units=("units", "sum"),
        sales_value=("sales_value", "sum"),
    )
    diagnostics = {
        "stage": "mechanic_strata",
        "week_range": list(week_range) if week_range is not None else None,
        "treated_rows": len(strata),
        "strata": {
            name: {
                "rows": int(row["rows"]),
                "rows_share": _share(row["rows"], len(strata)),
                "units": float(row["units"]),
                "sales_value": round(float(row["sales_value"]), 2),
            }
            for name, row in counts.iterrows()
        },
        "why": (
            "Adding in_mailer as a covariate fixes the control group; it does "
            "not purify the treated group. On a display_and_mailer row, holding "
            "in_mailer fixed at True holds an interaction fixed rather than "
            "removing it, so the estimate there is a joint display-and-mailer "
            "effect. Settled decision 8 requires the two strata be reported "
            "separately and never pooled into one 'display effect'."
        ),
    }
    return strata, diagnostics


def fit_baseline(
    panel: str | Path | pd.DataFrame = "data/interim/panel.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    features: tuple[str, ...] | list[str] | None = None,
    include_identity: bool = False,
    target: str = DEFAULT_TARGET,
    quantiles: tuple[float, ...] = QUANTILES,
    treatment_column: str = "treated",
    week_range: tuple[int, int] | None = ESTIMATION_WINDOW,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_data_in_leaf: int = 40,
    backtest_weeks: int = 8,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> tuple[BaselineModel, dict[str, Any]]:
    """Fit the counterfactual on control rows only.

    Predicts `log1p(units)` with LightGBM, plus one independent quantile fit per
    entry in `quantiles`. Everything the fit refuses to do — see treated rows,
    see the deal, see the outcome — is enforced here rather than documented.

    Args:
        panel: the Task 2.6 modelling panel, path or DataFrame. A DataFrame
            containing treated rows raises rather than being filtered.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        features: override the feature list. Defaults to `BASELINE_FEATURES`.
        include_identity: also fit on PRODUCT_ID, STORE_ID and COMMODITY_DESC as
            categoricals. Off by default: the lags carry the level, and a fit
            keyed on 300 particular products does not transfer beyond the scope.
        target: one of `TARGETS`. Decides what the mean model is fitted on and
            how its output becomes units. `log1p` is retained so the Task 4.4
            comparison stays reproducible; see `TARGETS` for why it is not the
            default any more.
        quantiles: alphas for the quantile variants.
        treatment_column: the boolean that must be False on every training row.
        week_range: inclusive `(first, last)` week. Settled decision 5.
        n_estimators, learning_rate, num_leaves, min_data_in_leaf: LightGBM.
            `max_bin` is pinned at 63 and `num_leaves` capped at 63 per
            CLAUDE.md.
        backtest_weeks: hold out the last N weeks of the training frame and
            refit, to report held-out error and interval coverage. 0 skips it.
        seed: explicit, per the project's randomness rule.
        out_dir: if given, the fitted boosters are also saved there.

    Returns:
        `(model, diagnostics)`.

    Raises:
        TreatedRowsError: a treated row reached the training frame.
        ForbiddenFeatureError: a requested feature is derived from the treatment
            or from the outcome.
        ValueError: `num_leaves` exceeds the CLAUDE.md ceiling, or the training
            frame is empty.
    """
    import lightgbm as lgb

    if num_leaves > 63:
        raise ValueError(
            f"num_leaves={num_leaves} exceeds the CLAUDE.md ceiling of 63 on "
            f"this machine."
        )
    if target not in TARGETS:
        raise ValueError(f"target must be one of {list(TARGETS)}, got {target!r}")
    names, identity = _resolve_features(features, include_identity)

    frame, frame_diag = control_rows(
        panel,
        con=con,
        features=names,
        treatment_column=treatment_column,
        week_range=week_range,
    )
    if frame.empty:
        raise ValueError(
            "the training frame is empty after excluding treated rows and "
            f"restricting to weeks {week_range}; there is nothing to fit"
        )

    coupling, leaking = missingness_coupling(frame, names)
    if leaking:
        raise OutcomeLeakError(
            f"{leaking} are null on almost exactly the rows where units == 0, so "
            f"their absence is a readout of the outcome rather than a gap in a "
            f"covariate. A model given them reads the answer instead of "
            f"predicting it. Measured: "
            f"{ {c: coupling[c] for c in leaking} }"
        )

    categories = {
        c: sorted(pd.Series(frame[c]).dropna().unique().tolist()) for c in identity
    }
    design = _prepare(frame[list(names)], identity, categories)
    observed = frame["units"].to_numpy(dtype="float64")
    # tweedie and poisson are fitted on units directly — their log link lives
    # inside LightGBM, so there is no retransformation to get wrong afterwards.
    fit_target = (
        observed if target in _DIRECT_TARGETS else np.log1p(observed)
    )

    params = {
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        # CLAUDE.md: max_bin 63, num_leaves no greater than 63, threads 2.
        "max_bin": 63,
        "num_threads": 2,
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }

    boosters = _fit_all(
        lgb, design, fit_target, params, n_estimators, quantiles, identity,
        seed, target,
    )
    # Measured for both log1p paths, applied by only one. The measurement on
    # the plain path is the size of the bias it is carrying uncorrected, which
    # is worth reporting even where nothing is done about it.
    measured_smearing = _smearing_factor(boosters["mean"], design, fit_target, target)
    smearing = measured_smearing if target == "log1p_smearing" else 1.0
    model = BaselineModel(
        boosters=boosters,
        features=names,
        categorical=identity,
        params={
            **params,
            "n_estimators": n_estimators,
            "objective": _mean_objective(target),
        },
        train_window=tuple(week_range) if week_range else (0, 0),  # type: ignore[arg-type]
        n_train_rows=len(frame),
        seed=seed,
        categories=categories,
        target=target,
        smearing_factor=smearing,
    )

    fitted = {
        key: np.asarray(booster.predict(design), dtype="float64")
        for key, booster in boosters.items()
    }
    crossing = _quantile_crossing(fitted, quantiles)
    backtest = _backtest(
        lgb, frame, names, identity, categories, params, n_estimators,
        quantiles, backtest_weeks, seed, target, names,
    )
    _, strata_diag = mechanic_strata(
        panel, con=con, treatment_column=treatment_column, week_range=week_range
    )

    importances = model.importances()
    diagnostics: dict[str, Any] = {
        "stage": "fit_baseline",
        "estimand": (
            "E[log1p(units) | features, display = 0], inverted with expm1. The "
            "counterfactual for a treated row holds in_mailer at its observed "
            "value: what the row would have done with the mailer that actually "
            "ran and without the display."
        ),
        "control_rows": frame_diag,
        "features": {
            "columns": list(names),
            "n": len(names),
            "identity_included": include_identity,
            "categorical": list(identity),
            "missingness": {
                c: round(float(design[c].isna().mean()), 6)
                for c in names
                if design[c].dtype.kind == "f"
            },
            "forbidden": list(FORBIDDEN_FEATURES),
            "why_forbidden": (
                "The treatment itself, anything derived from the deal, the "
                "outcome, and any column whose missingness is the outcome. "
                "Checked at fit time, not left to the caller."
            ),
        },
        "missingness_leak_check": {
            "threshold": MISSINGNESS_LEAK_THRESHOLD,
            "rule": (
                "A feature is leaking if P(units == 0 | feature null) and "
                "P(feature null | units == 0) both exceed the threshold: it is "
                "then observed if and only if the row sold, so its absence "
                "reports the outcome. Raises OutcomeLeakError; never repaired."
            ),
            "measured": coupling,
            "leaking": leaking,
            "why_this_check_exists": (
                "price_rel_category is null on exactly the 2,046,518 control "
                "rows in the window where units == 0 — no exception in either "
                "direction — and took 94.3% of the gain in the first fit of "
                "this module. Phase 2.6 asserts no lag uses the current week's "
                "outcome, but checks lag values and not column availability, so "
                "the leak passed that test. It is replaced here by "
                "price_rel_category_lag, and the raw column is in "
                "FORBIDDEN_FEATURES."
            ),
        },
        "partial_leak_caveats": {
            "n_stores_carrying": (
                "Counts stores with a sale of the product that week, including "
                "the focal store, so a selling row contributes 1 to its own "
                "feature. Not deterministic and not repaired here — the fix "
                "belongs in Phase 2.6, which builds the column — but it is a "
                "partial use of the current week's outcome and any reader of "
                "the importance table should know it."
            ),
            "store_traffic": (
                "Distinct baskets in the store that week, including any basket "
                "containing the focal product. The dilution is ~1 in 500, and "
                "Phase 2.6 already flags it as a possible mediator."
            ),
        },
        "mailer_covariate": {
            "decision": "settled decision 8, docs/data_findings.md",
            "in_features": "in_mailer" in names,
            "control_rows_with_mailer": frame_diag["training_frame"]["mailer_rows"],
            "control_mailer_share": frame_diag["training_frame"]["mailer_share"],
            "why": (
                "Those control rows were promoted. Told nothing about it, the "
                "model learns an inflated normal, the counterfactual is fitted "
                "too high, and every measured display effect is biased towards "
                "zero."
            ),
            "reconciliation": (
                "The no-promotion-features rule stops the counterfactual seeing "
                "the promotion being measured. in_mailer is a different "
                "mechanic, is not derived from display, is known before the "
                "display decision, and is not a post-treatment aggregate. "
                "Conditioning on a concurrent treatment avoids omitted-variable "
                "bias. Depth, on_deal and the discount columns remain excluded."
            ),
            "what_it_does_not_fix": strata_diag["why"],
        },
        "treated_strata": strata_diag,
        "model": {
            "estimator": "lgb.train",
            "objectives": [
                _mean_objective(target),
                *[f"quantile@{a}" for a in quantiles],
            ],
            "target": target,
            "fitted_on": "units" if target in _DIRECT_TARGETS else "log1p(units)",
            "inverse": {
                "log1p": "expm1, clipped at zero",
                "log1p_smearing": (
                    f"exp(raw) * {smearing:.4f} - 1 for the mean, expm1 for the "
                    f"quantiles, clipped at zero"
                ),
                "tweedie": "none — the booster predicts the conditional mean",
                "poisson": "none — the booster predicts the conditional mean",
            }[target],
            "smearing_factor_applied": round(smearing, 6),
            "smearing_factor_measured": round(measured_smearing, 6),
            "smearing_note": (
                "Duan's factor, mean(exp(training residual)) in log space, "
                "measured on both log1p paths and applied only by "
                "log1p_smearing. On the plain path the measured value is the "
                "retransformation bias the fit is carrying uncorrected: expm1 "
                "understates the conditional mean by roughly this factor. It is "
                "applied to the mean only — quantiles are equivariant under "
                "expm1 and need no correction."
            ),
            "n_estimators": n_estimators,
            **{k: v for k, v in params.items() if k != "verbose"},
        },
        "quantile_crossing": crossing,
        "feature_importances": importances.to_dict(orient="records"),
        "backtest": backtest,
        "in_sample": _error_summary(
            frame["units"].to_numpy(dtype="float64"), model.predict(frame)
        ),
        "in_sample_caveat": (
            "In-sample fit on control rows. It is not evidence the effect "
            "estimate is right — see the backtest caveat."
        ),
    }

    if out_dir is not None:
        diagnostics["written_to"] = str(model.save(out_dir))

    return model, diagnostics


def _mean_objective(target: str) -> str:
    """The LightGBM objective the mean model is fitted with."""
    if target in _DIRECT_TARGETS:
        return target
    return "regression"


def _fit_all(
    lgb: Any,
    design: pd.DataFrame,
    fit_target: np.ndarray,
    params: dict[str, Any],
    n_estimators: int,
    quantiles: tuple[float, ...],
    categorical: tuple[str, ...],
    seed: int,
    target: str = "log1p",
) -> dict[str, Any]:
    """The mean fit and one independent fit per quantile.

    The quantile models are always plain quantile regressions on whatever scale
    the mean model was given. Quantiles are equivariant under a monotone
    transform, so a quantile of `log1p(units)` inverts to that quantile of units
    with no correction — which is exactly why the smearing factor must not touch
    them.
    """
    cat = list(categorical) if categorical else "auto"
    dataset = lgb.Dataset(
        design,
        fit_target,
        params={"max_bin": params["max_bin"], "verbose": -1},
        categorical_feature=cat,
        free_raw_data=False,
    )
    mean_params = {**params, "objective": _mean_objective(target)}
    if target == "tweedie":
        mean_params["tweedie_variance_power"] = params.get(
            "tweedie_variance_power", TWEEDIE_VARIANCE_POWER
        )
    boosters = {"mean": lgb.train(mean_params, dataset, num_boost_round=n_estimators)}
    for alpha in quantiles:
        boosters[_quantile_key(alpha)] = lgb.train(
            {**params, "objective": "quantile", "alpha": alpha},
            dataset,
            num_boost_round=n_estimators,
        )
    return boosters


def _smearing_factor(
    booster: Any, design: pd.DataFrame, fit_target: np.ndarray, target: str
) -> float:
    """Duan's smearing factor: `mean(exp(residual))` in log space.

    For `log1p(y) = f(x) + e`, `E[y + 1 | x] = exp(f(x)) * E[exp(e)]`, so the
    conditional mean of units is `exp(f(x)) * S - 1`. Inverting with `expm1`
    alone silently sets `S = 1`, which is only true when the residuals have no
    spread — and a zero-inflated outcome has plenty.

    Estimated in-sample, which is the standard form and is stated here because
    it matters: an overfitted mean model has small residuals and returns a
    factor near 1, understating the correction it needs.
    """
    if target not in {"log1p", "log1p_smearing"}:
        return 1.0
    predicted = np.asarray(booster.predict(design), dtype="float64")
    return float(np.mean(np.exp(fit_target - predicted)))


def _quantile_crossing(
    fitted: dict[str, np.ndarray], quantiles: tuple[float, ...]
) -> dict[str, Any]:
    """How often the independently fitted quantiles come back out of order.

    Measured and reported, never sorted away. Re-ordering the predictions would
    make the interval look coherent while hiding that three separate fits
    disagree about the shape of the conditional distribution.
    """
    keys = [_quantile_key(a) for a in sorted(quantiles)]
    pairs = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
    crossings = {
        f"{lo}>{hi}": _share(int((fitted[lo] > fitted[hi] + 1e-9).sum()), len(fitted[lo]))
        for lo, hi in pairs
    }
    any_cross = np.zeros(len(fitted[keys[0]]), dtype=bool) if keys else np.zeros(0, bool)
    for lo, hi in pairs:
        any_cross |= fitted[lo] > fitted[hi] + 1e-9
    return {
        "pairs": crossings,
        "any_share": _share(int(any_cross.sum()), len(any_cross)),
        "rows": len(any_cross),
        "why_not_repaired": (
            "The quantile models are independent fits, not a decomposition of "
            "one model, so crossings are possible. Sorting them would make the "
            "interval look coherent while hiding that the three fits disagree "
            "about the conditional distribution."
        ),
    }


def _backtest(
    lgb: Any,
    frame: pd.DataFrame,
    names: tuple[str, ...],
    categorical: tuple[str, ...],
    categories: dict[str, list[Any]],
    params: dict[str, Any],
    n_estimators: int,
    quantiles: tuple[float, ...],
    backtest_weeks: int,
    seed: int,
    target: str = DEFAULT_TARGET,
    features: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Refit on all but the last N weeks and score the held-out ones.

    A time split, not a random one: a random split puts adjacent weeks of the
    same product-store on both sides, and they share their lagged units almost
    exactly.

    The held-out fit goes through a real `BaselineModel`, so it inverts exactly
    the way the shipped model does. An earlier version hardcoded `log1p` and
    `expm1` here: when the default target changed, the backtest silently went on
    measuring the old path and reported numbers identical to the previous run.
    A validation block that does not track the thing it validates is worse than
    none, because it looks like evidence.
    """
    caveat = (
        "Backtest error on untreated weeks is necessary and not sufficient. It "
        "says nothing about y_hat(0) where D = 1, which is the quantity the "
        "baseline exists to produce. Never report predictive accuracy as "
        "evidence the effect estimate is right."
    )
    if backtest_weeks <= 0:
        return {"ran": False, "why": "backtest_weeks = 0", "caveat": caveat}

    last = int(frame["WEEK_NO"].max())
    cutoff = last - backtest_weeks
    train_mask = frame["WEEK_NO"] <= cutoff
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        return {
            "ran": False,
            "why": (
                f"a {backtest_weeks}-week holdout leaves one side empty on a "
                f"frame spanning weeks {int(frame['WEEK_NO'].min())}-{last}"
            ),
            "caveat": caveat,
        }

    train = _prepare(frame.loc[train_mask, list(names)], categorical, categories)
    observed_train = frame.loc[train_mask, "units"].to_numpy(dtype="float64")
    y_train = (
        observed_train if target in _DIRECT_TARGETS else np.log1p(observed_train)
    )
    y_test = frame.loc[test_mask, "units"].to_numpy(dtype="float64")

    boosters = _fit_all(
        lgb, train, y_train, params, n_estimators, quantiles, categorical, seed,
        target,
    )
    measured = _smearing_factor(boosters["mean"], train, y_train, target)
    held_out = BaselineModel(
        boosters=boosters,
        features=tuple(features or names),
        categorical=categorical,
        categories=categories,
        target=target,
        smearing_factor=measured if target == "log1p_smearing" else 1.0,
    )
    test_frame = frame.loc[test_mask]
    predicted = {
        "mean": held_out.predict(test_frame),
        **{
            _quantile_key(alpha): held_out.predict(test_frame, alpha)
            for alpha in quantiles
        },
    }

    lo_key, hi_key = _quantile_key(min(quantiles)), _quantile_key(max(quantiles))
    inside = (y_test >= predicted[lo_key]) & (y_test <= predicted[hi_key])
    return {
        "ran": True,
        "scheme": "time split — the last weeks of the training frame, held out",
        "target": target,
        "why_time_split": (
            "Adjacent weeks of one product-store share their lagged units "
            "almost exactly, so a random split scores the model on near-copies "
            "of its own training rows."
        ),
        "train_weeks": [int(frame.loc[train_mask, "WEEK_NO"].min()), cutoff],
        "test_weeks": [cutoff + 1, last],
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        **_error_summary(y_test, predicted["mean"]),
        "interval": {
            "nominal": [min(quantiles), max(quantiles)],
            "nominal_coverage": round(max(quantiles) - min(quantiles), 4),
            "empirical_coverage": _share(int(inside.sum()), len(inside)),
            "mean_width_units": round(
                float(np.mean(predicted[hi_key] - predicted[lo_key])), 4
            ),
        },
        "caveat": caveat,
    }


def _error_summary(
    actual_units: np.ndarray, predicted_units: np.ndarray
) -> dict[str, float]:
    """Error on both scales, from predictions already inverted to units.

    Takes units rather than the fitted scale because the fitted scale is no
    longer one thing: `tweedie` and `poisson` predict units directly, and a
    smearing-corrected `log1p` prediction is not `expm1` of anything.
    """
    predicted_units = np.clip(predicted_units, 0.0, None)
    residual = actual_units - predicted_units
    log_residual = np.log1p(actual_units) - np.log1p(predicted_units)
    return {
        "mae_units": round(float(np.mean(np.abs(residual))), 6),
        "rmse_units": round(float(np.sqrt(np.mean(residual**2))), 6),
        "bias_units": round(float(np.mean(residual)), 6),
        "mae_log1p": round(float(np.mean(np.abs(log_residual))), 6),
        "rmse_log1p": round(float(np.sqrt(np.mean(log_residual**2))), 6),
        "mean_actual_units": round(float(np.mean(actual_units)), 6),
        "mean_predicted_units": round(float(np.mean(predicted_units)), 6),
    }


def rollout(
    model: BaselineModel,
    history: pd.DataFrame,
    exog_weeks: pd.DataFrame,
    *,
    quantile: float | None = None,
    feedback: str = "recursive",
    carry: tuple[str, ...] = CARRIED_BY_DEFAULT,
    outcome: str = "units",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A multi-week counterfactual whose own predictions become its lags.

    For any window longer than one week the baseline needs last week's units to
    predict this week's, and inside the window last week's *observed* units are
    contaminated by the promotion. Feeding them in makes the counterfactual
    chase the bump: the model is told demand was high, so it predicts high, so
    the residual shrinks. The measured effect then falls the longer the campaign
    runs, which is the opposite of the truth for a campaign that kept working.

    Each step therefore predicts, then pushes **its own prediction** into the
    series that the next step's lags and rolling means are computed from.
    Nothing observed inside the window ever reaches a feature.

    This is not circular. Every parameter was fitted on unpromoted weeks; the
    rollout only applies what was learned. The honest cost is compounding error
    down the horizon, which is what Task 4.4's synthetic truth and Task 4.5's
    placebo band exist to measure.

    Features split three ways, and the split is reported rather than assumed:

    - **recursive** — `RECURSIVE_FEATURES`, recomputed each step from the
      counterfactual path, seeded by the observed history before the window;
    - **exogenous** — supplied per week in `exog_weeks`, which is how
      `in_mailer` takes its **observed** value: the question is what the row
      would have done with the mailer that actually ran and without the display;
    - **carried** — everything else, held at its last pre-window value, plus
      anything named in `carry` even when `exog_weeks` supplies it.
      `price_rel_category_lag` is in `carry` by default on purpose. Its
      in-window value depends on whether the product sold, and whether it sold
      is exactly what the counterfactual is trying to say — settled decision 9.
      The natural call passes a slice of the panel, which carries the observed
      column, so the safe reading has to be the default rather than something
      the caller remembers. `carry=()` opts out, and the diagnostics record it.

    Args:
        model: a fitted `BaselineModel`.
        history: observed rows before the window, for one or many product-stores.
            Must carry `PRODUCT_ID`, `STORE_ID`, `WEEK_NO`, the outcome column,
            and every carried feature. Deep enough to seed the longest lag —
            52 weeks for the default feature set, less if the fit is shorter.
        exog_weeks: one row per counterfactual week per product-store, carrying
            the keys, `WEEK_NO`, and the exogenous features. Windows may be
            ragged across keys.
        quantile: predict this quantile's path instead of the mean. Each
            quantile is its own recursion — it feeds its own predictions back.
        feedback: `"recursive"` (the estimator) or `"observed"` (the naive
            version, which feeds observed units back). **`"observed"` is biased
            by construction and exists to make that bias measurable** — see
            `tests/test_rollout_contamination.py`. It is labelled in the
            diagnostics and must never produce a reported estimate.
        outcome: the units column, read from `history` and — under
            `feedback="observed"` only — from `exog_weeks`.

    Returns:
        `(path, diagnostics)`. `path` has one row per key-week with
        `counterfactual_units` and **every feature value the model was given**,
        so a contaminated lag or a carried price is visible in the output rather
        than inferred from the code.

    Raises:
        NonContiguousWeeksError: the window does not continue the history week
            by week.
        KeyError: a feature is neither recursive, nor in `exog_weeks`, nor in
            `history`.
        ValueError: `feedback` is not one of the two modes.
    """
    if feedback not in {"recursive", "observed"}:
        raise ValueError(
            f"feedback must be 'recursive' or 'observed', got {feedback!r}"
        )
    keys = ["PRODUCT_ID", "STORE_ID"]
    for name, frame in (("history", history), ("exog_weeks", exog_weeks)):
        missing = [c for c in (*keys, "WEEK_NO") if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} is missing {missing}")
    if outcome not in history.columns:
        raise KeyError(f"history is missing the outcome column {outcome!r}")
    if feedback == "observed" and outcome not in exog_weeks.columns:
        raise KeyError(
            f"feedback='observed' feeds observed units back as lags, so "
            f"exog_weeks must carry {outcome!r}"
        )

    recursive = [c for c in model.features if c in RECURSIVE_FEATURES]
    exogenous = [
        c
        for c in model.features
        if c not in recursive and c not in carry and c in exog_weeks.columns
    ]
    carried = [c for c in model.features if c not in recursive and c not in exogenous]
    absent = [c for c in carried if c not in history.columns]
    if absent:
        raise KeyError(
            f"{absent} are neither recursive nor in exog_weeks, so they would be "
            f"carried from history — but history does not have them either"
        )

    state = _rollout_state(history, exog_weeks, keys, outcome, carried)
    steps = max((len(s["weeks"]) for s in state.values()), default=0)

    rows: list[dict[str, Any]] = []
    for step in range(steps):
        # One predict per step across every key, so the loop is over weeks and
        # not over rows. It cannot be vectorised away entirely: step t's
        # features are a function of step t-1's prediction, which is what the
        # whole function is about.
        live = [k for k, s in state.items() if step < len(s["weeks"])]
        if not live:
            break
        frame = pd.DataFrame(
            [_rollout_row(state[k], step, recursive, exogenous, carried) for k in live]
        )
        predicted = model.predict(frame, quantile)
        for key, row_index, value in zip(live, range(len(live)), predicted, strict=True):
            s = state[key]
            week = s["weeks"][step]
            fed = float(s["observed"][step]) if feedback == "observed" else float(value)
            s["series"][week] = fed
            rows.append(
                {
                    # Every feature the model was actually given, not only the
                    # recursive ones: a counterfactual nobody can audit is a
                    # number without a derivation. The keys are written after,
                    # so an identity feature cannot shadow them.
                    **frame.iloc[row_index].to_dict(),
                    "PRODUCT_ID": key[0],
                    "STORE_ID": key[1],
                    "WEEK_NO": week,
                    "step": step,
                    "counterfactual_units": float(value),
                    "fed_back": fed,
                }
            )

    path = (
        pd.DataFrame(rows).sort_values([*keys, "WEEK_NO"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[*keys, "WEEK_NO", "step", "counterfactual_units", "fed_back"]
        )
    )

    diagnostics = {
        "stage": "rollout",
        "feedback": feedback,
        "quantile": quantile,
        "keys": len(state),
        "steps": steps,
        "rows": len(path),
        "features": {
            "recursive": recursive,
            "exogenous": exogenous,
            "carried": carried,
            "forced_carry": [c for c in carry if c in carried],
            "forced_carry_overrode_exog": [
                c for c in carry if c in carried and c in exog_weeks.columns
            ],
        },
        "why_recursive": (
            "Inside the window, last week's observed units carry the promotion. "
            "Feeding them in makes the counterfactual chase the bump and the "
            "measured effect shrink as the window grows. Each step feeds its "
            "own prediction back instead, so nothing observed inside the window "
            "reaches a feature."
        ),
        "carried_note": (
            "Carried features hold their last pre-window value. "
            "price_rel_category_lag is carried by default, and stays carried "
            "even when exog_weeks supplies it, because its in-window value "
            "depends on whether the product sold — which is the quantity the "
            "counterfactual exists to state, settled decision 9. "
            "forced_carry_overrode_exog names the columns where that happened. "
            "Pass carry=() to use the observed path instead."
        ),
        "exogenous_note": (
            "in_mailer takes its observed value here by design: the question is "
            "what the row would have done with the mailer that actually ran and "
            "without the display. The contemporaneous block "
            "(n_stores_carrying, category_units_ex_focal, store_traffic) can be "
            "affected by the promotion, so supplying its observed values "
            "conditions the counterfactual on a mediator. Task 4.4 reports "
            "recovery with and without that block."
        ),
        "compounding": (
            "Error compounds down the horizon because each step's features are "
            "built from earlier predictions. That is the honest cost of not "
            "contaminating the lags, and it is what the synthetic-truth and "
            "placebo harnesses measure."
        ),
    }
    if feedback == "observed":
        diagnostics["biased"] = True
        diagnostics["why_biased"] = (
            "feedback='observed' feeds the observed, promotion-contaminated "
            "units back as lags. It exists so the contamination can be measured "
            "against the recursive path and must never produce a reported "
            "estimate."
        )
    return path, diagnostics


def _rollout_state(
    history: pd.DataFrame,
    exog_weeks: pd.DataFrame,
    keys: list[str],
    outcome: str,
    carried: list[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Per product-store: the observed series, the window, and the carried row.

    Also where the contiguity invariant is enforced. Week arithmetic is what
    makes a lag a lag; a gap makes `units_lag_1` mean something different in
    every row.
    """
    state: dict[tuple[Any, ...], dict[str, Any]] = {}
    history_by_key = dict(tuple(history.groupby(keys, observed=True, sort=True)))
    for key, block in exog_weeks.groupby(keys, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        if key not in history_by_key:
            raise KeyError(f"no history for product-store {key}")
        past = history_by_key[key].sort_values("WEEK_NO")
        window = block.sort_values("WEEK_NO")
        weeks = window["WEEK_NO"].astype(int).tolist()

        expected = list(range(weeks[0], weeks[0] + len(weeks)))
        if weeks != expected:
            raise NonContiguousWeeksError(
                f"{key}: rollout weeks {weeks} are not consecutive"
            )
        last_observed = int(past["WEEK_NO"].max())
        if weeks[0] != last_observed + 1:
            raise NonContiguousWeeksError(
                f"{key}: history ends at week {last_observed} but the rollout "
                f"starts at week {weeks[0]}; the gap would make units_lag_1 "
                f"mean 'the last week we happened to have'"
            )

        state[key] = {
            "series": {
                int(w): float(u)
                for w, u in zip(past["WEEK_NO"], past[outcome], strict=True)
            },
            "weeks": weeks,
            "exog": window.reset_index(drop=True),
            "observed": (
                window[outcome].astype(float).tolist()
                if outcome in window.columns
                else [float("nan")] * len(weeks)
            ),
            "carried": {c: past.iloc[-1][c] for c in carried},
        }
    return state


def _rollout_row(
    state: dict[str, Any],
    step: int,
    recursive: list[str],
    exogenous: list[str],
    carried: list[str],
) -> dict[str, Any]:
    """One design row: recursive features from the path, the rest as supplied."""
    week = state["weeks"][step]
    series = state["series"]
    row: dict[str, Any] = {}
    for name in recursive:
        if name.startswith("units_lag_"):
            row[name] = series.get(week - int(name.rsplit("_", 1)[1]), np.nan)
        else:
            # Weeks w-W..w-1, matching Phase 2.6's frame exactly: the window
            # ends at 1 PRECEDING, so it never includes its own week, and a
            # partial window averages what exists rather than returning null.
            span = int(name.rsplit("_", 1)[1])
            values = [
                series[w] for w in range(week - span, week) if w in series
            ]
            row[name] = float(np.mean(values)) if values else np.nan
    exog_row = state["exog"].iloc[step]
    for name in exogenous:
        row[name] = exog_row[name]
    for name in carried:
        row[name] = state["carried"][name]
    return row


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
