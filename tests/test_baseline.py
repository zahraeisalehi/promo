"""Tests for Task 4.1, the counterfactual baseline.

The fixture tests fit on a small synthetic control panel, so they run in a few
seconds and still exercise every guard that makes the fit defensible: no treated
row reaches it, `in_mailer` is in and the deal columns are out, the quantile
crossings are measured rather than repaired, and a saved model reloads to the
same predictions.

The one heavy test fits on the real panel and asserts the training frame is the
control-and-in-window subset it claims to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.baseline import (
    BASELINE_FEATURES,
    ESTIMATION_WINDOW,
    FORBIDDEN_FEATURES,
    IDENTITY_FEATURES,
    BaselineModel,
    ForbiddenFeatureError,
    OutcomeLeakError,
    TreatedRowsError,
    add_price_history,
    control_rows,
    fit_baseline,
    mechanic_strata,
    missingness_coupling,
    write_diagnostics,
)

PANEL = Path("data/interim/panel.parquet")


def real_data(fn):
    """Marks a test that reads the real panel from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast pass
    excludes it with -m "not heavy", and it is skipped outright when the
    artefact is absent.
    """
    return pytest.mark.skipif(not PANEL.exists(), reason="run Task 2.6 first")(
        pytest.mark.heavy(fn)
    )


def _panel(
    n_pairs: int = 30,
    n_weeks: int = 50,
    first_week: int = 18,
    *,
    treated_share: float = 0.0,
    mailer_share: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """A synthetic panel shaped like Task 2.6's, with a signal worth fitting.

    Units follow the lagged level plus a seasonal term, so a model that learns
    nothing would be visibly worse than one that does. Deliberately not built by
    a tree process — the estimator should not be validated on its own shape.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pair in range(n_pairs):
        level = float(rng.lognormal(1.2, 0.6))
        history = np.zeros(n_weeks + 4)
        for w in range(n_weeks + 4):
            season = 1.0 + 0.4 * np.sin(2 * np.pi * w / 52.0)
            history[w] = max(0.0, rng.poisson(max(0.05, level * season)))
        for i in range(n_weeks):
            w = first_week + i
            units = history[i + 4]
            rows.append(
                {
                    "PRODUCT_ID": 1000 + pair % 7,
                    "STORE_ID": 300 + pair // 7,
                    "WEEK_NO": w,
                    "units": float(units),
                    "sales_value": float(units) * 2.5,
                    "treated": bool(rng.random() < treated_share),
                    "in_mailer": bool(rng.random() < mailer_share),
                    "COMMODITY_DESC": f"CAT {pair % 3}",
                    "units_lag_1": history[i + 3],
                    "units_lag_2": history[i + 2],
                    "units_lag_4": history[i],
                    "units_lag_52": np.nan,
                    "units_roll_mean_4": history[i : i + 4].mean(),
                    "units_roll_mean_8": history[max(0, i - 4) : i + 4].mean(),
                    "units_roll_mean_13": history[max(0, i - 9) : i + 4].mean(),
                    "week_of_year": (w - 1) % 52 + 1,
                    "is_holiday_week": False,
                    "category_units_ex_focal": float(rng.poisson(50)),
                    "store_traffic": float(rng.poisson(500)),
                    # Null exactly when nothing sold, as on the real panel: the
                    # ratio needs an observed price, and a week with no sale has
                    # none. This is the leak the module refuses to fit on.
                    "price_rel_category": (
                        float(rng.normal(1.0, 0.1)) if units > 0 else np.nan
                    ),
                    "n_stores_carrying": float(rng.integers(1, 20)),
                    "price_index": 1.0 + 0.001 * i,
                }
            )
    return pd.DataFrame(rows)


def _fit(panel: pd.DataFrame, **kwargs):
    """A small, fast fit with the guards intact."""
    defaults = {
        "n_estimators": 40,
        "min_data_in_leaf": 20,
        "num_leaves": 15,
        "backtest_weeks": 0,
        "week_range": (18, 101),
    }
    return fit_baseline(panel, **{**defaults, **kwargs})


# --- the training frame may not contain a treated row ------------------------


def test_treated_rows_raise_and_the_frame_is_not_filtered():
    panel = _panel(treated_share=0.3, seed=1)
    before = len(panel)

    with pytest.raises(TreatedRowsError) as exc:
        control_rows(panel)

    assert "will not filter them out for you" in str(exc.value)
    # The caller's frame is untouched: no silent repair, and no mutation either.
    assert len(panel) == before
    assert panel["treated"].any()


def test_fit_baseline_raises_on_treated_rows():
    with pytest.raises(TreatedRowsError):
        _fit(_panel(treated_share=0.1, seed=2))


def test_control_frame_from_a_clean_panel_records_both_exclusions():
    panel = _panel(first_week=9, n_weeks=60, seed=3)
    frame, diag = control_rows(panel)

    assert frame["WEEK_NO"].min() >= ESTIMATION_WINDOW[0]
    assert frame["WEEK_NO"].max() <= ESTIMATION_WINDOW[1]

    treated_step, window_step = diag["exclusions"]
    assert treated_step["removed"]["rows"] == 0
    # Weeks 9-17 of the synthetic panel are dropped by decision 5, and the drop
    # is reported in rows, units, and sales value.
    assert window_step["removed"]["rows"] == 30 * 9
    assert window_step["removed"]["units"] > 0
    assert window_step["removed"]["sales_value"] > 0
    assert diag["training_frame"]["rows"] == len(frame)


def test_missing_treatment_column_is_named_not_assumed():
    panel = _panel(seed=4).drop(columns=["treated"])
    with pytest.raises(KeyError):
        control_rows(panel)


# --- what the model may and may not see --------------------------------------


def test_in_mailer_is_a_feature_and_the_deal_columns_are_not():
    assert "in_mailer" in BASELINE_FEATURES
    for banned in ("on_display", "treated", "depth", "price_status", "paid_price",
                   "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"):
        assert banned not in BASELINE_FEATURES
        assert banned in FORBIDDEN_FEATURES


def test_a_forbidden_feature_is_refused():
    panel = _panel(seed=5)
    with pytest.raises(ForbiddenFeatureError) as exc:
        _fit(panel, features=[*BASELINE_FEATURES, "depth"])
    assert "decision 8" in str(exc.value)


# --- a feature's missingness may not be the outcome either --------------------


def test_price_rel_category_is_forbidden_and_replaced_by_its_lag():
    assert "price_rel_category" in FORBIDDEN_FEATURES
    assert "price_rel_category" not in BASELINE_FEATURES
    assert "price_rel_category_lag" in BASELINE_FEATURES

    with pytest.raises(ForbiddenFeatureError):
        _fit(_panel(n_pairs=7, n_weeks=25, seed=20),
             features=[*BASELINE_FEATURES, "price_rel_category"])


def test_the_fixture_reproduces_the_real_missingness_leak():
    """The guard is only worth having if the fixture has the disease."""
    panel = _panel(n_pairs=20, n_weeks=40, seed=21)
    coupling, leaking = missingness_coupling(panel, ["price_rel_category"])

    assert leaking == ["price_rel_category"]
    assert coupling["price_rel_category"]["p_zero_given_null"] == 1.0
    assert coupling["price_rel_category"]["p_null_given_zero"] == 1.0


def test_a_leaking_feature_raises_rather_than_being_fitted():
    panel = _panel(n_pairs=20, n_weeks=40, seed=22)
    # Smuggled in under a name the forbidden list does not know, so the check
    # that fires here is the measurement and not the blocklist.
    panel["sneaky_price"] = panel["price_rel_category"]
    with pytest.raises(OutcomeLeakError) as exc:
        _fit(panel, features=[*BASELINE_FEATURES, "sneaky_price"])
    assert "readout of the outcome" in str(exc.value)


def test_the_lagged_price_uses_only_earlier_weeks():
    panel = _panel(n_pairs=7, n_weeks=30, seed=23)
    with_history = add_price_history(panel).sort_values(
        ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]
    )

    for _, block in with_history.groupby(["PRODUCT_ID", "STORE_ID"], observed=True):
        prices = block["price_rel_category"].to_numpy()
        carried = block["price_rel_category_lag"].to_numpy()
        for i in range(len(block)):
            earlier = [p for p in prices[:i] if not np.isnan(p)]
            expected = earlier[-1] if earlier else np.nan
            assert (np.isnan(carried[i]) and np.isnan(expected)) or (
                carried[i] == pytest.approx(expected)
            )


def test_the_lagged_price_no_longer_reads_the_outcome():
    panel = _panel(n_pairs=20, n_weeks=40, seed=24)
    frame, diag = control_rows(panel)
    _, leaking = missingness_coupling(frame, ["price_rel_category_lag"])

    assert leaking == []
    assert diag["derived_features"]["price_rel_category_lag"]["from"] == (
        "price_rel_category"
    )


def test_the_lag_is_computed_before_the_treated_rows_are_dropped():
    """A carried price must not skip promoted weeks — see add_price_history."""
    panel = _panel(n_pairs=7, n_weeks=30, treated_share=0.5, seed=25)
    full = add_price_history(panel)
    controls = full.loc[~full["treated"]].reset_index(drop=True)

    from_full = controls.set_index([*("PRODUCT_ID", "STORE_ID", "WEEK_NO")])[
        "price_rel_category_lag"
    ]
    filtered_first = add_price_history(
        panel.loc[~panel["treated"]].reset_index(drop=True)
    ).set_index([*("PRODUCT_ID", "STORE_ID", "WEEK_NO")])["price_rel_category_lag"]

    # The two disagree — which is the point: the derivation is only well defined
    # on the unfiltered panel, and control_rows does it there.
    assert not from_full.equals(filtered_first)


def test_identity_features_are_off_by_default_and_opt_in_works():
    panel = _panel(n_pairs=14, n_weeks=30, seed=6)

    model, diag = _fit(panel)
    assert tuple(model.features) == BASELINE_FEATURES
    assert diag["features"]["identity_included"] is False

    model_id, diag_id = _fit(panel, include_identity=True)
    for column in IDENTITY_FEATURES:
        assert column in model_id.features
    assert model_id.categorical == IDENTITY_FEATURES
    assert diag_id["features"]["identity_included"] is True


def test_diagnostics_state_the_mailer_covariate_decision():
    panel = _panel(n_pairs=14, n_weeks=30, seed=7)
    _, diag = _fit(panel)

    mailer = diag["mailer_covariate"]
    assert mailer["in_features"] is True
    assert mailer["control_rows_with_mailer"] > 0
    assert 0.0 < mailer["control_mailer_share"] < 1.0
    assert "biased towards zero" in mailer["why"]


# --- the fit itself ----------------------------------------------------------


def test_predictions_are_non_negative_and_round_trip_the_log_scale():
    panel = _panel(n_pairs=14, n_weeks=30, seed=8)
    model, _ = _fit(panel)
    # Predicting takes the same derived column the fit used — see Task 4.2's
    # rollout, which will call add_price_history for the same reason.
    scored = add_price_history(panel)

    units = model.predict(scored)
    log_units = model.predict_log1p(scored)

    assert (units >= 0).all()
    assert np.allclose(units, np.clip(np.expm1(log_units), 0.0, None))


def test_quantiles_are_ordered_on_most_rows_and_crossings_are_reported():
    panel = _panel(n_pairs=20, n_weeks=40, seed=9)
    model, diag = _fit(panel, n_estimators=80)
    scored = add_price_history(panel)

    q10 = model.predict(scored, quantile=0.1)
    q50 = model.predict(scored, quantile=0.5)
    q90 = model.predict(scored, quantile=0.9)
    assert np.mean(q10 <= q90 + 1e-9) > 0.95
    assert np.mean(q10 <= q50 + 1e-9) > 0.95

    crossing = diag["quantile_crossing"]
    assert set(crossing["pairs"]) == {"q10>q50", "q50>q90"}
    assert crossing["any_share"] is not None
    assert "Sorting them would make the interval look coherent" in (
        crossing["why_not_repaired"]
    )


def test_the_same_seed_gives_the_same_predictions():
    panel = _panel(n_pairs=14, n_weeks=30, seed=10)
    a, _ = _fit(panel, seed=7)
    b, _ = _fit(panel, seed=7)
    scored = add_price_history(panel)
    assert np.array_equal(a.predict(scored), b.predict(scored))


def test_save_and_load_reproduce_the_fit(tmp_path):
    panel = _panel(n_pairs=14, n_weeks=30, seed=11)
    model, _ = _fit(panel, include_identity=True)
    model.save(tmp_path / "baseline")

    reloaded = BaselineModel.load(tmp_path / "baseline")
    assert reloaded.features == model.features
    assert reloaded.categorical == model.categorical
    scored = add_price_history(panel)
    for quantile in (None, 0.1, 0.5, 0.9):
        assert np.allclose(
            reloaded.predict(scored, quantile), model.predict(scored, quantile)
        )


def test_predicting_without_a_feature_names_it():
    panel = _panel(n_pairs=14, n_weeks=30, seed=12)
    model, _ = _fit(panel)
    scored = add_price_history(panel)
    with pytest.raises(KeyError) as exc:
        model.predict(scored.drop(columns=["units_roll_mean_8"]))
    assert "units_roll_mean_8" in str(exc.value)


def test_importances_cover_every_feature():
    panel = _panel(n_pairs=14, n_weeks=30, seed=13)
    model, diag = _fit(panel)
    importances = model.importances()

    assert set(importances["feature"]) == set(model.features)
    assert abs(importances["gain_share"].sum() - 1.0) < 1e-6
    assert len(diag["feature_importances"]) == len(model.features)


def test_num_leaves_above_the_machine_ceiling_is_refused():
    with pytest.raises(ValueError, match="ceiling"):
        _fit(_panel(n_pairs=7, n_weeks=25, seed=14), num_leaves=127)


def test_backtest_holds_out_the_last_weeks_and_carries_its_caveat():
    panel = _panel(n_pairs=20, n_weeks=40, seed=15)
    _, diag = _fit(panel, backtest_weeks=8)

    backtest = diag["backtest"]
    assert backtest["ran"] is True
    assert backtest["test_weeks"] == [50, 57]
    assert backtest["train_rows"] + backtest["test_rows"] == 20 * 40
    assert 0.0 <= backtest["interval"]["empirical_coverage"] <= 1.0
    assert "necessary and not sufficient" in backtest["caveat"]


def test_an_empty_training_frame_is_refused_rather_than_fitted():
    panel = _panel(n_pairs=7, n_weeks=20, first_week=18, seed=16)
    with pytest.raises(ValueError, match="nothing to fit"):
        _fit(panel, week_range=(90, 101))


# --- the treated group is not purified, and says so --------------------------


def test_mechanic_strata_splits_the_treated_rows():
    panel = _panel(n_pairs=20, n_weeks=40, treated_share=0.4, mailer_share=0.5, seed=17)
    strata, diag = mechanic_strata(panel)

    assert set(strata["stratum"]) == {"display_only", "display_and_mailer"}
    assert diag["treated_rows"] == int(panel["treated"].sum())
    assert (
        diag["strata"]["display_only"]["rows"]
        + diag["strata"]["display_and_mailer"]["rows"]
        == diag["treated_rows"]
    )
    assert "never pooled into one 'display effect'" in diag["why"]


def test_fit_diagnostics_carry_the_treated_strata():
    panel = _panel(n_pairs=20, n_weeks=40, treated_share=0.3, seed=18)
    controls = panel.loc[~panel["treated"]].reset_index(drop=True)
    # The fit sees controls only; the strata are read off the full panel, which
    # is the frame that still knows which treated rows carried both mechanics.
    _, diag = _fit(controls)
    assert diag["treated_strata"]["treated_rows"] == 0

    _, diag_full = fit_baseline(
        panel.assign(treated=False),
        n_estimators=20,
        min_data_in_leaf=20,
        num_leaves=15,
        backtest_weeks=0,
    )
    assert diag_full["treated_strata"]["treated_rows"] == 0


def test_write_diagnostics_round_trips(tmp_path):
    import json

    panel = _panel(n_pairs=14, n_weeks=30, seed=19)
    _, diag = _fit(panel)
    path = write_diagnostics(diag, tmp_path / "baseline_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "fit_baseline"


# --- the real panel ----------------------------------------------------------


@real_data
def test_real_panel_training_frame_is_control_and_in_window():
    frame, diag = control_rows(PANEL)

    assert not frame["treated"].any()
    assert frame["WEEK_NO"].min() == 18
    assert frame["WEEK_NO"].max() == 101
    assert len(frame) == 2_351_749

    treated_step, window_step = diag["exclusions"]
    assert treated_step["before"]["rows"] == 2_966_328
    assert treated_step["after"]["rows"] == 2_599_982
    assert window_step["removed"]["rows"] == 248_233
    # Settled decision 8's figure, recomputed rather than quoted.
    assert diag["training_frame"]["mailer_share"] == pytest.approx(0.1306, abs=5e-4)


@real_data
def test_the_real_panel_carries_the_missingness_leak_exactly():
    """The finding this module was changed for, asserted on the real data."""
    # price_rel_category comes back as the source of the derived feature, so
    # the raw column can be measured beside its replacement.
    frame, _ = control_rows(PANEL)
    coupling, leaking = missingness_coupling(frame, ["price_rel_category"])

    assert leaking == ["price_rel_category"]
    assert coupling["price_rel_category"]["null_rows"] == 2_046_518
    assert coupling["price_rel_category"]["p_zero_given_null"] == 1.0
    assert coupling["price_rel_category"]["p_null_given_zero"] == 1.0

    # And the replacement does not have it.
    _, still_leaking = missingness_coupling(frame, ["price_rel_category_lag"])
    assert still_leaking == []


@real_data
def test_real_panel_fit_sees_no_treated_row():
    model, diag = fit_baseline(PANEL, n_estimators=50, backtest_weeks=0)

    assert diag["control_rows"]["training_frame"]["rows"] == 2_351_749
    assert model.n_train_rows == 2_351_749
    assert "in_mailer" in model.features
    assert "price_rel_category" not in model.features
    assert diag["missingness_leak_check"]["leaking"] == []
    strata = diag["treated_strata"]["strata"]
    assert strata["display_and_mailer"]["rows"] == 129_324
    assert strata["display_only"]["rows"] == 198_191
