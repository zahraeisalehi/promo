"""Tests for Task 3.5, the refusal engine.

The gate-authoring skill requires that every gate have data guaranteed to
trigger it and data guaranteed not to: "a gate that has never fired in a test
does not work." Both directions are here for every one of the eleven codes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promo.audit import DEFAULT_COVARIATES
from promo.gates import (
    GATE_ORDER,
    REASON_CODES,
    CampaignSpec,
    GateResult,
    UnknownReasonCode,
    message_for,
    run_audit,
    write_audit,
)

PANEL = Path("data/interim/panel.parquet")
CYCLES = Path("data/interim/repurchase_cycles.parquet")

_NO_DATA = not PANEL.exists() or not CYCLES.exists()


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast
    pass excludes it with -m "not heavy", and it is skipped outright when
    the artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="run Phases 2-3 first")(
        pytest.mark.heavy(fn)
    )


#: The evidence each code's template needs, with values that render.
_EVIDENCE = {
    "NO_VARIATION": {"best_axis": "product", "mixed_units_share": 0.02,
                     "threshold": 0.10},
    "NO_OVERLAP": {"auc": 0.97, "outside_share": 0.42},
    "LEAKED_FEATURE": {"feature": "depth", "gain_share": 0.93, "auc": 0.99},
    "DEPTH_BOUNDED": {"product": 1234, "deal_share": 0.96, "threshold": 0.90},
    "INSUFFICIENT_SUPPORT": {"product": 1234, "weeks_priced": 3,
                             "minimum_weeks": 8},
    "KAPPA_IMPOSSIBLE": {"depth": 0.45, "margin": 0.30, "kappa": 1.5},
    "NO_MARGIN": {"minimum_viable_margin": 0.242},
    "PLACEBO_OVERLAP": {"estimate": 120.0, "band_low": -400.0, "band_high": 500.0},
    "OVERLAPPING_TREATMENTS": {"collision_share": 0.398,
                               "contaminated_share": 0.131},
    "ROI_UNBOUNDED": {"low": -300.0, "high": 900.0},
    "HORIZON_TOO_SHORT": {"commodity": "SOUP", "horizon_weeks": 2,
                          "required_weeks": 5},
}


# --------------------------------------------------------------------------
# The vocabulary.
# --------------------------------------------------------------------------


def test_every_code_the_plan_names_exists() -> None:
    expected = {
        "NO_VARIATION", "NO_OVERLAP", "LEAKED_FEATURE", "DEPTH_BOUNDED",
        "INSUFFICIENT_SUPPORT", "KAPPA_IMPOSSIBLE", "NO_MARGIN",
        "PLACEBO_OVERLAP", "OVERLAPPING_TREATMENTS", "ROI_UNBOUNDED",
        "HORIZON_TOO_SHORT",
    }
    assert set(REASON_CODES) == expected


@pytest.mark.parametrize("code", sorted(_EVIDENCE))
def test_every_code_renders_a_message(code) -> None:
    message = message_for(code, **_EVIDENCE[code])
    assert message and message[0].isupper() and message.rstrip().endswith(".")
    assert "{" not in message   # every placeholder was filled


@pytest.mark.parametrize("code", sorted(_EVIDENCE))
def test_no_message_blames_the_user_or_says_error(code) -> None:
    """The skill: never blame the user, never use the word error."""
    message = message_for(code, **_EVIDENCE[code]).lower()
    for banned in ("error", "invalid", "you failed", "your fault", "bad data"):
        assert banned not in message, banned


@pytest.mark.parametrize("code", sorted(_EVIDENCE))
def test_every_message_says_what_would_change_it(code) -> None:
    """The skill: name what is missing and what it would take to fix."""
    message = message_for(code, **_EVIDENCE[code]).lower()
    assert any(
        phrase in message
        for phrase in ("needs", "need", "would", "extending", "available", "instead")
    )


def test_placebo_message_does_not_claim_the_effect_is_absent() -> None:
    """The skill's sharpest rule, and the easiest one to break.

    An estimate inside the placebo band means this comparison cannot see the
    effect. It is not evidence the promotion did nothing.
    """
    message = message_for("PLACEBO_OVERLAP", **_EVIDENCE["PLACEBO_OVERLAP"])
    assert "not evidence that the promotion did nothing" in message
    assert "no effect" not in message.lower()


def test_every_code_declares_a_severity_and_a_reason_for_it() -> None:
    for code, spec in REASON_CODES.items():
        assert spec.severity in {"pass", "bounded", "refuse"}, code
        assert spec.why_this_severity, code
        assert spec.trigger, code


def test_bounded_codes_are_the_ones_leaving_a_weaker_quantity() -> None:
    bounded = {c for c, s in REASON_CODES.items() if s.severity == "bounded"}
    assert bounded == {
        "DEPTH_BOUNDED", "NO_MARGIN", "OVERLAPPING_TREATMENTS", "ROI_UNBOUNDED",
    }


def test_insufficient_support_and_depth_bounded_differ_in_severity() -> None:
    """Different diagnoses, different actions — the whole reason they split."""
    assert REASON_CODES["INSUFFICIENT_SUPPORT"].severity == "refuse"
    assert REASON_CODES["DEPTH_BOUNDED"].severity == "bounded"
    thin = message_for("INSUFFICIENT_SUPPORT", **_EVIDENCE["INSUFFICIENT_SUPPORT"])
    bounded = message_for("DEPTH_BOUNDED", **_EVIDENCE["DEPTH_BOUNDED"])
    assert "not a finding about the promotion" in thin
    assert "ranked against" in bounded


def test_unknown_code_raises() -> None:
    with pytest.raises(UnknownReasonCode, match="not a defined reason code"):
        message_for("NEEDS_MORE_DATA")


def test_missing_evidence_raises_rather_than_rendering_a_gap() -> None:
    with pytest.raises(KeyError, match="needs evidence"):
        message_for("KAPPA_IMPOSSIBLE", depth=0.4)


def test_gate_result_knows_when_it_refuses() -> None:
    refusing = GateResult(gate="g", status="refuse", reason_code="NO_VARIATION",
                          message="m")
    passing = GateResult(gate="g", status="pass", message="m")
    assert refusing.refuses is True
    assert passing.refuses is False


# --------------------------------------------------------------------------
# Panels built to fire, or not fire, each runnable gate.
# --------------------------------------------------------------------------


def _panel(rows: list[dict], *, covariates: bool = True) -> pd.DataFrame:
    base = {
        "PRODUCT_ID": 1, "STORE_ID": 1, "WEEK_NO": 1, "units": 5,
        "treated": False, "in_mailer": False, "treatment_observed": True,
    }
    frame = pd.DataFrame([{**base, **r} for r in rows])
    if covariates:
        rng = np.random.default_rng(0)
        for column in DEFAULT_COVARIATES:
            frame[column] = rng.normal(size=len(frame)).astype("float32")
    return frame


def _clean_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Variation everywhere, no collisions, treatment unrelated to covariates."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "PRODUCT_ID": int(rng.integers(1, 12)),
            "STORE_ID": int(rng.integers(1, 5)),
            "WEEK_NO": int(rng.integers(1, 20)),
            "units": int(rng.integers(1, 20)),
            "treated": bool(rng.random() < 0.4),
            "in_mailer": False,
        }
        for _ in range(n)
    ]
    return _panel(rows)


def _cycles(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["COMMODITY_DESC", "horizon_weeks", "low_support"]
    )


def _codes(results: list[GateResult]) -> list[str | None]:
    return [r.reason_code for r in results]


def test_no_variation_fires_when_everything_is_treated() -> None:
    panel = _panel([{"PRODUCT_ID": p, "treated": True} for p in (1, 2, 3)])
    results, diag = run_audit(
        CampaignSpec(name="c"), panel, run_overlap=False, stop_on_refuse=False
    )
    assert "NO_VARIATION" in _codes(results)
    assert diag["verdict"] == "not identified"


def test_no_variation_does_not_fire_on_a_mixed_panel() -> None:
    results, _ = run_audit(
        CampaignSpec(name="c"), _clean_panel(), run_overlap=False,
        stop_on_refuse=False,
    )
    assert "NO_VARIATION" not in _codes(results)


def test_overlapping_treatments_fires_and_is_bounded() -> None:
    rows = (
        [{"PRODUCT_ID": 1, "WEEK_NO": w, "treated": True, "in_mailer": True}
         for w in range(1, 6)]
        + [{"PRODUCT_ID": 2, "WEEK_NO": w} for w in range(1, 6)]
    )
    results, diag = run_audit(
        CampaignSpec(name="c"), _panel(rows), run_overlap=False,
        stop_on_refuse=False,
    )
    fired = [r for r in results if r.reason_code == "OVERLAPPING_TREATMENTS"]
    assert fired and fired[0].status == "bounded"
    assert diag["verdict"] in {"bounded", "not identified"}


def test_overlapping_treatments_does_not_fire_without_a_mailer() -> None:
    results, _ = run_audit(
        CampaignSpec(name="c"), _clean_panel(), run_overlap=False,
        stop_on_refuse=False,
    )
    assert "OVERLAPPING_TREATMENTS" not in _codes(results)


def test_kappa_impossible_fires_when_depth_exceeds_margin() -> None:
    campaign = CampaignSpec(name="c", depth=0.45, margin=0.30)
    results, _ = run_audit(
        campaign, _clean_panel(), run_overlap=False, stop_on_refuse=False
    )
    fired = [r for r in results if r.reason_code == "KAPPA_IMPOSSIBLE"]
    assert fired and fired[0].status == "refuse"
    assert "cannot cover its own discount" in fired[0].message


def test_kappa_impossible_does_not_fire_when_the_margin_covers_the_depth() -> None:
    campaign = CampaignSpec(name="c", depth=0.20, margin=0.40)
    results, _ = run_audit(
        campaign, _clean_panel(), run_overlap=False, stop_on_refuse=False
    )
    assert "KAPPA_IMPOSSIBLE" not in _codes(results)


def test_no_margin_fires_when_none_is_supplied_and_is_bounded() -> None:
    campaign = CampaignSpec(name="c", depth=0.242)
    results, _ = run_audit(
        campaign, _clean_panel(), run_overlap=False, stop_on_refuse=False
    )
    fired = [r for r in results if r.reason_code == "NO_MARGIN"]
    assert fired and fired[0].status == "bounded"
    assert "24.2%" in fired[0].message


def test_no_margin_does_not_fire_when_a_margin_is_supplied() -> None:
    campaign = CampaignSpec(name="c", depth=0.20, margin=0.40)
    results, _ = run_audit(
        campaign, _clean_panel(), run_overlap=False, stop_on_refuse=False
    )
    assert "NO_MARGIN" not in _codes(results)


def test_horizon_too_short_fires_against_the_cycle() -> None:
    campaign = CampaignSpec(name="c", commodity="SOUP", horizon_weeks=2)
    results, _ = run_audit(
        campaign, _clean_panel(), cycles=_cycles([("SOUP", 5, False)]),
        run_overlap=False, stop_on_refuse=False,
    )
    fired = [r for r in results if r.reason_code == "HORIZON_TOO_SHORT"]
    assert fired and fired[0].status == "refuse"
    assert "every 5 weeks" in fired[0].message


def test_horizon_too_short_does_not_fire_on_a_long_enough_window() -> None:
    campaign = CampaignSpec(name="c", commodity="SOUP", horizon_weeks=9)
    results, _ = run_audit(
        campaign, _clean_panel(), cycles=_cycles([("SOUP", 5, False)]),
        run_overlap=False, stop_on_refuse=False,
    )
    assert "HORIZON_TOO_SHORT" not in _codes(results)


def test_an_unknown_cycle_passes_but_says_it_was_not_checked() -> None:
    campaign = CampaignSpec(name="c", commodity="GHOST", horizon_weeks=2)
    results, _ = run_audit(
        campaign, _clean_panel(), cycles=_cycles([("SOUP", 5, False)]),
        run_overlap=False, stop_on_refuse=False,
    )
    horizon = next(r for r in results if r.gate == "horizon")
    assert horizon.status == "pass"
    assert "has not been shown to be long enough" in horizon.message


def test_leaked_feature_fires_when_a_covariate_is_the_treatment() -> None:
    panel = _clean_panel(600, seed=3)
    panel["store_traffic"] = panel["treated"].astype("float32")
    results, _ = run_audit(
        CampaignSpec(name="c"), panel, stop_on_refuse=False,
        n_folds=3, n_estimators=60,
    )
    fired = [r for r in results if r.reason_code == "LEAKED_FEATURE"]
    assert fired and fired[0].status == "refuse"
    assert "store_traffic" in fired[0].message


def test_no_overlap_fires_when_many_covariates_separate_the_groups() -> None:
    panel = _clean_panel(600, seed=4)
    score = sum(panel[c] for c in DEFAULT_COVARIATES[:6])
    panel["treated"] = score > score.median()
    results, _ = run_audit(
        CampaignSpec(name="c"), panel, stop_on_refuse=False,
        n_folds=3, n_estimators=400,
    )
    fired = [r for r in results if r.reason_code == "NO_OVERLAP"]
    assert fired and fired[0].status == "refuse"


def test_overlap_gate_passes_on_random_assignment() -> None:
    results, _ = run_audit(
        CampaignSpec(name="c"), _clean_panel(600, seed=5), stop_on_refuse=False,
        n_folds=3, n_estimators=30,
    )
    overlap = next(r for r in results if r.gate == "overlap")
    assert overlap.status == "pass"
    assert overlap.reason_code is None


# --------------------------------------------------------------------------
# Short-circuiting.
# --------------------------------------------------------------------------


def test_the_pipeline_stops_at_the_first_refusal() -> None:
    campaign = CampaignSpec(name="c", depth=0.45, margin=0.30)   # refuses first
    results, diag = run_audit(campaign, _clean_panel())
    assert diag["stopped_at"] == "break_even"
    assert [r.gate for r in results] == ["break_even"]
    # Everything after the stop is skipped, including the expensive fit.
    assert "overlap" in diag["gates_skipped"]
    assert "variation" in diag["gates_skipped"]


def test_gates_run_cheapest_first() -> None:
    """Cheapest first, so a campaign failing on arithmetic never pays for a fit.

    The tail order is the part that needs stating, because it is about
    dependencies rather than cost. `overlap` fits a model. `placebo` needs an
    estimate to already exist plus a band of at least 300 rollouts. `roi` is
    last because it needs **both** the Task 5.1 cost total and the Phase 4
    estimate — it cannot run until every other input exists.

    Written out so the intent survives the next gate someone adds.
    """
    assert GATE_ORDER[0] == "break_even"
    assert GATE_ORDER[-3:] == ("overlap", "placebo", "roi")


def test_stop_on_refuse_false_collects_every_verdict() -> None:
    campaign = CampaignSpec(name="c", depth=0.45, margin=0.30)
    stopped, stopped_diag = run_audit(campaign, _clean_panel(), run_overlap=False)
    full, full_diag = run_audit(
        campaign, _clean_panel(), run_overlap=False, stop_on_refuse=False
    )
    assert len(full) > len(stopped)
    assert stopped_diag["stopped_at"] == "break_even"
    assert full_diag["stopped_at"] is None


def test_a_bounded_gate_does_not_stop_the_pipeline() -> None:
    """Bounded means a weaker quantity survives, so the run continues."""
    campaign = CampaignSpec(name="c", depth=0.242)   # NO_MARGIN, bounded
    results, diag = run_audit(campaign, _clean_panel(), run_overlap=False)
    assert "NO_MARGIN" in _codes(results)
    assert diag["stopped_at"] is None
    assert len(results) > 1


def test_a_clean_campaign_is_measurable() -> None:
    campaign = CampaignSpec(
        name="c", depth=0.20, margin=0.40, commodity="SOUP", horizon_weeks=9
    )
    results, diag = run_audit(
        campaign, _clean_panel(600, seed=5), cycles=_cycles([("SOUP", 5, False)]),
        n_folds=3, n_estimators=30,
    )
    assert diag["verdict"] == "measurable"
    assert all(r.status == "pass" for r in results)
    assert diag["stopped_at"] is None


def test_missing_campaign_inputs_pass_with_a_stated_reason() -> None:
    """A gate whose input is absent must not be silently skipped."""
    results, _ = run_audit(
        CampaignSpec(name="c"), _clean_panel(), run_overlap=False
    )
    break_even = next(r for r in results if r.gate == "break_even")
    horizon = next(r for r in results if r.gate == "horizon")
    assert break_even.status == "pass" and "not run" in break_even.message
    assert horizon.status == "pass" and "not checked" in horizon.message


def test_audit_is_json_serialisable(tmp_path: Path) -> None:
    campaign = CampaignSpec(name="c", depth=0.45, margin=0.30)
    results, diag = run_audit(campaign, _clean_panel())
    path = write_audit(results, diag, tmp_path / "audit.json")
    payload = json.loads(path.read_text())
    assert payload["verdict"] == "not identified"
    assert payload["results"][0]["reason_code"] == "KAPPA_IMPOSSIBLE"


# --------------------------------------------------------------------------
# A real campaign.
# --------------------------------------------------------------------------


@real_data
def test_real_campaign_gets_a_full_verdict() -> None:
    """The Phase 3 done-when: a full verdict for a real campaign."""
    campaign = CampaignSpec(
        name="median-depth display on soup",
        commodity="SOUP",
        depth=0.242,
        horizon_weeks=9,
    )
    results, diag = run_audit(
        campaign, PANEL, cycles=CYCLES, stop_on_refuse=False, run_overlap=False
    )
    assert {r.gate for r in results} == {
        "break_even", "horizon", "price_status", "variation", "collisions",
    }
    assert diag["verdict"] in {"measurable", "bounded", "not identified"}
    # Known from Tasks 3.3 and 3.4 on this panel.
    assert "NO_MARGIN" in _codes(results)
    assert "OVERLAPPING_TREATMENTS" in _codes(results)
    assert "NO_VARIATION" not in _codes(results)


@real_data
def test_real_impossible_campaign_stops_before_any_model_runs() -> None:
    campaign = CampaignSpec(name="deep", depth=0.60, margin=0.30)
    results, diag = run_audit(campaign, PANEL, cycles=CYCLES)
    assert diag["stopped_at"] == "break_even"
    assert diag["gates_run"] == ["break_even"]
    assert results[0].reason_code == "KAPPA_IMPOSSIBLE"


# --------------------------------------------------------------------------
# The price-status gate.
# --------------------------------------------------------------------------


def _statuses(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["PRODUCT_ID", "price_status", "deal_share", "n_weeks_priced"]
    )


def test_depth_bounded_fires_for_an_always_on_deal_product() -> None:
    statuses = _statuses([(7, "bounded", 0.96, 40)])
    results, diag = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(), statuses=statuses,
        run_overlap=False, stop_on_refuse=False,
    )
    fired = [r for r in results if r.reason_code == "DEPTH_BOUNDED"]
    assert fired and fired[0].status == "bounded"
    assert "96%" in fired[0].message
    # Bounded leaves a weaker quantity, so the run continues.
    assert diag["stopped_at"] is None


def test_depth_bounded_does_not_fire_for_an_identified_product() -> None:
    statuses = _statuses([(7, "identified", 0.30, 40)])
    results, _ = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(), statuses=statuses,
        run_overlap=False, stop_on_refuse=False,
    )
    assert "DEPTH_BOUNDED" not in _codes(results)
    gate = next(r for r in results if r.gate == "price_status")
    assert gate.status == "pass"
    assert "usable number" in gate.message


def test_insufficient_support_fires_for_a_thinly_seen_product() -> None:
    statuses = _statuses([(7, "insufficient_support", 0.0, 2)])
    results, diag = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(), statuses=statuses,
        run_overlap=False, stop_on_refuse=False,
    )
    fired = [r for r in results if r.reason_code == "INSUFFICIENT_SUPPORT"]
    assert fired and fired[0].status == "refuse"
    assert "only 2 priced weeks" in fired[0].message
    assert diag["verdict"] == "not identified"


def test_insufficient_support_does_not_fire_for_a_well_seen_product() -> None:
    statuses = _statuses([(7, "identified", 0.30, 99)])
    results, _ = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(), statuses=statuses,
        run_overlap=False, stop_on_refuse=False,
    )
    assert "INSUFFICIENT_SUPPORT" not in _codes(results)


def test_the_two_price_codes_carry_different_severities() -> None:
    """The whole reason settled decision 6 split them."""
    bounded_run, _ = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(),
        statuses=_statuses([(7, "bounded", 0.96, 40)]),
        run_overlap=False, stop_on_refuse=False,
    )
    thin_run, _ = run_audit(
        CampaignSpec(name="c", product=7), _clean_panel(),
        statuses=_statuses([(7, "insufficient_support", 0.0, 2)]),
        run_overlap=False, stop_on_refuse=False,
    )
    bounded = next(r for r in bounded_run if r.gate == "price_status")
    thin = next(r for r in thin_run if r.gate == "price_status")
    assert (bounded.status, thin.status) == ("bounded", "refuse")


def test_no_product_passes_but_states_what_it_could_not_judge() -> None:
    """A commodity-level campaign still audits; it just gets no depth verdict."""
    statuses = _statuses(
        [(1, "identified", 0.3, 40), (2, "bounded", 0.95, 40),
         (3, "insufficient_support", 0.0, 2)]
    )
    results, diag = run_audit(
        CampaignSpec(name="c", commodity="SOUP"), _clean_panel(),
        statuses=statuses, run_overlap=False, stop_on_refuse=False,
    )
    gate = next(r for r in results if r.gate == "price_status")
    assert gate.status == "pass"
    assert gate.reason_code is None
    # It reports the portfolio distribution and says why it cannot judge.
    assert "No product was named" in gate.message
    assert "naming the product is what makes a verdict possible" in gate.message
    assert gate.detail["product_supplied"] is False
    assert diag["verdict"] != "not identified"


def test_an_unknown_product_is_not_a_pass_in_substance() -> None:
    statuses = _statuses([(7, "identified", 0.3, 40)])
    results, _ = run_audit(
        CampaignSpec(name="c", product=999), _clean_panel(), statuses=statuses,
        run_overlap=False, stop_on_refuse=False,
    )
    gate = next(r for r in results if r.gate == "price_status")
    assert gate.status == "pass"
    assert "has not been shown to be usable" in gate.message


@real_data
def test_real_price_status_gate_fires_on_real_products() -> None:
    """Each of the three statuses, drawn from the real price table."""
    statuses = pd.read_parquet(
        "data/interim/prices.parquet",
        columns=["PRODUCT_ID", "price_status", "deal_share", "n_weeks_priced"],
    ).drop_duplicates("PRODUCT_ID")
    expected = {
        "bounded": ("bounded", "DEPTH_BOUNDED"),
        "insufficient_support": ("refuse", "INSUFFICIENT_SUPPORT"),
        "identified": ("pass", None),
    }
    for label, (status, code) in expected.items():
        product = int(statuses[statuses.price_status == label].iloc[0].PRODUCT_ID)
        results, _ = run_audit(
            CampaignSpec(name=label, product=product), PANEL,
            cycles=CYCLES, run_overlap=False, stop_on_refuse=False,
        )
        gate = next(r for r in results if r.gate == "price_status")
        assert (gate.status, gate.reason_code) == (status, code), label


@real_data
def test_real_full_six_gate_run_including_overlap() -> None:
    """The Phase 3 done-when, pinned rather than demonstrated by hand.

    Heavy: the overlap fit is about two and a half minutes on the full panel.
    """
    campaign = CampaignSpec(
        name="soup display, median depth",
        commodity="SOUP",
        depth=0.242,
        horizon_weeks=9,
    )
    results, diag = run_audit(campaign, PANEL, cycles=CYCLES, stop_on_refuse=False)

    assert [r.gate for r in results] == list(GATE_ORDER)
    assert diag["gates_skipped"] == []
    assert diag["stopped_at"] is None
    assert diag["verdict"] == "bounded"

    by_gate = {r.gate: r for r in results}
    assert by_gate["overlap"].status == "pass"        # the fit actually ran
    assert by_gate["break_even"].reason_code == "NO_MARGIN"
    assert by_gate["collisions"].reason_code == "OVERLAPPING_TREATMENTS"
    assert by_gate["variation"].status == "pass"
