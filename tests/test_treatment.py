"""Tests for Task 2.5, the treatment join.

Fixture tests write a miniature `causal_data.csv` to a tmp dir so the collapse,
the duplicate rule, and the coverage envelope can be checked against arithmetic
done by hand. The `real_data` tests then assert the stage reproduces Task 1.4's
figures on the actual log.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.treatment import (
    DUPLICATE_RULES,
    TREATMENT_DEFINITIONS,
    build_treatment_panel,
    write_diagnostics,
)

PANEL = Path("data/interim/prices.parquet")
CAUSAL = Path("data/raw/causal_data.csv")

real_data = pytest.mark.skipif(
    not PANEL.exists() or not CAUSAL.exists(),
    reason="run Task 2.4 first, and data/raw must be populated",
)


def _panel(keys: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PRODUCT_ID": [k[0] for k in keys],
            "STORE_ID": [k[1] for k in keys],
            "WEEK_NO": [k[2] for k in keys],
            "units": 1,
            "sales_value": 1.0,
        }
    )


def _causal(tmp_path: Path, rows: list[tuple[int, int, int, str, str]]) -> Path:
    path = tmp_path / "causal_data.csv"
    pd.DataFrame(
        rows, columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"]
    ).to_csv(path, index=False)
    return path


def test_booleans_and_raw_codes_are_both_kept(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10), (2, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A"), (2, 1, 10, "0", "D")])
    out, _ = build_treatment_panel(panel, causal)

    first = out[out.PRODUCT_ID == 1].iloc[0]
    assert bool(first["on_display"]) is True
    assert bool(first["in_mailer"]) is True
    assert first["display_code"] == "9"  # the raw categorical code survives
    assert first["mailer_code"] == "A"

    second = out[out.PRODUCT_ID == 2].iloc[0]
    assert bool(second["on_display"]) is False
    assert second["display_code"] == "0"  # observed '0', not absence
    assert bool(second["in_causal_data"]) is True


def test_absent_key_is_untreated_with_no_code(tmp_path: Path) -> None:
    """Absence and an observed '0' must stay distinguishable."""
    panel = _panel([(1, 1, 10), (2, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A")])
    out, _ = build_treatment_panel(panel, causal)

    absent = out[out.PRODUCT_ID == 2].iloc[0]
    assert bool(absent["in_causal_data"]) is False
    assert bool(absent["on_display"]) is False
    assert bool(absent["treated"]) is False
    assert pd.isna(absent["display_code"])


def test_any_treated_wins_resolves_a_conflicting_key(tmp_path: Path) -> None:
    """The 15,208-key case: '0' against a real code, both rows present."""
    panel = _panel([(1, 1, 10)])
    causal = _causal(
        tmp_path, [(1, 1, 10, "0", "A"), (1, 1, 10, "5", "A")]
    )
    out, diag = build_treatment_panel(panel, causal)
    assert len(out) == 1  # collapse before join: no row duplication
    assert bool(out.iloc[0]["on_display"]) is True
    assert out.iloc[0]["display_code"] == "5"
    assert diag["duplicate_rule"]["rule"] == "any_treated_wins"
    assert diag["duplicate_rule"]["duplicate_keys"] == 1


def test_collapse_happens_before_the_join(tmp_path: Path) -> None:
    """A key appearing three times must still yield exactly one panel row."""
    panel = _panel([(1, 1, 10)])
    causal = _causal(
        tmp_path,
        [(1, 1, 10, "0", "A"), (1, 1, 10, "5", "A"), (1, 1, 10, "0", "D")],
    )
    out, _ = build_treatment_panel(panel, causal)
    assert len(out) == 1


def test_alternative_duplicate_rules_change_the_answer(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "0", "A"), (1, 1, 10, "5", "A")])

    any_wins, _ = build_treatment_panel(panel, causal, duplicate_rule="any_treated_wins")
    agree, _ = build_treatment_panel(panel, causal, duplicate_rule="all_must_agree")
    dropped, _ = build_treatment_panel(panel, causal, duplicate_rule="drop_conflicts")

    assert bool(any_wins.iloc[0]["treated"]) is True
    assert bool(agree.iloc[0]["treated"]) is False
    # Dropping the conflict leaves the key absent from the log entirely.
    assert bool(dropped.iloc[0]["in_causal_data"]) is False
    assert bool(dropped.iloc[0]["treated"]) is False


def test_unknown_duplicate_rule_raises(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A")])
    with pytest.raises(ValueError, match="duplicate_rule must be one of"):
        build_treatment_panel(panel, causal, duplicate_rule="coin_flip")


def test_treatment_definition_is_a_parameter(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10), (2, 1, 10), (3, 1, 10)])
    causal = _causal(
        tmp_path,
        [
            (1, 1, 10, "9", "0"),  # display only
            (2, 1, 10, "0", "A"),  # mailer only
            (3, 1, 10, "9", "A"),  # both
        ],
    )
    by = {}
    for name in TREATMENT_DEFINITIONS:
        out, diag = build_treatment_panel(panel, causal, definition=name)
        by[name] = set(out.loc[out["treated"], "PRODUCT_ID"])
        assert diag["definition"]["treated"] == name

    assert by["display"] == {1, 3}
    assert by["mailer"] == {2, 3}
    assert by["display_or_mailer"] == {1, 2, 3}
    assert by["display_and_mailer"] == {3}


def test_definition_accepts_a_callable(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10), (2, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "0"), (2, 1, 10, "0", "A")])
    out, diag = build_treatment_panel(
        panel, causal, definition=lambda f: f["in_mailer"] & ~f["on_display"]
    )
    assert set(out.loc[out["treated"], "PRODUCT_ID"]) == {2}
    assert diag["definition"]["treated"] == "<callable>"


def test_unknown_definition_name_raises(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A")])
    with pytest.raises(ValueError, match="definition must be one of"):
        build_treatment_panel(panel, causal, definition="coupon")


def test_rows_outside_the_envelope_are_unobserved_not_untreated(
    tmp_path: Path,
) -> None:
    """The distinction the plan's phrasing glosses, and Phase 3 depends on."""
    panel = _panel(
        [
            (1, 1, 10),  # in the log
            (2, 1, 10),  # store in the log, week in range, absent: a real zero
            (3, 9, 10),  # store the log never covers: unobserved
            (4, 1, 99),  # week outside the log's range: unobserved
        ]
    )
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A"), (5, 1, 20, "5", "A")])
    out, diag = build_treatment_panel(panel, causal)
    observed = out.set_index("PRODUCT_ID")["treatment_observed"]

    assert bool(observed[1]) is True
    assert bool(observed[2]) is True  # absence here is informative
    assert bool(observed[3]) is False  # store never in the log
    assert bool(observed[4]) is False  # week 99 is past the log's max of 20

    assumption = diag["absence_assumption"]
    assert assumption["rows_relying_on_it"] == 1  # product 2 only
    assert assumption["rows_outside_envelope"] == 2


def test_treated_rows_are_always_inside_the_envelope(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10), (2, 9, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A")])
    out, _ = build_treatment_panel(panel, causal)
    assert not out.loc[out["treated"], "treatment_observed"].eq(False).any()


def test_panel_row_count_is_never_changed(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10), (2, 1, 10), (3, 1, 11)])
    causal = _causal(
        tmp_path, [(1, 1, 10, "0", "A"), (1, 1, 10, "5", "A"), (2, 1, 10, "9", "0")]
    )
    out, _ = build_treatment_panel(panel, causal)
    assert len(out) == 3
    assert out[["PRODUCT_ID", "STORE_ID", "WEEK_NO"]].equals(
        panel[["PRODUCT_ID", "STORE_ID", "WEEK_NO"]]
    )


def test_writes_parquet_when_asked(tmp_path: Path) -> None:
    panel = _panel([(1, 1, 10)])
    causal = _causal(tmp_path, [(1, 1, 10, "9", "A")])
    out = tmp_path / "panel_treated.parquet"
    frame, diag = build_treatment_panel(panel, causal, out_path=out)
    assert diag["written_to"] == str(out)
    assert len(pd.read_parquet(out)) == len(frame)


# --------------------------------------------------------------------------
# The real log.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_treated():
    return build_treatment_panel(PANEL, CAUSAL)


@real_data
def test_real_join_conserves_panel_rows(real_treated) -> None:
    out, diag = real_treated
    assert len(out) == 2_349_560
    assert diag["coverage"]["panel_rows"] == len(out)
    assert not out.duplicated(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).any()


@real_data
def test_real_log_matches_task_1_4(real_treated) -> None:
    _, diag = real_treated
    log = diag["log"]
    assert log["rows"] == 36_786_524
    assert log["keys"] == 36_771_279
    assert (log["week_min"], log["week_max"]) == (9, 101)
    assert log["stores"] == 115


@real_data
def test_real_log_contains_no_untreated_row(real_treated) -> None:
    """The premise the duplicate rule and the absence reading both rest on."""
    _, diag = real_treated
    assert diag["log"]["untreated_rows"] == 0
    assert diag["log"]["is_a_treatment_log"] is True


@real_data
def test_real_conflict_evidence_is_total(real_treated) -> None:
    """Every zero-display conflict row carries a real mailer. No exception."""
    _, diag = real_treated
    conflicts = diag["duplicate_rule"]
    assert conflicts["duplicate_keys"] == 15_245
    assert conflicts["display_disagrees"] == 15_208
    assert conflicts["zero_display_rows"] == 15_208
    assert conflicts["zero_display_rows_with_a_real_mailer"] == 15_208
    assert conflicts["zero_display_with_mailer_share"] == 1.0
    # No display conflict is real-code-against-real-code, so the rule never has
    # to choose between two genuine display codes.
    assert conflicts["display_real_vs_real"] == 0
    # Nine keys carry two real mailer codes; that moves the code, not the flag.
    assert conflicts["mailer_real_vs_real"] == 9


@real_data
def test_real_duplicate_rule_moves_only_the_conflicted_keys(real_treated) -> None:
    out, diag = real_treated
    agree, _ = build_treatment_panel(PANEL, CAUSAL, duplicate_rule="all_must_agree")
    difference = int(out["treated"].sum()) - int(agree["treated"].sum())
    assert difference == diag["duplicate_rule"]["duplicate_keys_in_panel"] == 614


@real_data
def test_real_absence_assumption_share_is_recorded(real_treated) -> None:
    _, diag = real_treated
    assumption = diag["absence_assumption"]
    assert assumption["rows_relying_on_it"] > 0
    assert assumption["rows_outside_envelope"] > 0
    total = assumption["rows_relying_on_it"] + assumption["rows_outside_envelope"]
    in_log = diag["coverage"]["in_causal_data"]["rows"]
    assert total + in_log == diag["coverage"]["panel_rows"]


@real_data
def test_real_diagnostics_are_json_serialisable(real_treated, tmp_path: Path) -> None:
    import json

    _, diag = real_treated
    path = write_diagnostics(diag, tmp_path / "treatment_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "build_treatment_panel"


@real_data
def test_real_drop_conflicts_rule_also_runs() -> None:
    """The third rule; the other two are exercised by the tests above."""
    assert set(DUPLICATE_RULES) == {
        "any_treated_wins", "all_must_agree", "drop_conflicts",
    }
    out, diag = build_treatment_panel(PANEL, CAUSAL, duplicate_rule="drop_conflicts")
    assert len(out) == 2_349_560
    assert diag["duplicate_rule"]["rule"] == "drop_conflicts"
    # Dropping the 614 conflicted panel keys removes them from the log entirely.
    assert diag["coverage"]["in_causal_data"]["rows"] == 482_672 - 614


@real_data
def test_real_no_key_fires_neither_mechanic_under_the_chosen_rule(
    real_treated,
) -> None:
    """Under any_treated_wins the log's own premise must survive the collapse."""
    _, diag = real_treated
    assert diag["duplicate_rule"]["keys_in_log_firing_neither"] == 0
