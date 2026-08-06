"""Tests for Task 2.2 cleaning.

The fixture tests build a small transaction frame whose every flag is hit a
known number of times, so the diagnostics can be checked against arithmetic
done by hand. The `real_data` tests then assert the stage reproduces the Phase 1
totals on the actual file.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from promo.clean import (
    CASCADE,
    VOLUME_MEASURED_DEPARTMENTS,
    JoinError,
    clean_transactions,
    write_diagnostics,
)
from promo.io import load_raw

RAW = Path("data/raw")
INTERIM = Path("data/interim")

_NO_DATA = (
    not (RAW / "product.csv").exists()
    or not (INTERIM / "transactions.parquet").exists()
)


def real_data(fn):
    """Marks a test that reads a real artefact from data/interim.

    Heavy by definition — see "Test discipline" in CLAUDE.md — so the fast
    pass excludes it with -m "not heavy", and it is skipped outright when
    the artefact is absent.
    """
    return pytest.mark.skipif(_NO_DATA, reason="data/raw or the transactions mirror is not populated")(
        pytest.mark.heavy(fn)
    )



@pytest.fixture
def product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PRODUCT_ID": [1, 2, 3, 4, 5],
            "DEPARTMENT": ["GROCERY", "KIOSK-GAS", "MISC SALES TRAN", " ", "DRUG GM"],
            "COMMODITY_DESC": ["SOUP", "COUPON/MISC ITEMS", "COUPON/MISC ITEMS", " ", "VITAMINS"],
            "MANUFACTURER": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Eight rows, each chosen to hit one flag.

    | row | product | quantity | sales | flag                          |
    |-----|---------|----------|-------|-------------------------------|
    | 0,1 | 1       | 2, 3     | 4, 6  | clean                         |
    | 2   | 1       | 0        | 0.5   | nonpositive_quantity          |
    | 3   | 1       | 2        | 0     | free_good (retained)          |
    | 4   | 2       | 5000     | 12.5  | volume_measured               |
    | 5   | 3       | 4000     | 10.0  | volume_measured               |
    | 6   | 4       | 1        | 2.0   | blank_department (retained)   |
    | 7   | 9       | 1        | 3.0   | unmatched_product             |
    """
    return pd.DataFrame(
        {
            "household_key": [1, 1, 2, 2, 3, 3, 4, 4],
            "BASKET_ID": [10, 10, 11, 11, 12, 12, 13, 13],
            "DAY": [1, 1, 2, 2, 3, 3, 4, 4],
            "WEEK_NO": [1, 1, 1, 1, 2, 2, 2, 2],
            "PRODUCT_ID": [1, 1, 1, 1, 2, 3, 4, 9],
            "STORE_ID": [100, 100, 101, 101, 102, 102, 103, 103],
            "QUANTITY": [2, 3, 0, 2, 5000, 4000, 1, 1],
            "SALES_VALUE": [4.0, 6.0, 0.5, 0.0, 12.5, 10.0, 2.0, 3.0],
            "RETAIL_DISC": [0.0] * 8,
            "COUPON_DISC": [0.0] * 8,
            "COUPON_MATCH_DISC": [0.0] * 8,
        }
    )


def _by_name(diag: dict) -> dict[str, dict]:
    return {f["name"]: f for f in diag["filters"]}


def test_no_row_is_ever_dropped(transactions, product) -> None:
    cleaned, diag = clean_transactions(transactions, product)
    assert len(cleaned) == len(transactions)
    assert diag["totals_before"]["rows"] == len(transactions)


def test_each_flag_fires_on_the_expected_rows(transactions, product) -> None:
    cleaned, _ = clean_transactions(transactions, product)
    assert cleaned["nonpositive_quantity"].tolist() == [
        False, False, True, False, False, False, False, False,
    ]
    assert cleaned["free_good"].tolist() == [
        False, False, False, True, False, False, False, False,
    ]
    assert cleaned["volume_measured"].tolist() == [
        False, False, False, False, True, True, False, False,
    ]
    assert cleaned["blank_department"].tolist() == [
        False, False, False, False, False, False, True, False,
    ]
    assert cleaned["unmatched_product"].tolist() == [
        False, False, False, False, False, False, False, True,
    ]


def test_usable_excludes_the_cascade_and_keeps_the_retained_flags(
    transactions, product
) -> None:
    cleaned, _ = clean_transactions(transactions, product)
    # rows 2 (zero quantity), 4, 5 (volume), 7 (unmatched) are out;
    # row 3 (free good) and row 6 (blank department) stay in.
    assert cleaned["usable"].tolist() == [
        True, True, False, True, False, False, True, False,
    ]


def test_free_goods_survive_for_phase_5(transactions, product) -> None:
    """Task 5.1 reads these rows from here. Losing them understates cost."""
    cleaned, diag = clean_transactions(transactions, product)
    free = cleaned[cleaned["free_good"]]
    assert len(free) == 1
    assert bool(free["usable"].iloc[0]) is True
    entry = _by_name(diag)["free_good"]
    assert entry["action"] == "flag"
    assert entry["before"] == entry["after"]
    assert entry["rows_still_usable"] == 1


def test_diagnostics_account_for_every_removed_row(transactions, product) -> None:
    """The per-filter effects must sum to the total removed — no double count."""
    _, diag = clean_transactions(transactions, product)
    charged = sum(
        f["attributed_to_this_stage"]["rows"]
        for f in diag["filters"]
        if f["action"] == "exclude"
    )
    assert charged == diag["rows_removed"]
    assert (
        diag["totals_before"]["rows"] - charged == diag["totals_usable"]["rows"]
    )


def test_cascade_chains_before_to_after(transactions, product) -> None:
    _, diag = clean_transactions(transactions, product)
    excludes = [f for f in diag["filters"] if f["action"] == "exclude"]
    assert [f["name"] for f in excludes] == list(CASCADE)
    assert excludes[0]["before"] == diag["totals_before"]
    for earlier, later in pairwise(excludes):
        assert earlier["after"] == later["before"]
    assert excludes[-1]["after"] == diag["totals_usable"]


def test_units_and_sales_are_conserved_across_the_cascade(
    transactions, product
) -> None:
    _, diag = clean_transactions(transactions, product)
    for f in diag["filters"]:
        if f["action"] != "exclude":
            continue
        for key in ("rows", "units"):
            assert (
                f["before"][key] - f["attributed_to_this_stage"][key]
                == f["after"][key]
            ), f["name"]


def test_overlapping_rows_are_charged_once_to_the_first_rule(product) -> None:
    """A zero-quantity kiosk row belongs to the cascade's first rule, not both."""
    tx = pd.DataFrame(
        {
            "household_key": [1],
            "BASKET_ID": [1],
            "DAY": [1],
            "WEEK_NO": [1],
            "PRODUCT_ID": [2],  # KIOSK-GAS
            "STORE_ID": [1],
            "QUANTITY": [0],
            "SALES_VALUE": [1.0],
            "RETAIL_DISC": [0.0],
            "COUPON_DISC": [0.0],
            "COUPON_MATCH_DISC": [0.0],
        }
    )
    cleaned, diag = clean_transactions(tx, product)
    assert bool(cleaned["nonpositive_quantity"].iloc[0]) is True
    assert bool(cleaned["volume_measured"].iloc[0]) is True
    by_name = _by_name(diag)
    assert by_name["nonpositive_quantity"]["attributed_to_this_stage"]["rows"] == 1
    assert by_name["volume_measured"]["attributed_to_this_stage"]["rows"] == 0
    assert by_name["volume_measured"]["flagged_rows_total"] == 1
    assert diag["rows_removed"] == 1


def test_duplicate_product_ids_raise_rather_than_fan_out(transactions, product) -> None:
    doubled = pd.concat([product, product.iloc[:1]], ignore_index=True)
    with pytest.raises(JoinError, match="duplicate"):
        clean_transactions(transactions, doubled)


def test_category_columns_are_joined(transactions, product) -> None:
    cleaned, _ = clean_transactions(transactions, product)
    assert cleaned.loc[0, "DEPARTMENT"] == "GROCERY"
    assert cleaned.loc[0, "COMMODITY_DESC"] == "SOUP"
    assert pd.isna(cleaned.loc[7, "DEPARTMENT"])  # unmatched product


def test_writes_parquet_when_asked(transactions, product, tmp_path: Path) -> None:
    out = tmp_path / "clean.parquet"
    cleaned, diag = clean_transactions(transactions, product, out_path=out)
    assert diag["written_to"] == str(out)
    assert len(pd.read_parquet(out)) == len(cleaned)


# --------------------------------------------------------------------------
# The real file.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_clean():
    tables, _ = load_raw(RAW, INTERIM)
    return clean_transactions(tables["transaction_data"], tables["product"])


@real_data
def test_real_totals_match_phase_1(real_clean) -> None:
    _, diag = real_clean
    before = diag["totals_before"]
    assert before["rows"] == 2_595_732
    assert before["units"] == 260_685_622
    assert before["sales_value"] == pytest.approx(8_057_463.08, abs=0.01)


@real_data
def test_real_flag_counts_match_phase_1(real_clean) -> None:
    _, diag = real_clean
    by_name = _by_name(diag)
    assert by_name["nonpositive_quantity"]["flagged_rows_total"] == 14_466
    assert by_name["free_good"]["flagged_rows_total"] == 4_451
    assert by_name["free_good"]["attributed_to_this_stage"]["units"] == 4_544
    assert by_name["unmatched_product"]["flagged_rows_total"] == 0


@real_data
def test_real_free_goods_reach_phase_5_except_seven_in_a_kiosk_department(
    real_clean,
) -> None:
    """4,444 of 4,451 survive as usable.

    The other seven sit in MISC SALES TRAN and are excluded as volume-measured,
    so Task 5.1 must read free goods by the `free_good` flag rather than from
    the usable subset, or it silently loses them.
    """
    _, diag = real_clean
    free = _by_name(diag)["free_good"]
    assert free["flagged_rows_total"] == 4_451
    assert free["rows_still_usable"] == 4_444
    assert free["rows_already_excluded_by"] == {"volume_measured": 7}


@real_data
def test_real_blank_department_rows_retain_nothing(real_clean) -> None:
    """The 15 blank-department products carry only zero-quantity rows."""
    _, diag = real_clean
    blank = _by_name(diag)["blank_department"]
    assert blank["flagged_rows_total"] == 7_839
    assert blank["rows_still_usable"] == 0
    assert blank["rows_already_excluded_by"] == {"nonpositive_quantity": 7_839}


@real_data
def test_real_volume_exclusion_removes_the_units_phase_1_predicted(real_clean) -> None:
    """Decision 2's headline: about 1% of rows carrying about 99% of units."""
    _, diag = real_clean
    vol = _by_name(diag)["volume_measured"]
    assert vol["share_of_all"]["rows_share"] == pytest.approx(0.011, abs=0.002)
    assert vol["share_of_all"]["units_share"] == pytest.approx(0.987, abs=0.002)


@real_data
def test_real_cross_check_shows_the_two_rules_differ(real_clean) -> None:
    _, diag = real_clean
    cross = diag["volume_rule_cross_check"]
    # The threshold's extra rows are the zero-priced giveaways Phase 5 needs.
    assert cross["threshold_only"]["rows"] > 0
    assert cross["threshold_only"]["of_which_free_goods"] > 0
    # The department rule's extra rows are kiosk lines the threshold misses.
    assert cross["department_only"]["rows"] > 0
    assert set(cross["department_only"]["departments"]) <= set(
        VOLUME_MEASURED_DEPARTMENTS
    )


@real_data
def test_real_diagnostics_are_json_serialisable(real_clean, tmp_path: Path) -> None:
    import json

    _, diag = real_clean
    path = write_diagnostics(diag, tmp_path / "clean_diagnostics.json")
    assert json.loads(path.read_text())["stage"] == "clean_transactions"
