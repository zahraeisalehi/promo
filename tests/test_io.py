"""Tests for the Task 2.1 schema contract.

Two kinds of test here. The ones marked `real_data` assert facts about the eight
files in `data/raw`, and are skipped when those files are absent. The rest run
against tiny fixtures written to a tmp dir, so the contract's failure modes are
exercised without touching the real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.io import (
    TABLE_SPECS,
    IngestReport,
    LazyTable,
    MemoryPolicyError,
    SchemaContractError,
    load_raw,
    write_ingest_report,
)

RAW = Path("data/raw")
INTERIM = Path("data/interim")

real_data = pytest.mark.skipif(
    not all((RAW / s.filename).exists() for s in TABLE_SPECS.values()),
    reason="data/raw is not populated",
)


@pytest.fixture(scope="module")
def loaded() -> tuple[dict[str, object], IngestReport]:
    return load_raw(RAW, INTERIM)


# --------------------------------------------------------------------------
# Fixtures: a miniature data/raw, so contract violations can be constructed.
# --------------------------------------------------------------------------


def _write_minimal(root: Path, *, drop: tuple[str, str] | None = None) -> Path:
    """Write a one-row version of each of the eight files under `root`.

    `drop` names a (table, column) pair to omit, which is how the required- and
    optional-column paths are exercised.
    """
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "int64": 1,
        "float64": 1.0,
        "string": "x",
    }
    for spec in TABLE_SPECS.values():
        cols = {
            c.name: [values[c.dtype]]
            for c in spec.columns
            if not (drop is not None and drop == (spec.name, c.name))
        }
        pd.DataFrame(cols).to_csv(raw / spec.filename, index=False)
    return raw


def test_missing_required_column_raises(tmp_path: Path) -> None:
    raw = _write_minimal(tmp_path, drop=("product", "DEPARTMENT"))
    with pytest.raises(SchemaContractError, match="DEPARTMENT"):
        load_raw(raw, tmp_path / "interim")


def test_missing_optional_column_is_recorded_not_raised(tmp_path: Path) -> None:
    raw = _write_minimal(tmp_path, drop=("product", "CURR_SIZE_OF_PRODUCT"))
    _, report = load_raw(raw, tmp_path / "interim")
    product = report.table("product")
    assert product.missing_optional_columns == ["CURR_SIZE_OF_PRODUCT"]
    absent = [c for c in product.columns if c.name == "CURR_SIZE_OF_PRODUCT"]
    assert absent and absent[0].present is False


def test_missing_file_raises(tmp_path: Path) -> None:
    raw = _write_minimal(tmp_path)
    (raw / "coupon.csv").unlink()
    with pytest.raises(FileNotFoundError, match="coupon.csv"):
        load_raw(raw, tmp_path / "interim")


def test_declared_type_conflict_raises(tmp_path: Path) -> None:
    """A lazy table whose file contradicts the contract stops ingest."""
    raw = _write_minimal(tmp_path)
    bad = pd.read_csv(raw / "causal_data.csv")
    bad["WEEK_NO"] = "week-one"  # declared int64
    bad.to_csv(raw / "causal_data.csv", index=False)
    with pytest.raises(SchemaContractError, match="WEEK_NO"):
        load_raw(raw, tmp_path / "interim")


def test_falls_back_to_csv_and_says_so_when_mirror_absent(tmp_path: Path) -> None:
    raw = _write_minimal(tmp_path)
    tables, report = load_raw(raw, tmp_path / "interim")
    assert report.table("transaction_data").source_kind == "csv"
    assert any("mirror" in n and "absent" in n for n in report.notes)
    assert isinstance(tables["transaction_data"], LazyTable)


# --------------------------------------------------------------------------
# The real files.
# --------------------------------------------------------------------------


@real_data
def test_large_tables_are_never_dataframes(loaded) -> None:
    tables, _ = loaded
    for name in ("transaction_data", "causal_data"):
        assert isinstance(tables[name], LazyTable), name
        assert not isinstance(tables[name], pd.DataFrame), name


@real_data
def test_lazy_table_refuses_to_materialise(loaded) -> None:
    tables, _ = loaded
    with pytest.raises(MemoryPolicyError):
        tables["causal_data"].to_pandas()


@real_data
def test_lazy_table_refuses_an_unaggregated_query(loaded) -> None:
    tables, _ = loaded
    with pytest.raises(MemoryPolicyError, match="ceiling"):
        tables["causal_data"].sql("SELECT * FROM {t}")


@real_data
def test_lazy_table_answers_an_aggregate(loaded) -> None:
    tables, _ = loaded
    weeks = tables["causal_data"].sql(
        "SELECT WEEK_NO, COUNT(*) AS n FROM {t} GROUP BY 1 ORDER BY 1"
    )
    assert len(weeks) == 93  # weeks 9-101, settled decision 5
    assert int(weeks["WEEK_NO"].min()) == 9
    assert int(weeks["WEEK_NO"].max()) == 101


@real_data
def test_small_tables_are_dataframes_with_declared_dtypes(loaded) -> None:
    tables, report = loaded
    for spec in TABLE_SPECS.values():
        if spec.strategy != "eager":
            continue
        df = tables[spec.name]
        assert isinstance(df, pd.DataFrame), spec.name
        for c in spec.columns:
            if c.name not in df.columns:
                continue
            kind = df[c.name].dtype.kind
            if c.dtype == "int64":
                assert kind == "i", f"{spec.name}.{c.name} is {df[c.name].dtype}"
            elif c.dtype == "float64":
                assert kind == "f", f"{spec.name}.{c.name} is {df[c.name].dtype}"
            else:
                assert kind not in "if", f"{spec.name}.{c.name} is numeric"
    assert report.table("product").rows == len(tables["product"])


@real_data
def test_ids_are_never_float(loaded) -> None:
    """BASKET_ID loses precision as float64; check every declared key column."""
    _, report = loaded
    for table in report.tables:
        for col in table.columns:
            if col.name.endswith("_ID") or col.name in {"household_key", "COUPON_UPC"}:
                assert col.declared_dtype == "int64", f"{table.name}.{col.name}"
                assert "FLOAT" not in (col.observed_dtype or "").upper()
                assert "DOUBLE" not in (col.observed_dtype or "").upper()


@real_data
def test_row_counts_match_phase_1(loaded) -> None:
    _, report = loaded
    assert report.row_counts["transaction_data"] == 2_595_732
    assert report.row_counts["causal_data"] == 36_786_524
    assert report.row_counts["product"] == 92_353
    assert report.row_counts["hh_demographic"] == 801


@real_data
def test_ranges_match_phase_1(loaded) -> None:
    _, report = loaded
    tx = {r.column: (r.min, r.max) for r in report.table("transaction_data").ranges}
    assert tx["DAY"] == (1, 711)
    assert tx["WEEK_NO"] == (1, 102)
    causal = {r.column: (r.min, r.max) for r in report.table("causal_data").ranges}
    assert causal["WEEK_NO"] == (9, 101)


@real_data
def test_no_margin_is_established_not_asserted(loaded) -> None:
    _, report = loaded
    assert report.margin_candidate_columns == []
    assert report.has_margin is False
    assert report.has_cogs is False
    assert report.margin_reason_code == "NO_MARGIN"


@real_data
def test_transactions_read_from_the_mirror(loaded) -> None:
    _, report = loaded
    tx = report.table("transaction_data")
    assert tx.source_kind == "parquet_mirror"
    assert tx.source_path.endswith("transactions.parquet")


@real_data
def test_blank_departments_are_counted_not_repaired(loaded) -> None:
    """product.csv encodes a missing DEPARTMENT as ' '. Phase 2.2 must see it."""
    tables, report = loaded
    dept = next(c for c in report.table("product").columns if c.name == "DEPARTMENT")
    assert dept.blank_string_rows == 15
    raw_values = tables["product"]["DEPARTMENT"].astype("string")
    assert (raw_values == " ").sum() == 15  # not stripped, not nulled


@real_data
def test_report_round_trips_to_json(loaded, tmp_path: Path) -> None:
    _, report = loaded
    path = write_ingest_report(report, tmp_path / "ingest_report.json")
    assert IngestReport.model_validate_json(path.read_text()) == report
