"""Ingest and schema contract for the eight Dunnhumby CSVs.

`load_raw()` is the single door every later stage uses to reach raw data. It
enforces a declared schema rather than trusting inference, and returns a report
that records what was found, what was missing, and what was refused.

Two things about this module are deliberate and are not defects.

**The two large tables are not loaded.** `CLAUDE.md` forbids reading
`transaction_data.csv` into pandas (2.6M rows) and forbids loading
`causal_data.csv` into pandas at all (36.8M rows). The plan's phrasing for Task
2.1 says "into a dict of DataFrames"; the memory rule wins, and this is where
that is said out loud. Those two tables come back as `LazyTable` handles that
carry the schema, the row count, and a DuckDB source expression, and that refuse
to materialise. Every fact the report states about them was computed by one
streaming aggregate pass, never by holding the table in memory. The deviation is
recorded in `IngestReport.memory_policy` so a reader of the report sees it
without reading this docstring.

**Money stays float64.** The float32 rule in `CLAUDE.md` is about derived
feature columns in the Phase 2.6 panel, where the row count is the constraint.
Summing `SALES_VALUE` over 2.6M rows in float32 loses cents to accumulation
error, and the accounting layer in Phase 5 divides by those sums.

Nothing here cleans, filters, or repairs. Whitespace-only strings are counted
and reported, not stripped; `product.csv` encodes a missing `DEPARTMENT` as a
single space, and Phase 2.2 needs to see that rather than inherit a silent fix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel

__all__ = [
    "MARGIN_SEARCH_TERMS",
    "TABLE_SPECS",
    "ColumnReport",
    "ColumnSpec",
    "IngestReport",
    "LazyTable",
    "MemoryPolicyError",
    "RangeReport",
    "SchemaContractError",
    "TableReport",
    "TableSpec",
    "connect",
    "load_raw",
    "write_ingest_report",
]

DType = Literal["int64", "float64", "string"]

_DUCKDB_TYPE: dict[DType, str] = {
    "int64": "BIGINT",
    "float64": "DOUBLE",
    "string": "VARCHAR",
}

_DUCKDB_TO_DTYPE: dict[str, DType] = {
    "TINYINT": "int64",
    "SMALLINT": "int64",
    "INTEGER": "int64",
    "BIGINT": "int64",
    "HUGEINT": "int64",
    "UTINYINT": "int64",
    "USMALLINT": "int64",
    "UINTEGER": "int64",
    "UBIGINT": "int64",
    "FLOAT": "float64",
    "DOUBLE": "float64",
    "DECIMAL": "float64",
    "VARCHAR": "string",
}

#: Column-name substrings searched across all eight tables to establish, rather
#: than assert, that no cost or margin column exists. See `IngestReport`.
MARGIN_SEARCH_TERMS: tuple[str, ...] = ("cogs", "margin", "cost", "profit", "gross")

#: The DuckDB row ceiling from CLAUDE.md: anything larger is aggregated in SQL.
MAX_MATERIALISED_ROWS = 5_000_000


class SchemaContractError(Exception):
    """A raw file does not match its declared schema.

    Raised, not returned as a `GateResult`: a missing required column or a
    column of the wrong type is a broken input, not a data condition the
    pipeline can render a verdict about.
    """


class MemoryPolicyError(Exception):
    """An operation would breach the memory rules in CLAUDE.md."""


@dataclass(frozen=True)
class ColumnSpec:
    """One declared column.

    `required` means a settled Phase 1 decision depends on it, so its absence
    stops ingest. Optional columns are recorded as present or missing and never
    stop anything.
    """

    name: str
    dtype: DType
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    """One declared table, and how it is allowed to be read."""

    name: str
    filename: str
    columns: tuple[ColumnSpec, ...]
    strategy: Literal["eager", "lazy"]
    mirror: str | None = None
    range_columns: tuple[str, ...] = ()
    reason: str = ""

    @property
    def by_name(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)


TABLE_SPECS: dict[str, TableSpec] = {
    "transaction_data": TableSpec(
        name="transaction_data",
        filename="transaction_data.csv",
        strategy="lazy",
        mirror="transactions.parquet",
        range_columns=("DAY", "WEEK_NO"),
        reason="2.6M rows; CLAUDE.md forbids reading the CSV directly.",
        columns=(
            ColumnSpec("household_key", "int64"),
            ColumnSpec("BASKET_ID", "int64"),
            ColumnSpec("DAY", "int64"),
            ColumnSpec("PRODUCT_ID", "int64"),
            ColumnSpec("QUANTITY", "int64"),
            ColumnSpec("SALES_VALUE", "float64"),
            ColumnSpec("STORE_ID", "int64"),
            ColumnSpec("RETAIL_DISC", "float64"),
            ColumnSpec("TRANS_TIME", "string", required=False),
            ColumnSpec("WEEK_NO", "int64"),
            ColumnSpec("COUPON_DISC", "float64"),
            ColumnSpec("COUPON_MATCH_DISC", "float64"),
        ),
    ),
    "causal_data": TableSpec(
        name="causal_data",
        filename="causal_data.csv",
        strategy="lazy",
        range_columns=("WEEK_NO",),
        reason="36.8M rows; CLAUDE.md forbids loading it into pandas.",
        columns=(
            ColumnSpec("PRODUCT_ID", "int64"),
            ColumnSpec("STORE_ID", "int64"),
            ColumnSpec("WEEK_NO", "int64"),
            # Categorical codes, not booleans. '0' means none, and mailer uses
            # letters. Read as strings so the raw codes survive to Phase 2.5.
            ColumnSpec("display", "string"),
            ColumnSpec("mailer", "string"),
        ),
    ),
    "product": TableSpec(
        name="product",
        filename="product.csv",
        strategy="eager",
        columns=(
            ColumnSpec("PRODUCT_ID", "int64"),
            ColumnSpec("MANUFACTURER", "int64", required=False),
            # Required: settled decision 2 excludes volume-measured rows by
            # DEPARTMENT, so ingest cannot proceed without it.
            ColumnSpec("DEPARTMENT", "string"),
            ColumnSpec("BRAND", "string", required=False),
            ColumnSpec("COMMODITY_DESC", "string"),
            ColumnSpec("SUB_COMMODITY_DESC", "string", required=False),
            ColumnSpec("CURR_SIZE_OF_PRODUCT", "string", required=False),
        ),
    ),
    "hh_demographic": TableSpec(
        name="hh_demographic",
        filename="hh_demographic.csv",
        strategy="eager",
        columns=(
            ColumnSpec("household_key", "int64"),
            ColumnSpec("AGE_DESC", "string", required=False),
            ColumnSpec("MARITAL_STATUS_CODE", "string", required=False),
            ColumnSpec("INCOME_DESC", "string", required=False),
            ColumnSpec("HOMEOWNER_DESC", "string", required=False),
            ColumnSpec("HH_COMP_DESC", "string", required=False),
            ColumnSpec("HOUSEHOLD_SIZE_DESC", "string", required=False),
            ColumnSpec("KID_CATEGORY_DESC", "string", required=False),
        ),
    ),
    "campaign_desc": TableSpec(
        name="campaign_desc",
        filename="campaign_desc.csv",
        strategy="eager",
        range_columns=("START_DAY", "END_DAY"),
        columns=(
            ColumnSpec("DESCRIPTION", "string", required=False),
            ColumnSpec("CAMPAIGN", "int64"),
            ColumnSpec("START_DAY", "int64"),
            ColumnSpec("END_DAY", "int64"),
        ),
    ),
    "campaign_table": TableSpec(
        name="campaign_table",
        filename="campaign_table.csv",
        strategy="eager",
        columns=(
            ColumnSpec("DESCRIPTION", "string", required=False),
            ColumnSpec("household_key", "int64"),
            ColumnSpec("CAMPAIGN", "int64"),
        ),
    ),
    "coupon": TableSpec(
        name="coupon",
        filename="coupon.csv",
        strategy="eager",
        columns=(
            ColumnSpec("COUPON_UPC", "int64"),
            ColumnSpec("PRODUCT_ID", "int64"),
            ColumnSpec("CAMPAIGN", "int64"),
        ),
    ),
    "coupon_redempt": TableSpec(
        name="coupon_redempt",
        filename="coupon_redempt.csv",
        strategy="eager",
        range_columns=("DAY",),
        columns=(
            ColumnSpec("household_key", "int64"),
            ColumnSpec("DAY", "int64"),
            ColumnSpec("COUPON_UPC", "int64"),
            ColumnSpec("CAMPAIGN", "int64"),
        ),
    ),
}


class ColumnReport(BaseModel):
    """What the contract asked of one column, and what the file gave."""

    name: str
    declared_dtype: str | None
    observed_dtype: str | None
    required: bool
    present: bool
    #: Values that are whitespace-only rather than NA. `product.csv` encodes a
    #: missing DEPARTMENT this way. Counted, never stripped.
    blank_string_rows: int | None = None


class RangeReport(BaseModel):
    column: str
    min: int | float | None
    max: int | float | None


class TableReport(BaseModel):
    name: str
    source_path: str
    source_kind: Literal["csv", "parquet_mirror"]
    load_strategy: Literal["eager", "lazy"]
    rows: int
    columns: list[ColumnReport]
    missing_optional_columns: list[str] = []
    unexpected_columns: list[str] = []
    ranges: list[RangeReport] = []
    notes: list[str] = []

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.present]


class IngestReport(BaseModel):
    """The schema contract's verdict on `data/raw`.

    The margin fields are the ones Phase 5 depends on. `margin_candidate_columns`
    is the result of searching every column name in all eight tables for cost
    and margin terms; it comes back empty, which is why `has_margin` is False.
    The claim is therefore established at ingest rather than asserted, and it
    travels downstream as the reason code `NO_MARGIN` instead of surfacing as a
    crash when the accounting layer looks for a column that was never there.
    """

    raw_dir: str
    tables: list[TableReport]

    has_cogs: bool
    has_margin: bool
    margin_reason_code: str | None
    margin_note: str
    margin_search_terms: list[str]
    margin_candidate_columns: list[str]

    memory_policy: list[str] = []
    notes: list[str] = []

    def table(self, name: str) -> TableReport:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def row_counts(self) -> dict[str, int]:
        return {t.name: t.rows for t in self.tables}


@dataclass
class LazyTable:
    """A table too large to hold in memory, exposed as SQL only.

    `source_sql` is a DuckDB table function with the declared column types
    pinned, so a query against this handle reads the file under the same
    contract an eager load would enforce.
    """

    name: str
    path: Path
    source_kind: Literal["csv", "parquet_mirror"]
    source_sql: str
    rows: int
    columns: tuple[str, ...]
    max_rows: int = MAX_MATERIALISED_ROWS
    _notes: list[str] = field(default_factory=list)

    def sql(
        self,
        query: str,
        con: duckdb.DuckDBPyConnection | None = None,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Run `query` against this table and return the result as a DataFrame.

        Write `{t}` where the table belongs:

            causal.sql("SELECT WEEK_NO, COUNT(*) AS n FROM {t} GROUP BY 1")

        The result is counted before it is fetched, so a query that forgot to
        aggregate raises instead of exhausting memory.
        """
        limit = self.max_rows if max_rows is None else max_rows
        rendered = query.format(t=self.source_sql)
        own = con is None
        con = connect() if con is None else con
        try:
            n = con.execute(f"SELECT COUNT(*) FROM ({rendered})").fetchone()[0]
            if n > limit:
                raise MemoryPolicyError(
                    f"{self.name}: query would materialise {n:,} rows, over the "
                    f"{limit:,} ceiling. Aggregate it in SQL first."
                )
            return con.execute(rendered).df()
        finally:
            if own:
                con.close()

    def to_pandas(self) -> pd.DataFrame:
        raise MemoryPolicyError(
            f"{self.name} has {self.rows:,} rows and is never loaded into pandas. "
            f"Use .sql('SELECT ... FROM {{t}} GROUP BY ...') and materialise the "
            f"aggregate."
        )


def connect(memory_limit: str = "2GB", threads: int = 2) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with the CLAUDE.md limits applied."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET enable_progress_bar=false")
    return con


def _check_required(spec: TableSpec, present: list[str]) -> None:
    missing = [c for c in spec.required_names if c not in present]
    if missing:
        raise SchemaContractError(
            f"{spec.filename} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(present)}"
        )


def _load_eager(spec: TableSpec, path: Path) -> tuple[pd.DataFrame, TableReport]:
    header = pd.read_csv(path, nrows=0)
    present = list(header.columns)
    _check_required(spec, present)

    by_name = spec.by_name
    dtypes = {c: by_name[c].dtype for c in present if c in by_name}
    try:
        df = pd.read_csv(path, dtype=dtypes)
    except ValueError as exc:
        # Re-raised, not swallowed: pandas names the offending value, the
        # contract names the file and the declaration it contradicts.
        raise SchemaContractError(f"{spec.filename}: {exc}") from exc

    columns: list[ColumnReport] = []
    for c in spec.columns:
        if c.name not in present:
            columns.append(
                ColumnReport(
                    name=c.name,
                    declared_dtype=c.dtype,
                    observed_dtype=None,
                    required=c.required,
                    present=False,
                )
            )
            continue
        blank: int | None = None
        if c.dtype == "string":
            s = df[c.name].astype("string")
            blank = int((s.notna() & (s.str.strip() == "")).sum())
        columns.append(
            ColumnReport(
                name=c.name,
                declared_dtype=c.dtype,
                observed_dtype=str(df[c.name].dtype),
                required=c.required,
                present=True,
                blank_string_rows=blank,
            )
        )

    unexpected = [c for c in present if c not in by_name]
    for c in unexpected:
        columns.append(
            ColumnReport(
                name=c,
                declared_dtype=None,
                observed_dtype=str(df[c].dtype),
                required=False,
                present=True,
            )
        )

    ranges = [
        RangeReport(
            column=c,
            min=None if df[c].empty else df[c].min().item(),
            max=None if df[c].empty else df[c].max().item(),
        )
        for c in spec.range_columns
        if c in present
    ]

    notes: list[str] = []
    padded = [(c.name, c.blank_string_rows or 0) for c in columns if c.blank_string_rows]
    if padded:
        listed = ", ".join(f"{name} ({n:,})" for name, n in padded)
        notes.append(
            f"whitespace-only values present in {listed} — missingness encoded as "
            f"a space, not NA. Not stripped here."
        )

    report = TableReport(
        name=spec.name,
        source_path=str(path),
        source_kind="csv",
        load_strategy="eager",
        rows=len(df),
        columns=columns,
        missing_optional_columns=[
            c.name for c in spec.columns if not c.required and c.name not in present
        ],
        unexpected_columns=unexpected,
        ranges=ranges,
        notes=notes,
    )
    return df, report


def _describe(con: duckdb.DuckDBPyConnection, source_sql: str) -> dict[str, str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {source_sql} LIMIT 0").fetchall()
    return {r[0]: r[1] for r in rows}


def _profile_lazy(
    spec: TableSpec,
    path: Path,
    source_kind: Literal["csv", "parquet_mirror"],
    con: duckdb.DuckDBPyConnection,
) -> tuple[LazyTable, TableReport]:
    """Read the schema and one streaming aggregate pass. Never materialises."""
    probe = (
        f"read_parquet('{path.as_posix()}')"
        if source_kind == "parquet_mirror"
        else f"read_csv_auto('{path.as_posix()}')"
    )
    observed = _describe(con, probe)
    present = list(observed)
    _check_required(spec, present)

    by_name = spec.by_name
    unexpected = [c for c in present if c not in by_name]

    # Check the declared types before the scan, not after. Pinning a type the
    # file contradicts would otherwise surface as a DuckDB conversion error part
    # way through a full pass, which names the row rather than the contract.
    for c in spec.columns:
        if c.name not in present:
            continue
        mapped = _DUCKDB_TO_DTYPE.get(observed[c.name].split("(")[0])
        if mapped is not None and mapped != c.dtype:
            raise SchemaContractError(
                f"{spec.filename}.{c.name}: declared {c.dtype}, file holds "
                f"{observed[c.name]}"
            )

    # Pin the declared types on the source expression so every downstream query
    # reads under the contract rather than under DuckDB's inference. Columns the
    # contract does not know about are carried through as VARCHAR rather than
    # dropped, so an unexpected column is visible instead of silently gone.
    if source_kind == "csv":
        cols = ", ".join(
            f"'{c}': '{_DUCKDB_TYPE[by_name[c].dtype] if c in by_name else 'VARCHAR'}'"
            for c in present
        )
        source_sql = (
            f"read_csv('{path.as_posix()}', header=true, auto_detect=false, "
            f"columns={{{cols}}})"
        )
    else:
        source_sql = f"read_parquet('{path.as_posix()}')"

    parts = ["COUNT(*) AS n_rows"]
    for c in spec.range_columns:
        if c in present:
            parts += [f'MIN("{c}") AS "{c}__min"', f'MAX("{c}") AS "{c}__max"']
    string_cols = [
        c.name for c in spec.columns if c.name in present and c.dtype == "string"
    ]
    for c in string_cols:
        parts.append(
            f'SUM(CASE WHEN "{c}" IS NOT NULL AND TRIM("{c}") = \'\' THEN 1 ELSE 0 END) '
            f'AS "{c}__blank"'
        )
    agg = con.execute(f"SELECT {', '.join(parts)} FROM {source_sql}").df().iloc[0]

    columns: list[ColumnReport] = []
    for c in spec.columns:
        if c.name not in present:
            columns.append(
                ColumnReport(
                    name=c.name,
                    declared_dtype=c.dtype,
                    observed_dtype=None,
                    required=c.required,
                    present=False,
                )
            )
            continue
        columns.append(
            ColumnReport(
                name=c.name,
                declared_dtype=c.dtype,
                observed_dtype=observed[c.name],
                required=c.required,
                present=True,
                blank_string_rows=(
                    int(agg[f"{c.name}__blank"]) if c.name in string_cols else None
                ),
            )
        )
    for c in unexpected:
        columns.append(
            ColumnReport(
                name=c,
                declared_dtype=None,
                observed_dtype=observed[c],
                required=False,
                present=True,
            )
        )

    # A one-row aggregate frame carrying several DuckDB types comes back
    # upcast, so a week number arrives as 9.0. Put it back on its declared type.
    def _typed(value: object, dtype: DType) -> int | float:
        return int(value) if dtype == "int64" else float(value)  # type: ignore[arg-type]

    ranges = [
        RangeReport(
            column=c,
            min=_typed(agg[f"{c}__min"], by_name[c].dtype),
            max=_typed(agg[f"{c}__max"], by_name[c].dtype),
        )
        for c in spec.range_columns
        if c in present
    ]

    rows = int(agg["n_rows"])
    notes = [
        f"lazy: {spec.reason}",
        (
            "row count and ranges computed by one streaming DuckDB pass; the "
            "table was never held in memory."
        ),
    ]
    if source_kind == "parquet_mirror":
        notes.append(
            f"read from the interim mirror, not {spec.filename}, per CLAUDE.md."
        )

    handle = LazyTable(
        name=spec.name,
        path=path,
        source_kind=source_kind,
        source_sql=source_sql,
        rows=rows,
        columns=tuple(present),
        _notes=notes,
    )
    report = TableReport(
        name=spec.name,
        source_path=str(path),
        source_kind=source_kind,
        load_strategy="lazy",
        rows=rows,
        columns=columns,
        missing_optional_columns=[
            c.name for c in spec.columns if not c.required and c.name not in present
        ],
        unexpected_columns=unexpected,
        ranges=ranges,
        notes=notes,
    )
    return handle, report


def load_raw(
    raw_dir: str | Path = "data/raw",
    interim_dir: str | Path = "data/interim",
    *,
    verify_mirror: bool = False,
) -> tuple[dict[str, pd.DataFrame | LazyTable], IngestReport]:
    """Load the eight raw tables under an explicit schema contract.

    The six small tables come back as DataFrames with declared dtypes. The two
    large ones come back as `LazyTable` handles — see the module docstring for
    why, and `IngestReport.memory_policy` for the same statement in the report.

    Args:
        raw_dir: directory holding the eight CSVs.
        interim_dir: where the transactions parquet mirror lives.
        verify_mirror: if True, count the rows of `transaction_data.csv` itself
            and record whether the mirror agrees. Off by default because it
            costs a full scan of the CSV the mirror exists to avoid.

    Returns:
        `(tables, report)`.

    Raises:
        SchemaContractError: a required column is missing, or a column's type
            contradicts the contract.
        FileNotFoundError: a raw file is absent.
    """
    raw = Path(raw_dir)
    interim = Path(interim_dir)

    missing_files = [
        s.filename for s in TABLE_SPECS.values() if not (raw / s.filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"missing from {raw}: {', '.join(sorted(missing_files))}"
        )

    tables: dict[str, pd.DataFrame | LazyTable] = {}
    reports: list[TableReport] = []
    extra_notes: list[str] = []

    con = connect()
    try:
        for spec in TABLE_SPECS.values():
            if spec.strategy == "eager":
                df, rep = _load_eager(spec, raw / spec.filename)
                tables[spec.name] = df
                reports.append(rep)
                continue

            path = raw / spec.filename
            kind: Literal["csv", "parquet_mirror"] = "csv"
            if spec.mirror is not None:
                mirror_path = interim / spec.mirror
                if mirror_path.exists():
                    path, kind = mirror_path, "parquet_mirror"
                else:
                    extra_notes.append(
                        f"{spec.name}: mirror {mirror_path} absent, falling back to "
                        f"the CSV. Build the mirror before Phase 2.2."
                    )
            handle, rep = _profile_lazy(spec, path, kind, con)

            if verify_mirror and kind == "parquet_mirror":
                csv_rows = con.execute(
                    f"SELECT COUNT(*) FROM read_csv_auto("
                    f"'{(raw / spec.filename).as_posix()}')"
                ).fetchone()[0]
                agrees = int(csv_rows) == handle.rows
                rep.notes.append(
                    f"mirror verified against {spec.filename}: "
                    f"{handle.rows:,} vs {int(csv_rows):,} rows, "
                    f"{'agree' if agrees else 'DISAGREE'}."
                )
                if not agrees:
                    raise SchemaContractError(
                        f"{spec.name}: mirror holds {handle.rows:,} rows, "
                        f"{spec.filename} holds {int(csv_rows):,}."
                    )

            tables[spec.name] = handle
            reports.append(rep)
    finally:
        con.close()

    every_column = [c.name for r in reports for c in r.columns if c.present]
    candidates = sorted(
        {c for c in every_column for t in MARGIN_SEARCH_TERMS if t in c.lower()}
    )

    report = IngestReport(
        raw_dir=str(raw),
        tables=reports,
        has_cogs=bool(candidates),
        has_margin=bool(candidates),
        margin_reason_code=None if candidates else "NO_MARGIN",
        margin_note=(
            "No COGS or margin column exists in any of the eight tables. Searched "
            f"{len(every_column)} column names for {list(MARGIN_SEARCH_TERMS)} and "
            "found none. True ROI is therefore not computable on this dataset. The "
            "honest Phase 5 output is the break-even margin plus a sensitivity "
            "table; a margin is never imputed."
        )
        if not candidates
        else (
            "Candidate cost or margin columns found — the NO_MARGIN premise this "
            "project was built on may not hold for this input. Inspect before use."
        ),
        margin_search_terms=list(MARGIN_SEARCH_TERMS),
        margin_candidate_columns=candidates,
        memory_policy=[
            (
                "transaction_data and causal_data are returned as LazyTable "
                "handles, not DataFrames. CLAUDE.md forbids loading them into "
                "pandas; the plan's 'dict of DataFrames' phrasing yields to that "
                "rule."
            ),
            f"LazyTable.sql() refuses any result over {MAX_MATERIALISED_ROWS:,} rows.",
            "DuckDB connections are opened with memory_limit=2GB and threads=2.",
        ],
        notes=[
            (
                "Money columns are float64. The float32 rule applies to derived "
                "feature columns in Phase 2.6, not to raw values the accounting "
                "layer sums."
            ),
            (
                "No cleaning, filtering, or whitespace repair happens here. "
                "Whitespace-only strings are counted per column and left in place."
            ),
            *extra_notes,
        ],
    )
    return tables, report


def write_ingest_report(report: IngestReport, path: str | Path) -> Path:
    """Write the report as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(), indent=2) + "\n")
    return out
