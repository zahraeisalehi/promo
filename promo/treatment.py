"""Join the treatment log to the price panel.

`causal_data.csv` is 36.8M rows and is never loaded into pandas. It is collapsed
to one row per PRODUCT_ID x STORE_ID x WEEK_NO in DuckDB, restricted to keys the
panel actually contains, and only then joined.

Three decisions govern this module, all recorded in `docs/data_findings.md`:

**The treatment is `display`** (settled decision 4), with `mailer` kept as a
covariate and never as the treatment. The definition is a parameter here rather
than a constant, because Phase 3 needs to run the audit under alternatives to
show why the default was chosen.

**Duplicate keys resolve by "any treated wins."** 15,245 product-store-weeks
appear twice and every one disagrees with itself. The rule is structural, not
conservative: `causal_data` contains no untreated rows, so it is a treatment log
rather than a panel, and a row exists only because something was promoted. A
`display = '0'` row is therefore present on account of its mailer, and its zero
display field records absence of relevance, not absence of display. Reading it
as evidence of no-display misreads the file. Verified: **100.0000% of the 15,208
zero-display conflict rows carry a non-zero mailer**, with no exception.

**Absence from the log is untreated only inside the log's coverage.**
`causal_data` reaches 115 of 582 stores and weeks 9-101. Inside that envelope a
missing key is a real "not promoted"; outside it, absence carries no information
at all and calling it untreated would manufacture controls. The two cases get
separate columns — `treated` and `treatment_observed` — because collapsing them
is how an unobserved row becomes a clean control by accident.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.io import connect

__all__ = [
    "DUPLICATE_RULES",
    "TREATMENT_DEFINITIONS",
    "CoverageError",
    "build_treatment_panel",
    "write_diagnostics",
]

#: Named treatment definitions over the two booleans. The default is settled
#: decision 4; the others exist so Phase 3 can audit the alternatives.
TREATMENT_DEFINITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "display": lambda f: f["on_display"],
    "mailer": lambda f: f["in_mailer"],
    "display_or_mailer": lambda f: f["on_display"] | f["in_mailer"],
    "display_and_mailer": lambda f: f["on_display"] & f["in_mailer"],
}

#: How a product-store-week that appears twice in the log is collapsed.
DUPLICATE_RULES = ("any_treated_wins", "all_must_agree", "drop_conflicts")

_KEY = ["PRODUCT_ID", "STORE_ID", "WEEK_NO"]


class CoverageError(Exception):
    """The join changed the panel's row count or key set."""


def build_treatment_panel(
    panel: pd.DataFrame | str | Path = "data/interim/prices.parquet",
    causal: str | Path = "data/raw/causal_data.csv",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    definition: str | Callable[[pd.DataFrame], pd.Series] = "display",
    duplicate_rule: str = "any_treated_wins",
    out_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach `on_display`, `in_mailer`, the raw codes, and `treated`.

    Args:
        panel: the Task 2.3/2.4 price panel, as a DataFrame or a parquet path.
        causal: path to `causal_data.csv`. Never read into pandas.
        con: an existing DuckDB connection; one is opened and closed if omitted.
        definition: a key of `TREATMENT_DEFINITIONS`, or a callable taking the
            frame and returning a boolean Series. Not hardcoded, so the gate can
            re-run under alternatives.
        duplicate_rule: one of `DUPLICATE_RULES`. See the module docstring for
            why the default is what it is.
        out_path: if given, the panel is also written there as parquet.

    Returns:
        `(panel, diagnostics)`. The panel keeps every input row and gains
        `display_code`, `mailer_code`, `on_display`, `in_mailer`,
        `in_causal_data`, `treatment_observed`, and `treated`.

    Raises:
        ValueError: unknown `definition` name or `duplicate_rule`.
        CoverageError: the join changed the row count — a collapse failure.
    """
    if duplicate_rule not in DUPLICATE_RULES:
        raise ValueError(
            f"duplicate_rule must be one of {DUPLICATE_RULES}, got {duplicate_rule!r}"
        )
    if isinstance(definition, str) and definition not in TREATMENT_DEFINITIONS:
        raise ValueError(
            f"definition must be one of {sorted(TREATMENT_DEFINITIONS)} or a "
            f"callable, got {definition!r}"
        )

    own = con is None
    con = connect() if con is None else con
    try:
        return _build(
            panel, causal, con,
            definition=definition,
            duplicate_rule=duplicate_rule,
            out_path=out_path,
        )
    finally:
        if own:
            con.close()


def _build(
    panel: pd.DataFrame | str | Path,
    causal: str | Path,
    con: duckdb.DuckDBPyConnection,
    *,
    definition: str | Callable[[pd.DataFrame], pd.Series],
    duplicate_rule: str,
    out_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(panel, pd.DataFrame):
        con.register("_panel_frame", panel)
        panel_sql = "_panel_frame"
        panel_frame = panel
    else:
        panel_sql = f"read_parquet('{Path(panel).as_posix()}')"
        panel_frame = con.execute(f"SELECT * FROM {panel_sql}").df()

    causal_sql = (
        f"read_csv('{Path(causal).as_posix()}', header=true, auto_detect=false, "
        f"columns={{'PRODUCT_ID': 'BIGINT', 'STORE_ID': 'BIGINT', "
        f"'WEEK_NO': 'BIGINT', 'display': 'VARCHAR', 'mailer': 'VARCHAR'}})"
    )

    envelope = con.execute(
        f"""
        SELECT MIN(WEEK_NO) AS week_min, MAX(WEEK_NO) AS week_max,
               COUNT(DISTINCT STORE_ID) AS stores, COUNT(*) AS rows,
               COUNT(DISTINCT (PRODUCT_ID, STORE_ID, WEEK_NO)) AS keys
        FROM {causal_sql}
        """
    ).df().iloc[0]

    # The log's own honesty check: a treatment log must contain no untreated
    # row. If this ever fails, "absent means untreated" loses its footing and
    # the whole reading in the module docstring has to be revisited.
    untreated_rows = con.execute(
        f"SELECT COUNT(*) FROM {causal_sql} WHERE display = '0' AND mailer = '0'"
    ).fetchone()[0]

    collapsed = _collapse(con, causal_sql, panel_sql, duplicate_rule)
    conflicts = _conflict_report(con, causal_sql, panel_sql)

    n_collapsed = len(collapsed)
    n_keys = collapsed[_KEY].drop_duplicates().shape[0]
    if n_collapsed != n_keys:
        raise CoverageError(
            f"collapse produced {n_collapsed:,} rows for {n_keys:,} keys — "
            f"the join would duplicate panel rows"
        )

    before = len(panel_frame)
    out = panel_frame.merge(collapsed, on=_KEY, how="left", validate="one_to_one")
    if len(out) != before:
        raise CoverageError(
            f"join changed the row count: {before:,} in, {len(out):,} out"
        )

    out["in_causal_data"] = out["on_display"].notna()
    out["on_display"] = out["on_display"].fillna(False).astype(bool)
    out["in_mailer"] = out["in_mailer"].fillna(False).astype(bool)

    # Absence is informative only inside the log's coverage. Outside it the row
    # is unobserved, not untreated, and must never serve as a control.
    stores = set(
        con.execute(f"SELECT DISTINCT STORE_ID FROM {causal_sql}").df()["STORE_ID"]
    )
    out["treatment_observed"] = (
        out["STORE_ID"].isin(stores)
        & out["WEEK_NO"].between(int(envelope["week_min"]), int(envelope["week_max"]))
    )

    resolver = (
        TREATMENT_DEFINITIONS[definition] if isinstance(definition, str) else definition
    )
    out["treated"] = resolver(out).astype(bool)

    _assert_absence_is_untreated(out, duplicate_rule)

    diagnostics = _diagnose(
        out,
        envelope=envelope,
        untreated_rows=int(untreated_rows),
        conflicts=conflicts,
        duplicate_rule=duplicate_rule,
        definition=definition if isinstance(definition, str) else "<callable>",
    )

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        diagnostics["written_to"] = str(path)

    return out, diagnostics


def _collapse(
    con: duckdb.DuckDBPyConnection,
    causal_sql: str,
    panel_sql: str,
    duplicate_rule: str,
) -> pd.DataFrame:
    """One row per key, restricted to keys the panel holds.

    The SEMI JOIN is what keeps this affordable: the log has 36.8M rows and the
    panel matches under half a million of them, so only that slice is
    materialised. The collapse happens before the join, never after — joining
    raw `causal_data` duplicates a panel key for every extra log row.
    """
    booleans = {
        # Any real code in any row means the mechanic fired that week.
        "any_treated_wins": (
            "MAX(CASE WHEN display <> '0' THEN 1 ELSE 0 END) = 1 AS on_display, "
            "MAX(CASE WHEN mailer  <> '0' THEN 1 ELSE 0 END) = 1 AS in_mailer"
        ),
        "all_must_agree": (
            "MIN(CASE WHEN display <> '0' THEN 1 ELSE 0 END) = 1 AS on_display, "
            "MIN(CASE WHEN mailer  <> '0' THEN 1 ELSE 0 END) = 1 AS in_mailer"
        ),
        "drop_conflicts": (
            "MAX(CASE WHEN display <> '0' THEN 1 ELSE 0 END) = 1 AS on_display, "
            "MAX(CASE WHEN mailer  <> '0' THEN 1 ELSE 0 END) = 1 AS in_mailer"
        ),
    }[duplicate_rule]
    having = " HAVING COUNT(*) = 1" if duplicate_rule == "drop_conflicts" else ""

    return con.execute(
        f"""
        WITH scoped AS (
            SELECT c.PRODUCT_ID, c.STORE_ID, c.WEEK_NO, c.display, c.mailer
            FROM {causal_sql} c
            SEMI JOIN {panel_sql} p
              ON c.PRODUCT_ID = p.PRODUCT_ID
             AND c.STORE_ID  = p.STORE_ID
             AND c.WEEK_NO   = p.WEEK_NO
        )
        SELECT
            PRODUCT_ID, STORE_ID, WEEK_NO,
            {booleans},
            -- Raw codes preserved. MAX() puts a real code ahead of '0' because
            -- '0' sorts first, which matches the boolean rule above. The 9 keys
            -- carrying two real mailer codes take the lexicographic max; that
            -- moves the code only, never the boolean.
            MAX(display) AS display_code,
            MAX(mailer)  AS mailer_code
        FROM scoped
        GROUP BY 1, 2, 3{having}
        """
    ).df()


def _conflict_report(
    con: duckdb.DuckDBPyConnection, causal_sql: str, panel_sql: str
) -> dict[str, Any]:
    """What the duplicate keys look like, and what each rule would do."""
    row = con.execute(
        f"""
        WITH dup_keys AS (
            SELECT PRODUCT_ID, STORE_ID, WEEK_NO
            FROM {causal_sql} GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
        ),
        conflicting AS (
            SELECT c.* FROM {causal_sql} c SEMI JOIN dup_keys d
              ON c.PRODUCT_ID = d.PRODUCT_ID AND c.STORE_ID = d.STORE_ID
             AND c.WEEK_NO = d.WEEK_NO
        ),
        per_key AS (
            SELECT PRODUCT_ID, STORE_ID, WEEK_NO,
                COUNT(DISTINCT display) AS n_display,
                COUNT(DISTINCT mailer) AS n_mailer,
                COUNT(DISTINCT CASE WHEN display <> '0' THEN display END) AS n_disp_real,
                COUNT(DISTINCT CASE WHEN mailer <> '0' THEN mailer END) AS n_mail_real
            FROM conflicting GROUP BY 1, 2, 3
        )
        SELECT
            (SELECT COUNT(*) FROM dup_keys) AS duplicate_keys,
            (SELECT COUNT(*) FROM conflicting WHERE display = '0') AS zero_display_rows,
            (SELECT COUNT(*) FROM conflicting
              WHERE display = '0' AND mailer <> '0') AS zero_display_with_mailer,
            (SELECT COUNT(*) FROM per_key WHERE n_display > 1) AS display_disagrees,
            (SELECT COUNT(*) FROM per_key WHERE n_disp_real > 1) AS display_real_vs_real,
            (SELECT COUNT(*) FROM per_key WHERE n_mailer > 1) AS mailer_disagrees,
            (SELECT COUNT(*) FROM per_key WHERE n_mail_real > 1) AS mailer_real_vs_real,
            (SELECT COUNT(*) FROM dup_keys d SEMI JOIN {panel_sql} p
               ON d.PRODUCT_ID = p.PRODUCT_ID AND d.STORE_ID = p.STORE_ID
              AND d.WEEK_NO = p.WEEK_NO) AS duplicate_keys_in_panel
        """
    ).df().iloc[0]

    zero_rows = int(row["zero_display_rows"])
    with_mailer = int(row["zero_display_with_mailer"])
    return {
        "duplicate_keys": int(row["duplicate_keys"]),
        "duplicate_keys_in_panel": int(row["duplicate_keys_in_panel"]),
        "display_disagrees": int(row["display_disagrees"]),
        "display_real_vs_real": int(row["display_real_vs_real"]),
        "mailer_disagrees": int(row["mailer_disagrees"]),
        "mailer_real_vs_real": int(row["mailer_real_vs_real"]),
        "zero_display_rows": zero_rows,
        "zero_display_rows_with_a_real_mailer": with_mailer,
        "zero_display_with_mailer_share": (
            round(with_mailer / zero_rows, 6) if zero_rows else None
        ),
        "evidence_for_the_rule": (
            "Every zero-display row in a conflicting key carries a real mailer. "
            "The row exists because of the mailer, so its zero display field "
            "records absence of relevance, not absence of display. That is why "
            "any-treated-wins is structural rather than conservative."
        ),
        "first_wins_rejected": (
            "Considered and rejected. The '0' record appears first in the file "
            "99.76% of the time (a real code is first on only 37 of 15,245 "
            "keys), so file order is not arbitrary and 'first wins' would "
            "function as 'untreated always wins', stripping the treatment from "
            "15,208 keys."
        ),
    }


def _silent_keys(panel: pd.DataFrame) -> pd.Series:
    """Keys present in the log that end up firing neither mechanic."""
    return panel["in_causal_data"] & ~panel["on_display"] & ~panel["in_mailer"]


def _assert_absence_is_untreated(panel: pd.DataFrame, duplicate_rule: str) -> None:
    """Invariants the join must satisfy, checked rather than trusted."""
    absent = ~panel["in_causal_data"]
    assert not panel.loc[absent, "on_display"].any(), (
        "a key absent from causal_data is marked on display"
    )
    assert not panel.loc[absent, "in_mailer"].any(), (
        "a key absent from causal_data is marked in mailer"
    )
    assert not panel.loc[panel["treated"], "treatment_observed"].eq(False).any(), (
        "a row outside the log's coverage envelope is marked treated"
    )

    # The log holds no untreated rows, so under any_treated_wins a key that
    # survives the collapse must still fire something; if it does not, the
    # collapse dropped a code and the bug is here.
    #
    # This is *not* an invariant of the other rules. all_must_agree turns a key
    # whose display and mailer both disagree into a row that fires neither —
    # deliberately, that is what the rule means. So the check is scoped to the
    # rule it actually holds for, and the count is reported for every rule.
    if duplicate_rule == "any_treated_wins":
        silent = _silent_keys(panel)
        assert not silent.any(), (
            f"{int(silent.sum()):,} keys are in causal_data but fire neither "
            f"mechanic under any_treated_wins; the collapse lost a code"
        )


def _share(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def _diagnose(
    panel: pd.DataFrame,
    *,
    envelope: pd.Series,
    untreated_rows: int,
    conflicts: dict[str, Any],
    duplicate_rule: str,
    definition: str,
) -> dict[str, Any]:
    rows = len(panel)
    observed = panel["treatment_observed"]
    in_log = panel["in_causal_data"]
    treated = panel["treated"]
    inferred = observed & ~in_log

    def _units(mask: pd.Series) -> dict[str, Any]:
        return {
            "rows": int(mask.sum()),
            "rows_share": _share(int(mask.sum()), rows),
            "units": int(panel.loc[mask, "units"].sum()),
            "sales_value": round(float(panel.loc[mask, "sales_value"].sum()), 2),
        }

    return {
        "stage": "build_treatment_panel",
        "definition": {
            "treated": definition,
            "available": sorted(TREATMENT_DEFINITIONS),
            "why": (
                "Settled decision 4: display varies within a week across stores "
                "for 65.34% of treated products, mailer for 2.28%. mailer is "
                "kept as a covariate, never as the treatment."
            ),
        },
        "duplicate_rule": {
            "rule": duplicate_rule,
            "alternatives": list(DUPLICATE_RULES),
            "keys_in_log_firing_neither": int(_silent_keys(panel).sum()),
            "keys_in_log_firing_neither_note": (
                "Zero under any_treated_wins, and asserted so — the log has no "
                "untreated rows, so a surviving key must fire something. Under "
                "all_must_agree a key whose display and mailer both disagree "
                "collapses to neither, which is the rule working, not a fault."
            ),
            **conflicts,
        },
        "log": {
            "rows": int(envelope["rows"]),
            "keys": int(envelope["keys"]),
            "week_min": int(envelope["week_min"]),
            "week_max": int(envelope["week_max"]),
            "stores": int(envelope["stores"]),
            "untreated_rows": untreated_rows,
            "is_a_treatment_log": untreated_rows == 0,
            "note": (
                "Zero untreated rows is what makes causal_data a treatment log "
                "rather than a panel, and it is the premise the duplicate rule "
                "and the absence reading both rest on. It is measured here, not "
                "assumed."
            ),
        },
        "coverage": {
            "panel_rows": rows,
            "in_causal_data": _units(in_log),
            "treatment_observed": _units(observed),
            "untreated_by_inference": _units(inferred),
            "unobserved": _units(~observed),
        },
        "treated": {
            **_units(treated),
            "share_of_observed": _share(int(treated.sum()), int(observed.sum())),
            "share_of_in_log": _share(int(treated.sum()), int(in_log.sum())),
            "on_display": int(panel["on_display"].sum()),
            "in_mailer": int(panel["in_mailer"].sum()),
            "both": int((panel["on_display"] & panel["in_mailer"]).sum()),
        },
        "absence_assumption": {
            "claim": "a product-store-week absent from causal_data is untreated",
            "holds_where": (
                f"stores in the log ({int(envelope['stores'])} of them) and weeks "
                f"{int(envelope['week_min'])}-{int(envelope['week_max'])}"
            ),
            "defended_by": (
                "causal_data holds no untreated rows, so presence is treatment "
                "and absence inside the log's own coverage is a real negative."
            ),
            "rows_relying_on_it": int(inferred.sum()),
            "rows_relying_on_it_share": _share(int(inferred.sum()), rows),
            "does_not_hold_where": (
                "outside that envelope the log is silent, so absence carries no "
                "information. Those rows are unobserved, not untreated."
            ),
            "rows_outside_envelope": int((~observed).sum()),
            "rows_outside_envelope_share": _share(int((~observed).sum()), rows),
            "consequence": (
                "treatment_observed marks the difference. Phase 3 must build "
                "controls from treatment_observed rows only; an unobserved row "
                "used as an untreated control is a fabricated comparison."
            ),
        },
    }


def write_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    """Write a diagnostics dict as JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2) + "\n")
    return out
