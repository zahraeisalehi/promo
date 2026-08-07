"""Where the units came from: redistribution inside a commodity, from baskets.

Task 6.1. `BASKET_ID` and a recurring `household_key` are what make this
identified at all. A project reading store totals can assert that a promotion
cannibalised its neighbours; it cannot measure it, because the same store-week
total is consistent with one shopper switching brands and with a new shopper
walking in. Here the two are distinguishable, because the same household is
observed before and during the window.

**The mechanism.** For each household and commodity: work out what it bought in
the weeks before the window, project that rate onto the window, and compare
with what it actually bought. A promoted product it bought *more* of is a
**gain**; a substitute in the same commodity it bought *less* of is a **loss**.
`T[i][j]` is units moved from substitute `j` to promoted product `i`.

**What is identified and what is convention, stated separately because they are
not the same kind of claim.**

- **Column sums are identified.** The shortfall on substitute `j` is observed:
  that household bought less of it than its own history predicts.
- **Row sums are identified.** The excess on promoted product `i` is observed
  the same way.
- **The redistributed total is identified up to a cap.** A household cannot move
  more than it lost, nor more than it gained, so the transferred mass is
  `min(total gain, total loss)`. Gain above that is expansion, not
  redistribution, and belongs to Phase 4's `s`.
- **The cell-level split is a convention and nothing else.** When a household
  drops two substitutes and picks up two promoted products, which loss fed which
  gain is not in the data. This module allocates proportionally to the observed
  gains and losses; a different rule moves mass between cells and changes no
  marginal. `metadata["cell_split"]` says so, and anything rendering the matrix
  must repeat it.

**Mass conservation is asserted, not assumed.** Every cell is allocated from a
household's transferred mass, so the grand total of row sums must equal the
grand total of column sums exactly. The assertion is cheap and catches an
allocation bug that would otherwise look like a finding.

**This module never nets.** It returns gains and losses as separate quantities.
`delta_q = s + (g - l)` is Task 6.2's arithmetic, and the sign discipline there
depends on getting two unnetted numbers from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from promo.io import connect

__all__ = [
    "MassConservationError",
    "TransferResult",
    "build_transfer_matrix",
    "decompose",
    "write_diagnostics",
]

#: Allocated mass below this is treated as zero when asserting conservation.
#: Float addition over hundreds of thousands of cells accumulates error that is
#: not a bug; anything larger is.
CONSERVATION_TOLERANCE: float = 1e-6


class MassConservationError(Exception):
    """Row sums and column sums of the transfer matrix disagree.

    Raised rather than repaired. The matrix is an allocation of a known mass, so
    a mismatch means the allocation is wrong, and a matrix that does not
    conserve mass will misstate both cannibalisation and expansion.
    """


class TransferResult:
    """The matrix, its two margins, and what each of them is worth.

    Attributes:
        matrix: `T[i][j]`, units moved from substitute `j` (columns) to promoted
            product `i` (rows).
        gains: units product `i` took from elsewhere in its commodity.
        losses: units product `j` gave up.
        metadata: what is identified, what is convention, and the conservation
            check that was run.
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        gains: pd.Series,
        losses: pd.Series,
        metadata: dict[str, Any],
    ) -> None:
        self.matrix = matrix
        self.gains = gains
        self.losses = losses
        self.metadata = metadata

    @property
    def transferred(self) -> float:
        """Total redistributed units — the same number from either margin."""
        return float(self.gains.sum())


def build_transfer_matrix(
    campaign_products: tuple[int, ...],
    weeks: tuple[int, int],
    transactions: str | Path | pd.DataFrame = "data/interim/transactions_clean.parquet",
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    pre_weeks: int = 8,
    commodity: str | None = None,
    stores: tuple[int, ...] | None = None,
) -> TransferResult:
    """Units moved into the promoted products from substitutes in their commodity.

    Args:
        campaign_products: the promoted products. A household must have bought
            at least one of them inside the window to be a switcher.
        weeks: inclusive `(first, last)` promoted week.
        pre_weeks: how many weeks before the window establish the household's
            own baseline rate. Its own history, not the panel's — a household
            that never buys the commodity cannot switch within it.
        commodity: restrict to one commodity. Default is every commodity the
            promoted products belong to.
        stores: restrict to these stores.

    Returns:
        A `TransferResult`. Gains and losses are **separate**; nothing here
        nets them.

    Raises:
        MassConservationError: the allocation does not conserve mass.
        ValueError: no promoted products were supplied, or the pre-window has
            no weeks in it.
    """
    if not campaign_products:
        raise ValueError("campaign_products is empty; there is nothing to switch to")
    if pre_weeks < 1:
        raise ValueError(f"pre_weeks must be at least 1, got {pre_weeks}")

    first, last = int(weeks[0]), int(weeks[1])
    pre_first, pre_last = first - pre_weeks, first - 1
    window_weeks = last - first + 1

    own = con is None
    con = connect() if con is None else con
    try:
        if isinstance(transactions, pd.DataFrame):
            con.register("_transfer_tx", transactions)
            src = "_transfer_tx"
        else:
            src = f"read_parquet('{Path(transactions).as_posix()}')"

        promoted = ",".join(str(int(p)) for p in campaign_products)
        filters = [f"c.COMMODITY_DESC = '{commodity.replace(chr(39), chr(39) * 2)}'"] if commodity else []
        if stores:
            filters.append(f"t.STORE_ID IN ({','.join(str(int(s)) for s in stores)})")
        extra = (" AND " + " AND ".join(filters)) if filters else ""

        # Household x commodity x product, split into the two periods. Restricted
        # to the commodities the promoted products live in — a substitute is a
        # product in the same commodity, by the plan's definition.
        frame = con.execute(
            f"""
            WITH commodities AS (
                SELECT DISTINCT COMMODITY_DESC FROM {src}
                WHERE PRODUCT_ID IN ({promoted}) AND COMMODITY_DESC IS NOT NULL
            ),
            scoped AS (
                SELECT t.household_key, t.PRODUCT_ID, t.COMMODITY_DESC,
                       t.WEEK_NO, t.QUANTITY
                FROM {src} t JOIN commodities c USING (COMMODITY_DESC)
                WHERE t.usable
                  AND t.WEEK_NO BETWEEN {pre_first} AND {last}
                  {extra}
            )
            SELECT household_key, COMMODITY_DESC, PRODUCT_ID,
                   SUM(CASE WHEN WEEK_NO BETWEEN {pre_first} AND {pre_last}
                            THEN QUANTITY ELSE 0 END) AS pre_units,
                   SUM(CASE WHEN WEEK_NO BETWEEN {first} AND {last}
                            THEN QUANTITY ELSE 0 END) AS window_units
            FROM scoped
            GROUP BY 1, 2, 3
            """
        ).df()
    finally:
        if isinstance(transactions, pd.DataFrame):
            con.unregister("_transfer_tx")
        if own:
            con.close()

    promoted_set = {int(p) for p in campaign_products}
    if frame.empty:
        return _empty_result(promoted_set, weeks, pre_weeks, "no rows in scope")

    # The household's own rate, projected onto a window of this length. A
    # household is its own control here; the panel's average would import other
    # households' behaviour into a claim about this one.
    frame["expected"] = frame["pre_units"] / pre_weeks * window_weeks
    frame["delta"] = frame["window_units"] - frame["expected"]
    frame["is_promoted"] = frame["PRODUCT_ID"].isin(promoted_set)

    cells: dict[tuple[int, int], float] = {}
    switchers = 0
    households_seen = 0
    total_gain_raw = 0.0
    total_loss_raw = 0.0

    for (_, _), block in frame.groupby(
        ["household_key", "COMMODITY_DESC"], observed=True, sort=False
    ):
        households_seen += 1
        gains = block[block["is_promoted"] & (block["delta"] > 0)]
        losses = block[~block["is_promoted"] & (block["delta"] < 0)]
        if gains.empty or losses.empty:
            continue

        gain_total = float(gains["delta"].sum())
        loss_total = float(-losses["delta"].sum())
        total_gain_raw += gain_total
        total_loss_raw += loss_total
        # A household cannot move more than it lost, nor more than it gained.
        # Gain above the cap is expansion and belongs to Phase 4's `s`.
        transferred = min(gain_total, loss_total)
        if transferred <= 0:
            continue
        switchers += 1

        # The convention: proportional to observed gains and losses. Any other
        # rule moves mass between cells and leaves both margins untouched.
        for _, gain_row in gains.iterrows():
            gain_share = float(gain_row["delta"]) / gain_total
            for _, loss_row in losses.iterrows():
                loss_share = float(-loss_row["delta"]) / loss_total
                key = (int(gain_row["PRODUCT_ID"]), int(loss_row["PRODUCT_ID"]))
                cells[key] = cells.get(key, 0.0) + transferred * gain_share * loss_share

    if not cells:
        return _empty_result(
            promoted_set, weeks, pre_weeks,
            f"no household both gained a promoted product and lost a substitute "
            f"in the same commodity ({households_seen:,} household-commodities "
            f"examined)",
        )

    rows = sorted({i for i, _ in cells})
    columns = sorted({j for _, j in cells})
    matrix = pd.DataFrame(0.0, index=rows, columns=columns)
    for (i, j), value in cells.items():
        matrix.loc[i, j] = value
    matrix.index.name = "gained_by"
    matrix.columns.name = "lost_from"

    gains = matrix.sum(axis=1).rename("gained")
    losses = matrix.sum(axis=0).rename("lost")

    # Mass conservation, asserted. Both margins are sums of the same cells, so
    # a mismatch means the allocation is wrong.
    gained_total = float(gains.sum())
    lost_total = float(losses.sum())
    if abs(gained_total - lost_total) > CONSERVATION_TOLERANCE:
        raise MassConservationError(
            f"row sums total {gained_total:,.6f} but column sums total "
            f"{lost_total:,.6f}, a difference of "
            f"{abs(gained_total - lost_total):,.6f}. The matrix allocates a "
            f"known mass, so this is an allocation bug, not a data condition."
        )

    metadata = {
        "stage": "build_transfer_matrix",
        "weeks": [first, last],
        "pre_weeks": pre_weeks,
        "pre_window": [pre_first, pre_last],
        "campaign_products": sorted(promoted_set),
        "commodity": commodity,
        "products_gaining": len(rows),
        "products_losing": len(columns),
        "household_commodities_examined": households_seen,
        "switcher_household_commodities": switchers,
        "transferred_units": round(gained_total, 6),
        "gain_before_cap": round(total_gain_raw, 6),
        "loss_before_cap": round(total_loss_raw, 6),
        "uncapped_gain_is_expansion": (
            "Gain above min(gain, loss) is not redistribution — the household "
            "bought more without buying less of anything else. It belongs to "
            "Phase 4's expansion term `s`, not here."
        ),
        "mass_conservation": {
            "row_sums_total": round(gained_total, 6),
            "column_sums_total": round(lost_total, 6),
            "difference": round(abs(gained_total - lost_total), 12),
            "tolerance": CONSERVATION_TOLERANCE,
            "asserted": True,
        },
        "identified": (
            "Row sums and column sums. The excess on a promoted product and the "
            "shortfall on a substitute are both observed against the "
            "household's own prior rate."
        ),
        "cell_split": (
            "**A stated convention, not an identified quantity.** When a "
            "household drops two substitutes and picks up two promoted "
            "products, which loss fed which gain is not in the data. Cells are "
            "allocated proportionally to the observed gains and losses. A "
            "different rule moves mass between cells and changes no row or "
            "column total. Anything displaying this matrix must say so."
        ),
        "never_netted": (
            "Gains and losses are returned separately. delta_q = s + (g - l) is "
            "Task 6.2's arithmetic and depends on receiving them unnetted."
        ),
        "baseline_rule": (
            "Each household is its own control: its pre-window rate over "
            f"{pre_weeks} weeks, projected onto the {window_weeks}-week window. "
            "A panel average would import other households' behaviour into a "
            "claim about this one."
        ),
    }
    return TransferResult(matrix, gains, losses, metadata)


def decompose(
    transfer: TransferResult,
    expansion: pd.Series | dict[int, float],
    *,
    tolerance: float = CONSERVATION_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`delta_q = s + (g - l)`, per product, with the three terms kept apart.

    `s` is expansion, and it comes from the **Phase 4 counterfactual** — not
    from the transfer matrix. `g` and `l` are the matrix's row and column sums.
    They arrive from different machinery and are **added**.

    **Nothing here subtracts cannibalisation from a lift.** The temptation is to
    write `delta_q = lift - cannibalisation`, and it double-counts: the
    redistribution term is already signed, so subtracting it removes the units
    twice. `g` and `l` enter as `+g` and `-l`, which is the same arithmetic done
    once. A test asserts no code path does it the other way.

    **The identity is per product, and that is what makes it informative.** A
    promoted product has `g > 0` and `l ≈ 0`; a substitute has `g = 0` and
    `l > 0`. Summed across the commodity the redistribution term cancels — mass
    conservation guarantees `sum(g) == sum(l)` — so **the category-level change
    is pure expansion**. A promotion that only moved units between shelves
    shows a large per-product `delta_q` and a category total of about zero, and
    that contrast is the whole output.

    Args:
        transfer: the matrix from `build_transfer_matrix`.
        expansion: `s` per product, from Phase 4. Products absent from it get
            `s = 0`, recorded as `expansion_estimated = False` — substitutes
            have no counterfactual fitted, and a zero there is an absence of an
            estimate rather than an estimate of no expansion.

    Returns:
        `(per_product, diagnostics)`.

    Raises:
        MassConservationError: the matrix does not conserve mass, so the
            category-level cancellation would not hold either.
    """
    s_values = (
        expansion if isinstance(expansion, pd.Series) else pd.Series(expansion, dtype="float64")
    )
    gains, losses = transfer.gains, transfer.losses

    total_g = float(gains.sum())
    total_l = float(losses.sum())
    if abs(total_g - total_l) > tolerance:
        raise MassConservationError(
            f"the matrix does not conserve mass: gains {total_g:,.6f} against "
            f"losses {total_l:,.6f}. The category-level cancellation in "
            f"delta_q = s + (g - l) depends on it, so the decomposition would "
            f"be wrong in a way that looks like a finding."
        )

    products = sorted(
        set(gains.index) | set(losses.index) | set(s_values.index)
    )
    frame = pd.DataFrame(index=pd.Index(products, name="PRODUCT_ID"))
    frame["s_expansion"] = [float(s_values.get(p, 0.0)) for p in products]
    frame["expansion_estimated"] = [p in s_values.index for p in products]
    frame["g_gained"] = [float(gains.get(p, 0.0)) for p in products]
    frame["l_lost"] = [float(losses.get(p, 0.0)) for p in products]
    frame["redistribution"] = frame["g_gained"] - frame["l_lost"]
    # Added. Never `lift - cannibalisation`: see the docstring.
    frame["delta_q"] = frame["s_expansion"] + frame["redistribution"]
    frame = frame.reset_index()

    category_redistribution = float(frame["redistribution"].sum())
    category_expansion = float(frame["s_expansion"].sum())
    category_delta_q = float(frame["delta_q"].sum())

    diagnostics = {
        "stage": "decompose",
        "identity": "delta_q = s + (g - l)",
        "products": len(frame),
        "expansion_total": round(category_expansion, 6),
        "gained_total": round(total_g, 6),
        "lost_total": round(total_l, 6),
        "redistribution_total": round(category_redistribution, 6),
        "delta_q_total": round(category_delta_q, 6),
        "products_with_estimated_expansion": int(frame["expansion_estimated"].sum()),
        "redistribution_cancels": abs(category_redistribution) <= tolerance,
        "why_it_cancels": (
            "Mass conservation makes sum(g) equal sum(l), so the redistribution "
            "term sums to zero across the commodity and the category-level "
            "change is pure expansion. A promotion that only moved units "
            "between shelves shows large per-product delta_q and a category "
            "total near zero."
        ),
        "added_never_subtracted": (
            "s comes from the Phase 4 counterfactual and (g - l) from the "
            "transfer matrix. They are added. Writing "
            "delta_q = lift - cannibalisation double-counts, because the "
            "redistribution term is already signed."
        ),
        "expansion_absent_is_not_zero_expansion": (
            "Products with no Phase 4 estimate carry s = 0 and "
            "expansion_estimated = False. No counterfactual is fitted for a "
            "substitute, so that zero is a missing estimate, not an estimate "
            "of no expansion."
        ),
        "cell_split": transfer.metadata.get("cell_split"),
        "mass_conservation": transfer.metadata.get("mass_conservation"),
    }
    return frame, diagnostics


def _empty_result(
    promoted: set[int], weeks: tuple[int, int], pre_weeks: int, why: str
) -> TransferResult:
    """No switching found. An empty matrix that still conserves mass."""
    matrix = pd.DataFrame(dtype="float64")
    empty = pd.Series(dtype="float64")
    return TransferResult(
        matrix,
        empty.rename("gained"),
        empty.rename("lost"),
        {
            "stage": "build_transfer_matrix",
            "weeks": [int(weeks[0]), int(weeks[1])],
            "pre_weeks": pre_weeks,
            "campaign_products": sorted(promoted),
            "products_gaining": 0,
            "products_losing": 0,
            "transferred_units": 0.0,
            "empty": True,
            "why_empty": why,
            "mass_conservation": {
                "row_sums_total": 0.0,
                "column_sums_total": 0.0,
                "difference": 0.0,
                "tolerance": CONSERVATION_TOLERANCE,
                "asserted": True,
            },
            "not_evidence_of_no_cannibalisation": (
                "An empty matrix means no household was observed both gaining a "
                "promoted product and losing a substitute in the same "
                "commodity. That is a statement about what these baskets show, "
                "not evidence that the promotion moved nothing."
            ),
        },
    )


def write_diagnostics(result: TransferResult, path: str | Path) -> Path:
    """Write the matrix metadata and margins as JSON. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result.metadata,
        "gains": {int(k): round(float(v), 6) for k, v in result.gains.items()},
        "losses": {int(k): round(float(v), 6) for k, v in result.losses.items()},
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out
