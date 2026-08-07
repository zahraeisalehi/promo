"""Tests for Task 6.1, the basket-level transfer matrix.

The fixture households are hand-built so the answer is arithmetic: a household
that drops four units of one substitute and picks up four of a promoted product
has moved four units, and the matrix must say so in the one cell that can hold
them.

Two claims get their own tests because they are the ones a reader will lean on:
**mass conservation**, which is asserted rather than assumed, and the
**cell-level split being a convention**, which the metadata has to say in words
because no amount of precision in the number makes it identified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from promo.transfer import (
    CONSERVATION_TOLERANCE,
    MassConservationError,
    build_transfer_matrix,
    decompose,
    write_diagnostics,
)

CLEAN = Path("data/interim/transactions_clean.parquet")
PROMOTED, SUBSTITUTE, OTHER = 1, 2, 3
WEEKS = (10, 13)
PRE_WEEKS = 8


def real_data(fn):
    return pytest.mark.skipif(not CLEAN.exists(), reason="run Phase 2 first")(
        pytest.mark.heavy(fn)
    )


def _rows(spec: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """`(household, product, week, quantity)` into a clean-transactions frame."""
    return pd.DataFrame(
        [
            {
                "household_key": h,
                "PRODUCT_ID": p,
                "COMMODITY_DESC": "SOUP",
                "STORE_ID": 1,
                "WEEK_NO": w,
                "QUANTITY": q,
                "usable": True,
            }
            for h, p, w, q in spec
        ]
    )


def _steady(household: int, product: int, per_week: float) -> list[tuple]:
    """The same rate every pre-window week, so `expected` is exactly per_week."""
    return [(household, product, w, per_week) for w in range(2, 10)]


def _switcher() -> pd.DataFrame:
    """One household: drops the substitute entirely, picks the promoted up.

    Pre-window it buys 1/week of the substitute and none of the promoted
    product. In the four-week window it buys 4 of the promoted and 0 of the
    substitute — expected 4 of the substitute, so it lost 4 and gained 4.
    """
    rows = _steady(1, SUBSTITUTE, 1.0)
    rows += [(1, PROMOTED, w, 1.0) for w in range(10, 14)]
    return _rows(rows)


# --- the mechanism ------------------------------------------------------------


def test_a_clean_switch_moves_exactly_the_units_it_lost():
    result = build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS)

    assert result.matrix.loc[PROMOTED, SUBSTITUTE] == pytest.approx(4.0)
    assert result.gains[PROMOTED] == pytest.approx(4.0)
    assert result.losses[SUBSTITUTE] == pytest.approx(4.0)
    assert result.transferred == pytest.approx(4.0)


def test_gains_and_losses_come_back_separately_never_netted():
    """Task 6.2 needs two unnetted numbers to compute s + (g - l)."""
    result = build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS)

    assert result.gains.sum() > 0
    assert result.losses.sum() > 0
    assert "unnetted" in result.metadata["never_netted"]


def test_gain_above_the_loss_is_expansion_and_is_not_transferred():
    """A household that buys more without buying less has not switched.

    It loses 4 units of the substitute and gains 10 of the promoted product.
    Only 4 moved; the other 6 are expansion and belong to Phase 4's `s`.
    """
    rows = _steady(1, SUBSTITUTE, 1.0)
    rows += [(1, PROMOTED, w, 2.5) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    assert result.transferred == pytest.approx(4.0)
    assert result.metadata["gain_before_cap"] == pytest.approx(10.0)
    assert result.metadata["loss_before_cap"] == pytest.approx(4.0)
    assert "belongs to Phase 4's expansion term" in (
        result.metadata["uncapped_gain_is_expansion"]
    )


def test_loss_above_the_gain_is_capped_too():
    """A household buying less overall has not moved all of it to the promotion."""
    rows = _steady(1, SUBSTITUTE, 2.0)
    rows += [(1, PROMOTED, w, 0.5) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    assert result.metadata["loss_before_cap"] == pytest.approx(8.0)
    assert result.transferred == pytest.approx(2.0)


def test_a_household_with_no_substitute_loss_contributes_nothing():
    """Buying the promotion without dropping anything is not cannibalisation."""
    rows = _steady(1, SUBSTITUTE, 1.0)
    rows += [(1, SUBSTITUTE, w, 1.0) for w in range(10, 14)]  # unchanged
    rows += [(1, PROMOTED, w, 1.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    assert result.transferred == pytest.approx(0.0)
    assert result.metadata["empty"] is True


def test_each_household_is_its_own_control():
    """A heavy buyer and a light buyer are not compared with each other."""
    rows = _steady(1, SUBSTITUTE, 10.0) + [(1, PROMOTED, w, 10.0) for w in range(10, 14)]
    rows += _steady(2, SUBSTITUTE, 1.0) + [(2, PROMOTED, w, 1.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    # 40 units from the heavy household, 4 from the light one.
    assert result.transferred == pytest.approx(44.0)
    assert "its own control" in result.metadata["baseline_rule"]


# --- mass conservation --------------------------------------------------------


def test_mass_is_conserved_and_the_check_is_recorded():
    rows = []
    for household in range(1, 6):
        rows += _steady(household, SUBSTITUTE, 1.0)
        rows += _steady(household, OTHER, 2.0)
        rows += [(household, PROMOTED, w, 2.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    row_total = float(result.matrix.sum(axis=1).sum())
    column_total = float(result.matrix.sum(axis=0).sum())
    assert abs(row_total - column_total) < CONSERVATION_TOLERANCE

    check = result.metadata["mass_conservation"]
    assert check["asserted"] is True
    assert check["row_sums_total"] == pytest.approx(check["column_sums_total"])
    assert check["difference"] < CONSERVATION_TOLERANCE


def test_a_broken_allocation_raises_rather_than_reporting():
    """The assertion has to be able to fail, or it is decoration."""
    from promo import transfer

    original = transfer.CONSERVATION_TOLERANCE
    try:
        # A negative tolerance makes any difference, including exact zero,
        # count as a mismatch — the cheapest way to prove the branch is live.
        transfer.CONSERVATION_TOLERANCE = -1.0
        with pytest.raises(MassConservationError, match="allocation bug"):
            build_transfer_matrix(
                (PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS
            )
    finally:
        transfer.CONSERVATION_TOLERANCE = original


# --- the cell split is a convention, and says so ------------------------------


def test_the_cell_split_is_labelled_a_convention_not_a_measurement():
    """Two losses feeding two gains: the marginals are real, the cells are not."""
    rows = _steady(1, SUBSTITUTE, 1.0) + _steady(1, OTHER, 3.0)
    rows += [(1, PROMOTED, w, 2.0) for w in range(10, 14)]
    rows += [(1, 4, w, 2.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED, 4), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    assert result.matrix.shape == (2, 2)
    split = result.metadata["cell_split"]
    assert "convention" in split and "not an identified quantity" in split
    assert "changes no row or column total" in split
    assert "Row sums and column sums" in result.metadata["identified"]


def test_the_convention_moves_cells_but_not_margins():
    """The claim the metadata makes, checked rather than asserted in prose.

    Reallocating within a row leaves both margins untouched — which is exactly
    why the margins may be reported and the cells may not.
    """
    rows = _steady(1, SUBSTITUTE, 1.0) + _steady(1, OTHER, 3.0)
    rows += [(1, PROMOTED, w, 4.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    before_rows = result.matrix.sum(axis=1).copy()
    shifted = result.matrix.copy()
    moved = shifted.loc[PROMOTED, SUBSTITUTE]
    shifted.loc[PROMOTED, SUBSTITUTE] = 0.0
    shifted.loc[PROMOTED, OTHER] += moved

    assert shifted.sum(axis=1).equals(before_rows)      # row margin unchanged
    assert not shifted.sum(axis=0).equals(result.matrix.sum(axis=0))  # cells moved


# --- edges --------------------------------------------------------------------


def test_no_switching_returns_an_empty_matrix_that_says_what_that_means():
    rows = _steady(1, SUBSTITUTE, 1.0) + [(1, SUBSTITUTE, w, 1.0) for w in range(10, 14)]
    result = build_transfer_matrix((PROMOTED,), WEEKS, _rows(rows), pre_weeks=PRE_WEEKS)

    assert result.transferred == 0.0
    assert result.metadata["empty"] is True
    assert "not evidence that the promotion moved nothing" in (
        result.metadata["not_evidence_of_no_cannibalisation"]
    )


def test_an_empty_product_list_is_refused():
    with pytest.raises(ValueError, match="nothing to switch to"):
        build_transfer_matrix((), WEEKS, _switcher())


def test_a_zero_length_pre_window_is_refused():
    with pytest.raises(ValueError, match="pre_weeks must be at least 1"):
        build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=0)


def test_substitutes_are_products_in_the_same_commodity_only():
    """A product in another commodity is not a substitute, however it moved."""
    rows = _steady(1, SUBSTITUTE, 1.0)
    rows += [(1, PROMOTED, w, 1.0) for w in range(10, 14)]
    frame = _rows(rows)
    other_commodity = _rows(_steady(1, 99, 5.0))
    other_commodity["COMMODITY_DESC"] = "BREAD"
    combined = pd.concat([frame, other_commodity], ignore_index=True)

    result = build_transfer_matrix((PROMOTED,), WEEKS, combined, pre_weeks=PRE_WEEKS)
    assert 99 not in result.matrix.columns


def test_write_diagnostics_round_trips(tmp_path):
    import json

    result = build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS)
    path = write_diagnostics(result, tmp_path / "transfer.json")
    payload = json.loads(path.read_text())

    assert payload["stage"] == "build_transfer_matrix"
    assert payload["gains"][str(PROMOTED)] == pytest.approx(4.0)
    assert payload["losses"][str(SUBSTITUTE)] == pytest.approx(4.0)


# --- the real baskets ---------------------------------------------------------


@real_data
def test_a_real_campaign_conserves_mass():
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    product = con.execute(
        """
        SELECT PRODUCT_ID FROM read_parquet('data/interim/panel.parquet')
        WHERE treated AND WEEK_NO BETWEEN 80 AND 83
        GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1
        """
    ).fetchone()[0]
    con.close()

    result = build_transfer_matrix((int(product),), (80, 83), CLEAN)
    check = result.metadata["mass_conservation"]
    assert check["difference"] < CONSERVATION_TOLERANCE
    assert result.metadata["household_commodities_examined"] > 0
    # Whatever the number, gains and losses agree — that is the invariant.
    assert result.gains.sum() == pytest.approx(result.losses.sum())


# --- Task 6.2: the decomposition ----------------------------------------------


def _decomposed(expansion: dict[int, float] | None = None):
    result = build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS)
    return decompose(result, expansion if expansion is not None else {PROMOTED: 10.0})


def test_delta_q_is_expansion_plus_redistribution():
    frame, diag = _decomposed()

    promoted = frame[frame.PRODUCT_ID == PROMOTED].iloc[0]
    # 10 units of new demand, 4 taken from the substitute, nothing given up.
    assert promoted["s_expansion"] == pytest.approx(10.0)
    assert promoted["g_gained"] == pytest.approx(4.0)
    assert promoted["l_lost"] == pytest.approx(0.0)
    assert promoted["delta_q"] == pytest.approx(14.0)

    substitute = frame[frame.PRODUCT_ID == SUBSTITUTE].iloc[0]
    assert substitute["s_expansion"] == pytest.approx(0.0)
    assert substitute["l_lost"] == pytest.approx(4.0)
    assert substitute["delta_q"] == pytest.approx(-4.0)

    assert diag["identity"] == "delta_q = s + (g - l)"


def test_redistribution_cancels_so_the_category_change_is_pure_expansion():
    """The contrast the whole output exists to show."""
    frame, diag = _decomposed()

    assert diag["redistribution_total"] == pytest.approx(0.0)
    assert diag["redistribution_cancels"] is True
    assert diag["delta_q_total"] == pytest.approx(diag["expansion_total"])
    assert float(frame["delta_q"].sum()) == pytest.approx(10.0)


def test_a_promotion_that_only_moved_units_shows_a_category_total_of_zero():
    """No expansion at all: every unit gained came off another shelf."""
    frame, diag = _decomposed(expansion={PROMOTED: 0.0})

    assert frame.loc[frame.PRODUCT_ID == PROMOTED, "delta_q"].iloc[0] == pytest.approx(4.0)
    assert frame.loc[frame.PRODUCT_ID == SUBSTITUTE, "delta_q"].iloc[0] == pytest.approx(-4.0)
    assert diag["delta_q_total"] == pytest.approx(0.0)
    assert diag["expansion_total"] == pytest.approx(0.0)


def test_cannibalisation_is_never_subtracted_from_the_lift():
    """The error CLAUDE.md names, asserted rather than trusted.

    The wrong arithmetic is `delta_q = s - (g - l)`, or `lift - cannibalisation`.
    With a gain of 4 and no loss it gives 6 where the identity gives 14, so the
    two are distinguishable by value and this fails loudly if a path flips it.
    """
    frame, _ = _decomposed(expansion={PROMOTED: 10.0})
    promoted = frame[frame.PRODUCT_ID == PROMOTED].iloc[0]

    s, g, ell = promoted["s_expansion"], promoted["g_gained"], promoted["l_lost"]
    assert promoted["delta_q"] == pytest.approx(s + (g - ell))
    assert promoted["delta_q"] != pytest.approx(s - (g - ell))
    assert promoted["delta_q"] != pytest.approx(s - g)

    from promo import transfer as module

    assert "double-count" in module.decompose.__doc__
    source = Path(module.__file__).read_text()
    assert 's_expansion"] + frame["redistribution"]' in source
    assert 's_expansion"] - frame["redistribution"]' not in source


def test_a_missing_expansion_estimate_is_flagged_not_read_as_zero_expansion():
    frame, diag = _decomposed()

    assert not frame[frame.PRODUCT_ID == SUBSTITUTE].iloc[0]["expansion_estimated"]
    assert frame[frame.PRODUCT_ID == PROMOTED].iloc[0]["expansion_estimated"]
    assert diag["products_with_estimated_expansion"] == 1
    assert "missing estimate, not an estimate of no expansion" in (
        diag["expansion_absent_is_not_zero_expansion"]
    )


def test_a_matrix_that_lost_mass_refuses_to_decompose():
    """The cancellation depends on conservation, so it is checked again here."""
    result = build_transfer_matrix((PROMOTED,), WEEKS, _switcher(), pre_weeks=PRE_WEEKS)
    result.losses = result.losses * 2.0     # break it after the fact

    with pytest.raises(MassConservationError, match="does not conserve mass"):
        decompose(result, {PROMOTED: 10.0})


def test_the_decomposition_carries_the_convention_label_forward():
    """A reader of the decomposition must still know the cells are a convention."""
    _, diag = _decomposed()
    assert "convention" in diag["cell_split"]
    assert diag["mass_conservation"]["asserted"] is True
