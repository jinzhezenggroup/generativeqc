"""Native work transport is an observation, never an iteration-derived count."""

from __future__ import annotations

import pytest

from tools.qualify_libxc_molecular_scf import PHASES, SPINS, _parse_native


def _output(*, work: str = "1 19 0") -> str:
    records = ["META binding expression 5 3"]
    records.extend(f"GRID {phase} 1 0 0 0 1" for phase in ("cold", "changed-geometry"))
    records.extend(
        f"ROW {spin} {phase} 1 2 -1.0 1e-12 {int(phase != 'cold')} {work}"
        for spin in SPINS
        for phase in PHASES
    )
    return "\n".join(records)


def _parse(output: str) -> dict:
    rows, _grids = _parse_native(
        output,
        binding_identity="binding",
        expression_identity="expression",
        ingredient_mask=5,
        domain_version=3,
    )
    return rows


def test_observed_work_is_not_inferred_from_iterations() -> None:
    rows = _parse(_output())
    assert len(rows) == 6
    for row in rows.values():
        assert row["iterations"] == 2
        assert row["coulomb_exchange_work"] == {
            "complete": True,
            "coulomb_applications": 19,
            "exchange_applications": 0,
        }


def test_unknown_native_work_stays_unavailable_not_zero() -> None:
    rows = _parse(_output(work="0 0 0"))
    for row in rows.values():
        assert row["coulomb_exchange_work"] == {
            "complete": False,
            "coulomb_applications": None,
            "exchange_applications": None,
        }


def test_executed_exchange_is_not_relabelled_as_requested_j_only() -> None:
    rows = _parse(_output(work="1 19 7"))
    assert all(
        row["coulomb_exchange_work"]["exchange_applications"] == 7
        for row in rows.values()
    )


@pytest.mark.parametrize("work", ["", "1 19", "2 19 0", "1 -1 0", f"1 19 {2**64}"])
def test_missing_or_malformed_native_work_fails_closed(work: str) -> None:
    with pytest.raises(ValueError):
        _parse(_output(work=work))
