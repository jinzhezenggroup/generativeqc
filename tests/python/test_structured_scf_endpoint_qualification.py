"""Fail-closed scientific/shape gates without loading CUDA or PySCF."""

import copy
import typing

import pytest

from benchmarks import structured_scf_endpoint_qualification as qualification


def sample() -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    reference = {
        "energy": -1.0,
        "density": [[[1.0]]],
        "fock": [[[-0.5]]],
        "forces": [[0.0, 0.0, 0.0]],
        "oracle_accepted": True,
    }
    actual = {
        **copy.deepcopy(reference),
        "converged": True,
        "status": 0,
        "final_state_gates": {"accepted": True},
    }
    return actual, reference


def test_matched_complete_state_passes() -> None:
    actual, reference = sample()
    assert qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field", qualification.GATES)
def test_each_independent_gate_is_required(field: str) -> None:
    actual, reference = sample()
    if field == "energy":
        actual[field] += 2 * qualification.GATES[field]
    else:
        actual[field][0][0] = 99.0
    assert not qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field", qualification.GATES)
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), []])
def test_invalid_or_missing_values_cannot_qualify(
    field: str, invalid: typing.Any
) -> None:
    actual, reference = sample()
    actual[field] = invalid
    assert not qualification.assess(actual, reference)["accepted"]


@pytest.mark.parametrize("field,value", [("status", 1), ("converged", False)])
def test_solver_failure_cannot_supply_performance_evidence(
    field: str, value: int | bool
) -> None:
    actual, reference = sample()
    actual[field] = value
    assert not qualification.assess(actual, reference)["accepted"]


def test_failed_reference_is_not_an_oracle() -> None:
    actual, reference = sample()
    reference["oracle_accepted"] = False
    assert not qualification.assess(actual, reference)["accepted"]
