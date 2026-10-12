"""Independent complete native CCSD(T)/CBS E+F acceptance on CPU and CUDA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import CBSBasis, CCSDTCBSCalculator, load_basis

if TYPE_CHECKING:
    from generativeqc import CBSResult

REFERENCES = Path(__file__).resolve().parents[1] / "reference_data/cbs"


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def cbs_device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_CBS_CUDA_TEST") != "1":
            pytest.skip("requires explicitly allocated CUDA library")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("real GPU CBS qualification must run through srun")
    return request.param


@pytest.fixture(scope="module", params=("h2-dz-tz", "water-algebra"))
def cbs_case(request: pytest.FixtureRequest, cbs_device: str) -> dict:
    if (
        request.param == "h2-dz-tz"
        and cbs_device == "cpu"
        and os.environ.get("GENERATIVEQC_CBS_LARGE_TEST") != "1"
    ):
        pytest.skip("large conventional CPU force replay is explicit qualification")
    record = json.loads((REFERENCES / f"{request.param}.json").read_text())
    assert record["pyscf"] == "2.14.0"
    assert max(point["absolute_error"] for point in record["finite_difference"]) < 2e-6
    return record


@pytest.fixture(scope="module")
def workflow(cbs_device: str, cbs_case: dict) -> CCSDTCBSCalculator:
    low = load_basis(REFERENCES / cbs_case["low_basis"])
    high = load_basis(REFERENCES / cbs_case["high_basis"])
    assert [low.identity, high.identity] == cbs_case["basis_identities"]
    return CCSDTCBSCalculator((CBSBasis(low, 2), CBSBasis(high, 3)), device=cbs_device)


@pytest.fixture(scope="module")
def complete_result(workflow: CCSDTCBSCalculator, cbs_case: dict) -> CBSResult:
    return workflow.singlepoint(cbs_case["atoms"])


def test_complete_components_and_composite_against_independent_oracle(
    cbs_device: str, cbs_case: dict, complete_result: CBSResult
) -> None:
    result = complete_result
    oracle = cbs_case["reference"]
    assert (
        result.executed_backend == ("cuda" if cbs_device == "cuda" else "cpu_reference")
        and result.converged
    )
    assert result.energy == pytest.approx(oracle["energy"], abs=1e-9)
    np.testing.assert_allclose(result.forces, oracle["forces"], atol=2e-6, rtol=0)
    for field in ("hf_energy", "mp2_correlation_energy", "delta_ccsdt_energy"):
        assert getattr(result, field) == pytest.approx(oracle[field], abs=1e-9)
    assert len(result.components) == 5
    for component in result.components:
        reference = oracle["components"][component.roles[0]]
        assert component.result.energy == pytest.approx(reference["energy"], abs=1e-9)
        np.testing.assert_allclose(
            component.result.forces, reference["forces"], atol=2e-6, rtol=0
        )
        if component.method != "rhf":
            assert component.result.correlation is not None
            assert component.result.correlation.force_provenance_flags & 7 == 7
        if component.method == "ccsd(t)":
            assert component.result.correlation.ccsd_t_triples_energy == pytest.approx(
                reference["triples_energy"], abs=1e-9
            )
    if cbs_case["case"] == "water-algebra":
        assert abs(oracle["components"]["correction_ccsdt"]["triples_energy"]) > 1e-5
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-7, rtol=0)


def test_energy_only_parity(
    workflow: CCSDTCBSCalculator, cbs_case: dict, complete_result: CBSResult
) -> None:
    result = workflow.singlepoint(cbs_case["atoms"], properties=("energy",))
    assert result.forces is None
    assert all(component.result.forces is None for component in result.components)
    assert result.energy == pytest.approx(complete_result.energy, abs=1e-10)


def test_reconverged_complete_energy_derivative(
    workflow: CCSDTCBSCalculator, cbs_case: dict, complete_result: CBSResult
) -> None:
    coordinates = np.asarray([position for _, position in cbs_case["atoms"]])
    direction = np.asarray(cbs_case["direction"])
    analytic = -float(np.vdot(complete_result.forces, direction))
    errors = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (-1, 1):
            displaced = [
                (number, position)
                for (number, _), position in zip(
                    cbs_case["atoms"],
                    coordinates + sign * step * direction,
                    strict=True,
                )
            ]
            energies.append(
                workflow.singlepoint(displaced, properties=("energy",)).energy
            )
        errors.append(abs((energies[1] - energies[0]) / (2 * step) - analytic))
    assert max(errors) < 2e-6, errors


def test_tiny_native_budget_rejects_and_independent_workflow_recovers(
    cbs_device: str,
) -> None:
    low = load_basis(REFERENCES / "water-algebra-low.json")
    high = load_basis(REFERENCES / "water-algebra-high.json")
    record = json.loads((REFERENCES / "water-algebra.json").read_text())
    bases = CBSBasis(low, 2), CBSBasis(high, 3)
    with pytest.raises(RuntimeError, match="memory budget"):
        CCSDTCBSCalculator(
            bases,
            device=cbs_device,
            calculator_options={"correlation_memory_budget_bytes": 1024},
        ).singlepoint(record["atoms"])
    result = CCSDTCBSCalculator(bases, device=cbs_device).singlepoint(
        record["atoms"], properties=("energy",)
    )
    assert result.energy == pytest.approx(record["reference"]["energy"], abs=1e-9)


def test_physical_pair_cpu_energy() -> None:
    """Keep physical CBS energy coverage in routine CPU CI without large force replay."""
    record = json.loads((REFERENCES / "h2-dz-tz.json").read_text())
    low = load_basis(REFERENCES / record["low_basis"])
    high = load_basis(REFERENCES / record["high_basis"])
    result = CCSDTCBSCalculator((CBSBasis(low, 2), CBSBasis(high, 3))).singlepoint(
        record["atoms"], properties=("energy",)
    )
    assert result.energy == pytest.approx(record["reference"]["energy"], abs=1e-9)
