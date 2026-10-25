"""Independent complete native CCSD(T)/CBS E+F acceptance on CPU and CUDA."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, CBSBasis, CCSDTCBSCalculator, _native, load_basis

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
    return CCSDTCBSCalculator(
        (CBSBasis(low, 2), CBSBasis(high, 3)), device=cbs_device, policy="tight"
    )


@pytest.fixture(scope="module")
def completed_work_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cbs-completed-work") / "progress.jsonl"


@pytest.fixture(scope="module")
def complete_result(
    workflow: CCSDTCBSCalculator, cbs_case: dict, completed_work_path: Path
) -> CBSResult:
    completed_work_path.unlink(missing_ok=True)
    previous = os.environ.get("GENERATIVEQC_DF_PROGRESS_TRACE")
    os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(completed_work_path)
    try:
        return workflow.singlepoint(cbs_case["atoms"])
    finally:
        if previous is None:
            os.environ.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
        else:
            os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = previous


def test_complete_endpoint_reuses_reference_and_mo_blocks(
    cbs_device: str, complete_result: CBSResult, completed_work_path: Path
) -> None:
    """Require real consumer reuse, not just optional cache allocation."""
    assert complete_result.converged
    records = [
        json.loads(line) for line in completed_work_path.read_text().splitlines()
    ]
    values = {
        record["key"]: record["value"]
        for record in records
        if record["event"] == "VALUE"
        and record["name"] in ("ccsdt_endpoint", "mp2_endpoint")
    }
    assert values["mp2_companion_energy_source_reads"] == 0
    assert values["mp2_companion_energy_transform_fmas"] == 0
    assert values["mp2_companion_ovov_reused"] == 1
    assert values["mp2_companion_force_blocks_reused"] == 1
    if cbs_device == "cuda":
        assert values["mp2_energy_device_source"] == 1
        assert values["mp2_force_device_source"] == 1
        assert values["mp2_force_mo_cache_bytes"] > 0
        primary_ids = {
            record["id"]
            for record in records
            if record["event"] == "BEGIN" and record["name"] == "mp2_endpoint"
        }
        assert (
            sum(
                record["value"]
                for record in records
                if record["event"] == "VALUE"
                and record["name"] == "borrowed_mo_blocks"
                and record["key"] == "mo_block_reuses"
                and record["parent"] in primary_ids
            )
            > 0
        )


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
    assert len(result.execution_groups) == 2
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


def test_native_component_rejection_revokes_diagnostic_and_recovers(
    cbs_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real ABI's group publication and stale-record boundary."""
    atoms = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)))
    calculator = Calculator(
        method="ccsd(t)",
        basis="sto-3g",
        device=cbs_device,
        precision="fp64",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    library = calculator._library
    native_execute = library.generativeqc_calculation_execute_components_v1
    native_getter = library.generativeqc_calculation_get_mp2_component_diagnostic_v1
    observed = []

    def execute(owner: object, primary: object, hf: object, mp2: object) -> int:
        status = native_execute(owner, primary, hf, mp2)
        assert status == _native.STATUS_SUCCESS
        diagnostic = _native.CorrelationDiagnostic()
        diagnostic.struct_size = ctypes.sizeof(diagnostic)
        diagnostic.abi_version = _native.ABI_VERSION
        assert native_getter(owner, ctypes.byref(diagnostic)) == _native.STATUS_SUCCESS
        descriptors = [
            ctypes.cast(value, ctypes.POINTER(_native.ResultDescriptor)).contents
            for value in (primary, hf, mp2)
        ]
        saved = [
            (record.energy, np.ctypeslib.as_array(record.forces, shape=(6,)).copy())
            for record in descriptors
        ]
        hf_descriptor = descriptors[1]
        capacity = hf_descriptor.force_count
        hf_descriptor.force_count = 1
        assert (
            native_execute(owner, primary, hf, mp2) == _native.STATUS_INVALID_ARGUMENT
        )
        assert (
            native_getter(owner, ctypes.byref(diagnostic))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        for descriptor, (energy, forces) in zip(descriptors, saved, strict=True):
            assert descriptor.energy == energy
            np.testing.assert_array_equal(
                np.ctypeslib.as_array(descriptor.forces, shape=(6,)), forces
            )
        hf_descriptor.force_count = capacity
        status = native_execute(owner, primary, hf, mp2)
        assert status == _native.STATUS_SUCCESS
        assert native_getter(owner, ctypes.byref(diagnostic)) == _native.STATUS_SUCCESS
        observed.append(diagnostic.force_provenance_flags)
        return status

    monkeypatch.setattr(
        library, "generativeqc_calculation_execute_components_v1", execute
    )
    result = calculator.singlepoint_components(atoms, companions=("rhf", "mp2"))
    assert set(result) == {"ccsd(t)", "rhf", "mp2"}
    assert observed and observed[0] & 7 == 7
    assert result["mp2"].correlation.reference_energy == result["rhf"].energy
    calculator.clear_cache()


def test_energy_only_parity(
    workflow: CCSDTCBSCalculator, cbs_case: dict, complete_result: CBSResult
) -> None:
    result = workflow.singlepoint(cbs_case["atoms"], properties=("energy",))
    assert result.forces is None
    assert all(component.result.forces is None for component in result.components)
    assert result.energy == pytest.approx(complete_result.energy, abs=1e-10)


@pytest.mark.parametrize("properties", (("energy",), ("energy", "forces")))
def test_nonconverged_native_group_publishes_no_components(
    cbs_device: str,
    properties: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real CC nonconvergence must not expose its successful RHF subset."""
    calculator = Calculator(
        method="ccsd(t)",
        basis="sto-3g",
        device=cbs_device,
        precision="fp64",
        ccsd_max_iterations=1,
    )
    library = calculator._library
    native_execute = library.generativeqc_calculation_execute_components_v1
    observed = []

    def execute(owner: object, *outputs: object) -> int:
        descriptors = [
            ctypes.cast(value, ctypes.POINTER(_native.ResultDescriptor)).contents
            for value in outputs
        ]
        for descriptor in descriptors:
            descriptor.energy = 123.0
            if descriptor.forces:
                np.ctypeslib.as_array(descriptor.forces, shape=(6,)).fill(456.0)
        status = native_execute(owner, *outputs)
        assert status == _native.STATUS_NOT_CONVERGED
        for descriptor in descriptors:
            assert descriptor.energy == 123.0
            if descriptor.forces:
                np.testing.assert_array_equal(
                    np.ctypeslib.as_array(descriptor.forces, shape=(6,)), 456.0
                )
        assert (
            library.generativeqc_calculation_get_mp2_component_diagnostic_v1(
                owner, None
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )
        observed.append(status)
        return status

    monkeypatch.setattr(
        library, "generativeqc_calculation_execute_components_v1", execute
    )
    try:
        with pytest.raises(RuntimeError):
            calculator.singlepoint_components(
                (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))),
                companions=("rhf", "mp2"),
                properties=properties,
            )
        assert observed
    finally:
        calculator.clear_cache()


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
            policy="tight",
            calculator_options={"correlation_memory_budget_bytes": 1024},
        ).singlepoint(record["atoms"])
    result = CCSDTCBSCalculator(bases, device=cbs_device, policy="tight").singlepoint(
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
