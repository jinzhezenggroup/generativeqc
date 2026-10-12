"""Complete CUDA correlation-only DF-MP2/CBS acceptance against PySCF energies."""

from __future__ import annotations

import ctypes
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Atom, Calculator, CBSBasis, CCSDTCBSCalculator, _native

from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies

if TYPE_CHECKING:
    from pathlib import Path

HYDROGEN = ((1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7)))
WATER = ((8, (0.0, 0.0, 0.0)), (1, (1.4, 0.0, 1.1)), (1, (-1.4, 0.0, 1.1)))
METHANE = (
    (6, (0.0, 0.0, 0.0)),
    (1, (1.1892269365, 1.1892269365, 1.1892269365)),
    (1, (1.1892269365, -1.1892269365, -1.1892269365)),
    (1, (-1.1892269365, 1.1892269365, -1.1892269365)),
    (1, (-1.1892269365, -1.1892269365, 1.1892269365)),
)


@pytest.fixture(scope="module", autouse=True)
def allocated_cuda() -> None:
    """Refuse unscheduled real-GPU qualification even when a CUDA library exists."""
    if os.environ.get("GENERATIVEQC_DF_MP2_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA DF-MP2 qualification")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    pytest.importorskip("pyscf")


def calculator(
    method: str, atoms: tuple, orbital: str, auxiliary: str, **options: object
) -> Calculator:
    """Bind exactly matched orbital/auxiliary records, not approximate named packs."""
    elements = [atom[0] for atom in atoms]
    defaults = {"correlation_memory_budget_bytes": 2 << 30, **options}
    return Calculator(
        method=method,
        basis=basis_snapshot(orbital, elements),
        auxiliary_basis=basis_snapshot(auxiliary, elements),
        device="cuda",
        precision="fp64",
        density_fitting="correlation_cuda" if method == "mp2" else "cuda",
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-11,
        **defaults,
    )


def displaced(atoms: tuple, direction: np.ndarray, step: float) -> list:
    """Displace every nuclear and auxiliary center through the public geometry."""
    return [
        (number, np.asarray(position) + step * offset)
        for (number, position), offset in zip(atoms, direction, strict=True)
    ]


@pytest.mark.parametrize(
    "atoms,orbital,auxiliary",
    (
        (HYDROGEN, "sto-3g", "def2-svp"),
        (WATER, "cc-pvdz", "cc-pvdz-ri"),
        (METHANE, "def2-svp", "def2-svp"),
    ),
)
def test_complete_df_mp2_energy_and_reconverged_force(
    atoms: tuple,
    orbital: str,
    auxiliary: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "mp2-source.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    owner = calculator("mp2", atoms, orbital, auxiliary)
    try:
        result = owner.singlepoint_components(
            atoms, companions=("rhf",), properties=("energy", "forces")
        )
        expected = energies(atoms, orbital, auxiliary)
        for method, component in result.items():
            assert component.energy == pytest.approx(expected[method], abs=3e-9)
            assert component.forces is not None and component.converged
            assert component.executed_backend == "cuda"
            np.testing.assert_allclose(component.forces.sum(axis=0), 0.0, atol=3e-7)
        assert result["mp2"].correlation.force_provenance_flags & 7 == 7
        assert result["mp2"].correlation.ccsd_iterations == 0
        assert result["mp2"].correlation.response_absolute_residual < 1e-8
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        core_products = [
            record["value"]
            for record in records
            if record["event"] == "VALUE"
            and record["name"] == "df_correlation_source"
            and record["key"] == "core_block_gemms"
        ]
        assert core_products and all(value == 0 for value in core_products)
        direction = np.random.default_rng(219).normal(size=(len(atoms), 3))
        direction /= np.linalg.norm(direction)
        for step in (1e-4, 3e-5):
            oracle = [
                energies(displaced(atoms, direction, sign * step), orbital, auxiliary)
                for sign in (-1, 1)
            ]
            for method in result:
                analytic = -float(np.vdot(result[method].forces, direction))
                derivative = (oracle[1][method] - oracle[0][method]) / (2 * step)
                assert analytic == pytest.approx(derivative, abs=3e-6)
        if orbital == "sto-3g":
            step = 1e-4
            native = [
                owner.singlepoint(
                    displaced(atoms, direction, sign * step), properties=("energy",)
                ).energy
                for sign in (-1, 1)
            ]
            assert (native[1] - native[0]) / (2 * step) == pytest.approx(
                -float(np.vdot(result["mp2"].forces, direction)), abs=3e-6
            )
        energy_only = owner.singlepoint(atoms, properties=("energy",))
        assert energy_only.energy == pytest.approx(result["mp2"].energy, abs=1e-10)
        assert energy_only.forces is None
    finally:
        owner.clear_cache()


def test_df_ccsdt_companions_share_source_and_match_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "source-reuse.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(journal))
    owner = calculator("df-ccsd(t)", WATER, "sto-3g", "def2-svp")
    mp2 = calculator("mp2", WATER, "sto-3g", "def2-svp")
    try:
        result = owner.singlepoint_components(WATER, companions=("rhf", "mp2"))
        group_records = [json.loads(line) for line in journal.read_text().splitlines()]
        independent = energies(WATER, "sto-3g", "def2-svp", triples=True)
        standalone = mp2.singlepoint(WATER, properties=("energy", "forces"))
        for method in result:
            assert result[method].energy == pytest.approx(independent[method], abs=3e-9)
        np.testing.assert_allclose(
            result["mp2"].forces, standalone.forces, atol=3e-8, rtol=0
        )
        assert result["mp2"].correlation.ccsd_iterations == 0
        direction = np.random.default_rng(738).normal(size=(len(WATER), 3))
        direction /= np.linalg.norm(direction)
        step = 1e-4
        oracle = [
            energies(
                displaced(WATER, direction, sign * step),
                "sto-3g",
                "def2-svp",
                triples=True,
            )
            for sign in (-1, 1)
        ]
        for method in result:
            assert -float(np.vdot(result[method].forces, direction)) == pytest.approx(
                (oracle[1][method] - oracle[0][method]) / (2 * step), abs=3e-6
            )
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        replay = {
            record["key"]: record["value"]
            for record in records
            if record["event"] == "VALUE"
            and record["name"] == "df_ccsdt_response_replay"
        }
        assert replay["df_mp2_source_reused"] == replay["df_mp2_reference_reused"] == 1
        assert replay["df_mp2_source_identity"] != 0
        source_identity = next(
            record["value"]
            for record in group_records
            if record["event"] == "VALUE"
            and record["name"] == "df_correlation_source"
            and record["key"] == "source_identity"
        )
        assert source_identity == replay["df_mp2_source_identity"]
    finally:
        owner.clear_cache()
        mp2.clear_cache()


def test_complete_df_cbs_energy_and_force() -> None:
    elements = [atom[0] for atom in WATER]
    low, high = (basis_snapshot(name, elements) for name in ("cc-pvdz", "cc-pvtz"))
    auxiliaries = tuple(
        basis_snapshot(name, elements) for name in ("cc-pvdz-ri", "cc-pvtz-ri")
    )
    workflow = CCSDTCBSCalculator(
        (CBSBasis(low, 2), CBSBasis(high, 3)),
        auxiliary_bases=auxiliaries,
        device="cuda",
        calculator_options={"correlation_memory_budget_bytes": 2 << 30},
    )
    result = workflow.singlepoint(WATER)
    assert len(result.execution_groups) == 2
    assert len(result.components) == 5
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=3e-7)
    direction = np.random.default_rng(412).normal(size=(len(WATER), 3))
    direction /= np.linalg.norm(direction)

    def reference(atoms: list | tuple) -> float:
        low_result = energies(atoms, "cc-pvdz", "cc-pvdz-ri", triples=True)
        high_result = energies(atoms, "cc-pvtz", "cc-pvtz-ri")
        return float(
            sum(
                weight * component["rhf"]
                for weight, component in zip(
                    workflow.hf_weights, (low_result, high_result), strict=True
                )
            )
            + sum(
                weight * (component["mp2"] - component["rhf"])
                for weight, component in zip(
                    workflow.mp2_weights, (low_result, high_result), strict=True
                )
            )
            + low_result["df-ccsd(t)"]
            - low_result["mp2"]
        )

    assert result.energy == pytest.approx(reference(WATER), abs=5e-9)
    analytic = -float(np.vdot(result.forces, direction))
    for step in (1e-4, 3e-5):
        derivative = (
            reference(displaced(WATER, direction, step))
            - reference(displaced(WATER, direction, -step))
        ) / (2 * step)
        assert analytic == pytest.approx(derivative, abs=4e-6)
    energy_only = workflow.singlepoint(WATER, properties=("energy",))
    assert energy_only.energy == pytest.approx(result.energy, abs=1e-10)
    assert energy_only.forces is None


def test_failed_df_group_keeps_outputs_and_revokes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = calculator("df-ccsd(t)", WATER, "sto-3g", "def2-svp", ccsd_max_iterations=1)
    library = owner._library
    execute = library.generativeqc_calculation_execute_components_v1
    observed = []

    def failed(owner_handle: object, *outputs: object) -> int:
        descriptors = [
            ctypes.cast(output, ctypes.POINTER(_native.ResultDescriptor)).contents
            for output in outputs
        ]
        for descriptor in descriptors:
            descriptor.energy = 123.0
            np.ctypeslib.as_array(descriptor.forces, shape=(9,)).fill(456.0)
        status = execute(owner_handle, *outputs)
        assert status != _native.STATUS_SUCCESS
        for descriptor in descriptors:
            assert descriptor.energy == 123.0
            np.testing.assert_array_equal(
                np.ctypeslib.as_array(descriptor.forces, shape=(9,)), 456.0
            )
        assert (
            library.generativeqc_calculation_get_mp2_component_diagnostic_v1(
                owner_handle, None
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )
        observed.append(status)
        return status

    monkeypatch.setattr(
        library, "generativeqc_calculation_execute_components_v1", failed
    )
    try:
        with pytest.raises(RuntimeError):
            owner.singlepoint_components(WATER, companions=("rhf", "mp2"))
        assert observed
    finally:
        owner.clear_cache()


def test_correlation_auxiliary_g_handles_are_source_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission of the correlation fit must not widen orbital or DF-HF routes."""
    owner = calculator("mp2", WATER, "cc-pvtz", "cc-pvtz-ri")
    atoms = tuple(Atom.from_value(atom) for atom in WATER)
    library = owner._library
    context = ctypes.c_void_p()
    orbital = auxiliary = None
    try:
        _native.check(
            library,
            library.generativeqc_context_create(
                ctypes.byref(owner._context_descriptor()), ctypes.byref(context)
            ),
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                library,
                "generativeqc_correlation_auxiliary_system_create_v1",
                library.generativeqc_system_create,
            )
            with pytest.raises(NotImplementedError, match="l<=3"):
                owner._create_native_system(
                    context, atoms, 0, 1, owner._auxiliary_basis
                )
        orbital = owner._create_native_system(context, atoms, 0, 1)
        auxiliary = owner._create_native_system(
            context, atoms, 0, 1, owner._auxiliary_basis
        )
        descriptor = owner._method_descriptor(auxiliary)
        for source_as_orbital in (False, True):
            descriptor.density_fitting_mode = (
                _native.DENSITY_FITTING_CORRELATION_CUDA
                if source_as_orbital
                else _native.DENSITY_FITTING_CUDA
            )
            system = auxiliary if source_as_orbital else orbital
            plan = ctypes.c_void_p()
            status = library.generativeqc_calculation_prepare(
                context, system, ctypes.byref(descriptor), ctypes.byref(plan)
            )
            assert status == _native.STATUS_NOT_IMPLEMENTED and not plan.value
            fleet = (ctypes.c_void_p * 1)(system.value)
            status = library.generativeqc_batch_prepare(
                context, fleet, 1, ctypes.byref(descriptor), 0, ctypes.byref(plan)
            )
            assert status == _native.STATUS_NOT_IMPLEMENTED and not plan.value
    finally:
        for system in (auxiliary, orbital):
            if system is not None:
                library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)
        owner.clear_cache()


def test_tiny_df_mp2_budget_refuses_then_independent_owner_recovers() -> None:
    bounded = calculator(
        "mp2", HYDROGEN, "sto-3g", "def2-svp", correlation_memory_budget_bytes=1024
    )
    admitted = calculator("mp2", HYDROGEN, "sto-3g", "def2-svp")
    try:
        with pytest.raises(RuntimeError, match="budget"):
            bounded.singlepoint_components(HYDROGEN, companions=("rhf",))
        result = admitted.singlepoint(HYDROGEN, properties=("energy", "forces"))
        assert result.converged and result.forces is not None
        assert result.correlation.numeric_capacity_bytes <= 2 << 30
    finally:
        bounded.clear_cache()
        admitted.clear_cache()
