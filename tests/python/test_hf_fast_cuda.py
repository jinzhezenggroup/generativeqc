"""Budget-selected RHF derivatives: independent oracle and per-item provenance.

Run only through a finite Slurm allocation. SCF, one-electron/Pulay terms,
geometry, Boys moments and global accumulation are still FP64.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, PrecisionCalculator, TargetErrorBudget, _native
from generativeqc._model_resolution import snapshot_basis
from generativeqc.elements import SYMBOLS

from benchmarks._cases import benchmark_cases

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)


def settings(case_name: str) -> dict[str, Any]:
    """Keep physical inputs and convergence identical for strict and Fast."""
    case = benchmark_cases()[case_name]
    return {
        "method": "rhf",
        "device": "cuda",
        "basis": case.generativeqc_basis,
        "basis_representation": case.basis_representation,
        "initial_guess": None,
        "max_iterations": 180,
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
    }


def oracle(case_name: str, atoms: Any = None) -> tuple[float, np.ndarray]:
    """PySCF owns integrals, SCF and derivatives; only basis inputs are shared."""
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(4)
    case = benchmark_cases()[case_name]
    basis = snapshot_basis(case.generativeqc_basis, case.basis_representation)
    shells = {
        SYMBOLS[element.atomic_number - 1]: [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell.coefficients],
                    ]
                    for index, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in basis.elements
    }
    molecule = pyscf.gto.M(
        atom=case.atoms if atoms is None else atoms,
        basis=shells,
        unit="Bohr",
        cart=case.basis_representation == "cartesian",
        verbose=0,
    )
    solver = molecule.RHF()
    solver.conv_tol = 1e-13
    solver.conv_tol_grad = 1e-10
    solver.direct_scf_tol = 0
    solver.max_cycle = 180
    energy = solver.kernel()
    assert solver.converged
    return energy, -solver.nuc_grad_method().kernel()


def assert_endpoint(result: Any, expected: tuple[float, np.ndarray]) -> None:
    """Tighter-than-preset gates qualify the retained small direct domain."""
    energy, forces = expected
    assert result.converged
    assert abs(result.energy - energy) < 1e-7
    error = result.forces - forces
    assert np.max(np.abs(error)) < 1e-5
    assert np.sqrt(np.mean(error**2)) < 3e-6
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    work = result.precision["gradient"]
    assert work["schema_version"] == 3
    assert work["qualification"] == "rhf-budget-fp32-v1"
    assert work["complete"] and work["returned_mode"] == "mixed"
    assert work["weighted_task_counters_collected"]
    assert work["attempted_fp32_weighted_tasks"] > 0
    assert work["attempted_fp32_primitive_sources"] > 0
    assert work["weighted_task_retries"] == 0
    assert work["attempted_fp32_geometry_points"] == 0
    assert work["attempted_fp32_grid_points"] == 0


@pytest.mark.parametrize(
    "case_name",
    (
        "water-def2-svp-spherical",
        "water-tetramer-def2-svp-spherical",
        "water-def2-svp",
        "ammonia-def2-svp-spherical",
    ),
)
@pytest.mark.parametrize("initial_guess", (None, "auto"))
def test_hf_fast_complete_public_endpoint(
    case_name: str, initial_guess: str | None
) -> None:
    expected = oracle(case_name)
    calculator = PrecisionCalculator(
        policy="fast", **{**settings(case_name), "initial_guess": initial_guess}
    )
    try:
        result = calculator.singlepoint(benchmark_cases()[case_name].atoms)
        assert_endpoint(result, expected)
        assert result.precision_validation.solve_count == 1
        assert result.precision_validation.status == "qualified_mixed_gradient"
        energy = calculator.singlepoint(
            benchmark_cases()[case_name].atoms, properties=("energy",)
        )
        assert energy.precision.get("gradient") is None
    finally:
        calculator.clear_cache()


def test_hf_prepared_counts_are_per_item_and_reset_on_replay() -> None:
    """Fusing identical items cannot multiply each item's semantic work count."""
    name = "water-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    expected = oracle(name)
    options = settings(name)
    options["screening_tolerance"] = 1e-30
    calculator = Calculator(
        precision="fp64", _application_precision_budget=BUDGET, **options
    )
    try:
        with calculator.prepare_batch([atoms]) as single:
            individual = single.execute(
                properties=("energy", "forces"), strict=True
            ).items[0]
            assert_endpoint(individual, expected)
        with calculator.prepare_batch([atoms, atoms, atoms]) as batch:
            for _ in range(2):
                for result in batch.execute(
                    properties=("energy", "forces"), strict=True
                ).items:
                    assert_endpoint(result, expected)
                    assert (
                        result.precision["gradient"]["attempted_fp32_weighted_tasks"]
                        == individual.precision["gradient"][
                            "attempted_fp32_weighted_tasks"
                        ]
                    )
                    assert (
                        result.precision["gradient"]["attempted_fp32_primitive_sources"]
                        == individual.precision["gradient"][
                            "attempted_fp32_primitive_sources"
                        ]
                    )
            for result in batch.execute(properties=("energy",), strict=True).items:
                assert result.precision.get("gradient") is None
    finally:
        calculator.clear_cache()


def test_hf_frozen_seed_geometry_move_and_failure_isolation() -> None:
    """A move gets its own strict state; invalid peers inherit no old proof."""
    name = "water-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in atoms]
    moved[1][1][0] += 0.025
    positions = np.array([position for _, position in moved])
    original_oracle, moved_oracle = oracle(name), oracle(name, moved)
    calculator = Calculator(
        precision="fp64", _application_precision_budget=BUDGET, **settings(name)
    )
    try:
        with calculator.prepare_batch([atoms, atoms]) as batch:
            cold = batch.execute(properties=("energy", "forces"), strict=True)
            for item in cold.items:
                assert_endpoint(item, original_oracle)
            batch.set_warm_start_updates(False)
            for item in batch.execute(
                properties=("energy", "forces"), strict=True
            ).items:
                assert_endpoint(item, original_oracle)
            batch.set_warm_start_updates(True)
            moved_result = batch.execute(
                [positions, None], properties=("energy", "forces"), strict=True
            )
            assert_endpoint(moved_result.items[0], moved_oracle)
            assert_endpoint(moved_result.items[1], original_oracle)
            batch.set_warm_start_updates(False)
            replay = batch.execute(
                [positions, None], properties=("energy", "forces"), strict=True
            )
            assert_endpoint(replay.items[0], moved_oracle)
            assert_endpoint(replay.items[1], original_oracle)
            failed = batch.execute(
                [positions, np.zeros((1, 3))], properties=("energy", "forces")
            )
            assert failed.failure_indices == (1,)
            assert (failed.items[1].precision or {}).get("gradient") is None
            assert_endpoint(failed.items[0], moved_oracle)
    finally:
        calculator.clear_cache()


def test_hf_tight_request_does_not_narrow_derivatives() -> None:
    name = "water-def2-svp-spherical"
    calculator = Calculator(
        precision="fp64",
        _application_precision_budget=TargetErrorBudget(1e-8, 1e-7, 3e-8),
        **settings(name),
    )
    try:
        result = calculator.singlepoint(benchmark_cases()[name].atoms)
        assert result.precision.get("gradient") is None
        energy, forces = oracle(name)
        assert abs(result.energy - energy) < 1e-7
        np.testing.assert_allclose(result.forces, forces, atol=1e-6, rtol=0)
    finally:
        calculator.clear_cache()


def test_hf_gradient_transport_versions_and_bounds() -> None:
    """Old clients cannot mistake HF for PBE0; unpublished records stay absent."""
    name = "water-def2-svp-spherical"
    calculator = Calculator(
        precision="fp64", _application_precision_budget=BUDGET, **settings(name)
    )
    try:
        with calculator.prepare_batch([benchmark_cases()[name].atoms]) as batch:
            library = calculator._library
            getter = library.generativeqc_batch_gradient_precision_v3
            values = (ctypes.c_uint64 * 10)()
            assert getter(batch._batch, 0, values, 9) == _native.STATUS_INVALID_ARGUMENT
            assert getter(None, 0, values, 10) == _native.STATUS_INVALID_ARGUMENT
            assert (
                getter(batch._batch, 0, values, 10)
                == _native.STATUS_PRECISION_UNAVAILABLE
            )
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert getter(batch._batch, 0, values, 10) == _native.STATUS_SUCCESS
            assert (
                getter(batch._batch, 1, values, 10) == _native.STATUS_INVALID_ARGUMENT
            )
            assert list(values)[5:7] == [5, 1]
            assert (
                values[7]
                == result.precision["gradient"]["attempted_fp32_weighted_tasks"]
            )
            assert (
                library.generativeqc_batch_gradient_precision_v1(
                    batch._batch, 0, values, 5
                )
                == _native.STATUS_NOT_IMPLEMENTED
            )
            assert (
                library.generativeqc_batch_gradient_precision_v2(
                    batch._batch, 0, values, 6
                )
                == _native.STATUS_SUCCESS
            )
            assert values[5] == 5
            batch.execute(properties=("energy",), strict=True)
            assert (
                getter(batch._batch, 0, values, 10)
                == _native.STATUS_PRECISION_UNAVAILABLE
            )
    finally:
        calculator.clear_cache()
