"""Qualify the public Direct Cartesian RKS Hessian and its directional route."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc._stationary_cpu import complete_rks_gradient_diagnostic
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.dft import NativeAO

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU",
)

SYMBOLS = ("O", "H", "H", "O", "H", "H")
POSITIONS = np.array(
    [
        [0.00, 0.03, -0.05],
        [1.42, 0.14, 1.06],
        [-1.39, -0.09, 1.11],
        [4.99, 0.63, 1.18],
        [5.75, -0.76, 2.04],
        [5.60, 2.01, 2.19],
    ]
)
GRID = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


def _calculator(
    method: str, device: str, *, resource_budget: ResourceBudget | None = None
) -> Calculator:
    return Calculator(
        method=method,
        device=device,
        precision="fp64",
        basis="sto-3g",
        basis_representation="cartesian",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        resource_budget=resource_budget,
    )


def _direction(generator: np.random.Generator) -> np.ndarray:
    """Use a unit displacement with no rigid translation component."""
    motion = generator.normal(size=POSITIONS.shape)
    motion -= motion.mean(axis=0)
    return motion / np.linalg.norm(motion)


def _gradient(method: str, positions: np.ndarray, cache: Path) -> np.ndarray:
    """Reconverge a fresh CPU SCF and take its analytic gradient, not an HVP."""
    atoms = list(zip(SYMBOLS, positions, strict=True))
    with (
        _calculator(method, "cpu").prepare_batch([atoms]) as batch,
        NativeAO(atoms, basis="sto-3g", representation="cartesian") as basis,
    ):
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        try:
            # The 42 primitive AO entries give 42**4 ordered Direct quartet
            # records plus bounded one-electron work, just above the public
            # CPU force default. Raise only this independent oracle's budget.
            result = complete_rks_gradient_diagnostic(
                state,
                basis,
                cache=cache,
                execution="native",
                max_primitive_records=3_200_000,
                max_host_bytes=128 << 20,
            )
            return np.array(result.gradient, copy=True)
        finally:
            state._source.close()


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_public_cuda_rks_cartesian_water_dimer(
    method: str, tmp_path: Path, record_property: Callable[[str, float], None]
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = list(zip(SYMBOLS, POSITIONS, strict=True))
    generator = np.random.default_rng(1905)
    right, left = _direction(generator), _direction(generator)
    options = {"cache": tmp_path, "solver_options": SOLVER}
    cuda = _calculator(method, "cuda")
    cpu = _calculator(method, "cpu")
    action = cuda.hessian_vector_product(atoms, right, **options)
    transpose_probe = cuda.hessian_vector_product(atoms, left, **options)
    dense = cuda.hessian(atoms, block_size=1, **options)
    analytic = cpu.hessian_vector_product(atoms, right, **options)

    for result in (action, transpose_probe, dense):
        assert result.diagnostics["response_backend"] == "cuda"
        assert result.diagnostics["numerical_host_fallbacks"] == 0
        assert result.diagnostics["df_integral_ad_backend"] is None
        assert result.diagnostics["posthoc_symmetrization"] is False
    invariants = {
        "cpu_analytic_max_error": np.max(np.abs(action.value - analytic.value)),
        "raw_symmetry_max_error": np.max(np.abs(dense.matrix - dense.matrix.T)),
        "raw_dense_action_max_error": np.max(
            np.abs(dense.matrix @ right.ravel() - action.value.ravel())
        ),
        "raw_bilinear_error": abs(
            np.vdot(left, action.value) - np.vdot(right, transpose_probe.value)
        ),
    }
    for name, error in invariants.items():
        record_property(name, float(error))
        assert error <= 2e-8, name

    errors = []
    for step in (1.2e-3, 1.3e-4):
        plus = _gradient(method, POSITIONS + step * right, tmp_path / "gradient")
        minus = _gradient(method, POSITIONS - step * right, tmp_path / "gradient")
        finite_difference = (plus - minus) / (2 * step)
        assert np.max(np.abs(finite_difference)) > 1e-3
        error = float(np.max(np.abs(finite_difference - action.value)))
        record_property(f"reconverged_gradient_difference_{step:.1e}", error)
        errors.append(error)
    assert errors[-1] < 4e-4, errors
    assert errors[-1] < max(0.35 * errors[0], 2e-5), errors


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_public_cuda_rks_cartesian_water_dimer_live_peak(
    method: str, record_property: Callable[[str, int], None]
) -> None:
    """Measure response while the same prepared SCF still owns its storage."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = list(zip(SYMBOLS, POSITIONS, strict=True))
    primal = (
        _calculator(method, "cuda")
        .estimate_resources([atoms], properties=("energy",))
        .require_feasible()
    )
    budget = ResourceBudget(
        host_bytes=primal.peak_bytes["host"] + 16 * POSITIONS.size**2,
        device_bytes=primal.peak_bytes["device"] + (128 << 20),
    )
    calculator = _calculator(method, "cuda", resource_budget=budget)
    direction = _direction(np.random.default_rng(1905))
    action = calculator.hessian_vector_product(atoms, direction, solver_options=SOLVER)
    dense = calculator.hessian(atoms, block_size=1, solver_options=SOLVER)
    np.testing.assert_allclose(
        dense.matrix @ direction.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
    for endpoint, result in (("hvp", action), ("dense", dense)):
        resources = result.diagnostics["resources"]
        scf = resources["observation"]["device_ledger"]
        response = resources["hessian_response"]["device_ledger"]
        assert 0 < response["live_bytes"] == scf["live_bytes"]
        assert (
            response["live_bytes"] + result.diagnostics["device_bytes"]
            <= response["peak_bytes"]
            <= budget.device_bytes
        )
        assert response["rejected_allocations"] == 0
        record_property(f"{endpoint}_live_bytes", response["live_bytes"])
        record_property(f"{endpoint}_peak_bytes", response["peak_bytes"])


def test_dense_r2scan_cartesian_column_matches_signed_hvp() -> None:
    """Compare the all-row atom/AO contraction with the generic signed action."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    positions = np.array(
        [
            (5.8 * (pair % 3), 5.4 * (pair // 3), 0.08 * pair + side * 0.71)
            for pair in range(6)
            for side in (-1, 1)
        ]
    )
    atoms = list(zip(["H"] * len(positions), positions, strict=True))
    direction = np.random.default_rng(1905).normal(size=positions.shape)
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    calculator = _calculator("r2scan-rks", "cuda")
    dense = calculator.hessian(atoms, block_size=1, solver_options=SOLVER)
    action = calculator.hessian_vector_product(atoms, direction, solver_options=SOLVER)
    np.testing.assert_allclose(
        dense.matrix @ direction.ravel(), action.value.ravel(), atol=2e-8, rtol=0
    )
    np.testing.assert_allclose(dense.matrix, dense.matrix.T, atol=2e-8, rtol=0)
    assert dense.diagnostics["response_backend"] == action.diagnostics["response_backend"] == "cuda"
    assert not dense.diagnostics["numerical_host_fallbacks"]
