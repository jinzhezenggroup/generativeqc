"""Independent M06-2X fitted SCF/force qualification on allocated CUDA GPUs.

Run with GENERATIVEQC_DFT_CUDA_TEST=1 and an installed native CUDA build,
GPU4PySCF and the packaged CUDA stationary force kernels. Each population
checks cold, five frozen-density warm, moved, and five moved-warm endpoints.
The moving-grid comparator includes auxiliary-center/metric response.
This test does not claim qualification until the GPU matrix actually passes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks._retained_basis import load_retained_comparison_basis
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_wb97mv import reference_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires a native CUDA build, GPU4PySCF and an allocated GPU",
)
ROOT = Path(__file__).resolve().parents[2]
ORBITAL = ROOT / "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
AUXILIARY = (
    ROOT / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
)


def _oracle(atoms: tuple, basis: dict, auxiliary: dict, grid: GridSpec) -> Any:
    engine = reference_engine(atoms, basis, grid, xc="M06-2X")
    engine = engine.density_fit(auxbasis=auxiliary)
    engine.conv_tol = 1e-12
    engine.conv_tol_grad = 1e-10
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 200
    return engine


def _reference_endpoint(
    engine: Any, cupy: Any, density: Any = None
) -> tuple[float, np.ndarray]:
    energy = float(engine.kernel(dm0=None if density is None else density.copy()))
    assert engine.converged
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    force = cupy.asnumpy(-gradient.kernel())
    cupy.cuda.Stream.null.synchronize()
    return energy, force


def _assert_endpoint(native: Any, reference: tuple[float, np.ndarray]) -> None:
    energy, force = reference
    assert native.converged
    assert native.physical_residual_rms < 1e-9
    assert abs(native.energy - energy) <= 1e-8
    np.testing.assert_allclose(native.forces, force, atol=1e-7, rtol=0)


@pytest.mark.parametrize("atom_count", (3, 12))
def test_m06_2x_cuda_df_complete_cold_warm_and_moved_endpoints(atom_count: int) -> None:
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("gpu4pyscf")
    from generativeqc._ks_snapshot import NativeKsSnapshot

    case = scaling_cases()[f"water-{atom_count}"]
    orbital, reference_basis = load_comparison_basis(
        ORBITAL, case, role="orbital", compute_forces=True
    )
    auxiliary, reference_auxiliary = load_retained_comparison_basis(
        AUXILIARY, case, role="auxiliary", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    atoms = tuple(case.atoms)
    moved = tuple(
        (
            symbol,
            (
                position[0] + (0.016 if index == 1 else 0.0),
                position[1] - (0.009 if index == 1 else 0.0),
                position[2] + (0.013 if index == 1 else 0.0),
            ),
        )
        for index, (symbol, position) in enumerate(atoms)
    )
    calculator = Calculator(
        method="m06-2x-rks",
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        precision="fp64",
        ks_options=KsOptions(grid=grid),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
    )
    assert "forces" in calculator.capabilities.supported_properties
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        reference = _oracle(atoms, reference_basis, reference_auxiliary, grid)
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        _assert_endpoint(cold, _reference_endpoint(reference, cupy))

        snapshot = NativeKsSnapshot(batch, 0)
        try:
            coulomb, exchange, threshold = snapshot.fock_provider_proof()
            assert (coulomb, exchange) == ("density-fitted", "density-fitted")
            assert threshold == calculator._density_fitting_relative_threshold
        finally:
            snapshot.close()

        seed = reference.make_rdm1().copy()
        batch.set_warm_start_updates(False)
        for _ in range(5):
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert warm.warm_start_used
            _assert_endpoint(warm, _reference_endpoint(reference, cupy, seed))

        batch.set_warm_start_updates(True)
        reference_moved = _oracle(moved, reference_basis, reference_auxiliary, grid)
        changed = batch.execute(
            coordinates=[[position for _, position in moved]],
            strict=True,
            properties=("energy", "forces"),
        ).items[0]
        _assert_endpoint(changed, _reference_endpoint(reference_moved, cupy))
        moved_seed = reference_moved.make_rdm1().copy()
        batch.set_warm_start_updates(False)
        for _ in range(5):
            # Omitted coordinates replay the preparation geometry, so every
            # moved observation must explicitly retain the moved geometry.
            warm = batch.execute(
                coordinates=[[position for _, position in moved]],
                strict=True,
                properties=("energy", "forces"),
            ).items[0]
            assert warm.warm_start_used
            _assert_endpoint(
                warm, _reference_endpoint(reference_moved, cupy, moved_seed)
            )
