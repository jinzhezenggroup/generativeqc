"""Issue #2412: public CUDA DF B3LYP energy/force and state provenance.

Run with GENERATIVEQC_DFT_CUDA_TEST=1 in a finite Slurm GPU allocation.
"""

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from test_dft_df_public import WATER, pyscf_energy_gradient


def test_b3lyp_df_cold_warm_and_moved_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare all 12 E+F calls to independent DF/metric/moving-grid PySCF."""
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires a source-matched CUDA library/device")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"

    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    calc = Calculator(
        method="b3lyp-rks",
        basis="sto-3g",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis="def2-svp",
        precision="fp64",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
        ),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    assert "forces" in calc.capabilities.supported_properties

    xyz = np.asarray([position for _, position in WATER], dtype=np.float64)
    xyz[1] += (0.02, -0.01, 0.03)
    moved_atoms = [
        (symbol, tuple(map(float, position)))
        for (symbol, _), position in zip(WATER, xyz, strict=True)
    ]

    def independent(batch: Any, atoms: Any) -> tuple[float, np.ndarray]:
        with NativeAO(
            atoms,
            basis=calc._basis,
            representation=calc._representation_name,
            multiplicity=1,
        ) as basis:
            state = StationaryKsState.from_native(batch, basis)
            try:
                source = state._source
                assert state.identity.method == "b3lyp-rks"
                assert source.functional_code == 3
                assert source.metadata[7] == 2  # explicit B3LYP SCF domain v2
                assert source.coefficients == (1.0, 1.0, -0.1)
                assert source.fock_provider_proof() == (
                    "density-fitted",
                    "density-fitted",
                    calc._density_fitting_relative_threshold,
                )
                return pyscf_energy_gradient(calc, state, atoms, 1, "B3LYP")
            finally:
                state._source.close()

    def check(item: Any, oracle: tuple[float, np.ndarray]) -> None:
        reference_energy, reference_gradient = oracle
        assert item.succeeded and item.converged and item.forces is not None
        assert item.executed_backend == "cuda"
        assert item.energy == pytest.approx(reference_energy, abs=1e-8)
        np.testing.assert_allclose(item.forces, -reference_gradient, atol=1e-7, rtol=0)
        np.testing.assert_allclose(item.forces.sum(axis=0), 0, atol=2e-9, rtol=0)

    receipts: list[Any] = []
    with calc.prepare_batch([WATER], warm_start=True) as batch:
        consumer = batch._public_dft_cuda_force

        def record(index: int, atoms: Any) -> tuple[Any, Any]:
            forces, work = consumer(index, atoms)
            receipts.append(work)
            return forces, work

        monkeypatch.setattr(batch, "_public_dft_cuda_force", record)
        cold = batch.execute(properties=("energy", "forces"), strict=True).items[0]
        original_oracle = independent(batch, WATER)
        check(cold, original_oracle)
        for _ in range(5):
            warm = batch.execute(properties=("energy", "forces"), strict=True).items[0]
            assert warm.warm_start_used
            check(warm, original_oracle)

        moved = batch.execute(
            coordinates=(xyz,), properties=("energy", "forces"), strict=True
        ).items[0]
        moved_oracle = independent(batch, moved_atoms)
        check(moved, moved_oracle)
        for _ in range(5):
            moved_warm = batch.execute(
                coordinates=(xyz,), properties=("energy", "forces"), strict=True
            ).items[0]
            check(moved_warm, moved_oracle)

    assert len(receipts) == 12
    for work in receipts:
        assert (
            work["stationary_integral_derivative_route"] == "prepared-native-complete"
        )
        assert work["stationary_native_integral_sources"] == (
            "one_electron",
            "overlap_pulay",
            "coulomb",
            "exact_exchange",
        )
        assert work["density_fitted_response_resources_included"] is False
