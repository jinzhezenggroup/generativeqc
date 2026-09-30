"""Real-device parity for the non-resident final-feature capability fallback."""

import os

import numpy as np
import pytest


@pytest.mark.parametrize("spin", [0, 1])
def test_host_unfused_wb97mv_forces_match_resident_and_independent_engine(
    spin: int,
) -> None:
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks.readme_wb97mv import reference_engine, reference_sample

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    if spin:
        atoms.append(("H", (1.8, -0.1, -0.3)))
    grid = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
    results = []
    for schedule, passes, feature_source in (
        ("device_fused", 1, "exact-final-scf-device-binding"),
        ("host_unfused", 2, "bounded-grid-feature-collection"),
    ):
        calc = Calculator(
            method="wb97m-v-uks" if spin else "wb97m-v",
            basis="sto-3g",
            device="cuda",
            ks_options=KsOptions(grid=grid, xc_schedule=schedule),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
            max_iterations=200,
        )
        with calc.prepare_batch(
            [atoms], multiplicities=[spin + 1], warm_start=True
        ) as batch:
            cold = batch.execute(strict=True).items[0]
            warm = batch.execute(strict=True).items[0]
            work = batch._stationary_cuda_execution.last_work
            assert work["prepared_execution_reused"]
            assert work["nonlocal_feature_source"] == feature_source
            assert work["ao_collocation_point_visits"] == passes * work["grid_points"]
            assert work["geometry_point_visits"] == 2 * work["grid_points"]
            assert work["nonlocal_feature_d2h_bytes"] == 0
            assert work["nonlocal_seed_h2d_bytes"] == 0
            assert cold.executed_backend == warm.executed_backend == "cuda"
            np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
            assert abs(warm.energy - cold.energy) < 1e-9
            results.append(cold)
    oracle = reference_sample(reference_engine(atoms, "sto-3g", grid, spin=spin), cp)
    for result in results:
        assert abs(result.energy - oracle["energies_hartree"][0]) < 1e-8
        np.testing.assert_allclose(
            result.forces, oracle["forces_hartree_per_bohr"][0], atol=1e-7, rtol=1e-7
        )
        np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    assert abs(results[0].energy - results[1].energy) < 1e-8
    np.testing.assert_allclose(
        results[0].forces, results[1].forces, atol=1e-7, rtol=1e-7
    )
