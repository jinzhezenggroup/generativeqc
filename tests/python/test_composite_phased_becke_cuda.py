"""Admitted composite geometry schedules preserve reconverged energy derivatives."""

import os

import numpy as np
import pytest


def test_composite_phased_becke_matches_reconverged_energy_derivatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check both owners and a directional derivative at two finite-difference steps.

    Absolute independent E/F qualification uses the original README water-48
    grid and basis; this modest fixture isolates the geometry derivative.
    """
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator, GridSpec, KsOptions

    # A modest orbital/grid fixture reaches the real atom-count crossover
    # without turning this regression into the full README timing population.
    # Separated neutral H2 fragments have a well-conditioned small-basis SCF.
    # Twenty-four pairs cross the unmodified 48-atom automatic threshold.
    atoms = []
    for pair in range(24):
        center = (8.0 * (pair % 3), 8.0 * ((pair // 3) % 3), 8.0 * (pair // 9))
        for offset in (-0.7, 0.7):
            atoms.append(("H", (center[0], center[1], center[2] + offset)))
    grid = GridSpec(12, 4, 8)
    monkeypatch.setenv("GENERATIVEQC_STATIONARY_BECKE_PRIMITIVE", "off")
    calculator = Calculator(
        method="wb97m-v",
        basis="sto-3g",
        basis_representation="spherical",
        device="cuda",
        device_id=int(os.environ.get("GENERATIVEQC_TEST_DEVICE_ID", "0")),
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        for displaced in (False, True):
            positions = np.asarray([xyz for _, xyz in atoms], dtype=np.float64)
            if displaced:
                positions[-1] += (0.002, -0.001, 0.003)
            actual = batch.execute([positions], strict=True).items[0]
            # Omitting coordinates reselects the original prepared geometry.
            warm = batch.execute([positions], strict=True).items[0]
            np.testing.assert_allclose(warm.forces, actual.forces, atol=1e-8, rtol=0)
            observed = batch._stationary_cuda_execution.last_work
            owners = observed["stationary_source_work"]
            assert set(owners) == {"semilocal", "nonlocal"}
            for owner in owners.values():
                assert owner["becke_primitive_selected"] == 0
                assert owner["phased_becke_bytes"] > 0
                assert owner["becke_phase_batches"] > 0
                assert owner["becke_phase_points"] == observed["grid_points"]
            # Rebuild the complete SCF energy for each displaced point. The
            # analytic force is not used to construct the independent oracle.
            direction = np.zeros_like(positions)
            direction[-1] = (0.3, -0.2, 0.4)
            analytic = -float(np.sum(actual.forces * direction))
            for step in (2e-4, 1e-4):
                plus = batch.execute([positions + step * direction], strict=True).items[
                    0
                ]
                minus = batch.execute(
                    [positions - step * direction], strict=True
                ).items[0]
                derivative = (plus.energy - minus.energy) / (2 * step)
                np.testing.assert_allclose(derivative, analytic, atol=1e-7, rtol=0)
