"""Complete force parity, actual geometry hits and bounded failure recovery."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("unrestricted", [False, True])
def test_exact_geometry_response_reuse(
    monkeypatch: pytest.MonkeyPatch, unrestricted: bool
) -> None:
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly scheduled GPU")
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    atoms = [("O", (0.02, -0.03, 0.04)), ("H", (0.1, 1.43, 1.1))]
    if not unrestricted:
        atoms.append(("H", (-0.15, -1.45, 1.12)))
    moved = np.array([[xyz for _, xyz in atoms]])
    moved[0, -1, 0] += 0.001
    moved_atoms = [
        (symbol, tuple(xyz)) for (symbol, _), xyz in zip(atoms, moved[0], strict=True)
    ]
    arms = []
    for enabled in (False, True):
        monkeypatch.setenv("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", str(int(enabled)))
        calc = Calculator(
            method="wb97m-v-uks" if unrestricted else "wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
            ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
        )
        results, work = [], []
        with calc.prepare_batch(
            [atoms], multiplicities=[2 if unrestricted else 1], warm_start=True
        ) as batch:
            for geometry in (None, moved):
                batch.set_warm_start_updates(True)
                for repeat in range(3):
                    if repeat:
                        batch.set_warm_start_updates(False)
                    results.append(batch.execute(geometry, strict=True).items[0])
                    work.append(batch._stationary_cuda_execution.last_work)
            monkeypatch.setenv("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", "0")
            results.append(batch.execute(moved, strict=True).items[0])
            work.append(batch._stationary_cuda_execution.last_work)
            assert all(result.converged for result in results)
            assert (
                work[0]["geometry_response_evaluations"]
                == work[3]["geometry_response_evaluations"]
                == work[-1]["geometry_response_evaluations"]
                == 1
            )
            if enabled and not unrestricted:
                assert (
                    work[2]["geometry_response_cache_hits"]
                    == work[5]["geometry_response_cache_hits"]
                    == 1
                )
            if not enabled:
                assert all(item["geometry_response_evaluations"] == 1 for item in work)
            if enabled:
                monkeypatch.setenv("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", "1")
                # Exercise both spins at an identical authenticated density,
                # even when repeated UKS solves change its last few bits.
                with NativeAO(
                    moved_atoms,
                    basis="def2-svp",
                    representation="spherical",
                    multiplicity=2 if unrestricted else 1,
                ) as basis:
                    state = StationaryKsState.from_native(batch, basis, index=0)
                    owner = batch._stationary_cuda_execution
                    arguments = {
                        "compiler": batch._stationary_cuda_compiler(),
                        "cache": Path(os.environ["GENERATIVEQC_STATIONARY_CACHE"]),
                        "library": calc._library._name,
                    }
                    try:
                        first, first_work = owner.execute(state, basis, **arguments)
                        second, second_work = owner.execute(state, basis, **arguments)
                        assert first_work["geometry_response_evaluations"] == 1
                        assert second_work["geometry_response_cache_hits"] == 1
                        assert (
                            second_work["geometry_point_visits"]
                            == second_work["ao_collocation_point_visits"]
                            == second_work["nonlocal_pair_sweeps"]
                            == 0
                        )
                        assert second_work["geometry_response_retained_host_bytes"] > 0
                        np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
                        np.testing.assert_allclose(
                            second, results[-1].forces, rtol=0, atol=1e-7
                        )
                        # An inadmissible budget fails closed and discards the
                        # owner; a later valid call recomputes and can retain.
                        with pytest.raises(ValueError):
                            owner.execute(state, basis, max_host_bytes=1, **arguments)
                        recovered, recovered_work = owner.execute(
                            state, basis, **arguments
                        )
                        assert recovered_work["geometry_response_evaluations"] == 1
                        np.testing.assert_allclose(recovered, first, rtol=0, atol=1e-12)
                    finally:
                        state._source.close()
        arms.append(results)
    for baseline, candidate in zip(*arms, strict=True):
        assert abs(baseline.energy - candidate.energy) <= 1e-8
        np.testing.assert_allclose(baseline.forces, candidate.forces, rtol=0, atol=1e-7)
