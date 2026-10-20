"""Opt-in complete large-owner gates, independent of README benchmarks."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from test_treutler_endpoints import WATER, reference, require_endpoint


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize(
    "method,functional", [("pbe0-rks", "PBE0"), ("wb97m-v-rks", "WB97M_V")]
)
def test_large_treutler_phased_endpoint(
    method: str, functional: str, fitted: bool
) -> None:
    """Gate native/composite forces, reuse, moved geometry and bounded storage.

    Level 0 keeps the 48-atom independent oracle affordable. This checks the
    derivative of that exact discrete quadrature, not grid convergence or a
    production performance ranking. Both XC and VV10 retain Treutler defaults
    for the radial rule, pruning, angular rule and radius adjustment.
    """
    if os.environ.get("GENERATIVEQC_TREUTLER_PHASED_ENDPOINT") != "1":
        pytest.skip("explicit large Treutler endpoint qualification")
    require_endpoint("cuda")
    atoms = []
    for fragment in range(16):
        shift = np.array([8 * (fragment % 4), 8 * (fragment // 4), 0.0])
        atoms.extend((symbol, tuple(np.asarray(xyz) + shift)) for symbol, xyz in WATER)
    positions = np.array([xyz for _, xyz in atoms])
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda" if fitted else "none",
        auxiliary_basis="def2-svp" if fitted else None,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=150,
    )
    records = []
    start = perf_counter()
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        native = batch._native_dft_force_eligible(0)
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        records.append(
            {
                "state": "cold",
                "seconds": perf_counter() - start,
                "iterations": cold.iterations,
                "fock_builds": cold.fock_builds,
            }
        )
        assert cold.converged
        for repeat in range(2):
            start = perf_counter()
            warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            records.append(
                {
                    "state": f"warm-{repeat}",
                    "seconds": perf_counter() - start,
                    "iterations": warm.iterations,
                    "fock_builds": warm.fock_builds,
                }
            )
            assert warm.converged
            np.testing.assert_allclose(warm.forces, cold.forces, atol=1e-7, rtol=0)
        positions[-1] += (0.002, -0.001, 0.003)
        start = perf_counter()
        moved = batch.execute(
            [positions], strict=True, properties=("energy", "forces")
        ).items[0]
        records.append(
            {
                "state": "moved",
                "seconds": perf_counter() - start,
                "iterations": moved.iterations,
                "fock_builds": moved.fock_builds,
            }
        )
        assert moved.converged
        if not native:
            work = batch._stationary_cuda_execution.sources.metrics()
            assert work["becke_phase_batches"] > 0 and work["becke_phase_points"] > 0
            records.append({"work": work})
        direction = np.zeros_like(positions)
        direction[-1] = (0.3, -0.2, 0.4)
        analytic = -float(np.sum(moved.forces * direction))
        for step in (2e-4, 1e-4):
            energies = [
                batch.execute(
                    [positions + sign * step * direction],
                    strict=True,
                    properties=("energy",),
                )
                .items[0]
                .energy
                for sign in (1, -1)
            ]
            assert (energies[0] - energies[1]) / (2 * step) == pytest.approx(
                analytic, abs=2e-7
            )
    displaced = [
        (symbol, tuple(xyz)) for (symbol, _), xyz in zip(atoms, positions, strict=True)
    ]
    energy, forces = reference(
        calculator,
        fitted=fitted,
        functional=functional,
        atom_values=displaced,
        grid_level=0,
        nonlocal_level=0,
    )
    assert moved.energy == pytest.approx(energy, abs=2e-8)
    np.testing.assert_allclose(moved.forces, forces, atol=3e-7, rtol=0)
    if native:
        start = perf_counter()
        with calculator.prepare_batch([displaced], warm_start=False) as batch:
            query = batch._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
            query.argtypes = [
                ct.c_void_p,
                ct.c_uint32,
                ct.POINTER(ct.c_uint64),
                ct.c_size_t,
            ]
            capacity = (ct.c_uint64 * 9)()
            _native.check(
                batch._library, query(batch._batch, 0, capacity, len(capacity))
            )
            assert capacity[5] > capacity[4]
            limit = batch._library.generativeqc_ks_batch_stationary_cuda_limit_v1
            limit.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint64]
            _native.check(batch._library, limit(batch._batch, 0, capacity[4]))
            bounded = batch.execute(strict=True, properties=("energy", "forces")).items[
                0
            ]
            records.append(
                {
                    "state": "bounded-cold",
                    "seconds": perf_counter() - start,
                    "iterations": bounded.iterations,
                    "fock_builds": bounded.fock_builds,
                    "capacity": list(capacity),
                }
            )
            assert bounded.converged
            assert bounded.energy == pytest.approx(moved.energy, abs=1e-9)
            np.testing.assert_allclose(bounded.forces, moved.forces, atol=1e-7, rtol=0)
    if output := os.environ.get("GENERATIVEQC_TREUTLER_PHASED_EVIDENCE"):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / f"{method}-df-{fitted}.json").write_text(
            json.dumps(
                {
                    "scope": "complete 48-atom water/STO-3G level-0 schedule qualification",
                    "slurm_job": os.environ["SLURM_JOB_ID"],
                    "native_owner": native,
                    "method": method,
                    "density_fitting": fitted,
                    "xc_grid_points": calculator.ks_options.grid.point_count(
                        [8, 1, 1] * 16
                    ),
                    "vv10_grid_points": calculator.ks_options.grid.point_count(
                        [8, 1, 1] * 16
                    )
                    if functional == "WB97M_V"
                    else 0,
                    "records": records,
                    "energy_error": moved.energy - energy,
                    "max_force_error": float(np.max(np.abs(moved.forces - forces))),
                },
                indent=2,
            )
            + "\n"
        )
