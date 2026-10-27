"""Exercise staged partition publication on 48 atoms and its bounded fallback.

Independent reconverged energy differences, the incumbent binary, and the
minimum-workspace path must agree with the complete native force endpoint.
This is schedule qualification, not a broader scientific admission claim.
"""

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "tests/python")
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc.automatic_artifacts import read_automatic_artifact_diagnostic
from test_dft_complete_cuda import no_cpu_derivatives
from test_stationary_aot_cuda import _forbid_aot_generation

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
atoms = []
for pair in range(24):
    center = (8.0 * (pair % 4), 8.0 * ((pair // 4) % 4), 8.0 * (pair // 16))
    for offset in (-0.7, 0.7):
        atoms.append(("H", (center[0], center[1], center[2] + offset)))
records = []
for spins in (1, 2):
    calc = Calculator(
        method=f"libxc-{'rks' if spins == 1 else 'uks'}:GGA_X_APBE",
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec(12, 4, 8), tile_points=256),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    positions = np.asarray([p for _, p in atoms])
    positions[-1] += (0.002, -0.001, 0.003)
    geometry = [
        (symbol, tuple(p)) for (symbol, _), p in zip(atoms, positions, strict=True)
    ]
    with calc.prepare_batch([geometry], warm_start=True) as batch:
        with pytest.MonkeyPatch.context() as guard, no_cpu_derivatives():
            _forbid_aot_generation(guard, batch)
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
        receipt = read_automatic_artifact_diagnostic(batch)
        assert (
            receipt["capacity"]["device_upper_bytes"]
            > receipt["capacity"]["minimum_device_bytes"]
        )
        direction = np.zeros_like(positions)
        direction[-1] = (0.3, -0.2, 0.4)
        analytic = float(np.sum(result.forces * direction))
        differences = []
        for step in (2e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                value = batch.execute(
                    [positions + sign * step * direction],
                    strict=True,
                    properties=("energy",),
                ).items[0]
                energies.append(value.energy)
            finite_difference = -(energies[1] - energies[0]) / (2 * step)
            np.testing.assert_allclose(finite_difference, analytic, atol=1e-7, rtol=0)
            differences.append(
                dict(step=step, energies=energies, derivative=finite_difference)
            )
    with calc.prepare_batch([geometry], warm_start=False) as batch:
        query = batch._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        capacity = (ctypes.c_uint64 * 9)()
        _native.check(batch._library, query(batch._batch, 0, capacity, len(capacity)))
        assert capacity[5] > capacity[4]
        limit = batch._library.generativeqc_ks_batch_stationary_cuda_limit_v1
        limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
        limit.restype = ctypes.c_int
        _native.check(batch._library, limit(batch._batch, 0, capacity[4]))
        bounded = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        bounded_receipt = read_automatic_artifact_diagnostic(batch)
        assert bounded_receipt["fallback_reason"] == "minimum-workspace"
        np.testing.assert_allclose(bounded.energy, result.energy, atol=1e-9, rtol=0)
        np.testing.assert_allclose(bounded.forces, result.forces, atol=1e-7, rtol=0)
    records.append(
        dict(
            spins=spins,
            energy=result.energy,
            forces=result.forces.tolist(),
            artifact=receipt,
            analytic=analytic,
            finite_differences=differences,
            bounded_energy=bounded.energy,
            bounded_forces=bounded.forces.tolist(),
            bounded_artifact=bounded_receipt,
        )
    )
    print(spins, "phased / bounded / finite differences passed", flush=True)
Path(sys.argv[1]).write_text(
    json.dumps(
        dict(
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            library=os.environ["GENERATIVEQC_LIBRARY"],
            atoms=geometry,
            rows=records,
        ),
        indent=2,
    )
    + "\n"
)
