"""Fresh-process complete cold/warm CPU MP2 E+F observation."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from generativeqc import Calculator

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
cases = {
    "h2": [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))],
    "lih": [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 3.0))],
    "water": [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))],
}
observations = {}
for name, atoms in cases.items():
    calculator = Calculator(
        method="mp2",
        basis="sto-3g",
        device="cpu",
        precision="fp64",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    started = time.perf_counter()
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        prepare_seconds = time.perf_counter() - started
        samples = []
        for iteration in range(2):
            started = time.perf_counter()
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            samples.append(
                {
                    "execute_seconds": time.perf_counter() - started,
                    "energy": result.energy,
                    "forces": np.asarray(result.forces).tolist(),
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "warm_start_used": result.warm_start_used,
                    "planned_endpoint_peak_bytes": result.correlation.planned_endpoint_peak_bytes,
                }
            )
    observations[name] = {"prepare_seconds": prepare_seconds, "samples": samples}
Path(sys.argv[1]).write_text(
    json.dumps(
        {"affinity": sorted(os.sched_getaffinity(0)), "cases": observations}, indent=2
    )
    + "\n"
)
