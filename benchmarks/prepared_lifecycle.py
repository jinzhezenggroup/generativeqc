"""Matched complete-endpoint lifecycle qualification, one library per process.

Select GENERATIVEQC_LIBRARY externally so baseline and candidate never share
loaded code, contexts, streams, buffers or generated artifacts. CUDA invocations
must run inside a finite Slurm allocation, including profiler/sanitizer calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
HE = [("He", (0.0, 0.0, 0.0))]


def qualify(device: str, method: str, forces: bool, repeats: int) -> dict[str, Any]:
    """Time prepare/cold and full replay, not only SCF or response kernels."""
    if device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA qualification requires a finite Slurm allocation")
    calculator = Calculator(
        device=device,
        method=method,
        density_fitting="none",
        precision="fp64",
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    properties = ("energy", "forces") if forces else ("energy",)
    rows = []
    transport = []
    moved = np.array([[0.0, 0.0, -0.8], [0.0, 0.0, 0.8]])
    for repeat in range(repeats):
        started = perf_counter()
        with calculator.prepare_batch([H2, HE]) as batch:
            preparation = perf_counter() - started
            for case, coordinates in (
                ("cold", None),
                ("warm", None),
                ("changed_geometry", [moved, None]),
                ("changed_replay", [moved, None]),
                ("reset", None),
                ("ragged_failure", [np.full((2, 3), np.nan), None]),
                ("recovery", None),
            ):
                if case == "reset":
                    batch.clear_warm_starts()
                started = perf_counter()
                result = batch.execute(coordinates, properties=properties)
                seconds = perf_counter() - started
                if case != "ragged_failure" and result.failure_indices:
                    raise RuntimeError(
                        f"incomplete {case} endpoint: {result.failure_indices}"
                    )
                if case == "ragged_failure" and result.failure_indices != (0,):
                    raise RuntimeError("ragged failure was not isolated")
                rows.append(
                    {
                        "repeat": repeat,
                        "case": case,
                        "seconds": seconds + (preparation if case == "cold" else 0),
                        "preparation_seconds": preparation if case == "cold" else None,
                        "items": [
                            {
                                "status": item.status,
                                "energy": item.energy if item.succeeded else None,
                                "forces": None
                                if item.forces is None
                                else item.forces.tolist(),
                                "backend": item.executed_backend,
                                "iterations": item.iterations,
                                "fock_builds": item.fock_builds,
                                "physical_residual_rms": item.physical_residual_rms,
                                "warm_start_used": item.warm_start_used,
                            }
                            for item in result.items
                        ],
                    }
                )
            transport.append(
                [
                    None if value is None else asdict(value)
                    for value in batch.ks_transport_diagnostics
                ]
            )
    library = Path(calculator._library._name).resolve()
    return {
        "schema_version": 1,
        "device": device,
        "method": method,
        "forces": forces,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_identity": calculator._library.generativeqc_get_source_identity().decode(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rows": rows,
        "native_ks_transport": transport,
        "unmeasured": [
            "full host heap",
            "vendor-internal allocations",
            "non-KS native transfer/work counts",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--method", choices=("rhf", "pbe-rks"), required=True)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        parser.error("repeats must be positive")
    receipt = qualify(
        arguments.device, arguments.method, arguments.forces, arguments.repeats
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
