"""Compare complete native RHF energy/nuclear-gradient calls, not kernels.

GPU execution is opt-in and must run through a finite-time Slurm srun job on
an allowed node. Timing includes native method invocation and host result
publication. Cold JIT is separate; no peak-memory or universal CUDA-performance
claim is manufactured when those measurements are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import Calculator
from generativeqc.experimental import array_api as xp


def _scheduler_guard(backend: str) -> None:
    """Admit both SSH aliases and Slurm's verified node names, never node3."""
    if backend == "cuda" and (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURMD_NODENAME")
        not in {"n1", "n2", "n4", "n5", "node1", "node2", "node4", "node5"}
    ):
        raise RuntimeError(
            "CUDA benchmarks require srun on n1/n2/n4/n5, with finite --time"
        )


def _timed(function: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def main() -> None:
    """Run identical FP64 exact-J/K endpoints and retain honest qualification gaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--molecule", choices=("h2", "water"), default="water")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--cuda-arch", default="sm_120")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 3 <= arguments.repeats <= 100:
        parser.error("--repeats must lie in [3, 100]")
    _scheduler_guard(arguments.backend)
    if arguments.molecule == "h2":
        elements = ("H", "H")
        coordinates = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    else:
        elements = ("O", "H", "H")
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]]
        )
    operator = xp.RHF(elements, basis=arguments.basis, backend=arguments.backend)
    native = Calculator(
        method="rhf",
        basis=arguments.basis,
        device=arguments.backend,
        initial_guess=None,
        density_fitting="none",
        precision="fp64",
        **dict(operator._controls),
    )

    def energy(positions: Any) -> Any:
        return operator(positions)

    compiled = xp.value_and_grad(
        energy, backend=arguments.backend, cuda_arch=arguments.cuda_arch
    )

    def baseline() -> Any:
        return native.singlepoint(
            operator._atoms(coordinates), properties=("energy", "forces")
        )

    def generated() -> Any:
        return compiled(coordinates)

    try:
        reference, native_cold = _timed(baseline)
        transformed, generated_cold = _timed(generated)
        energy_error = abs(float(transformed[0]) - reference.energy)
        gradient_error = float(
            np.max(np.abs(np.asarray(transformed[1]) + reference.forces))
        )
        if energy_error > 1e-10 or gradient_error > 1e-8:
            raise RuntimeError("complete RHF endpoint failed numerical acceptance")
        samples: dict[str, list[float]] = {"native": [], "array_api": []}
        iterations: dict[str, list[int]] = {"native": [], "array_api": []}
        # Alternate order to avoid making the second path systematically warmer.
        for repeat in range(arguments.repeats):
            order = (
                ("native", "array_api") if repeat % 2 == 0 else ("array_api", "native")
            )
            for name in order:
                result, elapsed = _timed(baseline if name == "native" else generated)
                samples[name].append(elapsed)
                iterations[name].append(
                    result.iterations
                    if name == "native"
                    else compiled.compilation_info()["molecular_calls"][0]["iterations"]
                )
                if name == "native":
                    value, gradient = result.energy, -result.forces
                else:
                    value, gradient = result
                if (
                    abs(float(value) - reference.energy) > 1e-10
                    or float(np.max(np.abs(np.asarray(gradient) + reference.forces)))
                    > 1e-8
                ):
                    raise RuntimeError("warm RHF replay failed numerical acceptance")
        ratio = statistics.median(samples["array_api"]) / statistics.median(
            samples["native"]
        )
        info = compiled.compilation_info()
        matching_work = iterations["array_api"] == iterations["native"]
        record = {
            "schema": "generativeqc.array-api-rhf-endpoint/v1",
            "endpoint": "host Bohr coordinates -> converged total energy + nuclear gradient",
            "molecule": arguments.molecule,
            "elements": elements,
            "coordinates_bohr": coordinates.tolist(),
            "basis": arguments.basis,
            "basis_metadata": reference.basis_metadata,
            "backend": reference.executed_backend,
            "tensor_cuda_arch": (
                arguments.cuda_arch if arguments.backend == "cuda" else None
            ),
            "controls": dict(operator._controls),
            "precision": reference.precision,
            "initial_guess": "fresh hcore, no caller-provided density",
            "cold_seconds": {
                "native": native_cold,
                "array_api_including_jit": generated_cold,
            },
            "warm": {name: _summary(values) for name, values in samples.items()},
            "array_api_over_native": ratio,
            "observed_5pct_timing_target_met": matching_work and ratio <= 1.05,
            "energy_error_hartree": energy_error,
            "gradient_error_hartree_per_bohr": gradient_error,
            "scf_iterations": iterations,
            "matching_iteration_counts": matching_work,
            "compiled_execution": info,
            "peak_memory_bytes": None,
            "qualification_gaps": [
                "peak host/device memory not measured",
                "native ABI does not expose a complete exact-J/K operator-work inventory here",
            ],
            "gpu_performance_qualified": False,
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        rendered = json.dumps(record, indent=2, allow_nan=False) + "\n"
        if arguments.output is not None:
            arguments.output.write_text(rendered)
        print(rendered, end="")
    finally:
        compiled.close()
        native.clear_cache()


if __name__ == "__main__":
    main()
