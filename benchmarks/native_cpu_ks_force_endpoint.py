"""Paired complete KS force endpoints, including preparation and destruction.

The default comparison retains the Python stationary compatibility owner using
the same scientific library. An optional baseline library instead compares
native RKS endpoints with a previous binary. Neither comparison changes the
functional, spin, grid, convergence, integral provider or output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import typing
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc.batch import PreparedBatch
from generativeqc_compiler.common.resources import CpuResourceObservation

ATOMS = [("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5))]
MOVED = [[(0.11, -0.1, 0.02), (0.1, 0.22, 1.71), (1.61, -0.21, -0.5)]]


def endpoint(
    method: str, fitting: str, native: bool, library: Path, trace: Path | None = None
) -> dict[str, typing.Any]:
    """Measure one complete cold/warm/moved lifecycle without diagnostic exports."""
    environment = {"GENERATIVEQC_LIBRARY": str(library)}
    if trace is not None:
        trace.unlink(missing_ok=True)
        environment["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(trace)
    with patch.dict(os.environ, environment):
        observer = CpuResourceObservation(
            _native.load_library(device="cpu"), cpu_workers=1
        )
        compatibility = patch.object(
            PreparedBatch, "_native_dft_force_eligible", return_value=False
        )
        with (
            nullcontext() if native else compatibility,
            observer if trace else nullcontext(),
        ):
            started = time.perf_counter()
            calc = Calculator(
                method,
                "sto-3g",
                device="cpu",
                density_fitting=fitting,
                ks_options=KsOptions(grid=GridSpec(24, 8, 16)),
                max_iterations=200,
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-14,
            )
            uks = method.endswith("-uks")
            observations = []
            with calc.prepare_batch(
                [ATOMS], charges=[int(uks)], multiplicities=[1 + int(uks)]
            ) as batch:
                if native and not batch._native_dft_force_eligible():
                    raise RuntimeError(
                        f"{method}/{fitting} lacks a prepared native force owner"
                    )
                prepared_seconds = time.perf_counter() - started
                for stage, coordinates in [
                    ("cold", None),
                    ("warm", None),
                    ("moved", MOVED),
                ]:
                    stage_started = time.perf_counter()
                    item = batch.execute(
                        coordinates, properties=("energy", "forces"), strict=True
                    ).items[0]
                    observations.append(
                        {
                            "stage": stage,
                            "seconds": time.perf_counter() - stage_started,
                            "energy": item.energy,
                            "forces": item.forces.tolist(),
                            "iterations": item.iterations,
                            "fock_builds": item.fock_builds,
                            "warm_start_used": item.warm_start_used,
                            "force_coordinates": item.forces.size,
                        }
                    )
            del batch
            calc.clear_cache()
            del calc
            elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "construction_and_preparation_seconds": prepared_seconds,
        "observations": observations,
        "numeric_capacity_observation": observer.to_dict() if trace else None,
    }


def main() -> None:
    """Emit raw paired samples, semantic counts, provenance and separate traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spin", choices=("rks", "uks"), default="uks")
    parser.add_argument("--fitting", choices=("none", "cpu", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-library", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.baseline_library and arguments.spin != "rks":
        parser.error("the prior native baseline comparison is RKS-only")
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    baseline = (
        arguments.baseline_library.resolve() if arguments.baseline_library else library
    )
    if not library.is_file() or not baseline.is_file():
        parser.error("scientific and comparison libraries must exist")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for fitting in (
        ("none", "cpu") if arguments.fitting == "both" else (arguments.fitting,)
    ):
        for family in ("pbe", "pbe0"):
            method = f"{family}-{arguments.spin}"
            comparison = "baseline" if arguments.baseline_library else "python"
            modes = {
                "native": (True, library),
                comparison: (bool(arguments.baseline_library), baseline),
            }
            for native, binary in modes.values():
                endpoint(method, fitting, native, binary)
            pairs = []
            for repeat in range(arguments.repeats):
                order = list(modes) if repeat % 2 == 0 else list(reversed(modes))
                pair = {mode: endpoint(method, fitting, *modes[mode]) for mode in order}
                for actual, reference in zip(
                    pair["native"]["observations"],
                    pair[comparison]["observations"],
                    strict=True,
                ):
                    for key in (
                        "stage",
                        "iterations",
                        "fock_builds",
                        "warm_start_used",
                        "force_coordinates",
                    ):
                        assert actual[key] == reference[key], (method, fitting, key)
                    np.testing.assert_allclose(
                        actual["energy"], reference["energy"], atol=2e-9, rtol=0
                    )
                    np.testing.assert_allclose(
                        actual["forces"], reference["forces"], atol=2e-7, rtol=0
                    )
                pairs.append(pair)
            trace = arguments.output.with_name(
                f"{arguments.output.stem}-{method}-{fitting}-work.jsonl"
            )
            diagnostic = endpoint(method, fitting, True, library, trace)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            ratio = statistics.median(
                pair["native"]["seconds"] / pair[comparison]["seconds"]
                for pair in pairs
            )
            results.append(
                {
                    "method": method,
                    "fitting": fitting,
                    "basis": "sto-3g",
                    "nbf": 7,
                    "grid": [24, 8, 16],
                    "spin_blocks": 2 if arguments.spin == "uks" else 1,
                    "pairs": pairs,
                    "comparison": comparison,
                    "paired_median_ratio": ratio,
                    "diagnostic": diagnostic,
                    "actual_progress_trace": records,
                    "memory_scope": "named numeric capacities, not allocator/RSS/full endpoint peak",
                }
            )
            print(method, fitting, comparison, ratio, flush=True)
    arguments.output.write_text(
        json.dumps(
            {
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "comparison_library_sha256": hashlib.sha256(
                    baseline.read_bytes()
                ).hexdigest(),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
