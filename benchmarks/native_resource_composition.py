"""Time complete budgeted CPU endpoints for an explicitly selected Python tree.

Run with the same compiled library and either the baseline or candidate Python
source on PYTHONPATH. Measurements include calculator construction, preparation,
cold/warm/moved execution and prepared-owner close; imports and result formatting
are outside the timed scope. This is opt-in evidence, not a timing assertion in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import generativeqc
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget

from benchmarks._retention import raw_output_path


def endpoint(
    method: str, systems: list, properties: tuple[str, ...]
) -> tuple[float, dict]:
    """Run one actual lifecycle and retain observed numerical/work signatures."""
    moved = [[tuple(position) for _, position in system] for system in systems]
    moved[0][0] = (moved[0][0][0] + 0.013, *moved[0][0][1:])
    options = {
        "method": method,
        "resource_budget": ResourceBudget(host_bytes=1 << 30),
        "max_iterations": 150,
    }
    if method != "rhf":
        options["ks_options"] = KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        )
    started = time.perf_counter()
    calculator = Calculator(**options)
    with calculator.prepare_batch(systems) as prepared:
        cold = prepared.execute(properties=properties, strict=True)
        warm = prepared.execute(properties=properties, strict=True)
        replay = prepared.execute(moved, properties=properties, strict=True)
        plan = prepared.resource_plan
    del prepared, calculator
    elapsed = time.perf_counter() - started
    return elapsed, {
        "plan_identity": plan.identity,
        "peak_bytes": plan.peak_bytes,
        "phases": [
            [
                {
                    "energy": item.energy,
                    "forces": None if item.forces is None else item.forces.tolist(),
                    "status": item.status,
                    "converged": item.converged,
                    "iterations": item.iterations,
                    "energy_change": item.energy_change,
                    "density_rms": item.density_rms,
                    "physical_residual_rms": item.physical_residual_rms,
                    "fock_builds": item.fock_builds,
                    "warm_start_used": item.warm_start_used,
                    "warm_start_fallback": item.warm_start_fallback,
                }
                for item in result.items
            ]
            for result in (cold, warm, replay)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    if arguments.samples < 1 or arguments.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        parser.error("set GENERATIVEQC_LIBRARY to the explicit qualification library")
    h2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    water = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43, 1.11)),
        ("H", (0.0, 1.43, 1.11)),
    ]
    cases = (
        ("rhf-h2-energy-force", "rhf", [h2], ("energy", "forces")),
        ("pbe-h2-energy", "pbe-rks", [h2], ("energy",)),
        ("pbe-h2-energy-force", "pbe-rks", [h2], ("energy", "forces")),
        ("pbe-ragged-energy", "pbe-rks", [h2, water], ("energy",)),
    )
    results = []
    for name, method, systems, properties in cases:
        for _ in range(arguments.warmup):
            endpoint(method, systems, properties)
        durations = []
        observations = []
        reference = None
        for _ in range(arguments.samples):
            elapsed, signature = endpoint(method, systems, properties)
            if reference is None:
                reference = signature
            elif signature != reference:
                raise RuntimeError(
                    f"{name}: numerical/work signature changed between repetitions"
                )
            durations.append(elapsed)
            observations.append({"seconds": elapsed, "signature": signature})
        results.append(
            {
                "name": name,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "signature": reference,
                "observations": observations,
                "iteration_branch_histogram": {
                    json.dumps(
                        [
                            [item["iterations"] for item in phase]
                            for phase in reference["phases"]
                        ]
                    ): arguments.samples
                },
            }
        )
    with Path(os.environ["GENERATIVEQC_LIBRARY"]).open("rb") as library_file:
        library_sha256 = hashlib.file_digest(library_file, "sha256").hexdigest()
    record = {
        "schema": "generativeqc.native-resource-composition-endpoint.v1",
        "python_source_sha": arguments.source_sha,
        "python_package": str(Path(generativeqc.__file__).resolve()),
        "python_version": platform.python_version(),
        "native_library": os.environ.get("GENERATIVEQC_LIBRARY"),
        "native_library_sha256": library_sha256,
        "samples": arguments.samples,
        "warmup": arguments.warmup,
        "scope": "calculator+prepare+cold+warm+moved+prepared-close; import/formatting excluded",
        "warm_start_policy": "new prepared owner per repeat; warm from cold result, moved from warm result; every observed signature must match",
        "cases": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({case["name"]: case["median_seconds"] for case in results}, indent=2)
    )


if __name__ == "__main__":
    main()
