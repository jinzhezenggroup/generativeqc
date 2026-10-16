"""Compare complete CPU basis-construction/prepared lifecycles across source/library pairs.

Run this same harness with each checkout's PYTHONPATH and explicit native library.
Raw receipts retain all numerical/work signatures, not merely median timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def endpoint(
    options: dict[str, Any],
    systems: list[Any],
    charges: list[int],
    multiplicities: list[int],
) -> tuple[float, dict[str, Any]]:
    """Time model/resource admission through cold, warm, moved execution and close."""
    from generativeqc import Calculator, ResourceBudget

    moved = [
        [
            (xyz[0], xyz[1], xyz[2] + 0.02 * (index + 1))
            for index, (_, xyz) in enumerate(atoms)
        ]
        for atoms in systems
    ]
    started = time.perf_counter()
    calculator = Calculator(**options)
    plan = calculator.estimate_resources(
        systems,
        charges=charges,
        multiplicities=multiplicities,
        properties=("energy", "forces"),
    ).require_feasible()
    calculator = Calculator(
        **options, resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"])
    )
    with calculator.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities
    ) as prepared:
        cold = prepared.execute(properties=("energy", "forces"), strict=True)
        warm = prepared.execute(properties=("energy", "forces"), strict=True)
        replay = prepared.execute(moved, properties=("energy", "forces"), strict=True)
        admitted = prepared.resource_plan
    del prepared, calculator
    elapsed = time.perf_counter() - started
    return elapsed, {
        "plan_identity": admitted.identity,
        "peak_bytes": admitted.peak_bytes,
        "phases": [
            [
                {
                    "energy": item.energy,
                    "forces": item.forces.tolist(),
                    "status": item.status,
                    "converged": item.converged,
                    "iterations": item.iterations,
                    "energy_change": item.energy_change,
                    "density_rms": item.density_rms,
                    "physical_residual_rms": item.physical_residual_rms,
                    "fock_builds": item.fock_builds,
                    "warm_start_used": item.warm_start_used,
                    "warm_start_fallback": item.warm_start_fallback,
                    "executed_backend": item.executed_backend,
                }
                for item in result.items
            ]
            for result in (cold, warm, replay)
        ],
    }


def main() -> None:
    """Validate retention before imports/science, then retain every endpoint sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    if arguments.samples < 1 or arguments.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if arguments.output.exists():
        parser.error("refusing to overwrite an existing qualification receipt")
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        parser.error("set GENERATIVEQC_LIBRARY to the explicit qualification library")

    import generativeqc
    from generativeqc import KsOptions, import_bse
    from generativeqc_compiler.dft import GridSpec

    root = Path(__file__).resolve().parents[1]
    source = import_bse(
        root / "tests/data/external_basis/sto-3g-ho.json",
        source="pinned independent BSE input",
        source_version="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        license="BSD-3-Clause",
    )
    hydrogen = source.by_element[1]
    shell = hydrogen.shells[0]
    general = replace(
        source,
        elements=(
            replace(
                hydrogen,
                shells=(
                    replace(
                        shell, coefficients=(*shell.coefficients, ("0", "0.25", "0.75"))
                    ),
                ),
            ),
            *[element for element in source.elements if element.atomic_number != 1],
        ),
    )
    h2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    water = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, -1.43, 1.11)),
        ("H", (0.0, 1.43, 1.11)),
    ]
    cases = (
        ("bundled-rhf-h2", "rhf", "sto-3g", [h2], [0], [1]),
        ("general-rhf-h2", "rhf", general, [h2], [0], [1]),
        (
            "general-spherical-rhf-h2",
            "rhf",
            replace(general, representation="spherical"),
            [h2],
            [0],
            [1],
        ),
        ("general-uhf-ragged", "uhf", general, [h2, water], [1, 0], [2, 1]),
        ("general-pbe-h2", "pbe-rks", general, [h2], [0], [1]),
    )
    results = []
    for name, method, basis, systems, charges, multiplicities in cases:
        options = {"method": method, "basis": basis, "device": "cpu"}
        if method == "pbe-rks":
            options["ks_options"] = KsOptions(
                grid=GridSpec(radial_points=6, angular_polar=4, angular_azimuth=8)
            )
        for _ in range(arguments.warmup):
            endpoint(options, systems, charges, multiplicities)
        observations = []
        reference = None
        for _ in range(arguments.samples):
            elapsed, signature = endpoint(options, systems, charges, multiplicities)
            if reference is None:
                reference = signature
            elif signature != reference:
                raise RuntimeError(
                    f"{name}: numerical/work signature changed across samples"
                )
            observations.append({"seconds": elapsed, "signature": signature})
        results.append(
            {
                "name": name,
                "median_seconds": statistics.median(
                    item["seconds"] for item in observations
                ),
                "signature": reference,
                "observations": observations,
            }
        )
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    with library.open("rb") as stream:
        library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "schema": "generativeqc.native-basis-template-endpoint.v1",
        "python_source_sha": arguments.source_sha,
        "python_package": str(Path(generativeqc.__file__).resolve()),
        "python_version": platform.python_version(),
        "native_library": str(library),
        "native_library_sha256": library_sha256,
        "samples": arguments.samples,
        "warmup": arguments.warmup,
        "scope": "calculator+resource-query+budgeted-calculator+prepare+cold+warm+moved+close",
        "excluded": "imports, custom file parsing and signature formatting",
        "cases": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({case["name"]: case["median_seconds"] for case in results}, indent=2)
    )


if __name__ == "__main__":
    main()
