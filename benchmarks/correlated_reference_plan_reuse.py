"""Whole-endpoint CUDA correlated replay timing for issue #1856.

Warm-density reuse is deliberately disabled.  The stationary and changed
samples therefore differ from the cold sample only by retention of compatible
prepared executable state; every sample still performs and validates a fresh
RHF solve before correlation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from generativeqc import Calculator

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.43233673, 1.10715266)),
    ("H", (0.0, -1.43233673, 1.10715266)),
]


def _sample(
    batch: Any, *, coordinates: Any = None, properties: tuple[str, ...]
) -> dict[str, Any]:
    started = time.perf_counter()
    result = batch.execute(
        coordinates=[coordinates] if coordinates is not None else None,
        properties=properties,
        strict=True,
    ).items[0]
    elapsed = time.perf_counter() - started
    diagnostic = result.correlation
    if diagnostic is None:
        raise RuntimeError("correlated endpoint did not publish diagnostics")
    if result.warm_start_used:
        raise RuntimeError("benchmark must not use a retained HF density")
    return {
        "seconds": elapsed,
        "energy": result.energy,
        "plan_reused": diagnostic.reference_execution_plan_reused,
        "plan_owned_device_bytes": diagnostic.reference_execution_plan_owned_device_bytes,
        "numeric_capacity_bytes": diagnostic.numeric_capacity_bytes,
    }


def _run_method(
    method: str, repeats: int, properties: tuple[str, ...], budget: int
) -> dict[str, Any]:
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        correlation_memory_budget_bytes=budget,
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ccsd_max_iterations=150,
        ccsd_energy_tolerance=1e-13,
        ccsd_residual_tolerance=1e-11,
    )
    moved = [list(xyz) for _, xyz in WATER]
    moved[1][2] += 0.01

    records: list[dict[str, Any]] = []
    for _ in range(repeats):
        with calculator.prepare_batch([WATER], warm_start=False) as batch:
            cold = _sample(batch, properties=properties)
            stationary = _sample(batch, properties=properties)
            changed = _sample(batch, coordinates=moved, properties=properties)
        if cold["plan_reused"]:
            raise RuntimeError(
                "fresh correlated owner unexpectedly reused a CUDA RHF plan"
            )
        for name, row in (("stationary", stationary), ("changed_geometry", changed)):
            if not row["plan_reused"]:
                raise RuntimeError(f"{name} did not reuse the retained CUDA RHF plan")
            if row["plan_owned_device_bytes"] <= 0:
                raise RuntimeError(f"{name} retained-plan byte diagnostic is empty")
        if not all(math.isfinite(row["energy"]) for row in (cold, stationary, changed)):
            raise RuntimeError("nonfinite correlated energy")
        records.append(
            {"cold": cold, "stationary": stationary, "changed_geometry": changed}
        )

    def median(phase: str) -> float:
        return statistics.median(row[phase]["seconds"] for row in records)

    cold = median("cold")
    stationary = median("stationary")
    changed = median("changed_geometry")
    return {
        "method": method,
        "properties": properties,
        "repeats": repeats,
        "cold_median_seconds": cold,
        "stationary_replay_median_seconds": stationary,
        "changed_geometry_median_seconds": changed,
        "cold_over_stationary": cold / stationary,
        "cold_over_changed_geometry": cold / changed,
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["mp2", "rccsd", "ccsd(t)"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--forces", action="store_true")
    parser.add_argument("--budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    properties = ("energy", "forces") if args.forces else ("energy",)
    payload = {
        "issue": 1856,
        "device": "cuda",
        "warm_density_reuse": False,
        "timing_scope": "complete prepared correlated execute call",
        "methods": [
            _run_method(method, args.repeats, properties, args.budget_bytes)
            for method in args.methods
        ],
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
