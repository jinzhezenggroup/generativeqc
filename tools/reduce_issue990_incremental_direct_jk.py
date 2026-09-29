"""Reduce matched #990 Direct-J/K baseline/incremental endpoint records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SIZES = ((12, 96), (24, 192), (48, 384), (96, 768))


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("benchmark") != "compare_gpu4pyscf_batch":
        raise ValueError(f"{path} is not an audited direct-HF comparator record")
    return data


def _maximum_abs_difference(first: Any, second: Any) -> float:
    """Return the maximum finite difference; missing results are not evidence."""

    if isinstance(first, list) or isinstance(second, list):
        if not isinstance(first, list) or not isinstance(second, list):
            raise ValueError("baseline/incremental result shapes differ")
        if len(first) != len(second):
            raise ValueError("baseline/incremental result shapes differ")
        if not first:
            raise ValueError("empty benchmark result")
        return max(
            _maximum_abs_difference(a, b) for a, b in zip(first, second, strict=True)
        )
    if type(first) not in (int, float) or type(second) not in (int, float):
        raise ValueError("benchmark results require numeric values")
    a = float(first)
    b = float(second)
    difference = abs(a - b)
    if not all(math.isfinite(value) for value in (a, b, difference)):
        raise ValueError("non-finite benchmark result")
    return difference


def _positive_seconds(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("benchmark timings must be finite and positive")
    return float(value)


def _branches(record: dict[str, Any]) -> list[tuple[int, ...]]:
    return [
        tuple(int(item["iterations"]) for item in sample["convergence"])
        for sample in record["generativeqc"]["warm_samples"]
    ]


def _incremental_work(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = [
        sample["convergence"][0].get("incremental_direct_jk")
        for sample in record["generativeqc"]["warm_samples"]
    ]
    if not diagnostics or any(type(value) is not dict for value in diagnostics):
        raise ValueError("incremental benchmark is missing Direct-J/K work diagnostics")
    typed = [value for value in diagnostics if isinstance(value, dict)]
    if any(not value.get("active") for value in typed):
        raise ValueError("incremental benchmark did not activate Direct-J/K updates")
    if any(not value.get("quartet_work_counters_valid") for value in typed):
        raise ValueError("incremental quartet work counters are not complete")
    full_builds = sum(int(value["anchor_full_builds"]) for value in typed)
    delta_builds = sum(int(value["delta_builds"]) for value in typed)
    full_shells = sum(int(value["full_admitted_shell_quartets"]) for value in typed)
    delta_shells = sum(int(value["delta_admitted_shell_quartets"]) for value in typed)
    full_tiles = sum(int(value["full_admitted_quartet_tiles"]) for value in typed)
    delta_tiles = sum(int(value["delta_admitted_quartet_tiles"]) for value in typed)
    if full_builds == 0 or delta_builds == 0 or full_shells == 0 or full_tiles == 0:
        raise ValueError(
            "incremental benchmark lacks measurable full/delta Direct-J/K work"
        )
    full_shells_per_build = full_shells / full_builds
    delta_shells_per_build = delta_shells / delta_builds
    full_tiles_per_build = full_tiles / full_builds
    delta_tiles_per_build = delta_tiles / delta_builds
    return {
        "sample_count": len(typed),
        "full_builds": full_builds,
        "delta_builds": delta_builds,
        "full_admitted_shell_quartets_per_build": full_shells_per_build,
        "delta_admitted_shell_quartets_per_build": delta_shells_per_build,
        "shell_quartet_reduction_fraction": 1.0
        - delta_shells_per_build / full_shells_per_build,
        "full_admitted_quartet_tiles_per_build": full_tiles_per_build,
        "delta_admitted_quartet_tiles_per_build": delta_tiles_per_build,
        "quartet_tile_reduction_fraction": 1.0
        - delta_tiles_per_build / full_tiles_per_build,
        "samples": typed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-energy-difference", type=float, default=1.0e-8)
    parser.add_argument("--maximum-force-difference", type=float, default=1.0e-7)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    for name in ("maximum_energy_difference", "maximum_force_difference"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    _positive_seconds(args.minimum_speedup)

    rows: list[dict[str, Any]] = []
    library_sha: str | None = None
    for atoms, expected_aos in SIZES:
        baseline = _load(args.input / "baseline" / f"direct-{atoms}.json")
        incremental = _load(args.input / "incremental" / f"direct-{atoms}.json")
        for name, record in (("baseline", baseline), ("incremental", incremental)):
            aos = int(record["workload"]["ao_count"])
            if aos != expected_aos:
                raise ValueError(
                    f"{name} {atoms}-atom record has {aos} AOs, expected {expected_aos}"
                )
            if int(record["workload"]["batch_size"]) != 1:
                raise ValueError("issue #990 qualification requires batch size one")
            sha = str(record["native_build"]["library_sha256"])
            if library_sha is None:
                library_sha = sha
            elif sha != library_sha:
                raise ValueError(
                    "baseline/incremental matrix did not use one native library"
                )

        baseline_seconds = _positive_seconds(
            baseline["generativeqc"]["warm_median_seconds"]
        )
        incremental_seconds = _positive_seconds(
            incremental["generativeqc"]["warm_median_seconds"]
        )
        speedup = baseline_seconds / incremental_seconds
        if not math.isfinite(speedup):
            raise ValueError("non-finite benchmark speedup")
        energy_difference = _maximum_abs_difference(
            baseline["generativeqc"]["energies_hartree"],
            incremental["generativeqc"]["energies_hartree"],
        )
        force_difference = _maximum_abs_difference(
            baseline["generativeqc"]["forces_hartree_per_bohr"],
            incremental["generativeqc"]["forces_hartree_per_bohr"],
        )
        baseline_branches = _branches(baseline)
        incremental_branches = _branches(incremental)
        work = _incremental_work(incremental)
        rows.append(
            {
                "atoms": atoms,
                "ao_count": expected_aos,
                "baseline_warm_median_seconds": baseline_seconds,
                "incremental_warm_median_seconds": incremental_seconds,
                "speedup": speedup,
                "baseline_iteration_branches": baseline_branches,
                "incremental_iteration_branches": incremental_branches,
                "iteration_branches_match": baseline_branches == incremental_branches,
                "maximum_energy_difference_hartree": energy_difference,
                "maximum_force_difference_hartree_per_bohr": force_difference,
                "numerical_gate_passed": (
                    energy_difference <= args.maximum_energy_difference
                    and force_difference <= args.maximum_force_difference
                ),
                "speed_gate_passed": speedup >= args.minimum_speedup,
                "incremental_work": work,
            }
        )

    payload = {
        "schema_version": 1,
        "issue": 990,
        "workload": "README direct-HF nested water, spherical def2-SVP, batch one, warm energy+forces",
        "library_sha256": library_sha,
        "thresholds": {
            "maximum_energy_difference_hartree": args.maximum_energy_difference,
            "maximum_force_difference_hartree_per_bohr": args.maximum_force_difference,
            "minimum_speedup": args.minimum_speedup,
        },
        "rows": rows,
        "all_numerical_gates_passed": all(row["numerical_gate_passed"] for row in rows),
        "all_iteration_branches_match": all(
            row["iteration_branches_match"] for row in rows
        ),
        "all_speed_gates_passed": all(row["speed_gate_passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(
        "AOs  baseline_ms  incremental_ms  speedup  tile_reduce  shell_reduce  "
        "branch_match  numerical_gate"
    )
    for row in rows:
        print(
            f"{row['ao_count']:>3}  "
            f"{row['baseline_warm_median_seconds'] * 1e3:>11.3f}  "
            f"{row['incremental_warm_median_seconds'] * 1e3:>14.3f}  "
            f"{row['speedup']:>7.3f}  "
            f"{row['incremental_work']['quartet_tile_reduction_fraction']:>11.3%}  "
            f"{row['incremental_work']['shell_quartet_reduction_fraction']:>12.3%}  "
            f"{row['iteration_branches_match']!s:>12}  "
            f"{row['numerical_gate_passed']!s:>14}"
        )


if __name__ == "__main__":
    main()
