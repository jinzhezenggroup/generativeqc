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
    """Return the maximum finite scalar difference in equally nested lists."""

    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            raise ValueError("baseline/incremental result shapes differ")
        return max(
            (_maximum_abs_difference(a, b) for a, b in zip(first, second, strict=True)),
            default=0.0,
        )
    if first is None and second is None:
        return 0.0
    a = float(first)
    b = float(second)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("non-finite benchmark result")
    return abs(a - b)


def _branches(record: dict[str, Any]) -> list[tuple[int, ...]]:
    return [
        tuple(int(item["iterations"]) for item in sample["convergence"])
        for sample in record["generativeqc"]["warm_samples"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-energy-difference", type=float, default=1.0e-8)
    parser.add_argument("--maximum-force-difference", type=float, default=1.0e-7)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    library_sha: str | None = None
    for atoms, expected_aos in SIZES:
        baseline = _load(args.input / "baseline" / f"direct-{atoms}.json")
        incremental = _load(args.input / "incremental" / f"direct-{atoms}.json")
        for name, record in (("baseline", baseline), ("incremental", incremental)):
            aos = int(record["workload"]["ao_count"])
            if aos != expected_aos:
                raise ValueError(f"{name} {atoms}-atom record has {aos} AOs, expected {expected_aos}")
            if int(record["workload"]["batch_size"]) != 1:
                raise ValueError("issue #990 qualification requires batch size one")
            sha = str(record["native_build"]["library_sha256"])
            if library_sha is None:
                library_sha = sha
            elif sha != library_sha:
                raise ValueError("baseline/incremental matrix did not use one native library")

        baseline_seconds = float(baseline["generativeqc"]["warm_median_seconds"])
        incremental_seconds = float(incremental["generativeqc"]["warm_median_seconds"])
        speedup = baseline_seconds / incremental_seconds
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
        "all_iteration_branches_match": all(row["iteration_branches_match"] for row in rows),
        "all_speed_gates_passed": all(row["speed_gate_passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("AOs  baseline_ms  incremental_ms  speedup  branch_match  numerical_gate")
    for row in rows:
        print(
            f"{row['ao_count']:>3}  "
            f"{row['baseline_warm_median_seconds'] * 1e3:>11.3f}  "
            f"{row['incremental_warm_median_seconds'] * 1e3:>14.3f}  "
            f"{row['speedup']:>7.3f}  "
            f"{str(row['iteration_branches_match']):>12}  "
            f"{str(row['numerical_gate_passed']):>14}"
        )


if __name__ == "__main__":
    main()
