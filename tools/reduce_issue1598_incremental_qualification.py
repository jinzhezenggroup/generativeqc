"""Reduce the #1598 incremental Direct-J/K qualification matrix.

Consumes the audited comparator records produced by
`benchmarks/run_issue1598_incremental_qualification.sh` plus the PBE0 cold
re-verification controls, and emits one machine-readable summary with all
numerical gates, iteration-branch matching, work censuses, identities and the
resource-ledger receipt counts. It never substitutes a gate; a missing or
invalid record fails the reduction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

POPULATIONS = ("rhf", "uhf", "moved")
SIZES = ((12, 96), (24, 192), (48, 384), (96, 768))
MOVED_SIZES = ((24, 192), (96, 768))
KS_SAMPLES = (
    ("off", 96, 1),
    ("on", 96, 1),
    ("on", 96, 2),
    ("off", 96, 2),
    ("off", 48, 1),
    ("on", 48, 1),
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("benchmark") != "compare_gpu4pyscf_batch":
        raise ValueError(f"{path} is not an audited direct-HF comparator record")
    return data


def _maximum_abs_difference(first: Any, second: Any) -> float:
    if isinstance(first, list) or isinstance(second, list):
        if not isinstance(first, list) or not isinstance(second, list):
            raise TypeError("result shapes differ")
        if len(first) != len(second):
            raise ValueError("result shapes differ")
        if not first:
            raise ValueError("empty benchmark result")
        return max(
            _maximum_abs_difference(a, b) for a, b in zip(first, second, strict=True)
        )
    if type(first) not in (int, float) or type(second) not in (int, float):
        raise TypeError("benchmark results require numeric values")
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


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


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
        raise ValueError("incremental benchmark is missing Direct-J/K diagnostics")
    typed = [value for value in diagnostics if isinstance(value, dict)]
    if any(not value.get("active") for value in typed):
        raise ValueError("incremental benchmark did not activate Direct-J/K updates")
    full_builds = sum(int(value["anchor_full_builds"]) for value in typed)
    delta_builds = sum(int(value["delta_builds"]) for value in typed)
    full_shells = sum(int(value["full_admitted_shell_quartets"]) for value in typed)
    delta_shells = sum(int(value["delta_admitted_shell_quartets"]) for value in typed)
    full_tiles = sum(int(value["full_admitted_quartet_tiles"]) for value in typed)
    delta_tiles = sum(int(value["delta_admitted_quartet_tiles"]) for value in typed)
    if delta_builds == 0:
        # A population that converges before the first delta update has no
        # incremental opportunity; quartet counters may legitimately be
        # invalid there, and its timings remain valid matched endpoints.
        if full_builds == 0:
            raise ValueError("incremental benchmark lacks measurable full work")
        return {
            "sample_count": len(typed),
            "full_builds": full_builds,
            "delta_builds": 0,
            "full_admitted_shell_quartets_per_build": (
                full_shells / full_builds if full_shells else None
            ),
            "delta_admitted_shell_quartets_per_build": 0.0,
            "shell_quartet_reduction_fraction": 0.0,
            "full_admitted_quartet_tiles_per_build": (
                full_tiles / full_builds if full_tiles else None
            ),
            "delta_admitted_quartet_tiles_per_build": 0.0,
            "quartet_tile_reduction_fraction": 0.0,
            "no_incremental_opportunity": True,
        }
    if any(not value.get("quartet_work_counters_valid") for value in typed):
        raise ValueError("incremental quartet work counters are not complete")
    if full_builds == 0 or full_shells == 0 or full_tiles == 0:
        raise ValueError("incremental benchmark lacks measurable full work")
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
        "no_incremental_opportunity": False,
    }


def _outcome(root: Path, path: Path) -> dict[str, Any]:
    stem = path.name[: -len(".json")]
    return json.loads((path.parent / f"{stem}.outcome").read_text())


def _hf_rows(
    args: argparse.Namespace, root: Path
) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    library_sha: str | None = None
    for population in POPULATIONS:
        size_pairs = MOVED_SIZES if population == "moved" else SIZES
        for atoms, expected_aos in size_pairs:
            loaded: dict[str, dict[int, dict[str, Any]]] = {
                "baseline": {},
                "incremental": {},
            }
            failed_points: list[str] = []
            for name, mode in (
                ("baseline", "baseline"),
                ("incremental", "incremental"),
            ):
                for round in (1, 2):
                    path = root / population / f"direct-{atoms}-{mode}-{round}.json"
                    outcome = _outcome(root, path)
                    if outcome.get("exit_code") != 0:
                        failed_points.append(
                            f"{population}/{name}-{round} "
                            f"(exit {outcome.get('exit_code')}, "
                            f"attempt {outcome.get('attempt')})"
                        )
                        continue
                    record = _load(path)
                    aos = int(record["workload"]["ao_count"])
                    if aos != expected_aos:
                        raise ValueError(
                            f"{population} {name} {atoms}-atom record has {aos} AOs"
                        )
                    if int(record["workload"]["batch_size"]) != 1:
                        raise ValueError("qualification requires batch size one")
                    if not record["gate"].get("passed"):
                        raise ValueError(
                            f"{population} {name} {atoms}-atom record failed its "
                            "independent reference gates"
                        )
                    sha = str(record["native_build"]["library_sha256"])
                    if library_sha is None:
                        library_sha = sha
                    elif sha != library_sha:
                        raise ValueError("matrix did not use one native library")
                    loaded[name][round] = record
            if failed_points:
                failures.append(
                    {
                        "population": population,
                        "atoms": atoms,
                        "ao_count": expected_aos,
                        "failed_points": failed_points,
                    }
                )
            matched_cold: dict[str, list[float]] = {"baseline": [], "incremental": []}
            matched_warm: dict[str, list[float]] = {"baseline": [], "incremental": []}
            for round in (1, 2):
                baseline_record = loaded["baseline"].get(round)
                incremental_record = loaded["incremental"].get(round)
                if baseline_record is None or incremental_record is None:
                    continue
                baseline_cold = _positive_seconds(
                    baseline_record["generativeqc"]["cold_seconds"]
                )
                incremental_cold = _positive_seconds(
                    incremental_record["generativeqc"]["cold_seconds"]
                )
                baseline_warm = _positive_seconds(
                    baseline_record["generativeqc"]["warm_median_seconds"]
                )
                incremental_warm = _positive_seconds(
                    incremental_record["generativeqc"]["warm_median_seconds"]
                )
                matched_cold["baseline"].append(baseline_cold)
                matched_cold["incremental"].append(incremental_cold)
                matched_warm["baseline"].append(baseline_warm)
                matched_warm["incremental"].append(incremental_warm)
                energy_difference = _maximum_abs_difference(
                    baseline_record["generativeqc"]["energies_hartree"],
                    incremental_record["generativeqc"]["energies_hartree"],
                )
                force_difference = _maximum_abs_difference(
                    baseline_record["generativeqc"]["forces_hartree_per_bohr"],
                    incremental_record["generativeqc"]["forces_hartree_per_bohr"],
                )
                baseline_branches = _branches(baseline_record)
                incremental_branches = _branches(incremental_record)
                work = _incremental_work(incremental_record)
                rows.append(
                    {
                        "population": population,
                        "atoms": atoms,
                        "ao_count": expected_aos,
                        "round": round,
                        "baseline_cold_seconds": baseline_cold,
                        "incremental_cold_seconds": incremental_cold,
                        "cold_speedup": baseline_cold / incremental_cold,
                        "baseline_warm_median_seconds": baseline_warm,
                        "incremental_warm_median_seconds": incremental_warm,
                        "warm_speedup": baseline_warm / incremental_warm,
                        "baseline_iteration_branches": baseline_branches,
                        "incremental_iteration_branches": incremental_branches,
                        "iteration_branches_match": (
                            baseline_branches == incremental_branches
                        ),
                        "maximum_energy_difference_hartree": energy_difference,
                        "maximum_force_difference_hartree_per_bohr": force_difference,
                        "numerical_gate_passed": (
                            energy_difference <= args.maximum_energy_difference
                            and force_difference <= args.maximum_force_difference
                        ),
                        "incremental_work": work,
                    }
                )
            if matched_warm["baseline"] and matched_warm["incremental"]:
                rows.append(
                    {
                        "population": population,
                        "atoms": atoms,
                        "ao_count": expected_aos,
                        "round": "all-matched",
                        "baseline_cold_median_seconds": _median(
                            matched_cold["baseline"]
                        ),
                        "incremental_cold_median_seconds": _median(
                            matched_cold["incremental"]
                        ),
                        "cold_speedup": _median(matched_cold["baseline"])
                        / _median(matched_cold["incremental"]),
                        "baseline_warm_median_seconds": _median(
                            matched_warm["baseline"]
                        ),
                        "incremental_warm_median_seconds": _median(
                            matched_warm["incremental"]
                        ),
                        "warm_speedup": _median(matched_warm["baseline"])
                        / _median(matched_warm["incremental"]),
                    }
                )
    return rows, library_sha, failures


def _ks_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    samples: dict[tuple[str, int], list[float]] = {}
    for mode, atoms, round in KS_SAMPLES:
        path = root / "ks" / f"pbe0-{atoms}-{mode}-{round}.json"
        outcome = _outcome(root, path)
        if outcome.get("exit_code") != 0:
            failures.append(
                {
                    "mode": mode,
                    "atoms": atoms,
                    "round": round,
                    "exit_code": outcome.get("exit_code"),
                    "attempt": outcome.get("attempt"),
                }
            )
            continue
        record = json.loads(path.read_text())
        if record.get("schema") != "issue1598.ks-cold-control.v1":
            raise ValueError("KS control record has an unexpected schema")
        if record.get("status") != "measured":
            raise ValueError(f"KS {mode}-{atoms} record is not measured")
        seconds = _positive_seconds(record["complete_seconds"])
        samples.setdefault((mode, atoms), []).append(seconds)
        if not record["acceptance"].get("passed"):
            raise ValueError(f"KS {mode}-{atoms} independent acceptance failed")
        rows.append(
            {
                "mode": mode,
                "atoms": atoms,
                "complete_seconds": seconds,
                "iterations": record["iterations"],
                "fock_builds": record["fock_builds"],
                "incremental_active": bool(
                    record["incremental_direct_jk"]
                    and record["incremental_direct_jk"].get("active")
                ),
            }
        )
    for (mode, atoms), values in sorted(samples.items()):
        rows.append(
            {
                "mode": mode,
                "atoms": atoms,
                "median_complete_seconds": _median(values),
                "sample_count": len(values),
            }
        )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-energy-difference", type=float, default=1.0e-8)
    parser.add_argument("--maximum-force-difference", type=float, default=1.0e-7)
    args = parser.parse_args()

    hf_rows, library_sha, hf_failures = _hf_rows(args, args.input)
    ks_rows, ks_failures = _ks_rows(args.input)
    ledger_path = args.input / "ledger" / "nvidia-smi.tsv"
    ledger_lines = sum(1 for _ in ledger_path.open()) if ledger_path.exists() else 0

    per_round_rows = [row for row in hf_rows if isinstance(row.get("round"), int)]
    matched_rows = [
        row for row in hf_rows if row.get("round") == "all-matched"
    ]
    payload = {
        "schema_version": 1,
        "issue": 1598,
        "workload": (
            "nested-water def2-SVP spherical RHF/UHF-cation/moved populations, "
            "96/192/384/768 AOs, batch one, cold plus three fixed-density warm "
            "energy+force replays, interleaved A-B-B-A per size"
        ),
        "library_sha256": library_sha,
        "thresholds": {
            "maximum_energy_difference_hartree": args.maximum_energy_difference,
            "maximum_force_difference_hartree_per_bohr": args.maximum_force_difference,
        },
        "hf_rows": hf_rows,
        "hf_failures": hf_failures,
        "ks_rows": ks_rows,
        "ks_failures": ks_failures,
        "ledger_receipt_lines": ledger_lines,
        "any_point_failed": bool(hf_failures or ks_failures),
        "all_numerical_gates_passed": bool(per_round_rows)
        and all(row["numerical_gate_passed"] for row in per_round_rows),
        "all_iteration_branches_match": all(
            row["iteration_branches_match"] for row in per_round_rows
        ),
        "any_warm_endpoint_win": any(row["warm_speedup"] > 1.0 for row in matched_rows),
        "any_cold_endpoint_win": any(row["cold_speedup"] > 1.0 for row in matched_rows),
        "best_warm_speedup": (
            max(row["warm_speedup"] for row in matched_rows) if matched_rows else None
        ),
        "best_cold_speedup": (
            max(row["cold_speedup"] for row in matched_rows) if matched_rows else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(
        "pop    AOs  round  warm_base_ms  warm_inc_ms  speedup  cold_speedup  "
        "tile_reduce  shell_reduce  branches  numerical"
    )
    for row in hf_rows:
        if row.get("round") == "all-matched":
            print(
                f"{row['population']:<5}  {row['ao_count']:>4}  {row['round']!s:>10}  "
                f"{row['baseline_warm_median_seconds'] * 1e3:>11.3f}  "
                f"{row['incremental_warm_median_seconds'] * 1e3:>11.3f}  "
                f"{row['warm_speedup']:>7.3f}  {row['cold_speedup']:>12.3f}"
            )
            continue
        print(
            f"{row['population']:<5}  {row['ao_count']:>4}  {row['round']:>5}  "
            f"{row['baseline_warm_median_seconds'] * 1e3:>11.3f}  "
            f"{row['incremental_warm_median_seconds'] * 1e3:>11.3f}  "
            f"{row['warm_speedup']:>7.3f}  "
            f"{row['cold_speedup']:>12.3f}  "
            f"{row['incremental_work']['quartet_tile_reduction_fraction']:>11.3%}  "
            f"{row['incremental_work']['shell_quartet_reduction_fraction']:>12.3%}  "
            f"{row['iteration_branches_match']!s:>8}  "
            f"{row['numerical_gate_passed']!s:>9}"
        )
    for row in ks_rows:
        if "median_complete_seconds" in row:
            print(
                f"ks {row['atoms']:>4}-atom {row['mode']:<3} median_cold_s = "
                f"{row['median_complete_seconds']:.3f} "
                f"(n={row['sample_count']})"
            )


if __name__ == "__main__":
    main()
