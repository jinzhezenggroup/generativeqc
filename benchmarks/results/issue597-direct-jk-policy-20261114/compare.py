"""Qualify complete Direct-HF endpoint populations against independent fixtures."""

import argparse
import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# This retained campaign declares two fresh processes and three owners per
# process (provenance.json). A file is one process, named <variant>-<index>.jsonl.
EXPECTED_SAMPLES = {
    (process, repetition) for process in range(2) for repetition in range(3)
}


def population(pattern: str, errors: list[str]) -> dict[tuple, list[dict[str, Any]]]:
    """Retain every observed sample, including process/owner cold calls."""
    result = {}
    process_ids = []
    for filename in sorted(glob.glob(pattern)):
        try:
            process = int(Path(filename).stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            errors.append(f"invalid process filename: {filename}")
            continue
        process_ids.append(process)
        for line in Path(filename).read_text().splitlines():
            row = json.loads(line)
            row["process"] = process
            key = (row["spin"], row["nbf"], row["forces"], row["batch"], row["phase"])
            result.setdefault(key, []).append(row)
    if Counter(process_ids) != Counter({0: 1, 1: 1}):
        errors.append(f"incomplete or duplicate process population: {pattern}")
    for key, rows in result.items():
        identities = Counter((row["process"], row["repetition"]) for row in rows)
        if identities != Counter(dict.fromkeys(EXPECTED_SAMPLES, 1)):
            errors.append(f"incomplete or duplicate owner population: {pattern}: {key}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors = []
    baseline = population(args.baseline, errors)
    candidate = population(args.candidate, errors)
    references = json.loads(Path(args.reference).read_text())["fixtures"]
    lookup = {
        (row["unrestricted"], row["nbf"], row["moved"]): row for row in references
    }
    expected = {
        (spin, nbf, forces, batch, phase)
        for spin in ("RHF", "UHF")
        for nbf in (2, 18)
        for forces in (0, 1)
        for batch in (1, 2)
        for phase in ("cold", "warm", "moved-warm")
    }
    groups = []
    if set(baseline) != expected or set(candidate) != expected:
        errors.append("incomplete or unexpected endpoint populations")
    for key in sorted(expected & baseline.keys() & candidate.keys()):
        before, after = baseline[key], candidate[key]
        reference = lookup[(key[0] == "UHF", key[1], key[4] == "moved-warm")]
        expected_forces = [value for atom in reference["forces"] for value in atom]
        energy_error = force_error = 0.0
        for row in before + after:
            if not math.isfinite(row["endpoint_ms"]) or row["endpoint_ms"] <= 0:
                errors.append(f"invalid endpoint timing: {key}")
            if not math.isfinite(row["energy"]):
                errors.append(f"nonfinite energy: {key}")
            energy_error = max(energy_error, abs(row["energy"] - reference["energy"]))
            if key[2]:
                if len(row["force_values"]) != len(expected_forces):
                    errors.append(f"wrong force shape: {key}")
                for observed, expected_force in zip(
                    row["force_values"], expected_forces, strict=False
                ):
                    if not math.isfinite(observed):
                        errors.append(f"nonfinite force: {key}")
                    force_error = max(force_error, abs(observed - expected_force))
            if row["iterations"] <= 0 or row["fock_builds"] <= 0:
                errors.append(f"missing semantic work: {key}")
        if energy_error > 1e-8 or force_error > 1e-7:
            errors.append(f"independent scientific gate failed: {key}")
        before_ms = statistics.median(row["endpoint_ms"] for row in before)
        after_ms = statistics.median(row["endpoint_ms"] for row in after)
        regression = 100 * (after_ms / before_ms - 1)
        work = lambda rows: Counter(
            (row["process"], row["repetition"], row["iterations"], row["fock_builds"])
            for row in rows
        )
        if work(before) != work(after):
            errors.append(f"semantic work changed: {key}")
        if len(before) != len(after):
            errors.append(f"unmatched sample counts: {key}")
        if regression > 5:
            errors.append(
                f"median endpoint regression exceeds 5%: {key}: {regression:.3f}%"
            )
        before_peak = max(row["tracked_numeric_peak_bytes"] for row in before)
        after_peak = max(row["tracked_numeric_peak_bytes"] for row in after)
        if after_peak > before_peak:
            errors.append(f"numeric peak grew: {key}: {before_peak} -> {after_peak}")
        routes = lambda rows: sorted(
            {(row["matrix_provider"], row["persistent_eri"]) for row in rows}
        )
        if routes(before) != routes(after):
            errors.append(f"default route changed: {key}")
        groups.append(
            {
                "key": key,
                "baseline_samples": len(before),
                "candidate_samples": len(after),
                "baseline_median_ms": before_ms,
                "candidate_median_ms": after_ms,
                "median_regression_percent": regression,
                "baseline_work": sorted(
                    {(row["iterations"], row["fock_builds"]) for row in before}
                ),
                "candidate_work": sorted(
                    {(row["iterations"], row["fock_builds"]) for row in after}
                ),
                "baseline_numeric_peak_bytes": before_peak,
                "candidate_numeric_peak_bytes": after_peak,
                "default_routes": routes(after),
                "maximum_energy_error": energy_error,
                "maximum_force_error": force_error,
            }
        )
    report = {
        "energy_gate_hartree": 1e-8,
        "force_gate_hartree_per_bohr": 1e-7,
        "maximum_median_regression_percent": 5,
        "passed": not errors,
        "errors": errors,
        "groups": groups,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": not errors, "groups": len(groups), "errors": errors}, indent=2
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
