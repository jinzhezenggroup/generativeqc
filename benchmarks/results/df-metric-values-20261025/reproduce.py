"""Replay the complete metric-value endpoint and independent oracle receipts."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
CASES = (("pbe", 48), ("pbe", 96), ("r2scan-3c", 96))


def read(name: str) -> object:
    """Keep original producer identities while decoding the retained receipts."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def force_error(first: list, second: list) -> float:
    """Compare every public atom/Cartesian component, including warm samples."""
    return max(
        abs(a - b)
        for row_a, row_b in zip(first, second, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def assess() -> dict:
    """Fail on incomplete populations, changed work, or any numerical failure."""
    endpoints = read("endpoints.json.xz")
    assert len(endpoints) == 18
    rows, max_energy, max_force = [], 0.0, 0.0
    for method, atoms in CASES:
        populations = {variant: [] for variant in ("baseline", "candidate")}
        for repeat in range(3):
            pair = [endpoints[f"{repeat}-{method}-{atoms}-{v}"] for v in populations]
            before, after = pair
            for row in pair:
                assert row["status"] == "complete" and len(row["samples"]) == 3
                assert row["slurm_job_id"] and row["native_build"]["library_sha256"]
            for key in ("geometry_bohr", "protocol", "runner_sha256", "orbital_basis"):
                assert before[key] == after[key], key
            for b, a in zip(before["samples"], after["samples"], strict=True):
                assert b["converged"] and a["converged"]
                for key in ("phase", "repeat", "settings", "iterations", "fock_builds"):
                    assert b[key] == a[key], key
                energy = abs(a["energy"] - b["energy"])
                force = force_error(a["forces"], b["forces"])
                assert energy <= 1e-8 and force <= 1e-8
                max_energy, max_force = max(max_energy, energy), max(max_force, force)
            for variant, row in zip(populations, pair, strict=True):
                populations[variant].extend(row["samples"])
        row = {"method": method, "atoms": atoms}
        for variant, samples in populations.items():
            for phase in ("cold", "warm"):
                times = [s["seconds"] for s in samples if s["phase"] == phase]
                assert len(times) == (3 if phase == "cold" else 6)
                row[f"{variant}_{phase}"] = {
                    "median": median(times),
                    "minimum": min(times),
                    "maximum": max(times),
                }
        rows.append(row)
    moved = read("moved.json.xz")
    references = {
        (r["method"], r["atoms"], r["geometry"]): r["sample"]
        for r in moved
        if r["kind"] == "reference"
    }
    checks = [r for r in moved if r["kind"] == "native"]
    assert len(references) == 6 and len(checks) == 12
    for row in checks:
        ref = references[row["method"], row["atoms"], row["geometry"]]
        assert row["converged"] and row["errors"]["passed"]
        assert abs(row["energy"] - ref["energy"]) / row["atoms"] <= 1e-5
        assert force_error(row["forces"], ref["forces"]) <= 1e-4
        rms = (
            sum(
                (a - b) ** 2
                for aa, bb in zip(row["forces"], ref["forces"], strict=True)
                for a, b in zip(aa, bb, strict=True)
            )
            / (3 * row["atoms"])
        ) ** 0.5
        assert rms <= 3e-5
    native = read("native-oracles.json.xz")
    for name, count in (("values", 80), ("derivatives", 16)):
        report = native[name]
        assert report["passed"] and len(report["runs"]) == count
        assert all(row["passed"] and row["ranks_match"] for row in report["runs"])
    tests = read("tests.json.xz")
    assert "50 passed" in tests["tests.log"]
    assert "ERROR SUMMARY: 0 errors" in tests["memcheck.log"]
    return {
        "timings": rows,
        "maximum_paired_energy": max_energy,
        "maximum_paired_force": max_force,
        "moved_checks": len(checks),
        "native_oracle_cases": 96,
        "gpu_tests": 50,
        "memcheck_errors": 0,
    }


if __name__ == "__main__":
    print(json.dumps(assess(), indent=2))
