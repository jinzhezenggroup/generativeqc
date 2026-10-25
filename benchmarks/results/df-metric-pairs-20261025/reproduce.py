"""Replay metric-pair endpoint, independent reference and semantic-work gates."""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults_accuracy import errors

ROOT = Path(__file__).resolve().parent
CONTROL = "fba34e9192edf0ea0d5baf6e3cd1f0b9ab14dc5d"
CANDIDATE = "a55214c7c458694aaa93a5b533e71b842f3a73d6"
CASES = (("pbe", 48), ("pbe", 96), ("r2scan-3c", 96))


def read(name: str) -> object:
    """Decode full receipts without replacing original producer identities."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def assess() -> dict:
    """Require every population, original/moved oracle and bounded work count."""
    receipts = read("endpoints.json.xz")
    moved = read("moved.json.xz")
    assert len(receipts) == 36 and len(moved) == 30
    references = {
        (r["method"], r["atoms"]): r for r in moved if r["geometry"] == "original"
    }
    provenance = read("provenance.json.xz")["variants"]
    pairs, independent, rollbacks = [], [], []
    for method, atoms in CASES:
        for arm in ("standard", "fast"):
            force_gate = 3e-7 if arm == "fast" else 1e-8
            for repeat in range(3):
                prefix = f"{repeat}-{method}-{atoms}-{arm}"
                b, a = receipts[prefix + "-candidate"], receipts[prefix + "-metric"]
                for label, row, revision in (
                    ("candidate", b, CONTROL),
                    ("metric", a, CANDIDATE),
                ):
                    assert (
                        row["status"] == "complete"
                        and row["source_revision"] == revision
                    )
                    assert len(row["samples"]) == 4
                    assert (
                        row["native_build"]["library_sha256"]
                        in provenance[label]["libraries"].values()
                    )
                for field in (
                    "geometry_bohr",
                    "runner_sha256",
                    "auxiliary_input_sha256",
                    "protocol",
                ):
                    assert b[field] == a[field], field
                reference = references[method, atoms]
                assert reference["geometry_bohr"] == a["geometry_bohr"]
                changes = []
                for before, after in zip(b["samples"], a["samples"], strict=True):
                    assert before["converged"] and after["converged"]
                    for field in (
                        "phase",
                        "repeat",
                        "iterations",
                        "fock_builds",
                        "settings",
                    ):
                        assert before[field] == after[field], field
                    delta = {
                        "energy_abs": abs(after["energy"] - before["energy"]),
                        "force_max_abs": float(
                            np.max(abs(np.asarray(after["forces"]) - before["forces"]))
                        ),
                        "force_atol": force_gate,
                    }
                    assert (
                        delta["energy_abs"] <= 1e-8
                        and delta["force_max_abs"] <= force_gate
                    )
                    changes.append(delta)
                    checked = errors(
                        after["energy"], after["forces"], reference["reference"], atoms
                    )
                    assert checked["passed"]
                    independent.append(checked)
                pairs.append(
                    {
                        "method": method,
                        "atoms": atoms,
                        "arm": arm,
                        "repeat": repeat,
                        "changes": changes,
                        "control_seconds": [s["seconds"] for s in b["samples"]],
                        "candidate_seconds": [s["seconds"] for s in a["samples"]],
                        "fock_builds": [s["fock_builds"] for s in a["samples"]],
                    }
                )
    for index, row in enumerate(moved):
        checked = (
            errors(row["energy"], row["forces"], row["reference"], row["atoms"])
            if "forces" in row["properties"]
            else {
                "energy_abs_per_atom_hartree": abs(
                    row["energy"] - row["reference"]["energy"]
                )
                / row["atoms"],
                "passed": abs(row["energy"] - row["reference"]["energy"]) / row["atoms"]
                <= 1e-5,
            }
        )
        assert checked == row["errors"] and checked["passed"]
        if row["rollback"]:
            previous = moved[index - 1]
            assert not previous["rollback"] and previous["geometry"] == row["geometry"]
            force = float(np.max(abs(np.asarray(row["forces"]) - previous["forces"])))
            assert abs(previous["energy"] - row["energy"]) <= 1e-8
            assert force <= (3e-7 if row["arm"] == "fast" else 1e-8)
            rollbacks.append(force)
    work = []
    for name, records in read("work-traces.json.xz").items():
        if not name.endswith("-metric-trace.jsonl"):
            continue
        for row in records:
            if row["operation"] != "force_response":
                continue
            counters = row["counters"]
            n = int(counters["metric_derivative_dense_weights"] ** 0.5)
            assert n * n == counters["metric_derivative_dense_weights"]
            assert counters["metric_derivative_pair_tasks"] == n * (n + 1) // 2
            assert counters["metric_derivative_folded_panels"] == 1
            work.append(
                {"name": name, "dense_weights": n * n, "pair_tasks": n * (n + 1) // 2}
            )
    assert len(work) == 6
    return {
        "pairs": pairs,
        "independent_endpoint_errors": independent,
        "moved_checks": len(moved),
        "rollback_force_max_abs": rollbacks,
        "work": work,
    }


def table(summary: dict) -> str:
    """Report all median cold/warm observations with their complete ranges."""
    lines = [
        "# Complete DF energy-and-force endpoint timings",
        "",
        "Seconds. Three alternating fresh-process rounds, each with cold preparation and three frozen-density warm calls.",
        "Brackets give the complete observed range. Diagnostic traces do not enter these timings.",
        "",
        "| Method | Atoms | Profile | Control cold | Candidate cold | Control warm | Candidate warm |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for method, atoms in CASES:
        for arm in ("standard", "fast"):
            selected = [
                r
                for r in summary["pairs"]
                if (r["method"], r["atoms"], r["arm"]) == (method, atoms, arm)
            ]
            cells = []
            for phase in (0, 1):
                for variant in ("control", "candidate"):
                    values = (
                        [r[variant + "_seconds"][0] for r in selected]
                        if phase == 0
                        else [t for r in selected for t in r[variant + "_seconds"][1:]]
                    )
                    cells.append(
                        f"{median(values):.6f} [{min(values):.6f}, {max(values):.6f}]"
                    )
            lines.append(f"| {method} | {atoms} | {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    summary = assess()
    assert summary == read("comparisons.json.xz")
    assert table(summary) == (ROOT / "timings.md").read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparisons.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "timings.md").write_text(table(summary))
    print(
        "36 populations / 144 calls, 72 independent sample gates, 30 moved/property checks and 6 work traces pass"
    )


if __name__ == "__main__":
    main()
