"""Replay figures, accuracy gates and work counts without running a GPU solve."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import MEASUREMENT_ARMS, METHODS
from benchmarks.readme_defaults_report import ARM_LABELS, LABELS, load_records

ROOT = Path(__file__).resolve().parent


def work(sample: dict, arm: str) -> str:
    """Include every exposed Fast solve, including independent precision audits.

    Native Fock counts and reference effective-potential calls have different
    semantics. Preserve their names in the table explanation; they are useful
    work diagnostics, not interchangeable operation counts.
    """
    solves = (sample.get("precision_validation") or {}).get("solves")
    if arm == "fast":
        assert solves
        counts = [
            sum(s[key] for s in solves)
            if all(s.get(key) is not None for s in solves)
            else None
            for key in ("iterations", "fock_builds", "force_evaluations")
        ]
        counts.append(len(solves))
    else:
        counts = [
            sample["iterations"],
            sample.get("fock_builds", sample.get("effective_potential_builds")),
            1,
            1,
        ]
    return "/".join("?" if value is None else str(value) for value in counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    receipts = [ROOT / f"{arm}.json.xz" for arm in MEASUREMENT_ARMS]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.readme_defaults_report",
            "--receipts",
            *map(str, receipts),
            "--oracles",
            str(ROOT / "oracles.json.xz"),
            "--include-unqualified",
            "--reference-reuse",
            str(ROOT / "reference-reuse.json"),
            "--output",
            str(args.output),
        ],
        check=True,
    )
    lines = [
        "# Executed SCF and precision work",
        "",
        (
            "Each entry is iterations / Fock builds / force evaluations / solves. "
            "GPU4PySCF reports effective-potential calls in the Fock column; these "
            "are not identical operation counts. Fast totals include all exposed "
            "candidate, audit and fallback solves. Warm entries retain every call "
            "in order, including partial populations. Missing entries have no "
            "completed endpoint; no work is inferred for a failed call."
        ),
        "",
        "| Mode | Method | Atoms | Engine | Cold | Warm calls |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for record in sorted(
        load_records(receipts),
        key=lambda r: (
            r["mode"],
            list(METHODS).index(r["method"]),
            r["atoms"],
            r["arm"],
        ),
    ):
        phases = [
            "; ".join(
                work(s, record["arm"]) for s in record["samples"] if s["phase"] == phase
            )
            or "—"
            for phase in ("cold", "warm")
        ]
        lines.append(
            f"| {record['mode']} | {LABELS[record['method']]} | {record['atoms']} | "
            f"{ARM_LABELS[record['arm']]} | {' | '.join(phases)} |"
        )
    (args.output / "work.md").write_text("\n".join(lines) + "\n")
    from assessment import assess, read, table

    comparisons = assess(ROOT)
    assert comparisons == read(ROOT / "comparisons.json.xz")
    assert table(comparisons) == (ROOT / "changes.md").read_text()
    (args.output / "changes.md").write_text(table(comparisons))
    qualifications = json.loads((args.output / "qualification.json").read_text())
    print(
        {
            state: sum(q["status"] == state for q in qualifications)
            for state in ("qualified", "failed", "unverified")
        }
    )


if __name__ == "__main__":
    main()
