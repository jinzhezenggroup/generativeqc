"""Recompute every paired endpoint and independent observable acceptance gate."""

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
CONTROL = "7b2ab52fdb37ebdd8d149664530e77f9b8cc8cc0"
CANDIDATE = "fba34e9192edf0ea0d5baf6e3cd1f0b9ab14dc5d"


def read(name: str) -> object:
    """Decode full receipts without rewriting their original identities."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def assess() -> dict:
    """Keep three independent cold populations and all nine warm calls per case."""
    receipts = read("endpoints.json.xz")
    references = {r["atoms"]: r for r in read("references.json.xz")}
    provenance = read("provenance.json.xz")["variants"]
    assert len(receipts) == 24
    pairs, independent = [], []
    for arm in ("standard", "fast"):
        for atoms in (48, 96):
            for repeat in range(3):
                b = receipts[f"{repeat}-{arm}-{atoms}-baseline"]
                a = receipts[f"{repeat}-{arm}-{atoms}-candidate"]
                for label, record, revision in (
                    ("baseline", b, CONTROL),
                    ("candidate", a, CANDIDATE),
                ):
                    assert record["status"] == "complete"
                    assert record["source_revision"] == revision
                    assert len(record["samples"]) == 4
                    binary = record["native_build"]["library_sha256"]
                    assert binary in provenance[label]["libraries"].values()
                for field in ("geometry_bohr", "runner_sha256", "protocol"):
                    assert b[field] == a[field], field
                oracle = references[atoms]
                assert oracle["geometry_bohr"] == a["geometry_bohr"]
                assert oracle["orbital_basis"] == a["orbital_basis"]
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
                    force = np.asarray(after["forces"]) - np.asarray(before["forces"])
                    delta = {
                        "energy_abs": abs(after["energy"] - before["energy"]),
                        "force_max_abs": float(abs(force).max()),
                    }
                    assert (
                        delta["energy_abs"] <= 1e-8 and delta["force_max_abs"] <= 1e-9
                    )
                    changes.append(delta)
                    checked = errors(
                        after["energy"], after["forces"], oracle["samples"][0], atoms
                    )
                    assert checked["passed"]
                    independent.append(checked)
                pairs.append(
                    {
                        "arm": arm,
                        "atoms": atoms,
                        "repeat": repeat,
                        "changes": changes,
                        "control_seconds": [s["seconds"] for s in b["samples"]],
                        "candidate_seconds": [s["seconds"] for s in a["samples"]],
                        "fock_builds": [s["fock_builds"] for s in a["samples"]],
                    }
                )
    moved = read("moved.json.xz")
    assert len(moved) == 24
    for row in moved:
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
    return {
        "pairs": pairs,
        "independent_endpoint_errors": independent,
        "moved_checks": len(moved),
    }


def table(summary: dict) -> str:
    """Report medians and ranges; every cold/warm observation remains retained."""
    lines = [
        "# Complete HF energy-and-force endpoint timings",
        "",
        "Seconds. Three alternating fresh-process rounds; each cold includes construction and preparation,",
        "followed by three warm calls from the frozen cold density. Brackets show the full range.",
        "",
        "| Profile | Atoms | Control cold | Candidate cold | Control warm | Candidate warm |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for arm in ("standard", "fast"):
        for atoms in (48, 96):
            selected = [
                r for r in summary["pairs"] if r["arm"] == arm and r["atoms"] == atoms
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
            lines.append(f"| {arm} | {atoms} | " + " | ".join(cells) + " |")
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
        "24 complete populations / 96 calls, 48 independent sample gates and 24 moved/property/rollback gates pass"
    )


if __name__ == "__main__":
    main()
