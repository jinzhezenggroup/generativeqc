"""Replay complete endpoint comparisons and independent moved-geometry gates."""

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
CANDIDATE = "0c26bbf11af084dabceb7b825bbfdd0f624ec03a"
CONTROL = "2ed9b394a2889b3a373dc91ebbaaed03931056eb"


def read(name: str) -> object:
    """Read lossless receipts, preserving their originally recorded revisions."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def key(record: dict) -> tuple:
    return tuple(record[name] for name in ("arm", "method", "mode", "atoms"))


def timing(record: dict) -> dict:
    """Include setup in cold and every frozen-density warm observation."""
    assert record["status"] == "complete"
    cold = [s["seconds"] for s in record["samples"] if s["phase"] == "cold"]
    warm = [s["seconds"] for s in record["samples"] if s["phase"] == "warm"]
    assert len(cold) == 1 and len(warm) == 3
    return {"cold": cold[0], "warm": median(warm), "warm_range": [min(warm), max(warm)]}


def compare(before: dict, after: dict) -> dict:
    """Check identical scientific inputs and unchanged complete solve work."""
    assert key(before) == key(after)
    for field in ("geometry_bohr", "runner_sha256", "auxiliary_input_sha256"):
        assert before.get(field) == after.get(field), field
    differences = []
    for b, a in zip(before["samples"], after["samples"], strict=True):
        assert (b["phase"], b["repeat"]) == (a["phase"], a["repeat"])
        assert b["iterations"] == a["iterations"]
        assert b["fock_builds"] == a["fock_builds"]
        energy = abs(b["energy"] - a["energy"])
        force = np.asarray(a["forces"]) - np.asarray(b["forces"])
        maximum = float(np.max(abs(force)))
        force_gate = 3e-7 if after["arm"] == "fast" else 1e-9
        assert energy <= 1e-8 and maximum <= force_gate
        differences.append(
            {
                "phase": a["phase"],
                "repeat": a["repeat"],
                "energy_abs": energy,
                "force_max_abs": maximum,
                "force_rms": float(np.sqrt(np.mean(force**2))),
                "iterations": a["iterations"],
                "fock_builds": a["fock_builds"],
            }
        )
    return {
        **{name: after[name] for name in ("arm", "method", "mode", "atoms")},
        "control": timing(before),
        "candidate": timing(after),
        "differences": differences,
    }


def assessment() -> dict:
    controls = {key(r): r for r in read("control.json.xz")}
    candidates = {key(r): r for r in read("candidate.json.xz")}
    assert len(controls) == len(candidates) == 18
    assert controls.keys() == candidates.keys()
    provenance = read("provenance.json.xz")
    assert provenance["source_revision"] == CANDIDATE
    assert provenance["sources"]["candidate"]["verified_files"] == 12761
    for r in controls.values():
        assert r["source_revision"] == CONTROL
    for r in candidates.values():
        # The runner recorded an unambiguous short SHA. Resolve it only here,
        # against the full archive/build binding; never rewrite raw receipts.
        assert r["source_revision"] in (CANDIDATE, CANDIDATE[:9])
    comparisons = [compare(controls[k], candidates[k]) for k in sorted(candidates)]
    same_binary = read("same-binary.json.xz")
    paired = []
    for atoms in (36, 42):
        before = same_binary[f"panels-standard-r2scan-3c-{atoms}"]
        after = same_binary[f"candidate-standard-r2scan-3c-{atoms}"]
        assert before["source_revision"] == after["source_revision"] == CANDIDATE
        assert before["native_build"] == after["native_build"]
        paired.append(compare(before, after))
    moved = read("moved.json.xz")
    assert len(moved) == 12
    for row in moved:
        n = row["atoms"]
        if "forces" in row["properties"]:
            result = errors(row["energy"], row["forces"], row["reference"], n)
        else:
            delta = abs(row["energy"] - row["reference"]["energy"]) / n
            result = {"energy_abs_per_atom_hartree": delta, "passed": delta <= 1e-5}
        assert result == row["errors"] and result["passed"]
    return {
        "comparisons": comparisons,
        "same_binary": paired,
        "moved_checks": len(moved),
    }


def table(summary: dict) -> str:
    """Keep unchanged and slower observations alongside the affected points."""
    lines = [
        "# Complete energy-and-force timings",
        "",
        "Seconds; cold includes preparation. Warm is the median of three calls, with their full range.",
        "",
        "| Arm | Method | Atoms | Control cold | Candidate cold | Control warm [range] | Candidate warm [range] |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in summary["comparisons"]:
        b, a = r["control"], r["candidate"]

        def warm(value: dict) -> str:
            low, high = value["warm_range"]
            return f"{value['warm']:.6f} [{low:.6f}, {high:.6f}]"

        lines.append(
            f"| {r['arm']} | {r['method']} | {r['atoms']} | {b['cold']:.6f} | "
            f"{a['cold']:.6f} | {warm(b)} | {warm(a)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    summary = assessment()
    assert summary == read("comparisons.json.xz")
    assert table(summary) == (ROOT / "timings.md").read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "timings.md").write_text(table(summary))
    (args.output / "comparisons.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "18 paired populations, 2 same-binary controls and 12 moved/property checks pass"
    )


if __name__ == "__main__":
    main()
