"""Replay all endpoint/error comparisons from the retained full-precision receipts."""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_qualification import identity, qualify

BASELINE = "194f12d788d0073b4e158968d0eaa694283b0d34"


def read(root: Path, name: str) -> dict | list:
    """Read a retained JSON member without changing its provenance."""
    return json.loads(lzma.decompress((root / (name + ".json.xz")).read_bytes()))


def normalized(record: dict) -> dict:
    """Expand the first launcher's abbreviated SHA in a receipt-bound copy."""
    if record["source_revision"] != BASELINE[:9]:
        return record
    return {
        **record,
        "raw_receipt_sha256": identity(record),
        "recorded_source_revision": record["source_revision"],
        "source_revision": BASELINE,
    }


def key(record: dict) -> tuple:
    """Identify the scientific population and precision profile."""
    return tuple(record[k] for k in ("arm", "method", "mode", "atoms"))


def timing(record: dict) -> dict:
    """Keep the sole cold observation and every warm observation's range."""
    warm = [s["seconds"] for s in record["samples"] if s["phase"] == "warm"]
    cold = [s["seconds"] for s in record["samples"] if s["phase"] == "cold"]
    assert len(cold) == 1 and len(warm) == 3
    return {
        "cold": cold[0],
        "warm": median(warm),
        "warm_min": min(warm),
        "warm_max": max(warm),
    }


def assess(root: Path) -> dict:
    """Recompute independent gates, baseline deltas, and measured GPU4 comparisons."""
    receipts = read(root, "receipts")
    rows = {
        phase: [normalized(r) for r in values] for phase, values in receipts.items()
    }
    oracles = read(root, "oracles")
    reuse = read(root, "reference-reuse")
    qualified = {
        phase: qualify(values, oracles, reference_reuse=reuse)
        for phase, values in rows.items()
    }
    before_index = {key(r): r for r in rows["baseline"]}
    gpu_index = {key(r)[1:]: r for r in rows["reference"]}
    comparisons = []
    for after in sorted(rows["candidate"], key=key):
        before = before_index[key(after)]
        gpu4 = gpu_index[key(after)[1:]]
        for field in ("geometry_bohr", "runner_sha256", "auxiliary_input_sha256"):
            assert before.get(field) == after.get(field) == gpu4.get(field), field
        changes = []
        for old, new in zip(before["samples"], after["samples"], strict=True):
            assert (old["phase"], old["repeat"]) == (new["phase"], new["repeat"])
            delta = np.asarray(new["forces"]) - np.asarray(old["forces"])
            changes.append(
                {
                    "phase": new["phase"],
                    "repeat": new["repeat"],
                    "energy_abs": abs(new["energy"] - old["energy"]),
                    "force_max_abs": float(np.max(np.abs(delta))),
                    "force_rms": float(np.sqrt(np.mean(delta * delta))),
                    "before_iterations": old["iterations"],
                    "after_iterations": new["iterations"],
                    "before_fock_builds": old["fock_builds"],
                    "after_fock_builds": new["fock_builds"],
                    "before_solves": (old.get("precision_validation") or {}).get(
                        "solve_count", 1
                    ),
                    "after_solves": (new.get("precision_validation") or {}).get(
                        "solve_count", 1
                    ),
                }
            )
        comparisons.append(
            {
                **{k: after[k] for k in ("arm", "method", "mode", "atoms")},
                "before": timing(before),
                "after": timing(after),
                "gpu4": timing(gpu4),
                "changes": changes,
                "same_model_gpu4_errors": [
                    errors(s["energy"], s["forces"], gpu4["samples"][0], after["atoms"])
                    for s in after["samples"]
                ],
            }
        )
    moved = read(root, "qualification")["moved"]
    moved_checks = []
    for row in moved:
        checked = (
            errors(row["energy"], row["forces"], row["reference"], 48)
            if "forces" in row["properties"]
            else {
                "energy_abs_per_atom_hartree": abs(
                    row["energy"] - row["reference"]["energy"]
                )
                / 48,
                "passed": abs(row["energy"] - row["reference"]["energy"]) / 48 <= 1e-5,
            }
        )
        assert checked == row["errors"]
        moved_checks.append(
            {**{k: row[k] for k in ("method", "arm", "properties")}, **checked}
        )
    return {
        "comparisons": comparisons,
        "qualification": qualified,
        "counts": {
            phase: dict(Counter(r["status"] for r in values))
            for phase, values in qualified.items()
        },
        "moved": moved_checks,
    }


def table(assessment: dict) -> str:
    """Render all observations; failed accuracy gates never filter timings."""
    lines = [
        "# Complete endpoint observations",
        "",
        "Seconds; cold is one fresh-process observation. Warm is median [minimum, maximum] of three frozen-density calls. Fock counts describe the cold endpoint, including exposed audits. Accuracy outcomes are retained separately in assessment.json.xz.",
        "",
        "| Arm | Method | Mode | Atoms | Before cold | After cold | GPU4 cold | Before warm [range] | After warm [range] | GPU4 warm [range] | Focks before → after |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for r in assessment["comparisons"]:
        cells = [r["arm"], r["method"], r["mode"], str(r["atoms"])]
        cells += [f"{r[k]['cold']:.4f}" for k in ("before", "after", "gpu4")]
        cells += [
            f"{r[k]['warm']:.4f} [{r[k]['warm_min']:.4f}, {r[k]['warm_max']:.4f}]"
            for k in ("before", "after", "gpu4")
        ]
        changes = r["changes"][0]
        cells += [f"{changes['before_fock_builds']} → {changes['after_fock_builds']}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Validate the stored assessment and write an inspectable offline replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = assess(root)
    assert result == read(root, "assessment"), "retained assessment does not replay"
    assert table(result) == (root / "timings.md").read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "timings.md").write_text(table(result))
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
