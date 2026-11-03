"""Replay every frozen production pair, keeping failed and unstable runs visible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_accuracy import errors

SOURCE = "bbf5033f320dec96157df65e5f07e660d4f5de58b2946c34a6128814a361922e"


def report(folder: Path) -> dict:
    """Authenticate the OFF core and recompute all sixteen vector comparisons."""
    bundle, files = load(folder)
    pairs = []
    exits = {
        p: int(t) for p, t in files.items() if p.endswith(("/exit-code.txt", ".exit"))
    }
    for path in sorted(files):
        if not path.endswith("/standard.json"):
            continue
        prefix = path.removesuffix("standard.json")
        if prefix + "gpu4pyscf.json" not in files:
            continue
        a, b = (
            json.loads(files[prefix + name + ".json"])
            for name in ("standard", "gpu4pyscf")
        )
        proof = json.loads(files[prefix + "identity.json"])
        require(
            proof["source_identity"] == SOURCE and not proof["build_tests"],
            "production identity",
        )
        require(
            a["native_build"]["library_sha256"]
            == proof["binaries"]["libgenerativeqc.so"],
            "measured core",
        )
        require(a["native_build"]["source_identity"] == SOURCE, "measured source")
        require(a["status"] == b["status"] == "complete", "complete pair")
        require(
            a["geometry_bohr"] == b["geometry_bohr"]
            and a["auxiliary_input_sha256"] == b["auxiliary_input_sha256"],
            "same scientific inputs",
        )
        require(len(a["samples"]) == len(b["samples"]) == 4, "cold plus three warm")
        gates = [
            errors(x["energy"], x["forces"], y, a["atoms"])
            for x in a["samples"]
            for y in b["samples"]
        ]
        native = [s["seconds"] for s in a["samples"]]
        reference = [s["seconds"] for s in b["samples"]]
        pairs.append(
            {
                "run": prefix.rstrip("/"),
                "source_identity": SOURCE,
                "native_seconds": native,
                "reference_seconds": reference,
                "cold_reference_over_native": reference[0] / native[0],
                "warm_reference_over_native": statistics.median(reference[1:])
                / statistics.median(native[1:]),
                "reference_warm_range_over_median": (
                    max(reference[1:]) - min(reference[1:])
                )
                / statistics.median(reference[1:]),
                "nonstationary_reference": "15482" in prefix,
                "native_iterations": [s["iterations"] for s in a["samples"]],
                "reference_iterations": [s["iterations"] for s in b["samples"]],
                "paired_errors": gates,
                "paired_pass": all(x["passed"] for x in gates),
            }
        )
    require(len(pairs) == 6, "three process pairs per device")
    groups = {}
    for device in ("a100", "sm120"):
        rows = [x for x in pairs if x["run"].startswith(device + "/")]
        stable = [x for x in rows if not x["nonstationary_reference"]]
        groups[device] = {
            "all_pairs_median_ratio": statistics.median(
                x["warm_reference_over_native"] for x in rows
            ),
            "stable_pairs_median_ratio": statistics.median(
                x["warm_reference_over_native"] for x in stable
            ),
            "pair_count": len(rows),
            "stable_pair_count": len(stable),
        }
    return {
        "scope": "Static matched fitted-LR endpoints on the frozen source; no qualification of newer review fixes or all-GPU ranking",
        "source_identity": SOURCE,
        "input_archives": bundle["input_archives"],
        "pairs": pairs,
        "device_populations": groups,
        "original_exit_codes": exits,
        "all_paired_gates_pass": all(x["paired_pass"] for x in pairs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(report(Path(__file__).resolve().parent), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
