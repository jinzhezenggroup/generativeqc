"""Authenticate and replay both frozen cold-repeat cohorts without a GPU."""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from collections import Counter
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[2]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / "python")]

from benchmarks._retention import raw_output_path
from benchmarks.readme_cold_repeats_replay import replay
from benchmarks.readme_cold_repeats_report import render, sha
from benchmarks.readme_defaults import save
from tools.generativeqc_validation.publication import validate_publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--output", type=raw_output_path)
    args = parser.parse_args()
    manifest = json.loads((DIRECTORY / "publication.json").read_bytes())
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    oracle_path = (
        REPOSITORY / "benchmarks/results/readme-refresh-20261026/oracles.json.xz"
    )
    oracles = json.loads(lzma.decompress(oracle_path.read_bytes()))
    results = {}
    for device in ("h100", "rtx5090"):
        payload = json.loads(lzma.decompress(files[device + ".json.xz"]))
        if sha(oracle_path) != payload["originals"]["oracles_sha256"]:
            raise ValueError("frozen independent oracle bytes changed")
        results[device] = replay(payload, oracles)
    revisions = {
        r["source_revision"]
        for device in ("h100", "rtx5090")
        for r in json.loads(lzma.decompress(files[device + ".json.xz"]))["records"]
    }
    if revisions != {manifest["source"]["revision"]}:
        raise ValueError("paired cohorts differ from publication source")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for device, result in results.items():
            target = args.output / device
            target.mkdir()
            save(target / "summary.json", result)
            render(result["populations"], result["label"], target / "results-hfdft")
    print(
        json.dumps(
            {
                "status": "REPLAY_EQUIVALENT_WITH_RETAINED_NUMERICAL_FAILURES",
                "devices": {
                    device: {
                        "populations": len(result["populations"]),
                        "process_gate_counts": dict(
                            Counter(r["status"] for r in result["accuracy"])
                        ),
                        "cold_median_decreases": result["diagnostics"][
                            "cold_decreases"
                        ],
                        "median_wins": result["diagnostics"]["wins"],
                    }
                    for device, result in results.items()
                },
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
