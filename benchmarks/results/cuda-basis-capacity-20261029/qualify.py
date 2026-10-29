"""ABBA/BAAB complete endpoint comparison with independent worker processes."""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    workers, metadata = {}, {}
    try:
        for name, source, library in [
            ("A", args.baseline, args.baseline_library),
            ("B", args.candidate, args.candidate_library),
        ]:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("endpoint_worker.py")),
                    "--source",
                    str(source),
                    "--library",
                    str(library),
                    "--backend",
                    args.backend,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env={
                    **environment,
                    "BASIS_QUALIFY_NATIVE_COUNTS": "1" if name == "B" else "0",
                },
            )
            workers[name] = worker
            metadata[name] = json.loads(worker.stdout.readline())["ready"]

        def sample(name: str) -> list[dict[str, Any]]:
            worker = workers[name]
            worker.stdin.write("{}\n")
            worker.stdin.flush()
            return json.loads(worker.stdout.readline())["rows"]

        # Both processes prime their ordinary artifact/model caches. Every
        # measured protocol still constructs fresh owners and cold densities.
        for name in workers:
            sample(name)
        samples = []
        for group in range(args.groups):
            for name in "ABBA" if group % 2 == 0 else "BAAB":
                samples.append({"group": group, "engine": name, "rows": sample(name)})
        values, science = {}, {}
        max_energy_error = max_force_error = 0.0
        import numpy as np

        for sample_record in samples:
            for row in sample_record["rows"]:
                key = row["case"] + "/" + row["phase"]
                values.setdefault(key, {"A": [], "B": []})[
                    sample_record["engine"]
                ].append(row["seconds"])
                if key not in science:
                    science[key] = row["items"]
                expected = science[key]
                if len(expected) != len(row["items"]):
                    raise RuntimeError("endpoint item count changed")
                for actual, reference in zip(row["items"], expected, strict=True):
                    for field in (
                        "iterations",
                        "fock_builds",
                        "grid_points",
                        "occupation",
                        "coulomb_exchange_work",
                        "policy_work",
                    ):
                        if actual[field] != reference[field]:
                            raise RuntimeError(f"{key}: semantic work changed: {field}")
                    for field in ("energy",):
                        difference = abs(actual[field] - reference[field])
                        max_energy_error = max(max_energy_error, difference)
                        if difference > 2e-9:
                            raise RuntimeError(f"{key}: energy changed")
                    for field in ("forces",):
                        if actual[field] is None or reference[field] is None:
                            if actual[field] != reference[field]:
                                raise RuntimeError("derivative publication changed")
                            continue
                        difference = float(
                            np.max(
                                np.abs(
                                    np.asarray(actual[field])
                                    - np.asarray(reference[field])
                                )
                            )
                        )
                        max_force_error = max(max_force_error, difference)
                        if difference > 2e-8:
                            raise RuntimeError(f"{key}: forces changed")
        ratios = {
            key: statistics.median(value["B"]) / statistics.median(value["A"])
            for key, value in values.items()
        }
        report = {
            "metadata": metadata,
            "groups": args.groups,
            "endpoint_count": sum(len(s["rows"]) - 8 for s in samples),
            "max_energy_difference": max_energy_error,
            "max_force_difference": max_force_error,
            "ratios": ratios,
            "maximum_ratio": max(ratios.values()),
            "all_within_1_05": all(r <= 1.05 for r in ratios.values()),
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        opener = lzma.open if args.output.suffix == ".xz" else gzip.open
        with opener(args.output, "wt") as stream:
            json.dump(report, stream, separators=(",", ":"))
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("samples", "ratios")},
                indent=2,
            )
        )
        if not report["all_within_1_05"]:
            raise SystemExit("complete endpoint non-regression gate failed")
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n')
                worker.stdin.flush()
                worker.wait(timeout=30)


if __name__ == "__main__":
    main()
