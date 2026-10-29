"""ABBA/BAAB complete endpoint comparison with independent worker processes."""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import math
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


def _stop_worker(worker: subprocess.Popen) -> None:
    """Stop only this harness-owned session, including blocked pipe owners."""
    if os.name == "posix":
        try:
            os.killpg(worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif worker.poll() is None:
        worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            worker.kill()
    finally:
        # A descendant may retain stdout after its direct parent exits.
        if os.name == "posix":
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        worker.wait(timeout=5)


def _response(worker: subprocess.Popen, timeout: float) -> dict[str, Any]:
    """Bound a complete line, not merely the first readable pipe byte."""
    messages: queue.Queue = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            messages.put(worker.stdout.readline())
        except (OSError, UnicodeError) as error:
            messages.put(error)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        response = messages.get(timeout=timeout)
    except queue.Empty as error:
        _stop_worker(worker)
        reader.join(timeout=5)
        raise TimeoutError("qualification worker response deadline exceeded") from error
    if isinstance(response, Exception):
        raise response
    if not response:
        raise RuntimeError("qualification worker exited before a complete response")
    return json.loads(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not math.isfinite(args.response_timeout) or args.response_timeout <= 0:
        parser.error("--response-timeout must be finite and positive")
    if args.groups <= 0:
        parser.error("--groups must be positive")
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
                start_new_session=True,
                env=environment,
            )
            workers[name] = worker
            metadata[name] = _response(worker, args.response_timeout)["ready"]

        def sample(name: str) -> list[dict[str, Any]]:
            worker = workers[name]
            worker.stdin.write("{}\n")
            worker.stdin.flush()
            return _response(worker, args.response_timeout)["rows"]

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
                if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                    raise RuntimeError(f"{key}: invalid measured time")
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
                        if not math.isfinite(actual[field]) or not math.isfinite(
                            reference[field]
                        ):
                            raise RuntimeError(f"{key}: non-finite {field}")
                        difference = abs(actual[field] - reference[field])
                        max_energy_error = max(max_energy_error, difference)
                        if difference > 2e-9:
                            raise RuntimeError(f"{key}: energy changed")
                    for field in ("forces",):
                        if actual[field] is None or reference[field] is None:
                            if actual[field] != reference[field]:
                                raise RuntimeError("derivative publication changed")
                            continue
                        if not np.all(np.isfinite(actual[field])) or not np.all(
                            np.isfinite(reference[field])
                        ):
                            raise RuntimeError(f"{key}: non-finite {field}")
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
        medians = {
            key: {role: statistics.median(times) for role, times in value.items()}
            for key, value in values.items()
        }
        if any(
            not math.isfinite(t) or t <= 0
            for value in medians.values()
            for t in value.values()
        ):
            raise RuntimeError("invalid reduced measured time")
        ratios = {key: value["B"] / value["A"] for key, value in medians.items()}
        if any(not math.isfinite(r) or r <= 0 for r in ratios.values()):
            raise RuntimeError("invalid measured time ratio")
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
            json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("samples", "ratios")},
                indent=2,
                allow_nan=False,
            )
        )
        if not report["all_within_1_05"]:
            raise SystemExit("complete endpoint non-regression gate failed")
    finally:
        cleanup_error = None
        for worker in workers.values():
            try:
                _stop_worker(worker)
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError("qualification worker cleanup failed") from cleanup_error


if __name__ == "__main__":
    main()
