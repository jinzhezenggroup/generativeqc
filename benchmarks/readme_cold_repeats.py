"""Acquire independent application-cold processes; retain all repeats and failures.

Imports/context startup and compilation-cache prewarming are excluded, as in
the paper. Each formal process owns one cold call and three frozen-density warm
calls. Cold is summarized over three processes; warm over their three medians.
The error bars describe min/max, never confidence intervals. Historical samples
are not inputs to the new timing population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import METHODS, SIZES, allocation_metadata, save
from benchmarks.readme_defaults_report import checked_sample

SCHEMA = "generativeqc.readme-cold-repeats.v1"
ARMS = ("standard", "gpu4pyscf")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population_order(repeat: int) -> list[tuple[str, str, int, str]]:
    """Balance engine order in every round and reverse the middle round."""
    pairs = [(m, mode, n) for m in METHODS for mode in ("direct", "df") for n in SIZES]
    if repeat == 1:
        pairs.reverse()
    return [
        (method, mode, atoms, arm)
        for index, (method, mode, atoms) in enumerate(pairs)
        for arm in (ARMS if (index + repeat) % 2 == 0 else ARMS[::-1])
    ]


def aggregate(records: list[dict]) -> list[dict]:
    """Reject missing/duplicate/nonfinite populations instead of selecting survivors."""
    expected = {
        (a, m, mode, n)
        for a in ARMS
        for m in METHODS
        for mode in ("direct", "df")
        for n in SIZES
    }
    grouped: dict[tuple, dict[int, dict]] = {}
    sources, devices, libraries, runners = set(), set(), set(), set()
    for record in records:
        key = tuple(record[k] for k in ("arm", "method", "mode", "atoms"))
        repeat = record.get("process_repeat")
        if key not in expected or type(repeat) is not int or repeat not in range(3):
            raise ValueError("unexpected population or process repeat")
        group = grouped.setdefault(key, {})
        if repeat in group:
            raise ValueError("duplicate process repeat")
        group[repeat] = record
        if record.get("status") != "complete":
            raise ValueError("failed/incomplete process cannot produce a median")
        samples = record["samples"]
        if len(samples) != 4 or [s["phase"] for s in samples] != [
            "cold",
            "warm",
            "warm",
            "warm",
        ]:
            raise ValueError("each process requires one cold and three warm calls")
        if [s["repeat"] for s in samples] != [0, 0, 1, 2]:
            raise ValueError("duplicate or reordered warm repeats")
        for sample in samples:
            if (
                type(sample["seconds"]) not in (int, float)
                or not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
                or sample.get("converged") is not True
            ):
                raise ValueError("invalid or unconverged endpoint")
            checked_sample(sample, record)
        scientific_inputs = (
            "geometry_bohr",
            "orbital_basis",
            "basis_representation",
            "charge",
            "spin",
            "auxiliary_input_sha256",
        )
        for previous in list(group.values())[:-1]:
            if any(previous.get(k) != record.get(k) for k in scientific_inputs):
                raise ValueError("scientific input changed between processes")
        sources.add(record["source_revision"])
        devices.add(record["accelerator"]["nvidia_smi"]["uuid"])
        runners.add(record["runner_sha256"])
        if key[0] == "standard":
            libraries.add(record["native_build"]["library_sha256"])
    if set(grouped) != expected or any(set(g) != {0, 1, 2} for g in grouped.values()):
        raise ValueError("incomplete three-process matrix")
    if any(len(values) != 1 for values in (sources, devices, libraries, runners)):
        raise ValueError("mixed source, runner, library or physical GPU")
    rows = []
    for key in sorted(grouped):
        rr = [grouped[key][i] for i in range(3)]
        cold = [r["samples"][0]["seconds"] for r in rr]
        warm = [statistics.median(s["seconds"] for s in r["samples"][1:]) for r in rr]
        rows.append(
            {
                "key": list(key),
                "cold": {
                    "seconds": cold,
                    "median": statistics.median(cold),
                    "min": min(cold),
                    "max": max(cold),
                },
                "warm": {
                    "process_medians": warm,
                    "median": statistics.median(warm),
                    "min": min(warm),
                    "max": max(warm),
                    "all_seconds": [s["seconds"] for r in rr for s in r["samples"][1:]],
                },
                "cold_iterations": [r["samples"][0]["iterations"] for r in rr],
                "cold_focks": [
                    r["samples"][0].get(
                        "fock_builds", r["samples"][0].get("effective_potential_builds")
                    )
                    for r in rr
                ],
            }
        )
    return rows


def acquire(config: dict, output: Path) -> None:
    """Execute a fixed bounded matrix, preserving the first failed attempt."""
    allocation_metadata(config.get("qz_job_name"))
    output.mkdir(parents=True, exist_ok=False)
    source = Path(config["source"])
    inputs = Path(config["inputs"])
    deadline = time.monotonic() + config["deadline_seconds"]
    receipt = {
        "schema": SCHEMA,
        "status": "running",
        "config": config,
        "commands": [],
        "allocation": allocation_metadata(config.get("qz_job_name")),
    }
    save(output / "campaign.json", receipt)
    records = []

    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    previous_handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for stage, repeats, warms in (("prewarm", [-1], 1), ("formal", range(3), 3)):
            for repeat in repeats:
                for method, mode, atoms, arm in population_order(repeat):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("campaign monotonic deadline exhausted")
                    name = f"{method}-{mode}-{atoms}-{arm}-r{repeat}"
                    destination = output / stage / (name + ".json")
                    destination.parent.mkdir(exist_ok=True)
                    env = os.environ.copy()
                    for inherited in list(env):
                        if inherited.startswith("GENERATIVEQC_"):
                            del env[inherited]
                    env.update(config["environment"], **config["arm_environment"][arm])
                    argv = [
                        config["interpreters"][arm],
                        "-m",
                        "benchmarks.readme_defaults",
                        arm,
                        "--method",
                        method,
                        "--mode",
                        mode,
                        "--atoms",
                        str(atoms),
                        "--warm-repeats",
                        str(warms),
                        "--source-revision",
                        config["source_revision"],
                        "--inputs",
                        str(inputs),
                        "--output",
                        str(destination),
                    ]
                    if config.get("qz_job_name"):
                        argv += ["--qz-job-name", config["qz_job_name"]]
                    row = {
                        "name": name,
                        "stage": stage,
                        "process_repeat": repeat,
                        "argv": argv,
                        "status": "running",
                    }
                    receipt["commands"].append(row)
                    save(output / "campaign.json", receipt)
                    started = time.monotonic()
                    with destination.with_suffix(".log").open("x") as log:
                        child = subprocess.Popen(
                            argv,
                            cwd=source,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        try:
                            code = child.wait(
                                timeout=min(
                                    config["process_timeout_seconds"], remaining
                                )
                            )
                        except BaseException:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            row.update(
                                status="interrupted", seconds=time.monotonic() - started
                            )
                            raise
                    row.update(
                        returncode=code,
                        seconds=time.monotonic() - started,
                        status="complete" if code == 0 else "failed",
                    )
                    if destination.exists():
                        row["raw_sha256"] = digest(destination)
                    save(output / "campaign.json", receipt)
                    if code:
                        raise RuntimeError(f"endpoint failed; no retry: {name}")
                    record = json.loads(destination.read_text())
                    if (
                        record["status"] != "complete"
                        or len(record["samples"]) != warms + 1
                    ):
                        raise ValueError(f"incomplete endpoint: {name}")
                    if stage == "formal":
                        records.append(
                            {
                                **record,
                                "process_repeat": repeat,
                                "raw_file": str(destination),
                                "raw_sha256": row["raw_sha256"],
                            }
                        )
                    print(
                        stage,
                        name,
                        round(record["samples"][0]["seconds"], 6),
                        flush=True,
                    )
        summary = aggregate(records)
        save(output / "records.json", {"schema": SCHEMA, "records": records})
        save(output / "summary.json", {"schema": SCHEMA, "populations": summary})
        receipt.update(
            status="complete",
            formal_processes=len(records),
            formal_endpoints=sum(len(r["samples"]) for r in records),
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "campaign.json", receipt)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    acquire(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
