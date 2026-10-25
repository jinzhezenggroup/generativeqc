"""Complete same-binary E+F comparison of authenticated DF source/metric sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from generativeqc import Calculator

from tools.generativeqc_validation.correlation_df_oracle import basis_snapshot, energies


def main() -> None:
    """Retain cold/warm timings, independent gates and actual construction receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mp2", "df-ccsd(t)"), default="mp2")
    parser.add_argument("--orbital", default="cc-pvtz")
    parser.add_argument("--auxiliary", default="cc-pvtz-jkfit")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID") and args.repeats > 0
    atoms = [(8, (0, 0, 0)), (1, (1.4, 0, 1.1)), (1, (-1.4, 0, 1.1))]
    orbital = basis_snapshot(args.orbital, [8, 1])
    auxiliary = basis_snapshot(args.auxiliary, [8, 1])
    direction = np.random.default_rng(724).normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    step = 3e-5

    def oracle(shift: float) -> float:
        geometry = [
            (z, np.asarray(x) + shift * d)
            for (z, x), d in zip(atoms, direction, strict=True)
        ]
        return energies(
            geometry,
            args.orbital,
            args.auxiliary,
            triples=args.method == "df-ccsd(t)",
            df_reference=True,
        )[args.method]

    expected = oracle(0)
    slope = (oracle(step) - oracle(-step)) / (2 * step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_sha": args.source_sha,
        "library_sha256": hashlib.sha256(
            Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
        ).hexdigest(),
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "method": args.method,
        "orbital": args.orbital,
        "auxiliary": args.auxiliary,
        "samples": [],
        "work": [],
    }

    def execute(shared: bool, repeat: int, *, traced: bool = False) -> None:
        if shared:
            os.environ.pop("GENERATIVEQC_DF_SHARED_SOURCE", None)
        else:
            os.environ["GENERATIVEQC_DF_SHARED_SOURCE"] = "0"
        trace = args.output.with_name(f"{args.output.stem}-{int(shared)}-trace.jsonl")
        if traced:
            trace.unlink(missing_ok=True)
            os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"] = str(trace)
        else:
            os.environ.pop("GENERATIVEQC_DF_PROGRESS_TRACE", None)
        started = time.perf_counter()
        owner = Calculator(
            method=args.method,
            basis=orbital,
            auxiliary_basis=auxiliary,
            density_fitting="full_cuda",
            device="cuda",
            precision="fp64",
            correlation_memory_budget_bytes=2 << 30,
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-11,
        )
        try:
            for phase in ("cold", "warm"):
                if phase == "warm":
                    started = time.perf_counter()
                result = owner.singlepoint(atoms, properties=("energy", "forces"))
                seconds = time.perf_counter() - started
                energy_error = abs(result.energy - expected)
                derivative_error = abs(-np.vdot(result.forces, direction) - slope)
                assert (
                    result.converged and energy_error < 5e-9 and derivative_error < 3e-6
                )
                if repeat >= 0 and not traced:
                    row = {
                        "shared": shared,
                        "repeat": repeat,
                        "phase": phase,
                        "seconds": seconds,
                        "energy_error": energy_error,
                        "directional_force_error": float(derivative_error),
                    }
                    payload["samples"].append(row)
                    print(json.dumps(row), flush=True)
                if traced:
                    break
        finally:
            owner.clear_cache()
        if traced:
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            values = {
                row["key"]: row["value"]
                for row in records
                if row.get("name") == "df_correlation_source"
                and row.get("event") == "VALUE"
            }
            assert values["source_constructions"] == int(not shared)
            assert values["metric_factorizations"] == int(not shared)
            payload["work"].append(
                {"shared": shared, "values": values, "trace": trace.name}
            )
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for shared in (False, True):
        execute(shared, -1)
    for repeat in range(args.repeats):
        for shared in (False, True) if repeat % 2 == 0 else (True, False):
            execute(shared, repeat)
    for shared in (False, True):
        execute(shared, 0, traced=True)


if __name__ == "__main__":
    main()
