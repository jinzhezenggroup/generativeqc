"""Complete same-source DF E+F qualification for bounded host raw panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from generativeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks._retention import raw_output_path


def main() -> None:
    """Time fresh owners after initialization, then retain independent work receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="water-tetramer-def2-svp-spherical")
    parser.add_argument("--budget-mib", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 1:
        parser.error("requires finite Slurm allocation and positive repeat count")
    from pyscf import gto, scf

    atoms = benchmark_cases()[args.case].atoms
    moved = [
        (symbol, tuple(np.asarray(xyz) + (0, 0, 0.025) if i == 1 else xyz))
        for i, (symbol, xyz) in enumerate(atoms)
    ]
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis="def2-svp")
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        references.append((mf.e_tot, -mf.nuc_grad_method().kernel()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    payload = {
        "source_sha": args.source_sha,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "case": args.case,
        "budget_mib": args.budget_mib,
        "timing": "Fresh complete owners after unreported per-policy initialization; force endpoints; trace disabled for timing.",
        "samples": [],
        "work": [],
    }

    def execute(
        policy: str, storage: str, repeat: int, *, traced: bool = False
    ) -> None:
        os.environ["GENERATIVEQC_DF_VALUE_STORAGE"] = storage
        if policy == "auto":
            os.environ.pop("GENERATIVEQC_DF_HOST_MAX_BYTES", None)
        else:
            os.environ["GENERATIVEQC_DF_HOST_MAX_BYTES"] = policy
        trace = args.output.with_name(
            f"{args.output.stem}-{storage}-{policy}-trace.jsonl"
        )
        if traced:
            trace.unlink(missing_ok=True)
            os.environ["GENERATIVEQC_DF_TRACE"] = str(trace)
        else:
            os.environ.pop("GENERATIVEQC_DF_TRACE", None)
        preparation_started = time.perf_counter()
        calculator = Calculator(
            method="rhf",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            density_fitting="cuda",
            density_fitting_memory_budget_bytes=args.budget_mib << 20,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            max_iterations=100,
        )
        with calculator.prepare_batch([atoms]) as batch:
            preparation_seconds = time.perf_counter() - preparation_started
            for phase, geometry, ref in (
                ("cold", None, references[0]),
                ("warm", None, references[0]),
                ("moved", [np.array([xyz for _, xyz in moved])], references[1]),
            ):
                start = time.perf_counter()
                item = batch.execute(
                    geometry, properties=("energy", "forces"), strict=True
                ).items[0]
                elapsed = time.perf_counter() - start
                if phase == "cold":
                    elapsed += preparation_seconds
                energy_error = abs(item.energy - ref[0])
                force_error = float(np.max(abs(item.forces - ref[1])))
                assert energy_error < 1e-9 and force_error < 1e-8
                if repeat >= 0 and not traced:
                    row = {
                        "policy": policy,
                        "storage": storage,
                        "repeat": repeat,
                        "phase": phase,
                        "seconds": elapsed,
                        "preparation_seconds": preparation_seconds
                        if phase == "cold"
                        else 0,
                        "iterations": item.iterations,
                        "fock_builds": item.fock_builds,
                        "energy": item.energy,
                        "forces": item.forces.tolist(),
                        "energy_error": energy_error,
                        "force_error": force_error,
                    }
                    payload["samples"].append(row)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "forces"}),
                        flush=True,
                    )
                if traced:
                    diagnostics = [
                        d.to_dict()
                        for d in batch.last_density_fitting_metric_diagnostics()
                    ]
                    break
        if traced:
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            counts = {}
            for row in records:
                for key, value in row.get("counters", {}).items():
                    if key in (
                        "host_raw_panel_hits",
                        "host_raw_panel_h2d_bytes",
                        "raw_panel_source_auxiliary_evaluations",
                        "fused_source_auxiliary_evaluations",
                    ):
                        counts[key] = counts.get(key, 0) + value
            payload["work"].append(
                {
                    "policy": policy,
                    "storage": storage,
                    "counters": counts,
                    "trace": trace.name,
                    "df_diagnostics": diagnostics,
                    "execution_records": dict(Counter(r["execution"] for r in records)),
                }
            )
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    # Warm ordinary process/library initialization in every compared policy.
    if not args.trace_only:
        for policy, storage in (
            ("0", "dense"),
            ("auto", "dense"),
            ("0", "auto"),
            ("auto", "auto"),
        ):
            execute(policy, storage, -1)
        for repeat in range(args.repeats):
            for policy in ("0", "auto") if repeat % 2 == 0 else ("auto", "0"):
                execute(policy, "dense", repeat)
            for policy in ("0", "auto") if repeat % 2 == 0 else ("auto", "0"):
                execute(policy, "auto", repeat)
    for policy, storage in (
        ("0", "dense"),
        ("auto", "dense"),
        ("0", "auto"),
        ("auto", "auto"),
    ):
        execute(policy, storage, 0, traced=True)
    cached = next(
        r for r in payload["work"] if r["storage"] == "dense" and r["policy"] == "auto"
    )
    assert cached["counters"].get("host_raw_panel_hits", 0) > 0, "RAM was not consumed"


if __name__ == "__main__":
    main()
