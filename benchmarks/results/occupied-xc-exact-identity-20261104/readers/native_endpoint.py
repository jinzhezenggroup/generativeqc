"""Measure admission-inclusive native E+F without claiming default promotion.

Run in a finite Slurm allocation with the source-matched native library selected.
Independent default-grid PySCF vectors are inputs, never a production fallback.
Fresh-owner cold measurements include construction and preparation, but share a
process, CUDA context and existing compiler/artifact caches. They are not process
startup or cache-miss compilation measurements.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from generativeqc import Calculator
from test_dft_complete_cuda import no_cpu_derivatives

from benchmarks.readme_hf_scaling import scaling_cases


def main() -> None:
    """Retain every interleaved cold/warm/moved observation and unchanged gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        parser.error("a finite Slurm GPU allocation is required")
    if args.repeats < 10 or args.output.exists():
        parser.error("require at least ten repetitions and a fresh output path")
    references = {
        (row["spin"], row["geometry"]): row
        for row in json.loads(args.oracle.read_text())
    }
    atoms = scaling_cases()["water-3"].atoms
    moved = np.asarray([position for _, position in atoms]).copy()
    moved[1, 2] += 0.02
    record = {
        "schema": "generativeqc.occupied-xc-exact-endpoint.v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "library": os.environ["GENERATIVEQC_LIBRARY"],
        "repeats": args.repeats,
        "energy_gate_eh": 2e-9,
        "force_gate_eh_per_bohr": 1e-7,
        "observations": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for spin in ("rks", "uks"):
        charge = int(spin == "uks")
        for repetition in range(args.repeats):
            paired = {}
            for selector in ("0", "1") if repetition % 2 == 0 else ("1", "0"):
                os.environ["GENERATIVEQC_CUDA_KS_OCCUPIED_XC"] = selector
                started = perf_counter()
                calculator = Calculator(
                    method=f"r2scan-{spin}",
                    device="cuda",
                    basis="def2-svp",
                    basis_representation="spherical",
                    energy_tolerance=1e-12,
                    density_tolerance=1e-10,
                    max_iterations=200,
                )
                constructed = perf_counter()
                try:
                    with calculator.prepare_batch(
                        [atoms],
                        charges=[charge],
                        multiplicities=[charge + 1],
                        warm_start=True,
                    ) as owner:
                        prepared = perf_counter()
                        for phase in ("cold", "warm", "moved", "moved_warm"):
                            geometry = int(phase.startswith("moved"))
                            reference = references[spin, geometry]
                            with no_cpu_derivatives():
                                execution_started = perf_counter()
                                result = owner.execute(
                                    [moved] if geometry else None,
                                    properties=("energy", "forces"),
                                    strict=True,
                                ).items[0]
                                execution_seconds = perf_counter() - execution_started
                            difference = np.asarray(result.forces) - reference["forces"]
                            row = {
                                "spin": spin,
                                "repeat": repetition,
                                "selector": selector,
                                "phase": phase,
                                "execution_seconds": execution_seconds,
                                "energy": result.energy,
                                "forces": np.asarray(result.forces).tolist(),
                                "iterations": result.iterations,
                                "fock_builds": result.fock_builds,
                                "ks_diagnostic": (
                                    asdict(result.ks_diagnostic)
                                    if result.ks_diagnostic is not None
                                    else None
                                ),
                                "density_rms": result.density_rms,
                                "physical_residual_rms": result.physical_residual_rms,
                                "energy_error": abs(
                                    result.energy - reference["energy"]
                                ),
                                "force_max_error": float(np.max(np.abs(difference))),
                                "force_rms_error": float(
                                    np.sqrt(np.mean(difference**2))
                                ),
                                "resource_diagnostics": copy.deepcopy(
                                    owner.resource_diagnostics
                                ),
                            }
                            if phase == "cold":
                                row.update(
                                    construction_seconds=constructed - started,
                                    preparation_seconds=prepared - constructed,
                                    complete_seconds=prepared
                                    - started
                                    + execution_seconds,
                                )
                            record["observations"].append(row)
                            args.output.write_text(json.dumps(record, indent=2) + "\n")
                            assert result.succeeded and result.converged
                            assert result.executed_backend == "cuda"
                            assert row["energy_error"] <= record["energy_gate_eh"]
                            assert (
                                row["force_max_error"]
                                <= record["force_gate_eh_per_bohr"]
                            )
                            if phase in paired:
                                other = paired[phase]
                                assert result.iterations == other["iterations"]
                                assert abs(result.energy - other["energy"]) <= 2e-9
                                np.testing.assert_allclose(
                                    result.forces, other["forces"], atol=1e-7, rtol=0
                                )
                            else:
                                paired[phase] = row
                finally:
                    calculator.clear_cache()
    record["completed"] = True
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
