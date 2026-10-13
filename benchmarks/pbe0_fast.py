"""Compare budget-selected Fast against already-optimized FP64 endpoints.

An independently produced pbe0_precision reference is mandatory; no oracle or
alternate initial density enters native production or its timing interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

from generativeqc import Calculator, GridSpec, InitialGuessSpec, KsOptions

from benchmarks._cases import benchmark_cases
from benchmarks._support import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.pbe0_precision import CASES, errors, native_sample, protocol

MODES = ("fp64", "fast")
CONTROL = "GENERATIVEQC_KS_RESIDENT_VALUES"


def oracle_gate(mode: str, measured: dict, scientific: dict) -> bool:
    """Freeze separate strict/mixed force gates before collecting timings."""
    maximum = 1e-5 if mode == "fast" else scientific["force_gate_Eh_per_bohr"]
    return (
        measured["energy_Eh"] <= scientific["energy_gate_Eh"]
        and measured["max_force_Eh_per_bohr"] <= maximum
        and measured["rms_force_Eh_per_bohr"] <= 3e-6
    )


def sample(mode: str, options: dict[str, Any], case_name: str) -> dict[str, Any]:
    """Use real public endpoints without process-global arithmetic overrides."""
    if mode not in MODES:
        raise ValueError("unknown Fast benchmark mode")
    if CONTROL in os.environ:
        raise ValueError("benchmark requires unset resident-source override")
    row = native_sample(mode, options, case_name)
    row["mode"] = mode
    return row


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep first calls and failed samples visible; never select by iteration count."""
    measured = [record for record in records if record["phase"] == "measured"]
    if not measured or not all(record["oracle_gate"] for record in records):
        raise ValueError("missing or rejected independent endpoint evidence")
    times = {
        mode: median(record["seconds"] for record in measured if record["mode"] == mode)
        for mode in MODES
    }
    return {
        "median_seconds": times,
        "speedup_vs_current_fp64": {
            mode: times["fp64"] / times[mode] for mode in MODES
        },
    }


def main() -> None:
    """Persist every first call, repeat, numerical rejection and source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument(
        "--initial-guess", choices=("hcore", "automatic", "minao"), default="hcore"
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 1:
        parser.error("requires finite srun and positive repeat count")
    oracle = json.loads(args.reference.read_text())
    if oracle.get("engine") != "reference" or oracle.get("status") != "measured":
        parser.error("requires a completed independent reference")
    spec = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    report: dict[str, Any] = {
        "schema": "generativeqc.pbe0-fast.v3",
        "status": "running",
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing_contract": "fresh calculator, SCF, analytic forces, host export, release, CUDA synchronization",
        "cache_contract": f"warm process caches, fresh {args.initial_guess} seed on every endpoint",
        "arithmetic": "Fast: budget-selected FP32 XC warmup and stationary gradients, strict FP64 final SCF; FP64: ordinary optimized public baseline",
        "mixed_gates": {
            "energy_Eh": 1e-7,
            "max_force_Eh_per_bohr": 1e-5,
            "rms_force_Eh_per_bohr": 3e-6,
        },
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        for case_name in args.cases:
            scientific = protocol(case_name, spec)
            if scientific != oracle["cases"][case_name]["protocol"]:
                raise ValueError("independent reference scientific target differs")
            # Both timed modes use the same seed policy. The independent oracle
            # may reach this target from Hcore; it never supplies a native seed.
            oracle_protocol = dict(scientific)
            scientific["initial_guess"] = args.initial_guess
            case = benchmark_cases()[case_name]
            options = {
                "method": "pbe0-rks",
                "basis": case.generativeqc_basis,
                "device": "cuda",
                "basis_representation": case.basis_representation,
                "ks_options": KsOptions(grid=spec),
                **{
                    key: scientific[key]
                    for key in (
                        "energy_tolerance",
                        "density_tolerance",
                        "screening_tolerance",
                        "max_iterations",
                    )
                },
            }
            if args.initial_guess != "automatic":
                options["initial_guess"] = (
                    InitialGuessSpec("minao") if args.initial_guess == "minao" else None
                )
            probe = Calculator(**options)
            build = native_build_metadata(probe)
            require_tuned_native_build(build)
            probe.clear_cache()
            entry = {
                "protocol": scientific,
                "oracle_protocol": oracle_protocol,
                "build": build,
                "records": [],
            }
            report["cases"][case_name] = entry
            save()
            for repeat in range(-1, args.repeats):
                for mode in MODES if repeat % 2 else tuple(reversed(MODES)):
                    print(case_name, repeat, mode, flush=True)
                    row = sample(mode, options, case_name)
                    row.update(
                        repeat=repeat, phase="first-call" if repeat < 0 else "measured"
                    )
                    row["oracle_errors"] = errors(
                        row, oracle["cases"][case_name]["reference"]
                    )
                    row["oracle_gate"] = oracle_gate(
                        mode, row["oracle_errors"], scientific
                    )
                    entry["records"].append(row)
                    save()
                    if not row["oracle_gate"]:
                        raise ValueError("independent numerical gate failed")
                    if row["precision_evidence_error"]:
                        raise ValueError(row["precision_evidence_error"])
                    print(row["seconds"], row["oracle_errors"], flush=True)
            entry["summary"] = summarize(entry["records"])
            save()
        report["status"] = "measured"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
