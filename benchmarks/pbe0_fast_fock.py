"""Measure budget-selected FP32 J/K against optimized strict PBE0 endpoints.

All CUDA execution requires a finite Slurm allocation. A separate GPU4PySCF
process owns the strict oracle; source-matched native calls never supply it.
Keep first calls and alternating repeats, including failed attempts. Timings
include construction, SCF, complete moving-grid forces, export and teardown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from statistics import median
from typing import Any

from generativeqc import Calculator, GridSpec

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import native_build_metadata
from benchmarks.pbe0_mixed_gradient import options
from benchmarks.pbe0_precision import errors, native_sample, protocol, reference


def portable_report(report: dict) -> dict:
    """Preserve rejected nonfinite diagnostics as null plus explicit value tags.

    Qualification always examines the original endpoint, never this JSON-only
    representation. A rejected native iteration may legitimately contain NaN.
    """
    nonfinite = {}

    def convert(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: convert(child, f"{path}.{key}") for key, child in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                convert(child, f"{path}[{index}]") for index, child in enumerate(value)
            ]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] = repr(value)
            return None
        return value

    portable = convert(report, "$report")
    if nonfinite:
        portable["nonfinite_values"] = nonfinite
    return portable


def check_sample(row: dict, oracle: dict, mode: str) -> dict:
    """Require independent observables and actual nonempty FP32 exchange work."""
    measured = errors(row, oracle)
    limits = (1e-7, 1e-6, 3e-6) if mode == "fp64" else (1e-5, 1e-4, 3e-5)
    if any(
        not math.isfinite(measured[name]) or measured[name] > limit
        for name, limit in zip(
            ("energy_Eh", "max_force_Eh_per_bohr", "rms_force_Eh_per_bohr"),
            limits,
            strict=True,
        )
    ):
        raise RuntimeError(f"{mode} independent energy/force budget failed: {measured}")
    if row.get("precision_evidence_error"):
        raise RuntimeError(row["precision_evidence_error"])
    if mode == "fast":
        validation = row.get("precision_validation") or {}
        if validation.get("status") != "qualified_budget_fock":
            raise RuntimeError("Fast did not execute an admitted budget-Fock endpoint")
        if len(validation.get("solves", ())) != 1:
            raise RuntimeError("Fast used an additional full endpoint")
        work = row["precision"]
        for name in ("coulomb_j", "exchange_k", "coulomb_recurrence"):
            if not any(
                operator.get("name") == name
                and operator.get("compute") == "fp32"
                and operator.get("accumulation") == "fp64"
                and operator.get("count", 0) > 0
                for operator in work.get("operators", ())
            ):
                raise RuntimeError(f"missing executed FP32 {name}")
    return measured


def main() -> None:
    """Retain endpoint evidence before evaluating any performance claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("reference", "native"), required=True)
    parser.add_argument(
        "--case",
        choices=tuple(benchmark_cases()),
        default="water-32mer-4s4-def2-svp-spherical",
    )
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("requires finite srun allocation")
    if args.output.exists() or args.repeats < 1:
        parser.error("output must be new and repeats must be positive")
    grid = GridSpec(48, 16, 32)
    science = protocol(args.case, grid)
    report = {
        "schema": "generativeqc.pbe0-budget-fock.v1",
        "status": "running",
        "protocol": science,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing_scope": "fresh-owner complete energy/force endpoint; Hcore seed",
        "performance_claim": False,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(
            json.dumps(portable_report(report), indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        if args.engine == "reference":
            report["reference"] = reference(args.case, grid, 4, backend="gpu4pyscf")
            report["status"] = "reference-completed"
        else:
            if args.oracle is None:
                raise ValueError("native measurement requires --oracle")
            independent = json.loads(args.oracle.read_text())
            if independent.get("status") not in (
                "reference-completed",
                "completed-unqualified",
            ):
                raise ValueError("independent reference did not complete")
            if independent["protocol"] != science:
                raise ValueError("independent scientific protocol differs")
            oracle = independent["reference"]
            if (
                not oracle.get("converged")
                or oracle.get("reference_backend") != "gpu4pyscf"
            ):
                raise ValueError("a converged independent GPU4PySCF oracle is required")
            report["oracle_sha256"] = hashlib.sha256(
                args.oracle.read_bytes()
            ).hexdigest()
            report["native_library_sha256"] = hashlib.sha256(
                Path(os.environ["GENERATIVEQC_LIBRARY"]).read_bytes()
            ).hexdigest()
            settings = options(args.case, grid)
            settings.pop("precision")
            calculator = Calculator(**settings, precision="fp64")
            try:
                report["build"] = native_build_metadata(calculator)
            finally:
                calculator.clear_cache()
            for repeat in range(-1, args.repeats):
                modes = ("fp64", "fast") if repeat % 2 else ("fast", "fp64")
                for mode in modes:
                    selected = dict(settings)
                    if mode == "fast":
                        for control in (
                            "energy_tolerance",
                            "density_tolerance",
                            "screening_tolerance",
                        ):
                            selected.pop(control)
                    row = native_sample(mode, selected, args.case)
                    row.update(
                        repeat=repeat, phase="first-call" if repeat < 0 else "measured"
                    )
                    report["records"].append(row)
                    save()
                    row["oracle_errors"] = check_sample(row, oracle, mode)
                    row["oracle_gate"] = True
                    save()
                    print(
                        mode, repeat, row["seconds"], row["oracle_errors"], flush=True
                    )
            times = {
                mode: median(
                    row["seconds"]
                    for row in report["records"]
                    if row["mode"] == mode and row["repeat"] >= 0
                )
                for mode in ("fp64", "fast")
            }
            report.update(
                status="measured",
                median_seconds=times,
                speedup=times["fp64"] / times["fast"],
                performance_claim=True,
            )
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
