"""Separate public Fast fallback from experimental XC-only size scaling.

The experimental route does not widen production admission. Both timed routes
start from Hcore and include strict SCF refinement and all analytic forces.
An independent moving-grid PySCF reference is mandatory for accepted timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks._cases import benchmark_cases
from benchmarks._support import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import (
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.pbe0_precision import errors, native_sample, protocol, reference

if TYPE_CHECKING:
    from collections.abc import Iterator

CASES = (
    "water-octamer-s4-def2-svp-spherical",
    "water-hexadecamer-2s4-def2-svp-spherical",
    "water-32mer-4s4-def2-svp-spherical",
)
CONTROLS = (
    "GENERATIVEQC_EXPERIMENTAL_PBE_POINT_FP32",
    "GENERATIVEQC_EXPERIMENTAL_PBE_POINT_KEEP_DIIS",
)
CONFLICTS = (
    *CONTROLS,
    "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL",
    "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS",
    "GENERATIVEQC_KS_RESIDENT_VALUES",
)
MODES = ("fp64", "experimental_xc", "fast")


@contextmanager
def experimental_mode(mode: str) -> Iterator[str]:
    """Keep diagnostic overrides scoped and never leak them into public Fast."""
    if mode not in MODES:
        raise ValueError("unknown scaling mode")
    if any(name in os.environ for name in CONFLICTS):
        raise ValueError(
            "scaling experiment requires unset precision/residency overrides"
        )
    if mode != "experimental_xc":
        yield mode
        return
    try:
        os.environ.update(dict(zip(CONTROLS, ("3", "1"), strict=True)))
        yield "auto"
    finally:
        for name in CONTROLS:
            os.environ.pop(name, None)


def summarize(rows: list[dict], modes: tuple[str, ...]) -> dict:
    """Reject missing or failed evidence; retain every sample including outliers."""
    if not rows or not all(row["oracle_gate"] for row in rows):
        raise ValueError("missing or rejected independent endpoint evidence")
    times = {
        mode: median(
            row["seconds"]
            for row in rows
            if row["mode"] == mode and row["phase"] == "measured"
        )
        for mode in modes
    }
    return {
        "median_seconds": times,
        "speedup_vs_fp64": {
            mode: times["fp64"] / value for mode, value in times.items()
        }
        if "fp64" in times
        else None,
    }


def main() -> None:
    """Run one size at a time so timeouts and first-call costs remain visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("reference", "native"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES[:2])
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("requires positive repeat and thread counts")
    if args.engine == "native" and (
        not os.environ.get("SLURM_JOB_ID") or not args.reference
    ):
        parser.error(
            "native execution requires finite srun and an independent reference"
        )
    grid = GridSpec(48, 16, 32)
    science = protocol(args.case, grid)
    report: dict[str, Any] = {
        "schema": "generativeqc.pbe0-fast-scaling.v1",
        "status": "running",
        "engine": args.engine,
        "protocol": science,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "experimental_scope": "diagnostic-only XC warmup; production AO cap unchanged",
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.engine == "reference":
            report["reference"] = reference(args.case, grid, args.threads)
        else:
            oracle = json.loads(args.reference.read_text())
            if oracle["status"] != "measured" or oracle["engine"] != "reference":
                raise ValueError("requires completed independent reference")
            if oracle["protocol"] != science:
                raise ValueError("reference scientific target differs")
            report["reference_sha256"] = hashlib.sha256(
                args.reference.read_bytes()
            ).hexdigest()
            case = benchmark_cases()[args.case]
            options = {
                "method": "pbe0-rks",
                "basis": case.generativeqc_basis,
                "basis_representation": case.basis_representation,
                "device": "cuda",
                "ks_options": KsOptions(grid=grid),
                "initial_guess": None,
                **{
                    key: science[key]
                    for key in (
                        "energy_tolerance",
                        "density_tolerance",
                        "screening_tolerance",
                        "max_iterations",
                    )
                },
            }
            probe = Calculator(**options)
            report["build"] = native_build_metadata(probe)
            require_tuned_native_build(report["build"])
            probe.clear_cache()
            report["timing_contract"] = (
                "fresh construction, SCF/refinement, analytic forces, export, release, CUDA synchronization"
            )
            report["cache_contract"] = (
                "warm process caches, fresh Hcore density per endpoint"
            )
            report["aos"] = case.expected_ao_count
            for repeat in range(-1, args.repeats):
                for mode in args.modes if repeat % 2 else reversed(args.modes):
                    print(args.case, repeat, mode, flush=True)
                    with experimental_mode(mode) as native_mode:
                        row = native_sample(native_mode, options, args.case)
                    row.update(
                        mode=mode,
                        repeat=repeat,
                        phase="first-call" if repeat < 0 else "measured",
                    )
                    row["oracle_errors"] = errors(row, oracle["reference"])
                    row["oracle_gate"] = (
                        row["oracle_errors"]["energy_Eh"] <= science["energy_gate_Eh"]
                        and row["oracle_errors"]["max_force_Eh_per_bohr"]
                        <= science["force_gate_Eh_per_bohr"]
                    )
                    report["records"].append(row)
                    save()
                    print(
                        row["seconds"],
                        row["iterations"],
                        row["oracle_errors"],
                        flush=True,
                    )
                    if not row["oracle_gate"] or row["precision_evidence_error"]:
                        raise ValueError(
                            "independent numerical or precision-work gate failed"
                        )
            report["summary"] = summarize(report["records"], tuple(args.modes))
        report["status"] = "measured"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
