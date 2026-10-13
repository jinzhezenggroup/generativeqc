"""Qualify exact-J/K FP64 SCF plus typed FP32 stationary-gradient candidates.

This private native-controller endpoint is not the public Fast policy and is
not interchangeable with the Python stationary endpoint in older campaigns.
Both comparators include the same complete moving-grid gradient, preparation,
SCF, exports and teardown. Instrumented same-state calls are kept out of clean
timings. No density fitting, relaxed SCF or screening changes are permitted.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc_compiler.integral.precision import weighted_force_precision_schedule

from benchmarks._cases import benchmark_cases
from benchmarks._retention import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import native_build_metadata
from benchmarks.pbe0_precision import errors, protocol, reference


class GradientProbe:
    """Bind the test-only bridge; returned float64 arrays do not imply FP64 compute."""

    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path.resolve()))
        self.function = self.library.mixed_force_endpoint_v2
        self.function.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        )
        self.function.restype = ctypes.c_int

    def __call__(
        self, batch: Any, atoms: int, mixed: bool | int, collect: bool = False
    ) -> tuple[np.ndarray, list[int]]:
        """Evaluate all gradient sources at the owner's exact final-state token."""
        forces = np.empty((atoms, 3))
        work = np.zeros(12, dtype=np.uint64)
        error = ctypes.create_string_buffer(512)
        status = self.function(
            batch._batch,
            mixed,
            collect,
            forces,
            forces.size,
            work,
            work.size,
            error,
            len(error),
        )
        if status:
            raise RuntimeError(error.value.decode())
        self.last_semantic_work = dict(
            zip(
                (
                    "points",
                    "pair_visits",
                    "pair_state_evaluations",
                    "ao_jet_values",
                    "projection_fma_pairs",
                    "projection_output_values",
                ),
                work[6:].tolist(),
                strict=True,
            )
        )
        return forces, work[:6].tolist()


def options(case_name: str, grid: GridSpec) -> dict[str, Any]:
    """Freeze the same non-DF PBE0 Hamiltonian and strict SCF controls."""
    case = benchmark_cases()[case_name]
    science = protocol(case_name, grid)
    return dict(
        method="pbe0-rks",
        precision="fp64",
        device="cuda",
        basis=case.generativeqc_basis,
        basis_representation=case.basis_representation,
        ks_options=KsOptions(grid=grid),
        initial_guess=None,
        **{
            key: science[key]
            for key in (
                "energy_tolerance",
                "density_tolerance",
                "screening_tolerance",
                "max_iterations",
            )
        },
    )


def sample(
    probe: GradientProbe,
    case_name: str,
    grid: GridSpec,
    mixed: bool | int,
    audit: bool = False,
) -> dict:
    """Fresh-owner, synchronized construction-to-release endpoint; no warm SCF."""
    case = benchmark_cases()[case_name]
    runtime = ctypes.CDLL("libcudart.so.12")
    if runtime.cudaDeviceSynchronize():
        raise RuntimeError("pre-endpoint synchronization failed")
    started = perf_counter()
    calculator = Calculator(**options(case_name, grid))
    try:
        with calculator.prepare_batch(
            [case.atoms], charges=[case.charge], multiplicities=[case.multiplicity]
        ) as batch:
            result = batch.execute(properties=("energy",), strict=True).items[0]
            scf_end = perf_counter()
            forces, work = probe(batch, len(case.atoms), mixed, audit)
            semantic_work = probe.last_semantic_work.copy()
            gradient_seconds = perf_counter() - scf_end
            if audit:
                strict, _ = probe(batch, len(case.atoms), False)
                replay, replay_work = probe(batch, len(case.atoms), mixed, True)
                final_strict, _ = probe(batch, len(case.atoms), False)
                errors = forces - strict
                qualification = {
                    "maximum_force_difference": float(np.max(np.abs(errors))),
                    "rms_force_difference": float(np.sqrt(np.mean(errors**2))),
                    "strict_replay_difference": float(
                        np.max(np.abs(strict - final_strict))
                    ),
                    "mixed_replay_difference": float(np.max(np.abs(forces - replay))),
                    "mixed_replay_work": replay_work,
                }
            else:
                qualification = None
    finally:
        calculator.clear_cache()
    if runtime.cudaDeviceSynchronize():
        raise RuntimeError("post-endpoint synchronization failed")
    return {
        "mixed_weighted_derivatives": bool(int(mixed) & 1),
        "precision_mask": int(mixed),
        "instrumented": audit,
        "seconds": perf_counter() - started,
        "gradient_seconds": gradient_seconds,
        "energy": result.energy,
        "forces": forces.tolist(),
        "iterations": result.iterations,
        "fock_builds": result.fock_builds,
        "scf_precision": result.precision,
        "mixed_tasks_primitives_retries": work,
        "same_state": qualification,
        "semantic_work": semantic_work,
        "gradient_endpoint_invocations": 4 if audit else 1,
    }


def qualify(report: dict) -> dict:
    """Gate all results before reporting clean same-route endpoint speed ratios.

    The strict comparator retains the reference protocol's tighter force gate.
    Mixed gates describe this measured domain, not a general error guarantee.
    Instrumented and warmup endpoints must pass numerically but cannot improve
    timing medians. Counts expose empty mixed routes and differing SCF work.
    """
    qualification = report["qualification"]
    candidate_mask = qualification.get("precision_mask", 1)
    for row in [qualification, *report["samples"]]:
        if row.get("precision_mask", int(row["mixed_weighted_derivatives"])) not in (
            0,
            candidate_mask,
        ):
            raise ValueError("candidate arithmetic changed between samples")
        measured = errors(row, report["reference"])
        maximum = (
            report["gates"]["maximum_force"]
            if row.get("precision_mask", row["mixed_weighted_derivatives"])
            else report["protocol"]["force_gate_Eh_per_bohr"]
        )
        if (
            measured["energy_Eh"] > report["gates"]["energy"]
            or measured["max_force_Eh_per_bohr"] > maximum
            or measured["rms_force_Eh_per_bohr"] > report["gates"]["rms_force"]
        ):
            raise ValueError("independent energy/force gate failed")
        if row["scf_precision"]["mixed_stage_fock_builds"] != 0:
            raise ValueError("gradient qualification requires unchanged FP64 SCF")
        if any(row[key] != qualification[key] for key in ("iterations", "fock_builds")):
            raise ValueError("strict/mixed endpoint SCF work differs")
        if (
            "semantic_work" in qualification
            and row.get("semantic_work") != qualification["semantic_work"]
        ):
            raise ValueError("strict/mixed endpoint geometry or grid work differs")
        row["oracle_errors"] = measured
    work = qualification["mixed_tasks_primitives_retries"]
    same_state = qualification["same_state"]
    mask = qualification.get("precision_mask", 1)
    if (
        (mask & 1 and (work[0] <= 0 or work[1] <= 0 or work[2] != 0))
        or (mask & 2 and (len(work) < 4 or work[3] <= 0))
        or (mask & 4 and (len(work) < 5 or work[4] <= 0))
        or (len(work) > 5 and work[5])
    ):
        raise ValueError("empty mixed work or unqualified ordinary-case retry")
    if same_state["mixed_replay_work"] != work:
        raise ValueError("mixed replay work differs")
    for key, limit in (
        ("maximum_force_difference", report["gates"]["maximum_force"]),
        ("rms_force_difference", report["gates"]["rms_force"]),
        ("strict_replay_difference", 1e-10),
        ("mixed_replay_difference", 1e-10),
    ):
        if not np.isfinite(same_state[key]) or same_state[key] > limit:
            raise ValueError("same-state or replay gate failed")
    timing = {}
    for mixed, name in ((False, "strict"), (True, "mixed")):
        rows = [
            row
            for row in report["samples"]
            if bool(row.get("precision_mask", row["mixed_weighted_derivatives"]))
            == mixed
            and not row["excluded_warmup_pair"]
            and not row["instrumented"]
        ]
        if not rows:
            raise ValueError("missing clean strict/mixed endpoint samples")
        timing[name] = {}
        for key in ("seconds", "gradient_seconds"):
            values = [row[key] for row in rows]
            if not all(np.isfinite(value) and value > 0 for value in values):
                raise ValueError("invalid endpoint timing")
            timing[name][key] = {
                "median": median(values),
                "minimum": min(values),
                "maximum": max(values),
                "count": len(values),
            }
    return {
        "timing": timing,
        "speedup": {
            key: timing["strict"][key]["median"] / timing["mixed"][key]["median"]
            for key in ("seconds", "gradient_seconds")
        },
    }


def main() -> None:
    """Persist every endpoint before applying independent numerical gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=benchmark_cases())
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--precision-mask",
        type=int,
        choices=range(1, 8),
        default=7,
        help="private candidate bits: weighted ERI=1, AO/XC/Becke pullback=2, grid=4",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 1:
        parser.error("requires finite srun and positive repeats")
    grid = GridSpec(48, 16, 32)
    probe = GradientProbe(args.probe)
    if args.reference:
        retained = json.loads(args.reference.read_text())
        if retained.get("protocol") != protocol(args.case, grid):
            parser.error(
                "retained reference does not match the complete scientific protocol"
            )
        oracle = retained["reference"]
    else:
        oracle = reference(args.case, grid, 4)
    calculator = Calculator(**options(args.case, grid))
    report = {
        "scope": "private-complete-native-stationary-endpoint-not-public-Fast",
        "protocol": protocol(args.case, grid),
        "build": native_build_metadata(calculator),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reference": oracle,
        "reference_file_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()
        if args.reference
        else None,
        "samples": [],
        "status": "pending",
        "gates": {"energy": 1e-7, "maximum_force": 1e-5, "rms_force": 3e-6},
        "candidate_precision_schedule": weighted_force_precision_schedule(
            mixed_recurrence=bool(args.precision_mask & 1),
            qualification="private-gradient-qualification-only",
        ).to_payload(),
        "candidate_precision_mask": args.precision_mask,
        "work_receipt_fields": [
            "weighted_tasks",
            "weighted_primitives",
            "weighted_retries",
            "geometry_points",
            "grid_points",
            "complete_gradient_retries",
        ],
        "fp32_regions": [
            *(
                ["weighted-ERI-derivatives-orders-0-through-5"]
                if args.precision_mask & 1
                else []
            ),
            *(
                [
                    "AO-translation-pullback",
                    "XC-interior-exchange-and-correlation",
                    "Becke-point-pair-forward-and-reverse",
                ]
                if args.precision_mask & 2
                else []
            ),
            *(
                ["AO-spatial-jets", "TensorIR-density-projection", "density-features"]
                if args.precision_mask & 4
                else []
            ),
        ],
        "strict_remainder": [
            "SCF",
            *([] if args.precision_mask & 2 else ["XC-and-Becke-pullback"]),
            *([] if args.precision_mask & 4 else ["grid-AO-and-density-projection"]),
            "coordinate-differences/physical-normalization/global-force-sums",
            "XC-boundary-and-density-feature-range-fallbacks",
            "one-electron/Pulay",
            "order-6-and-higher-ERI-derivatives",
        ],
    }
    calculator.clear_cache()

    def save() -> None:
        output = raw_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["qualification"] = sample(
            probe, args.case, grid, args.precision_mask, True
        )
        save()
        for repeat in range(args.repeats + 1):
            modes = (
                (0, args.precision_mask)
                if repeat % 2 == 0
                else (args.precision_mask, 0)
            )
            for mixed in modes:
                row = sample(probe, args.case, grid, mixed)
                row["excluded_warmup_pair"] = repeat == 0
                report["samples"].append(row)
                save()
                print(mixed, row["seconds"], row["gradient_seconds"], flush=True)
        report["summary"] = qualify(report)
        report["status"] = "numerically-qualified-private-probe"
    except Exception as failure:
        report["status"] = "failed"
        report["error"] = repr(failure)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
