"""Time complete production bounded RHF E+F phases, never isolated Fock replay.

Each process loads one provider DSO. A paired launcher must use the same Slurm
device and interleave baseline/candidate processes. Optional PySCF E+F work is
an independent acceptance oracle, explicitly outside all production timing.
"""

from __future__ import annotations

import argparse
import csv
import ctypes as ct
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from benchmarks._retention import raw_output_path


def digest(path: Path) -> str:
    """Stream large native DSOs without charging an endpoint-sized host buffer."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def gpu_identity() -> dict:
    """Bind paired receipts to the physical Slurm-assigned device, not index zero."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={os.environ['CUDA_VISIBLE_DEVICES']}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("endpoint requires exactly one identified Slurm GPU")
    return dict(zip(["uuid", "name", "driver_version"], rows[0], strict=True))


def measure(args: argparse.Namespace) -> dict:
    """Retain progress, exceptions, full forces and exact Fock/iteration counts."""
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("requires finite srun with preserved GPU visibility")
    if (
        not 1 <= args.repeats <= 32
        or args.budget_bytes <= 0
        or args.max_iterations <= 0
    ):
        raise ValueError("invalid bounded endpoint request")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("invalid SCF tolerance")
    from benchmarks.rhf_resident_jk import read_input
    from tools.generativeqc_posthf.fixtures import source_arguments
    from tools.generativeqc_posthf.sources import NativeSource

    input_path, probe_path = (
        args.input.resolve(strict=True),
        args.probe.resolve(strict=True),
    )
    metadata = read_input(input_path, 0.0)
    library = ct.CDLL(str(probe_path))
    pointer = ct.POINTER(ct.c_double)
    create = library.selective_eri_rhf_create
    create.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_uint,
        ct.c_double,
        ct.POINTER(ct.c_void_p),
        ct.c_void_p,
        ct.c_size_t,
    ]
    create.restype = ct.c_int
    run = library.selective_eri_rhf_run
    run.argtypes = [
        ct.c_void_p,
        ct.c_uint,
        pointer,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_uint64),
        pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    run.restype = ct.c_int
    destroy = library.selective_eri_rhf_destroy
    destroy.argtypes = [ct.c_void_p]
    destroy.restype = None
    provider = library.selective_eri_rhf_native_library
    provider.restype = ct.c_char_p
    report = {
        "schema": "selective-eri-rhf-endpoint-v1",
        "status": "running",
        "samples": [],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": gpu_identity(),
        "input_sha256": digest(input_path),
        "probe_sha256": digest(probe_path),
        "budget_bytes": args.budget_bytes,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "precision": "fp64",
        "screening_tolerance": 0.0,
        "method": "bounded-physical-reference-RHF",
        "moved_displacement_bohr": 0.05,
        "phase_scope": args.phase_scope,
        "controls": {
            name: os.environ.get(name)
            for name in [
                "GENERATIVEQC_RHF_RESIDENT_VALUES",
                "GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK",
                "GENERATIVEQC_RHF_RESIDENT_MAX_BYTES",
            ]
        },
        "timing_scope": "complete run_rhf_cuda_cached SCF + reference validation + "
        "rhf_frame_response_cuda electronic gradient + nuclear assembly; "
        "cold also includes native metadata/context construction; "
        "excludes process startup/input deserialization/PySCF oracle",
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for repeat in range(args.repeats):
            context = ct.c_void_p()
            error = ct.create_string_buffer(4096)
            cold_started = time.perf_counter()
            with NativeSource(**source_arguments(metadata)) as source:
                source_seconds = time.perf_counter() - cold_started
                native = Path(source._library._name).resolve(strict=True)
                linked = provider()
                if (
                    not linked
                    or Path(os.fsdecode(linked)).resolve(strict=True) != native
                ):
                    raise ValueError("mixed native endpoint/provider identity")
                report.update(
                    native_sha256=digest(native),
                    native_library=str(native),
                    geometry_hash=source.geometry_hash,
                    basis_hash=source.basis_hash,
                    nbf=source.nbf,
                )
                cold_started = time.perf_counter() - source_seconds
                if create(
                    source._handle,
                    args.budget_bytes,
                    args.max_iterations,
                    args.tolerance,
                    ct.byref(context),
                    error,
                    len(error),
                ):
                    raise RuntimeError(error.value.decode())
                try:
                    phases = ["cold", "warm", "moved", "moved-warm"]
                    if args.phase_scope == "cold-only":
                        phases = phases[:1]
                    for phase, name in enumerate(phases):
                        energy = ct.c_double(math.nan)
                        forces = (
                            ct.c_double
                            * (3 * len(metadata["inputs"]["atomic_numbers"]))
                        )()
                        counts, residuals = (ct.c_uint64 * 9)(), (ct.c_double * 3)()
                        started = cold_started if phase == 0 else time.perf_counter()
                        if run(
                            context,
                            phase,
                            ct.byref(energy),
                            forces,
                            len(forces),
                            counts,
                            residuals,
                            error,
                            len(error),
                        ):
                            raise RuntimeError(f"{name}: {error.value.decode()}")
                        report["samples"].append(
                            {
                                "repeat": repeat,
                                "phase": name,
                                "seconds": time.perf_counter() - started,
                                "energy": energy.value,
                                "forces": list(forces),
                                "iterations": counts[0],
                                "physical_fock_builds": counts[1],
                                "execution_plan_reused": bool(counts[2]),
                                "response_jk_actions": counts[3],
                                "response_derivative_passes": counts[4],
                                "response_orbital_actions": counts[5],
                                "response_quartet_visits": counts[6],
                                "response_eri_evaluations": counts[7],
                                "response_resident_bytes": counts[8],
                                "energy_change": residuals[0],
                                "density_rms": residuals[1],
                                "physical_residual_rms": residuals[2],
                            }
                        )
                        save()
                finally:
                    destroy(context)
        report["status"] = "complete"
    except BaseException as failure:
        report.update(status="failed", failure=str(failure))
        raise
    finally:
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--budget-bytes", type=int, default=2 << 30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--phase-scope", choices=["complete", "cold-only"], default="complete"
    )
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
