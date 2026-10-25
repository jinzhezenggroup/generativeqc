"""Standard-only complete endpoint timing; retain work and startup separately."""

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.readme_cbs import checkpoint, digest, geometry, native_call, setup

parser = argparse.ArgumentParser()
parser.add_argument("--carbons", nargs="+", type=int, default=[1, 2, 4, 6])
parser.add_argument("--arms", nargs="+", default=["standard:auto"])
parser.add_argument("--samples", type=int, default=2)
parser.add_argument("--budget-mib", type=int, default=32768)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--patch", type=Path, required=True)
parser.add_argument("--energy-only", action="store_true")
parser.add_argument("--base-source", required=True)
args = parser.parse_args()
assert os.environ.get("SLURM_JOB_ID"), "real GPU endpoints require srun"
task_root = Path("/data/jzzeng/cbs-standard-20261025")
payload = {
    "schema": "generativeqc.cbs-independent-execution.v1",
    "host": platform.node(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "library_sha256": digest(Path(os.environ["GENERATIVEQC_LIBRARY"])),
    "source_artifact": str(args.patch),
    "source_artifact_sha256": digest(args.patch),
    "base_source": args.base_source,
    "driver_source": Path(__file__).read_text(),
    "gpu": subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip(),
    "protocols": {},
    "samples": [],
    "status": "running",
}
checkpoint(args.output, payload)


def endpoint(atoms, orbital, auxiliary, policy, preconvergence):
    if not args.energy_only:
        return native_call(
            atoms,
            orbital,
            auxiliary,
            args.budget_mib << 20,
            "df",
            policy,
            preconvergence,
        )
    from generativeqc import CBSBasis, CCSDTCBSCalculator

    started = time.perf_counter()
    owner = CCSDTCBSCalculator(
        tuple(CBSBasis(b, n) for b, n in zip(orbital, (2, 3), strict=True)),
        auxiliary_bases=auxiliary,
        hf_reference="df",
        policy=policy,
        preconvergence=preconvergence,
        device="cuda",
        calculator_options={"correlation_memory_budget_bytes": args.budget_mib << 20},
    )
    result = owner.singlepoint(atoms, properties=("energy",))
    seconds = time.perf_counter() - started
    return {**asdict(result), "complete_seconds": seconds}


try:
    # Keep process/CUDA initialization visible, but separate from balanced
    # fresh-owner trials. No converged density is supplied to any measured call.
    startup_atoms = geometry(Path("/data/jzzeng/readme-cbs-20261013/qm7.mat"), 1)
    _, startup_orbital, startup_auxiliary = setup(
        startup_atoms, 1, args.budget_mib << 20, "df"
    )
    payload["startup"] = endpoint(
        startup_atoms, startup_orbital, startup_auxiliary, "standard", "auto"
    )
    payload["timing"] = (
        "fresh complete owners; process/CUDA startup separately retained; no retained density"
    )
    checkpoint(args.output, payload)
    for carbon in args.carbons:
        atoms = geometry(Path("/data/jzzeng/readme-cbs-20261013/qm7.mat"), carbon)
        protocol, orbital, auxiliary = setup(atoms, carbon, args.budget_mib << 20, "df")
        if args.energy_only:
            protocol["endpoint"] = "energy_host"
        payload["protocols"][str(carbon)] = protocol
        for repeat in range(args.samples):
            arms = args.arms if repeat % 2 == 0 else list(reversed(args.arms))
            for arm in arms:
                policy, preconvergence = arm.split(":")
                payload["stage"] = f"C{carbon}/{repeat}/{arm}"
                checkpoint(args.output, payload)
                print("START", payload["stage"], flush=True)
                result = endpoint(atoms, orbital, auxiliary, policy, preconvergence)
                payload["samples"].append(
                    {
                        "carbon": carbon,
                        "repeat": repeat,
                        "policy": policy,
                        "preconvergence": preconvergence,
                        **result,
                    }
                )
                checkpoint(args.output, payload)
                print("DONE", payload["stage"], result["complete_seconds"], flush=True)
    payload["status"] = "completed"
except Exception as error:  # noqa: BLE001 -- retain a failed endpoint
    payload.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    checkpoint(args.output, payload)
