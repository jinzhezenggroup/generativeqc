"""Fresh cold 48/96-atom PBE0 RKS owner for the #1598 KS re-verification control.

Adapted from the retained
`benchmarks/results/pbe0-incremental-refinement-20261007/cold.py.txt`
protocol. One invocation records one complete cold energy-plus-force endpoint
for the current `GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK` selection; the driver
interleaves off/on samples in the same process/device epoch. Independent
GPU4PySCF acceptance comes from the retained composed-baseline oracle.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import cupy as cp
from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks.compare_df_direct_endpoint import check_endpoint
from benchmarks.compare_gpu4pyscf_batch import (
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.ks_preliminary_density import read_ao_work
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import protocol
from benchmarks.readme_pbe0 import PBE0


def _allocation_identity() -> dict:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inspire_notebook": os.environ.get("ISSUE1598_ALLOCATION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": os.environ.get("ISSUE1598_HOSTNAME"),
    }


def main() -> None:
    """Measure one fresh cold endpoint; exit nonzero when acceptance fails."""

    output = Path(sys.argv[1])
    atoms = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    if atoms not in (48, 96):
        raise ValueError("KS control requires 48 or 96 atoms")
    case = scaling_cases()[f"water-{atoms}"]
    basis_path = Path("benchmarks/results/pbe0-def2-svp-20261001/def2-svp-ho.json")
    basis, _ = load_comparison_basis(
        basis_path, case, role="orbital", compute_forces=True
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    scientific = protocol(atoms, basis, grid, 5, benchmark=PBE0)
    reference_path = Path(
        f"benchmarks/results/pbe0-composed-baseline-20261005/reference-{atoms}.json.gz"
    )
    reference = json.loads(gzip.decompress(reference_path.read_bytes()))
    if reference["status"] != "measured" or reference["protocol"] != scientific:
        raise ValueError("KS reference does not match the measured protocol")
    oracle = next(row for row in reference["records"] if row["phase"] == "cold")
    record = {
        "schema": "issue1598.ks-cold-control.v1",
        "status": "running",
        "protocol": scientific,
        "scope": "independent cold owner only; no warm/moved populations",
        "allocation": _allocation_identity(),
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("GENERATIVEQC_")
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    owner = None

    def save() -> None:
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    try:
        calculator = Calculator(
            method="pbe0-rks",
            basis=basis,
            device="cuda",
            basis_representation="spherical",
            ks_options=KsOptions(grid=grid),
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-12,
            max_iterations=100,
        )
        record["native_build"] = native_build_metadata(calculator)
        # Intentional portable AOT baseline: both arms share one binary, so the
        # A/B comparison is unaffected; the build identity is recorded.
        require_tuned_native_build(record["native_build"], allow_portable=True)
        record["stage"] = "prepare"
        save()
        cp.cuda.Stream.null.synchronize()
        started = perf_counter()
        owner = calculator.prepare_batch([case.atoms], warm_start=True)
        cp.cuda.Stream.null.synchronize()
        prepare_seconds = perf_counter() - started
        original_force = owner._public_dft_cuda_force
        force_samples = []

        def observed_force(*arguments: object) -> object:
            """Include synchronization and all force preparation in its sub-timer."""
            cp.cuda.Stream.null.synchronize()
            force_started = perf_counter()
            result = original_force(*arguments)
            cp.cuda.Stream.null.synchronize()
            force_samples.append(
                {"seconds": perf_counter() - force_started, "work": result[1]}
            )
            return result

        owner._public_dft_cuda_force = observed_force
        record.update(stage="execute", prepare_seconds=prepare_seconds)
        save()
        cp.cuda.Stream.null.synchronize()
        started = perf_counter()
        item = owner.execute(strict=False, properties=("energy", "forces")).items[0]
        cp.cuda.Stream.null.synchronize()
        execute_seconds = perf_counter() - started
        acceptance = check_endpoint(item, oracle)
        record.update(
            status="measured",
            stage="complete",
            execute_seconds=execute_seconds,
            complete_seconds=prepare_seconds + execute_seconds,
            force_samples=force_samples,
            energy=item.energy,
            forces=item.forces.tolist() if item.forces is not None else None,
            converged=item.converged,
            native_status=item.status,
            status_message=item.status_message,
            iterations=item.iterations,
            fock_builds=item.fock_builds,
            density_rms=item.density_rms,
            physical_residual_rms=item.physical_residual_rms,
            warm_start_used=item.warm_start_used,
            warm_start_fallback=item.warm_start_fallback,
            incremental_direct_jk=item.incremental_direct_jk,
            ks_diagnostic=(
                item.ks_diagnostic.to_payload()
                if item.ks_diagnostic is not None
                else None
            ),
            scf_ao_work=read_ao_work(owner) if item.status == 0 else None,
            acceptance=acceptance,
        )
        save()
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "complete_seconds",
                        "prepare_seconds",
                        "iterations",
                        "fock_builds",
                        "incremental_direct_jk",
                        "acceptance",
                    )
                }
            ),
            flush=True,
        )
        if acceptance.get("gate") is not True:
            raise SystemExit(3)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
