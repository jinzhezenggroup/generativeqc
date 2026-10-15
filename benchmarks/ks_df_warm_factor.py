"""Source-matched DF hybrid E+force qualification for #2474.

Run reference, baseline and candidate in separate finite Slurm jobs. Each uses
the same explicit spherical orbital/auxiliary bases and moving quadrature.
Clean wall times and opt-in component traces require separate invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.compare_df_direct_endpoint import check_endpoint, reference_work_counter
from benchmarks.compare_gpu4pyscf_batch import (
    gpu_convergence_payload,
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.df_component_ledger import read_trace
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_wb97mv import configure_reference_full_fock, reference_engine

if TYPE_CHECKING:
    from collections.abc import Callable


def source_provenance() -> dict[str, Any]:
    """Do not label a dirty checkout or a historical binary source-matched."""
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", *arguments], cwd=root)

    paths = (
        "src/dft/cuda_ks.cpp",
        "src/dft/cuda_ks.hpp",
        "src/scf/cuda_fock_execution.cpp",
        "src/scf/cuda/df_occupied_exchange.cpp",
        "src/scf/cuda/df_exchange.cpp",
        "benchmarks/ks_df_warm_factor.py",
    )
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--short").decode(),
        "tracked_diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--full-index", "--binary")
        ).hexdigest(),
        "files": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in paths
        },
    }


def ks_ledger(records: list[dict]) -> list[dict]:
    """Retain additive KS scopes, keeping nested provider timings in raw traces.

    Provider operation roots overlap ks_jk and must not be added to these totals.
    Host and GPU clocks are alternative views, never additive to each other.
    """
    return [
        {
            "operation": row["operation"],
            "counters": row["counters"],
            "regions": row["regions"],
            "host_completion_ms": row["host_completion_ms"],
            "diagnostic_synchronization_ms": row["final_synchronization_ms"],
        }
        for row in records
        if row["operation"] in ("ks_begin", "ks_iteration", "ks_finish")
    ]


def prepare_reference(
    geometry: Any, basis: Any, auxiliary: Any, grid: Any, functional: str
) -> Any:
    """Bind full-density get_veff only after creating the DF owner.

    Installing the closure before density_fit can retain a bound non-DF method
    through the reference engine's shallow copy and silently compare Direct K.
    """
    engine = reference_engine(
        geometry,
        basis,
        grid,
        xc=functional.upper().replace("-", "_"),
        full_fock=False,
    ).density_fit(auxbasis=auxiliary, only_dfj=False)
    configure_reference_full_fock(engine)
    engine.conv_tol, engine.conv_tol_grad = 1e-12, 1e-10
    engine.direct_scf_tol, engine.max_cycle = 1e-14, 200
    return engine


def main() -> None:
    """Journal failures and all cold/frozen/moved endpoints without filtering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("reference", "native"))
    parser.add_argument(
        "--variant", choices=("baseline", "candidate"), default="baseline"
    )
    parser.add_argument(
        "--functional", choices=("m06-2x", "pbe0", "b3lyp"), default="m06-2x"
    )
    parser.add_argument("--atoms", type=int, choices=(3, 6, 48, 96), required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--auxiliary-file", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=3, default=(48, 16, 32))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--trace-directory", type=raw_output_path)
    parser.add_argument("--audit-first-k", action="store_true")
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("GPU work requires srun and its unmodified CUDA_VISIBLE_DEVICES")
    if min(args.grid) <= 0:
        parser.error("grid dimensions must be positive")
    if args.engine == "native" and args.reference is None:
        parser.error(
            "native endpoints require an independently reconverged DF --reference"
        )
    if args.engine == "reference" and (
        args.variant != "baseline" or args.trace_directory is not None
    ):
        parser.error(
            "variant selection and native component tracing apply only to native"
        )
    if args.audit_first_k and (
        args.engine != "native"
        or args.variant != "candidate"
        or args.trace_directory is None
    ):
        parser.error("--audit-first-k requires a traced native candidate run")
    if os.environ.get("GENERATIVEQC_DF_TRACE") or os.environ.get(
        "GENERATIVEQC_DF_PROGRESS_TRACE"
    ):
        parser.error(
            "use --trace-directory in a separate diagnostic run, not ambient tracing"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve once before importing GPU packages or preparing scientific inputs.
    # The owned stream also avoids reopening a path that another writer replaces.
    with args.output.open("x", encoding="utf-8") as output:
        record: dict[str, Any] = {
            "schema": "generativeqc.ks-df-warm-factor.v1",
            "engine": args.engine,
            "variant": args.variant,
            "functional": args.functional,
            "atoms": args.atoms,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "status": "running",
            "records": [],
        }

        def save(stage: str) -> None:
            record["stage"] = stage
            output.seek(0)
            json.dump(record, output, indent=2)
            output.write("\n")
            output.truncate()
            output.flush()

        try:
            save("initialize")
            _execute(args, record, save)
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            save(record["stage"])
            raise


def _execute(
    args: argparse.Namespace, record: dict[str, Any], save: Callable[[str], None]
) -> None:
    """Run only after acquiring the fresh output journal."""
    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    auxiliary, reference_auxiliary = load_comparison_basis(
        args.auxiliary_file, case, role="auxiliary", compute_forces=True
    )
    grid = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    coordinates = np.asarray([position for _, position in case.atoms])
    moved = coordinates.copy()
    moved[-1, 0] += 0.01
    geometries = [
        case.atoms,
        [
            (atom[0], tuple(position))
            for atom, position in zip(case.atoms, moved, strict=True)
        ],
    ]
    protocol = {
        "functional": args.functional,
        "atoms": args.atoms,
        "grid": asdict(grid),
        "geometries_bohr": geometries,
        "properties": ["energy", "forces"],
        "basis_sha256": hashlib.sha256(args.basis_file.read_bytes()).hexdigest(),
        "auxiliary_sha256": hashlib.sha256(
            args.auxiliary_file.read_bytes()
        ).hexdigest(),
        "density_fitting": True,
        "df_relative_threshold": 1e-10,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "reference_gradient_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "reference_direct_scf_tolerance": 1e-14,
        "max_iterations": 200,
        "force_grid_response": True,
        "force_auxiliary_response": True,
        "frozen_repeats": 5,
    }
    record.update(
        {
            "schema": "generativeqc.ks-df-warm-factor.v1",
            "protocol": protocol,
            "engine": args.engine,
            "variant": args.variant,
            "source": source_provenance(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "diagnostic": args.trace_directory is not None,
            "first_k_matrix_audit": args.audit_first_k,
            "timer": "native cold includes Calculator construction, prepare_batch and complete host-returned E/F; excludes fixture loading and source/library hashing; moved rebuild is inside execute",
            "records": [],
            "status": "running",
        }
    )
    oracle = json.loads(args.reference.read_text()) if args.reference else None
    if args.engine == "native" and (
        oracle is None
        or oracle.get("engine") != "reference"
        or oracle.get("status") != "measured"
        or oracle.get("protocol") != json.loads(json.dumps(protocol))
    ):
        raise ValueError(
            "reference must be measured independently on this exact DF protocol"
        )
    owner = None
    construction_seconds = 0.0
    try:
        save("prepare")
        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        if args.engine == "native":
            os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR"] = str(
                int(args.variant == "candidate")
            )
            os.environ["GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT"] = str(
                int(args.audit_first_k)
            )
            calculator = Calculator(
                method=f"{args.functional}-rks",
                basis=basis,
                auxiliary_basis=auxiliary,
                basis_representation="spherical",
                device="cuda",
                precision="fp64",
                density_fitting="cuda",
                density_fitting_relative_threshold=1e-10,
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=200,
                ks_options=KsOptions(grid=grid),
            )
            cp.cuda.runtime.deviceSynchronize()
            construction_seconds = perf_counter() - started
            record["calculator_construction_seconds"] = construction_seconds
            record["native_build"] = native_build_metadata(calculator)
            require_tuned_native_build(record["native_build"])
            started = perf_counter()
            owner = calculator.prepare_batch([geometries[0]], warm_start=True)
        cp.cuda.runtime.deviceSynchronize()
        prepare_seconds = perf_counter() - started + construction_seconds
        for geometry_index, geometry in enumerate(geometries):
            if args.engine == "reference":
                started = perf_counter()
                engine = prepare_reference(
                    geometry,
                    reference_basis,
                    reference_auxiliary,
                    grid,
                    args.functional,
                )
                cp.cuda.runtime.deviceSynchronize()
                prepare_seconds = perf_counter() - started
                frozen = None
            else:
                owner.set_warm_start_updates(True)
            for replay in range(6):
                phase = (
                    ("cold" if geometry_index == 0 else "moved")
                    if replay == 0
                    else ("warm" if geometry_index == 0 else "moved-warm")
                )
                label = f"{phase}-{max(0, replay - 1)}"
                save(label)
                trace_path = None
                if args.trace_directory is not None:
                    args.trace_directory.mkdir(parents=True, exist_ok=True)
                    trace_path = (
                        args.trace_directory
                        / f"{args.engine}-{args.variant}-{label}.jsonl"
                    )
                    if trace_path.exists():
                        raise FileExistsError(trace_path)
                    os.environ["GENERATIVEQC_DF_TRACE"] = str(trace_path.resolve())
                cp.cuda.runtime.deviceSynchronize()
                started = perf_counter()
                if args.engine == "reference":
                    with reference_work_counter(engine) as work:
                        energy = float(
                            engine.kernel(dm0=None if frozen is None else frozen.copy())
                        )
                        if not engine.converged:
                            raise RuntimeError(
                                "independent DF reference did not converge"
                            )
                        gradient = engine.nuc_grad_method()
                        gradient.grid_response = True
                        gradient.auxbasis_response = True
                        forces = cp.asnumpy(-gradient.kernel())
                        cp.cuda.runtime.deviceSynchronize()
                        seconds = perf_counter() - started
                        if (
                            not np.isfinite(energy)
                            or forces.shape != coordinates.shape
                            or not np.isfinite(forces).all()
                        ):
                            raise RuntimeError(
                                "independent DF reference returned invalid E/F"
                            )
                        convergence = gpu_convergence_payload(
                            [engine], [work["tracker"]]
                        )[0]
                    row = {
                        "energy": energy,
                        "forces": forces.tolist(),
                        "converged": True,
                        "iterations": convergence["iterations"],
                        "fock_builds": work["scf_jk_builds"],
                        "final_residuals": convergence["final_residuals"],
                    }
                    if replay == 0:
                        frozen = engine.make_rdm1().copy()
                else:
                    item = owner.execute(
                        None if geometry_index == 0 else [moved],
                        strict=False,
                        properties=("energy", "forces"),
                    ).items[0]
                    cp.cuda.runtime.deviceSynchronize()
                    seconds = perf_counter() - started
                    baseline = next(
                        row
                        for row in oracle["records"]
                        if row["geometry"] == geometry_index
                    )
                    row = {
                        "energy": item.energy,
                        "forces": None if item.forces is None else item.forces.tolist(),
                        "converged": item.converged,
                        "status": item.status,
                        "detail": item.status_message,
                        "iterations": item.iterations,
                        "fock_builds": item.fock_builds,
                        "ks_diagnostic": item.ks_diagnostic.to_payload()
                        if item.ks_diagnostic
                        else None,
                        "warm_start_used": item.warm_start_used,
                        "warm_start_fallback": item.warm_start_fallback,
                        "df_metric_diagnostics": [
                            diagnostic.to_dict()
                            for diagnostic in owner.last_density_fitting_metric_diagnostics()
                        ],
                        **check_endpoint(item, baseline),
                    }
                    if replay == 0:
                        owner.set_warm_start_updates(False)
                prepare = prepare_seconds if replay == 0 else 0.0
                if args.engine == "native" and geometry_index == 1:
                    prepare = 0.0
                row.update(
                    phase=phase,
                    geometry=geometry_index,
                    replay=max(0, replay - 1),
                    seconds=seconds,
                    prepare_seconds=prepare,
                    complete_seconds=seconds + prepare,
                )
                if trace_path is not None:
                    os.environ.pop("GENERATIVEQC_DF_TRACE", None)
                    row["trace"] = str(trace_path)
                    row["ks_ledger"] = ks_ledger(read_trace(trace_path))
                record["records"].append(row)
                save(label)
                if args.engine == "native" and not row["gate"]:
                    raise RuntimeError(
                        "DF energy/force gate failed; no endpoints were discarded"
                    )
        record["status"] = "measured"
        save("complete")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record["stage"])
        raise
    finally:
        if args.trace_directory is not None:
            os.environ.pop("GENERATIVEQC_DF_TRACE", None)
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
