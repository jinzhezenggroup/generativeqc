"""Compare fresh-owner PBE0-DF endpoints with identical basis and quadrature.

Each invocation uses a fresh process and SCF density. Timers include preparation
and synchronized host-returned results, but exclude imports, CUDA initialization,
and native Calculator construction. Persistent compiler/runtime caches remain.
Energy-only and energy-plus-force endpoints are independent cold calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def main() -> None:
    """Journal every stage and retain scientific/work provenance on failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("native", "reference"))
    parser.add_argument("--properties", choices=("energy", "forces"), required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("real GPU measurements require Slurm-assigned visibility")

    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks._retained_basis import load_retained_comparison_basis
    from benchmarks.compare_df_direct_endpoint import reference_work_counter
    from benchmarks.compare_gpu4pyscf_batch import (
        gpu_convergence_payload,
        load_comparison_basis,
        native_build_metadata,
        require_tuned_native_build,
    )
    from benchmarks.ks_preliminary_density import read_ao_work
    from benchmarks.readme_hf_scaling import scaling_cases
    from benchmarks.readme_wb97mv import reference_engine

    case = scaling_cases()["water-96"]
    basis_path = Path("benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json")
    auxiliary_path = Path(
        "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    include_forces = args.properties == "forces"
    basis, reference_basis = load_comparison_basis(
        basis_path, case, role="orbital", compute_forces=include_forces
    )
    auxiliary, reference_auxiliary = load_retained_comparison_basis(
        auxiliary_path, case, role="auxiliary", compute_forces=include_forces
    )
    grid = GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
    record: dict[str, Any] = {
        "schema": "generativeqc.pbe0-df-cold.v1",
        "source": os.environ["GENERATIVEQC_BENCHMARK_SOURCE"],
        "engine": args.engine,
        "properties": ["energy", "forces"] if include_forces else ["energy"],
        "protocol": {
            "atoms": 96,
            "aos": 768,
            "geometry_bohr": case.atoms,
            "orbital_basis": basis.name,
            "orbital_identity": basis.identity,
            "auxiliary_basis": auxiliary.name,
            "auxiliary_identity": auxiliary.identity,
            "basis_representation": "spherical",
            "method": "PBE0-RKS",
            "density_fitting": "cuda",
            "precision": "FP64",
            "grid": asdict(grid),
            "grid_points": 96 * 48 * 16 * 32,
            "grid_pruning": False,
            "force_grid_response": include_forces,
            "energy_tolerance": 1e-12,
            "native_density_tolerance": 1e-10,
            "reference_gradient_tolerance": 1e-10,
            "native_screening_tolerance": 1e-12,
            "reference_direct_scf_tolerance": 1e-14,
            "df_relative_threshold": 1e-10,
            "df_memory_budget_bytes": 0,
            "reference_incremental_policy": "stock GPU4PySCF DF",
            "max_iterations": 200,
        },
        "cold_definition": "fresh process, owner and density; persistent caches retained",
        "timer": "prepare plus first synchronized public endpoint; excludes imports, CUDA context, native Calculator construction",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "df_execution_controls": {
            name: os.environ.get(name)
            for name in (
                "GENERATIVEQC_DF_VALUE_STORAGE",
                "GENERATIVEQC_DF_EXCHANGE",
                "GENERATIVEQC_DF_RESIDENT_EXCHANGE",
                "GENERATIVEQC_DF_COULOMB_RESPONSE",
                "GENERATIVEQC_DF_TRACE",
                "GENERATIVEQC_DF_PROGRESS_TRACE",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gpu4pyscf-cuda12x", "cupy-cuda12x", "pyscf", "numpy")
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running",
        "stage": "context",
    }
    owner = None

    def save(stage: str) -> None:
        """Write a receipt before a potentially long preparation/SCF stage."""
        record["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    try:
        save("context")
        cp.cuda.runtime.deviceSynchronize()
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        record["gpu"] = {
            "name": device["name"].decode(),
            "memory_bytes": device["totalGlobalMem"],
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        if args.engine == "native":
            construct_started = perf_counter()
            calculator = Calculator(
                method="pbe0-rks",
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                density_fitting="cuda",
                auxiliary_basis=auxiliary,
                ks_options=KsOptions(grid=grid),
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=200,
            )
            record["calculator_construct_seconds_excluded"] = (
                perf_counter() - construct_started
            )
            record["native_build"] = native_build_metadata(calculator)
            require_tuned_native_build(record["native_build"])
            save("prepare")
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
            owner = calculator.prepare_batch([case.atoms], warm_start=True)
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - started
            force_times: list[float] = []
            original_force = owner._public_dft_cuda_force

            def observed_force(*arguments: Any, **keywords: Any) -> Any:
                """Time the ordinary force consumer without changing its work."""
                force_started = perf_counter()
                result = original_force(*arguments, **keywords)
                force_times.append(perf_counter() - force_started)
                record["force_work"] = dict(result[1])
                return result

            if include_forces:
                owner._public_dft_cuda_force = observed_force
            save("execute")
            started = perf_counter()
            item = owner.execute(
                strict=False, properties=tuple(record["properties"])
            ).items[0]
            cp.cuda.runtime.deviceSynchronize()
            record["execute_seconds"] = perf_counter() - started
            record["force_seconds"] = sum(force_times)
            record["scf_and_publication_seconds"] = (
                record["execute_seconds"] - record["force_seconds"]
            )
            record.update(
                energy=float(item.energy),
                forces=item.forces.tolist() if item.forces is not None else None,
                converged=bool(item.converged),
                status_code=int(item.status),
                status_message=item.status_message,
                iterations=int(item.iterations),
                fock_builds=int(item.fock_builds)
                if item.fock_builds is not None
                else None,
                density_rms=float(item.density_rms),
                physical_residual_rms=float(item.physical_residual_rms),
                warm_start_used=bool(item.warm_start_used),
                ks_diagnostic=item.ks_diagnostic.to_payload()
                if item.ks_diagnostic
                else None,
                ao_work=read_ao_work(owner) if item.status == 0 else None,
                df_metric=[
                    entry.to_dict()
                    for entry in owner.last_density_fitting_metric_diagnostics()
                ],
            )
            if not item.succeeded or not item.converged:
                raise RuntimeError(f"native endpoint failed: {item.status_message}")
            if item.warm_start_used:
                raise RuntimeError("a cold endpoint unexpectedly used a warm density")
        else:
            save("prepare")
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
            engine = reference_engine(case.atoms, reference_basis, grid, xc="PBE0")
            engine = engine.density_fit(auxbasis=reference_auxiliary)
            engine.conv_tol = 1e-12
            engine.conv_tol_grad = 1e-10
            engine.direct_scf_tol = 1e-14
            engine.max_cycle = 200
            cp.cuda.runtime.deviceSynchronize()
            record["prepare_seconds"] = perf_counter() - started
            save("execute")
            with reference_work_counter(engine) as work:
                started = perf_counter()
                energy = float(engine.kernel())
                cp.cuda.runtime.deviceSynchronize()
                record["scf_seconds"] = perf_counter() - started
                forces = None
                force_started = perf_counter()
                if include_forces:
                    gradient = engine.nuc_grad_method()
                    gradient.grid_response = True
                    forces = cp.asnumpy(-gradient.kernel())
                    cp.cuda.runtime.deviceSynchronize()
                record["force_seconds"] = (
                    perf_counter() - force_started if include_forces else 0.0
                )
                record["execute_seconds"] = perf_counter() - started
                convergence = gpu_convergence_payload(
                    [engine], [work["tracker"]], warm_start_used=False
                )[0]
            record.update(
                energy=energy,
                forces=forces.tolist() if forces is not None else None,
                converged=bool(engine.converged),
                iterations=convergence["iterations"],
                fock_builds=work["scf_jk_builds"],
                final_residuals=convergence["final_residuals"],
                grid_points=len(engine.grids.weights),
                initial_guess=engine.init_guess,
                auxiliary_aos=int(engine.with_df.auxmol.nao_nr()),
            )
            if not engine.converged:
                raise RuntimeError("independent GPU4PySCF DF did not converge")
        record["complete_seconds"] = (
            record["prepare_seconds"] + record["execute_seconds"]
        )
        record["status"] = "measured"
        save("complete")
        print(
            json.dumps(
                {
                    name: record.get(name)
                    for name in (
                        "engine",
                        "properties",
                        "complete_seconds",
                        "prepare_seconds",
                        "scf_seconds",
                        "force_seconds",
                        "iterations",
                        "fock_builds",
                        "converged",
                        "energy",
                        "physical_residual_rms",
                    )
                }
            ),
            flush=True,
        )
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record["stage"])
        raise
    finally:
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
