"""Cross-functional CUDA DFT energy/force phase matrix for issue #1480.

Clean endpoint timing and intrusive CUDA component tracing are separate passes.
The runner uses one method-neutral stationary diagnostic boundary so semilocal,
global-hybrid, meta-GGA and WB97M-V force work can share the same report shape.
Unsupported RSH/nonlocal combinations remain explicit evidence instead of being
silently dropped or replaced by a different scientific method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import typing
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.df_component_ledger import read_trace, trace_identity
from benchmarks.dft_force_components import (
    expected_scf_components,
    normalize_force_work,
    normalize_scf_trace,
)

DEFAULT_METHODS = (
    "pbe-rks",
    "pbe0-rks",
    "r2scan-rks",
    "cam-b3lyp-rks",
    "wb97m-v",
)
DEFAULT_SYSTEMS = ("water-3", "formaldehyde")
QUALIFICATION_SYSTEMS = ("water-3", "water-6", "water-12", "formaldehyde")
FORMALDEHYDE = (
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 2.28)),
    ("H", (1.75, 0.0, -1.05)),
    ("H", (-1.75, 0.0, -1.05)),
)


def _jsonable(value: typing.Any) -> typing.Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *command], text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_identity() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _method_configuration(name: str, grid: typing.Any) -> tuple[str, typing.Any]:
    from vibeqc import KsOptions
    from vibeqc_compiler.method import resolve_method

    if name == "cam-b3lyp-rks":
        return (
            "pbe-rks",
            KsOptions(
                composition=resolve_method("CAM-B3LYP", spin="unpolarized"),
                grid=grid,
            ),
        )
    if name not in DEFAULT_METHODS:
        raise ValueError(f"unknown benchmark method: {name}")
    return name, KsOptions(grid=grid)


def _atoms(name: str) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if name == "formaldehyde":
        return FORMALDEHYDE
    if name in {"water-3", "water-6", "water-12"}:
        from benchmarks.readme_hf_scaling import scaling_cases

        return tuple(scaling_cases()[name].atoms)
    raise ValueError(f"unknown benchmark system: {name}")


def _changed_atoms(
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
) -> tuple[
    tuple[tuple[str, tuple[float, float, float]], ...],
    np.ndarray,
]:
    xyz = np.asarray([position for _, position in atoms], dtype=np.float64)
    moved = xyz.copy()
    moved[-1] += np.asarray((0.0010, -0.0005, 0.0003))
    changed = tuple(
        (symbol, tuple(float(value) for value in position))
        for (symbol, _), position in zip(atoms, moved, strict=True)
    )
    return changed, moved


def _exchange_operators(calculator: typing.Any) -> tuple[str, ...]:
    options = calculator.ks_options
    if options is None:
        return ()
    return tuple(term.operator for term in options.execution_plan.exchange)


def _has_nonlocal_correlation(calculator: typing.Any) -> bool:
    options = calculator.ks_options
    return bool(
        options is not None and options.execution_plan.nonlocal_correlation is not None
    )


def _force_diagnostic(
    batch: typing.Any,
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """Execute the existing internal stationary CUDA owner without public-ABI promotion."""

    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc._stationary_cuda import (
        PreparedStationaryCudaExecution,
        PreparedStationaryCudaTopologyMismatch,
        complete_rks_cuda_gradient_diagnostic,
    )
    from vibeqc_compiler.dft import NativeAO

    calculator = batch._calculator
    if (
        calculator.ks_options.has_range_exchange
        and not calculator._method_name.startswith("wb97m-v")
    ):
        raise NotImplementedError(
            "cross-functional matrix retains generic RSH force as an explicit missing owner"
        )
    if calculator._method_name.startswith("wb97m-v"):
        from vibeqc._stationary_wb97mv_cuda import PreparedWb97mvCudaGradient

        prepared = batch._stationary_cuda_execution
        if not isinstance(prepared, PreparedWb97mvCudaGradient):
            if prepared is not None:
                prepared.close()
            prepared = PreparedWb97mvCudaGradient()
            batch._stationary_cuda_execution = prepared
        with NativeAO(
            atoms,
            basis=calculator._basis,
            representation=calculator._representation_name,
            charge=batch._charges[0],
            multiplicity=batch._multiplicities[0],
        ) as basis:
            state = StationaryKsState.from_native(batch, basis, index=0)
            try:
                return prepared.execute(
                    state,
                    basis,
                    compiler=batch._stationary_cuda_compiler(),
                    cache=Path(
                        os.environ.get(
                            "VIBEQC_STATIONARY_CACHE", ".cache/stationary-cuda"
                        )
                    ),
                    library=Path(str(batch._library._name)).resolve(),
                )
            finally:
                state._source.close()

    prepared = batch._stationary_cuda_execution
    if not isinstance(prepared, PreparedStationaryCudaExecution):
        if prepared is not None:
            prepared.close()
        prepared = PreparedStationaryCudaExecution()
        batch._stationary_cuda_execution = prepared
    with NativeAO(
        atoms,
        basis=calculator._basis,
        representation=calculator._representation_name,
        charge=batch._charges[0],
        multiplicity=batch._multiplicities[0],
    ) as basis:
        state = StationaryKsState.from_native(batch, basis, index=0)
        try:
            native_library = Path(str(batch._library._name)).resolve()
            packaged = (
                state._source.hamiltonian == "all-electron"
                and not state._source.method_ir.full_range_exact_exchange
            )
            kwargs = {
                "compiler": None if packaged else batch._stationary_cuda_compiler(),
                "target": batch._stationary_cuda_target(),
                "cache": Path(
                    os.environ.get("VIBEQC_STATIONARY_CACHE", ".cache/stationary-cuda")
                ),
                "aot_directory": native_library.parent if packaged else None,
                "native_grid_library": native_library,
            }
            try:
                result = complete_rks_cuda_gradient_diagnostic(
                    state,
                    basis,
                    prepared=prepared,
                    **kwargs,
                )
            except PreparedStationaryCudaTopologyMismatch:
                prepared.close()
                prepared = PreparedStationaryCudaExecution()
                batch._stationary_cuda_execution = prepared
                result = complete_rks_cuda_gradient_diagnostic(
                    state,
                    basis,
                    prepared=prepared,
                    **kwargs,
                )
            return -np.asarray(result.gradient).copy(), dict(result.work)
        finally:
            state._source.close()


def _clean_sample(
    batch: typing.Any,
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
    cupy_module: typing.Any,
    *,
    scenario: str,
    coordinates: np.ndarray | None = None,
) -> dict[str, typing.Any]:
    cupy_module.cuda.Stream.null.synchronize()
    started = perf_counter()
    result = batch.execute(
        coordinates=None if coordinates is None else [coordinates],
        strict=True,
        properties=("energy",),
    )
    cupy_module.cuda.Stream.null.synchronize()
    scf_seconds = perf_counter() - started
    item = result.items[0]
    if not item.converged or not math.isfinite(item.energy):
        raise RuntimeError(f"{scenario}: DFT SCF did not converge to a finite energy")

    force_status = "ok"
    force_error = None
    force_seconds = None
    force = None
    work = None
    components = None
    cupy_module.cuda.Stream.null.synchronize()
    force_started = perf_counter()
    try:
        force, work = _force_diagnostic(batch, atoms)
        cupy_module.cuda.Stream.null.synchronize()
        force_seconds = perf_counter() - force_started
        if force.shape != (len(atoms), 3) or not np.isfinite(force).all():
            raise RuntimeError("stationary force must be finite atom-by-three")
        components = normalize_force_work(work)
    except NotImplementedError as error:
        cupy_module.cuda.Stream.null.synchronize()
        force_seconds = perf_counter() - force_started
        force_status = "unsupported"
        force_error = str(error)

    return {
        "scenario": scenario,
        "scf_seconds": scf_seconds,
        "force_seconds": force_seconds,
        "endpoint_seconds": scf_seconds + force_seconds,
        "converged": bool(item.converged),
        "iterations": int(item.iterations),
        "energy_hartree": float(item.energy),
        "physical_residual_rms": item.physical_residual_rms,
        "warm_start_used": item.warm_start_used,
        "force_status": force_status,
        "force_error": force_error,
        "forces_hartree_per_bohr": None if force is None else force.tolist(),
        "force_work": work,
        "force_components": components,
    }


def _scf_trace_profile(
    batch: typing.Any,
    cupy_module: typing.Any,
    path: Path,
    exchange_operators: tuple[str, ...],
    *,
    nonlocal_correlation: bool,
) -> dict[str, typing.Any]:
    if path.exists():
        raise FileExistsError(f"refusing to append prior SCF trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    previous = os.environ.get("VIBEQC_DF_TRACE")
    os.environ["VIBEQC_DF_TRACE"] = str(path.resolve())
    try:
        cupy_module.cuda.Stream.null.synchronize()
        result = batch.execute(strict=True, properties=("energy",))
        cupy_module.cuda.Stream.null.synchronize()
        if not result.items[0].converged:
            raise RuntimeError("traced SCF replay did not converge")
    finally:
        if previous is None:
            os.environ.pop("VIBEQC_DF_TRACE", None)
        else:
            os.environ["VIBEQC_DF_TRACE"] = previous

    if path.stat().st_size == 0:
        return {
            "status": "unavailable",
            "reason": "selected SCF provider emitted no VIBEQC_DF_TRACE roots",
            "expected_exchange_operators": list(exchange_operators),
            "expected_components": list(
                expected_scf_components(
                    exchange_operators,
                    nonlocal_correlation=nonlocal_correlation,
                )
            ),
        }
    records = read_trace(path)
    return {
        "status": "measured",
        "trace": trace_identity(path),
        "profile": normalize_scf_trace(
            records,
            exchange_operators=exchange_operators,
            nonlocal_correlation=nonlocal_correlation,
        ),
    }


def _calculator(
    method: str,
    grid: typing.Any,
    basis: str,
    density_fitting: str,
) -> typing.Any:
    from vibeqc import Calculator

    selector, options = _method_configuration(method, grid)
    return Calculator(
        method=selector,
        basis=basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting=density_fitting,
        ks_options=options,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
        screening_tolerance=1e-12,
        max_iterations=180,
    )


def benchmark_case(
    *,
    method: str,
    system: str,
    grid: typing.Any,
    basis: str,
    density_fitting: str,
    repeats: int,
    trace_directory: Path | None,
    cupy_module: typing.Any,
) -> dict[str, typing.Any]:
    atoms = _atoms(system)
    batch = None
    try:
        calculator = _calculator(method, grid, basis, density_fitting)
        library = Path(str(calculator._library._name)).resolve()
        exchange_operators = _exchange_operators(calculator)
        nonlocal_correlation = _has_nonlocal_correlation(calculator)
        started = perf_counter()
        batch = calculator.prepare_batch([atoms], warm_start=True)
        prepare_seconds = perf_counter() - started

        cold = _clean_sample(batch, atoms, cupy_module, scenario="cold")
        batch.set_warm_start_updates(False)
        priming = _clean_sample(batch, atoms, cupy_module, scenario="priming")
        warm = [
            _clean_sample(
                batch,
                atoms,
                cupy_module,
                scenario=f"same_geometry_warm_{repeat}",
            )
            for repeat in range(repeats)
        ]
        scf_profile = None
        if trace_directory is not None:
            scf_profile = _scf_trace_profile(
                batch,
                cupy_module,
                trace_directory / f"{method}-{system}.jsonl",
                exchange_operators,
                nonlocal_correlation=nonlocal_correlation,
            )
        changed_atoms, coordinates = _changed_atoms(atoms)
        changed = _clean_sample(
            batch,
            changed_atoms,
            cupy_module,
            scenario="changed_geometry",
            coordinates=coordinates,
        )
        return _jsonable(
            {
                "status": "measured",
                "method": method,
                "selector": calculator._method_name,
                "method_identity": calculator.method_ir.identity,
                "ks_options_identity": calculator.ks_options.identity,
                "execution_plan": calculator.ks_options.execution_plan.to_payload(),
                "exchange_operators": list(exchange_operators),
                "nonlocal_correlation": nonlocal_correlation,
                "system": system,
                "atoms": len(atoms),
                "basis": basis,
                "density_fitting": density_fitting,
                "prepare_seconds": prepare_seconds,
                "cold": cold,
                "priming": priming,
                "warm": warm,
                "changed_geometry": changed,
                "scf_profile": scf_profile,
                "library": str(library),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "resource_diagnostics": batch.resource_diagnostics,
            }
        )
    except (
        NotImplementedError,
        ValueError,
        RuntimeError,
        MemoryError,
        OSError,
    ) as error:
        return {
            "status": (
                "unsupported" if isinstance(error, NotImplementedError) else "failed"
            ),
            "method": method,
            "system": system,
            "basis": basis,
            "density_fitting": density_fitting,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        if batch is not None:
            batch.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("RADIAL", "POLAR", "AZIMUTH"),
        default=(24, 8, 16),
    )
    parser.add_argument(
        "--density-fitting",
        choices=("none", "cuda"),
        default="none",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--qualification",
        action="store_true",
        help="run the full water-3/6/12 plus non-water qualification system set",
    )
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.qualification and args.repeats < 3:
        parser.error("--qualification requires at least three repeats")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run requires a finite Slurm GPU allocation")
    if os.environ.get("VIBEQC_DF_TRACE"):
        parser.error("ambient VIBEQC_DF_TRACE would contaminate clean timing")
    if args.trace_directory is not None and args.trace_directory.exists():
        parser.error("--trace-directory must be a fresh path")

    import cupy as cp
    from vibeqc import GridSpec

    grid = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    methods = tuple(args.methods or DEFAULT_METHODS)
    systems = tuple(
        args.systems
        or (QUALIFICATION_SYSTEMS if args.qualification else DEFAULT_SYSTEMS)
    )
    unknown_methods = set(methods) - set(DEFAULT_METHODS)
    unknown_systems = set(systems) - {
        "water-3",
        "water-6",
        "water-12",
        "formaldehyde",
    }
    if unknown_methods:
        parser.error(f"unknown methods: {sorted(unknown_methods)}")
    if unknown_systems:
        parser.error(f"unknown systems: {sorted(unknown_systems)}")

    records = []
    for method in methods:
        for system in systems:
            print(f"#1480: {method} {system}", flush=True)
            records.append(
                benchmark_case(
                    method=method,
                    system=system,
                    grid=grid,
                    basis=args.basis,
                    density_fitting=args.density_fitting,
                    repeats=args.repeats,
                    trace_directory=args.trace_directory,
                    cupy_module=cp,
                )
            )

    payload = {
        "schema": "vibeqc.dft-force-matrix.v1",
        "issue": 1480,
        "provenance": {
            "head": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
            "gpu": _gpu_identity(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "grid": {
                "radial_points": args.grid[0],
                "angular_polar": args.grid[1],
                "angular_azimuth": args.grid[2],
            },
        },
        "measurement_policy": {
            "clean": "synchronized wall timing with VIBEQC_DF_TRACE disabled",
            "profile": (
                "separate optional VIBEQC_DF_TRACE replay; CUDA-event timings are "
                "diagnostic and never added to clean endpoint seconds"
            ),
            "unsupported": (
                "unsupported functional/provider combinations are retained as records"
            ),
            "qualification_preset": args.qualification,\n        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
