"""Time each engine's production-default cold E+F, without numerical alignment.

Run native and GPU4PySCF in separate processes in a finite Slurm allocation.
Shared geometry/basis/functional are the scientific input, not SCF/grid knobs.
The default force modes can differ; ratios are not accuracy certificates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.production-default-ef.v1"
METHODS = {
    "lda": ("lda-rks", "LDA_X,LDA_C_PW"),
    "pbe": ("pbe-rks", "PBE"),
    "pbe0": ("pbe0-rks", "PBE0"),
    "r2scan": ("r2scan-rks", "R2SCAN"),
    "wb97m-v": ("wb97m-v-rks", "WB97M_V"),
}
REQUIRED_NATIVE_GRID = frozenset(("pbe0", "wb97m-v"))
FORCE_POLICIES = ("engine-default", "full-moving-grid")
SIZES = (3, 6, 12, 24, 48, 96)


def problem_identity(method: str, atoms: Any) -> dict[str, Any]:
    """Match the Hamiltonian input, not the execution/numerical defaults."""
    if method not in METHODS:
        raise ValueError("unsupported default-comparison method")
    return {
        "method": method,
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "density_fitting": "none",
        "atoms": len(atoms),
        "geometry_sha256": hashlib.sha256(
            json.dumps(atoms, sort_keys=True).encode()
        ).hexdigest(),
    }


def native_constructor_kwargs(method: str) -> dict[str, Any]:
    """Pass only scientific inputs and the mandatory hybrid API grid spec."""
    if method not in METHODS:
        raise ValueError("unknown native method")
    kwargs: dict[str, Any] = {
        "method": METHODS[method][0],
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
    }
    if method in REQUIRED_NATIVE_GRID:
        from generativeqc import GridSpec, KsOptions

        # No qualified automatic hybrid grid exists in the public native API.
        kwargs["ks_options"] = KsOptions(grid=GridSpec())
    return kwargs


def _finite_endpoint(
    energy: float, forces: Any, atoms: int, seconds: float
) -> list[list[float]]:
    import numpy as np

    array = np.asarray(forces, dtype=float)
    if (
        not math.isfinite(energy)
        or not math.isfinite(seconds)
        or seconds <= 0
        or array.shape != (atoms, 3)
        or not np.isfinite(array).all()
    ):
        raise RuntimeError("incomplete/nonfinite cold E+F endpoint")
    return array.tolist()


def _function_name(value: Any) -> str | None:
    return getattr(value, "__name__", None) if value is not None else None


def _grid_point_count(grid: Any) -> int | None:
    points = getattr(grid, "coords", None)
    return len(points) if points is not None else None


def native_sample(method: str, atoms: Any, force_policy: str) -> dict[str, Any]:
    """Use the public CUDA Calculator with no SCF/seed/threshold overrides."""
    import cupy as cp
    from generativeqc import Calculator
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_POLICY,
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )

    if force_policy not in FORCE_POLICIES:
        raise ValueError("unknown force policy")
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    calculator = Calculator(**native_constructor_kwargs(method))
    result = calculator.singlepoint(atoms, properties=("energy", "forces"))
    cp.cuda.runtime.deviceSynchronize()
    seconds = perf_counter() - start
    if not result.converged or result.executed_backend != "cuda":
        raise RuntimeError("native CUDA default SCF did not converge")
    if result.forces is None:
        raise RuntimeError("native force endpoint is unavailable")
    forces = _finite_endpoint(float(result.energy), result.forces, len(atoms), seconds)
    grid = calculator.ks_options.grid
    seed = calculator.initial_guess
    diagnostic = result.initial_guess if isinstance(result.initial_guess, dict) else None
    settings = {
        "initial_guess": seed.kind if seed is not None else "hcore",
        "initial_guess_policy": "auto",
        "initial_guess_executed": (
            diagnostic.get("kind")
            if diagnostic is not None and diagnostic.get("outcome") == "used"
            else "hcore-or-existing-density"
        ),
        "initial_guess_outcome": (
            diagnostic.get("outcome") if diagnostic is not None else None
        ),
        "max_iterations": calculator._max_iterations,
        "energy_tolerance": calculator._energy_tolerance,
        "density_tolerance": calculator._density_tolerance,
        "diis_history": calculator._diis_history,
        "screening_tolerance": calculator._screening_tolerance,
        "xc_grid": {
            "rule": "native-resolved-GridSpec",
            "version": grid.version,
            "radial_points": grid.radial_points,
            "angular_polar": grid.angular_polar,
            "angular_azimuth": grid.angular_azimuth,
            "pruning": grid.pruning,
            "points": getattr(result.ks_diagnostic, "grid_points", None),
        },
        "mandatory_explicit_hybrid_GridSpec": method in REQUIRED_NATIVE_GRID,
        "vv10_density": (
            {
                "policy": MOLECULAR_VV10_DENSITY_POLICY,
                "threshold": float(MOLECULAR_VV10_DENSITY_THRESHOLD),
            }
            if method == "wb97m-v"
            else None
        ),
    }
    sample = {
        "seconds": seconds,
        "energy": float(result.energy),
        "forces": forces,
        "converged": True,
        "iterations": int(result.iterations),
        "grid_response": True,
        "settings": settings,
        "native_library_path": str(Path(calculator._library._name).resolve()),
    }
    del calculator
    gc.collect()
    return sample


def gpu4_sample(method: str, atoms: Any, force_policy: str) -> dict[str, Any]:
    """Preserve RKS/SCF/grid defaults; optional full gradient is explicit."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto

    if force_policy not in FORCE_POLICIES:
        raise ValueError("unknown force policy")
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        cart=False,
        charge=0,
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    engine = rks.RKS(mol)
    engine.xc = METHODS[method][1]
    energy = float(engine.kernel())
    if not engine.converged:
        raise RuntimeError("GPU4PySCF default SCF did not converge")
    gradient = engine.nuc_grad_method()
    gradient_default = bool(getattr(gradient, "grid_response", False))
    if force_policy == "full-moving-grid":
        gradient.grid_response = True
    actual_response = bool(getattr(gradient, "grid_response", False))
    forces = -cp.asnumpy(gradient.kernel())
    cp.cuda.runtime.deviceSynchronize()
    seconds = perf_counter() - start
    forces = _finite_endpoint(energy, forces, len(atoms), seconds)
    settings = {
        "initial_guess": str(getattr(engine, "init_guess", "unavailable")),
        "max_iterations": int(engine.max_cycle),
        "energy_tolerance": float(engine.conv_tol),
        "orbital_gradient_tolerance": (
            float(engine.conv_tol_grad) if engine.conv_tol_grad is not None else None
        ),
        "diis_history": int(engine.diis_space),
        "direct_scf": bool(engine.direct_scf),
        "direct_scf_tolerance": float(engine.direct_scf_tol),
        "xc_grid": {
            "rule": "GPU4PySCF-unmodified-default",
            "level": int(engine.grids.level),
            "radial": _function_name(engine.grids.radi_method),
            "angular_pruning": _function_name(engine.grids.prune),
            "radii_adjust": _function_name(engine.grids.radii_adjust),
            "becke": _function_name(engine.grids.becke_scheme),
        },
    }
    settings["xc_grid"]["points"] = _grid_point_count(engine.grids)
    settings["nonlocal_grid_points"] = (
        _grid_point_count(engine.nlcgrids) if method == "wb97m-v" else None
    )
    settings["force_grid_response_default"] = gradient_default
    settings["force_grid_response_executed"] = actual_response
    if method == "wb97m-v":
        from gpu4pyscf.dft import numint
        from gpu4pyscf.grad import rks as grad_rks

        settings["vv10_density"] = {
            "scf": getattr(numint, "NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD", None),
            "forces": getattr(grad_rks, "NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD", None),
        }
    sample = {
        "seconds": seconds,
        "energy": energy,
        "forces": forces,
        "converged": True,
        "iterations": (
            int(engine.cycles) if getattr(engine, "cycles", None) is not None else None
        ),
        "grid_response": actual_response,
        "settings": settings,
    }
    del gradient, engine, mol
    gc.collect()
    return sample


def _checked_samples(record: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    if record.get("schema") != SCHEMA or record.get("engine") != engine:
        raise ValueError("wrong engine or schema")
    if record.get("status") != "complete":
        raise ValueError("uncompleted default benchmark cannot be ranked")
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("missing timed samples")
    if len(samples) != record.get("repeats"):
        raise ValueError("missing or extra repetitions")
    atoms = record.get("problem", {}).get("atoms")
    if not isinstance(atoms, int) or atoms <= 0:
        raise ValueError("invalid atom count in measured scientific input")
    for sample in samples:
        seconds = sample.get("seconds")
        energy = sample.get("energy")
        forces = sample.get("forces")
        if (
            not sample.get("converged")
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
            or not isinstance(energy, (int, float))
            or not math.isfinite(energy)
            or not isinstance(forces, list)
            or len(forces) != atoms
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in row
                )
                for row in forces
            )
        ):
            raise ValueError("failed or partial default benchmark sample")
    return samples


def compare(native: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Return the observed default-runtime ratio, never an accuracy assertion."""
    left = _checked_samples(native, "generativeqc")
    right = _checked_samples(reference, "gpu4pyscf")
    for key in ("problem", "force_policy", "phase", "endpoint"):
        if native.get(key) != reference.get(key):
            raise ValueError(f"mismatched default comparison: {key}")
    if native["force_policy"] not in FORCE_POLICIES:
        raise ValueError("unknown default force policy")
    # An apples-to-apples device comparison needs at least the same GPU model.
    native_gpu, reference_gpu = native.get("accelerator"), reference.get("accelerator")
    checked_hardware = isinstance(native_gpu, dict) and isinstance(reference_gpu, dict)
    if checked_hardware:
        for property_name in (
            "name", "compute_capability", "total_global_memory_bytes",
            "multiprocessor_count",
        ):
            if native_gpu.get(property_name) != reference_gpu.get(property_name):
                raise ValueError(f"different GPU hardware: {property_name}")
    n_time = statistics.median(s["seconds"] for s in left)
    g_time = statistics.median(s["seconds"] for s in right)
    n_response = {s["grid_response"] for s in left}
    g_response = {s["grid_response"] for s in right}
    if len(n_response) != 1 or len(g_response) != 1:
        raise ValueError("force definition changed within one engine")
    return {
        "schema": "generativeqc.production-default-compare.v1",
        "problem": native["problem"],
        "force_policy": native["force_policy"],
        "native_median_seconds": n_time,
        "gpu4pyscf_median_seconds": g_time,
        "observed_default_runtime_ratio_gpu4_over_native": g_time / n_time,
        "native_iterations": [s.get("iterations") for s in left],
        "gpu4pyscf_iterations": [s.get("iterations") for s in right],
        "native_grid": left[0]["settings"]["xc_grid"],
        "gpu4pyscf_grid": right[0]["settings"]["xc_grid"],
        "native_initial_guess": left[0]["settings"]["initial_guess"],
        "gpu4pyscf_initial_guess": right[0]["settings"]["initial_guess"],
        "same_moving_grid_response": n_response == g_response,
        "force_definition_warning": (
            None
            if n_response == g_response
            else "Different analytic-force definitions under stock defaults; "
            "performance ratio is not an equal-work force speedup."
        ),
        "accuracy": "unverified: no independent grid-converged E/F oracle",
        "science_equivalence_proved": False,
        "equal_fock_work_proved": False,
        "gpu_hardware_model_verified": checked_hardware,
        "runtime_scope": "fresh owner, initialized CUDA context, "
        "constructor + SCF + forces + host export",
        "native_settings": left[0]["settings"],
        "gpu4pyscf_settings": right[0]["settings"],
    }


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("generativeqc", "gpu4pyscf", "compare"))
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--atoms", type=int, choices=SIZES, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--force-policy", choices=FORCE_POLICIES, default="engine-default"
    )
    parser.add_argument("--native", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=raw_output_path)
    args = parser.parse_args(argv)
    if args.engine == "compare":
        if args.native is None or args.reference is None:
            parser.error("compare requires both --native and --reference")
        receipt = compare(
            json.loads(args.native.read_text()),
            json.loads(args.reference.read_text()),
        )
        _save(args.output, receipt)
        print(
            json.dumps(
                {
                    "runtime_ratio": receipt[
                        "observed_default_runtime_ratio_gpu4_over_native"
                    ],
                    "accuracy": receipt["accuracy"],
                }
            ),
            flush=True,
        )
        return
    if args.method is None:
        parser.error("--method is required to measure a GPU engine")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU measurement requires a finite Slurm allocation")
    import cupy as cp

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    sampler = native_sample if args.engine == "generativeqc" else gpu4_sample
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "engine": args.engine,
        "problem": problem_identity(args.method, atoms),
        "phase": "fresh-owner",
        "endpoint": "constructor-scf-energy-analytic-force-host-export",
        "force_policy": args.force_policy,
        "repeats": args.repeats,
        "samples": [],
        "controls": {
            "scf_overrides": False,
            "initial_guess_overrides": False,
            "reference_grid_overrides": False,
            "native_mandatory_hybrid_grid": args.method in REQUIRED_NATIVE_GRID,
        },
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
                "numpy": ("numpy",),
            }
        ),
        "accelerator": cuda_accelerator_metadata(cp),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _save(args.output, record)
    try:
        for repeat in range(args.repeats):
            sample = sampler(args.method, atoms, args.force_policy)
            sample["repeat"] = repeat
            record["samples"].append(sample)
            _save(args.output, record)
        if args.engine == "generativeqc":
            library_path = Path(record["samples"][0]["native_library_path"])
            digest = hashlib.sha256()
            with library_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            record["native_library_sha256"] = digest.hexdigest()
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _save(args.output, record)
    print(
        json.dumps(
            {
                "engine": args.engine,
                "seconds": [s["seconds"] for s in record["samples"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
