"""Benchmark GPU4PySCF defaults and GenerativeQC Standard/Fast defaults.

Each arm runs in its own process on the same GPU. Scientific inputs are shared;
SCF convergence, initial guesses, grids, pruning and screening are not copied
between programs, and Fast uses its public default application error budget.
Complete cold E+F timings include any real Fast strict/audit fallback solves.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.production-default-ef.v2"
METHODS = {
    "lda": ("lda-rks", "LDA_X,LDA_C_PW"),
    "pbe": ("pbe-rks", "PBE"),
    "pbe0": ("pbe0-rks", "PBE0"),
    "r2scan": ("r2scan-rks", "R2SCAN"),
    "wb97m-v": ("wb97m-v-rks", "WB97M_V"),
}
REQUIRED_NATIVE_GRID = frozenset(("pbe0", "wb97m-v"))
ARMS = ("gpu4pyscf", "standard", "fast")
NATIVE_ARMS = ("standard", "fast")
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


def make_native_calculator(method: str, arm: str) -> Any:
    """Select real public default implementations; do not pass numerical flags."""
    from generativeqc import Calculator, PrecisionCalculator

    options = native_constructor_kwargs(method)
    if arm == "standard":
        return Calculator(**options)
    if arm == "fast":
        return PrecisionCalculator(policy="fast", **options)
    raise ValueError("native arm must be standard or fast")


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


def native_sample(method: str, atoms: Any, arm: str) -> dict[str, Any]:
    """Time Standard/Fast constructor, cold SCF, complete forces and host output."""
    import cupy as cp
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_POLICY,
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )

    if arm not in NATIVE_ARMS:
        raise ValueError("unsupported native precision profile")
    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    calculator = make_native_calculator(method, arm)
    result = calculator.singlepoint(atoms, properties=("energy", "forces"))
    cp.cuda.runtime.deviceSynchronize()
    seconds = perf_counter() - started

    if not result.converged or result.executed_backend != "cuda":
        raise RuntimeError(f"GenerativeQC {arm} CUDA SCF did not converge")
    if result.forces is None:
        raise RuntimeError(f"GenerativeQC {arm} force endpoint unavailable")
    forces = _finite_endpoint(float(result.energy), result.forces, len(atoms), seconds)

    # Fast is a public PrecisionCalculator wrapper; its internally selected
    # Calculator may have DIFFERENT default controls and may perform an FP64
    # reference audit. Read the result's actual status, never assume FP32.
    candidate = calculator if arm == "standard" else calculator._candidate
    validation = result.precision_validation
    if arm == "fast":
        if validation is None or validation.profile != "fast":
            raise RuntimeError("Fast result lacks its actual precision validation")
        validation_info: dict[str, Any] | None = {
            "profile": validation.profile,
            "status": validation.status,
            "scope": validation.scope,
            "solve_count": validation.solve_count,
            "gradient_admission": validation.gradient_admission,
            "fallback_reasons": list(validation.fallback_reasons),
            "budget": asdict(validation.budget)
            if validation.budget is not None
            else None,
            "solves": [
                {
                    "role": solve.role,
                    "requested_precision": solve.requested_precision,
                    "iterations": solve.iterations,
                    "fock_builds": solve.fock_builds,
                    "seconds": solve.seconds,
                }
                for solve in validation.solves
            ],
        }
    else:
        validation_info = None
    grid = candidate.ks_options.grid
    guess = candidate.initial_guess
    initial = result.initial_guess if isinstance(result.initial_guess, dict) else None
    work = result.precision or {}
    gradient = work.get("gradient") if isinstance(work, dict) else None
    settings = {
        "precision_profile": "fp64" if arm == "standard" else "fast",
        "selected_candidate_precision": (
            "fp64" if arm == "standard" else calculator._candidate_precision
        ),
        "initial_guess": guess.kind if guess is not None else "hcore",
        "initial_guess_policy": "auto",
        "initial_guess_executed": (
            initial.get("kind")
            if initial is not None and initial.get("outcome") == "used"
            else "hcore-or-existing-density"
        ),
        "initial_guess_outcome": initial.get("outcome") if initial else None,
        "max_iterations": candidate._max_iterations,
        "energy_tolerance": candidate._energy_tolerance,
        "density_tolerance": candidate._density_tolerance,
        "diis_history": candidate._diis_history,
        "screening_tolerance": candidate._screening_tolerance,
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
        "force_grid_response": True,
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
        "precision_validation": validation_info,
        "mixed_fock_builds": work.get("mixed_stage_fock_builds")
        if isinstance(work, dict)
        else None,
        "gradient_arithmetic": gradient.get("returned_mode")
        if isinstance(gradient, dict)
        else None,
        "native_library_path": str(Path(candidate._library._name).resolve()),
    }
    del calculator, candidate
    gc.collect()
    return sample


def gpu4_sample(method: str, atoms: Any) -> dict[str, Any]:
    """Time stock GPU4PySCF, without SCF, grid or force-response overrides."""
    import cupy as cp
    from gpu4pyscf.dft import rks
    from pyscf import gto

    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
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
    # Important: even grid_response remains at GPU4PySCF's REAL default.
    grid_response = bool(getattr(gradient, "grid_response", False))
    forces = -cp.asnumpy(gradient.kernel())
    cp.cuda.runtime.deviceSynchronize()
    seconds = perf_counter() - started
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
            "points": _grid_point_count(engine.grids),
        },
        "nonlocal_grid_points": (
            _grid_point_count(engine.nlcgrids) if method == "wb97m-v" else None
        ),
        "force_grid_response_default": grid_response,
        "force_grid_response_executed": grid_response,
    }
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
        "grid_response": grid_response,
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
        if engine == "fast":
            report = sample.get("precision_validation")
            if (
                not isinstance(report, dict)
                or report.get("profile") != "fast"
                or not isinstance(report.get("status"), str)
                or not isinstance(report.get("solve_count"), int)
                or report["solve_count"] < 1
            ):
                raise ValueError("Fast execution/audit provenance is missing")
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


def compare(
    gpu4pyscf: dict[str, Any],
    standard: dict[str, Any],
    fast: dict[str, Any],
) -> dict[str, Any]:
    """One three-way default comparison; never claim equal accuracy/work."""
    records = {"gpu4pyscf": gpu4pyscf, "standard": standard, "fast": fast}
    samples = {arm: _checked_samples(records[arm], arm) for arm in ARMS}
    for key in ("problem", "phase", "endpoint"):
        if (
            len({json.dumps(records[arm].get(key), sort_keys=True) for arm in ARMS})
            != 1
        ):
            raise ValueError(f"mismatched default comparison: {key}")
    # One real GPU model is mandatory for a reported speed ratio. Host-only
    # synthetic receipts may omit all accelerator metadata, but live records do not.
    hardware = [records[arm].get("accelerator") for arm in ARMS]
    present = [isinstance(device, dict) for device in hardware]
    if any(present) and not all(present):
        raise ValueError("missing GPU hardware for one default-comparison arm")
    checked_hardware = all(present)
    if checked_hardware:
        for name in (
            "name",
            "compute_capability",
            "total_global_memory_bytes",
            "multiprocessor_count",
        ):
            if len({json.dumps(device.get(name)) for device in hardware}) != 1:
                raise ValueError(f"different GPU hardware: {name}")
    medians = {
        arm: statistics.median(row["seconds"] for row in samples[arm]) for arm in ARMS
    }
    responses = {arm: {row["grid_response"] for row in samples[arm]} for arm in ARMS}
    if any(len(response) != 1 for response in responses.values()):
        raise ValueError("force response policy varied within an arm")
    if responses["standard"] != responses["fast"]:
        raise ValueError("Standard and Fast returned different force definitions")
    equivalent_response = responses["standard"] == responses["gpu4pyscf"]
    fast_status = [row["precision_validation"]["status"] for row in samples["fast"]]
    fast_solve_counts = [
        row["precision_validation"]["solve_count"] for row in samples["fast"]
    ]
    problem = gpu4pyscf["problem"]
    row = {
        "method": problem["method"],
        "atoms": problem["atoms"],
        "gpu4pyscf_default_seconds": medians["gpu4pyscf"],
        "generativeqc_standard_seconds": medians["standard"],
        "generativeqc_fast_seconds": medians["fast"],
        "standard_speedup_vs_gpu4pyscf": medians["gpu4pyscf"] / medians["standard"],
        "fast_speedup_vs_gpu4pyscf": medians["gpu4pyscf"] / medians["fast"],
        "fast_speedup_vs_standard": medians["standard"] / medians["fast"],
    }
    return {
        "schema": "generativeqc.production-default-comparison.v2",
        "problem": problem,
        "main_table": [row],
        "medians_seconds": medians,
        "iterations": {
            arm: [sample["iterations"] for sample in samples[arm]] for arm in ARMS
        },
        "fast_statuses": fast_status,
        "fast_internal_solve_counts": fast_solve_counts,
        "engine_settings": {arm: samples[arm][0]["settings"] for arm in ARMS},
        "same_gpu_hardware_verified": checked_hardware,
        "same_moving_grid_force_definition": equivalent_response,
        "force_definition_warning": (
            None
            if equivalent_response
            else "GPU4PySCF uses its stock force grid_response default; "
            "GenerativeQC Standard/Fast use full moving-grid response. "
            "This is default end-user E+F runtime, not equal-force-work speedup."
        ),
        "accuracy": "not_qualified_against_independent_grid_converged_oracle",
        "science_equivalence_proved": False,
        "equal_fock_work_proved": False,
        "runtime_scope": "new calculator owner, initialized CUDA context, "
        "constructor+SCF+analytic force+host export",
    }


def main_table_markdown(comparisons: list[dict[str, Any]]) -> str:
    """Render exactly one performance table, never a separate accuracy table."""
    headers = (
        "| Method | Atoms | GPU4PySCF default (s) | "
        "GQC Standard (s) | GQC Fast (s) | Standard / GPU4 | Fast / GPU4 |"
    )
    lines = [headers, "|---|---:|---:|---:|---:|---:|---:|"]
    rows = [row for item in comparisons for row in item["main_table"]]
    for row in sorted(rows, key=lambda item: (item["method"], item["atoms"])):
        lines.append(
            f"| {row['method']} | {row['atoms']} | "
            f"{row['gpu4pyscf_default_seconds']:.3f} | "
            f"{row['generativeqc_standard_seconds']:.3f} | "
            f"{row['generativeqc_fast_seconds']:.3f} | "
            f"{row['standard_speedup_vs_gpu4pyscf']:.2f}x | "
            f"{row['fast_speedup_vs_gpu4pyscf']:.2f}x |"
        )
    return "\n".join(lines)


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=(*ARMS, "compare", "all", "render"))
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--atoms", type=int, choices=SIZES, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu4pyscf", type=Path)
    parser.add_argument("--standard", type=Path)
    parser.add_argument("--fast", type=Path)
    parser.add_argument("--comparisons", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=raw_output_path)
    args = parser.parse_args(argv)

    if args.engine == "render":
        if not args.comparisons:
            parser.error("render requires --comparisons")
        comparisons = [json.loads(path.read_text()) for path in args.comparisons]
        if any(
            record.get("schema") != "generativeqc.production-default-comparison.v2"
            for record in comparisons
        ):
            raise ValueError("cannot render an unverified/older comparison schema")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table = main_table_markdown(comparisons)
        args.output.write_text(table + "\n")
        print(table, flush=True)
        return

    if args.engine == "compare":
        if args.gpu4pyscf is None or args.standard is None or args.fast is None:
            parser.error("compare requires --gpu4pyscf, --standard and --fast")
        result = compare(
            json.loads(args.gpu4pyscf.read_text()),
            json.loads(args.standard.read_text()),
            json.loads(args.fast.read_text()),
        )
        _save(args.output, result)
        print(main_table_markdown([result]), flush=True)
        return

    if args.method is None:
        parser.error("--method is required for measured engines and all mode")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU runs require a finite Slurm GPU allocation")

    if args.engine == "all":
        # Keep all three execution arms in independent Python processes, so
        # each has its own GPU/PySCF native state and clean cold seed.
        receipts = {
            arm: args.output.with_name(f"{args.output.stem}-{arm}.json") for arm in ARMS
        }
        for arm in ARMS:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.production_default_compare",
                    arm,
                    "--method",
                    args.method,
                    "--atoms",
                    str(args.atoms),
                    "--repeats",
                    str(args.repeats),
                    "--output",
                    str(receipts[arm]),
                ],
                check=True,
            )
        result = compare(
            json.loads(receipts["gpu4pyscf"].read_text()),
            json.loads(receipts["standard"].read_text()),
            json.loads(receipts["fast"].read_text()),
        )
        _save(args.output, result)
        print(main_table_markdown([result]), flush=True)
        return

    import cupy as cp

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    arm = args.engine
    sample = (
        (lambda: gpu4_sample(args.method, atoms))
        if arm == "gpu4pyscf"
        else (lambda: native_sample(args.method, atoms, arm))
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "engine": arm,
        "problem": problem_identity(args.method, atoms),
        "phase": "fresh-owner",
        "endpoint": "constructor-scf-energy-analytic-force-host-export",
        "repeats": args.repeats,
        "samples": [],
        "controls": {
            "scf_overrides": False,
            "initial_guess_overrides": False,
            "reference_grid_overrides": False,
            "native_required_hybrid_GridSpec": (
                arm in NATIVE_ARMS and args.method in REQUIRED_NATIVE_GRID
            ),
            "fast_default_policy": "fast" if arm == "fast" else None,
            "force_grid_response_override": False,
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
            observation = sample()
            observation["repeat"] = repeat
            record["samples"].append(observation)
            _save(args.output, record)
        if arm in NATIVE_ARMS:
            library_path = Path(record["samples"][0]["native_library_path"])
            if any(
                row["native_library_path"] != str(library_path)
                for row in record["samples"]
            ):
                raise RuntimeError("native library path changed across repetitions")
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
                "engine": arm,
                "seconds": [row["seconds"] for row in record["samples"]],
                "status": record["status"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
