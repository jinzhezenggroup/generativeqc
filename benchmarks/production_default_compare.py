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
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.production-default-ef.v3"
COMPARISON_SCHEMA = "generativeqc.production-default-comparison.v3"
PHASE = "fresh-owner"
ENDPOINT = "constructor-scf-energy-analytic-force-host-export"
# Receipt protocol identity, not an override passed to a calculator. A host
# regression compares this snapshot with PrecisionPolicy("fast").budget.
DEFAULT_FAST_BUDGET = {"energy_abs": 1e-5, "force_max_abs": 1e-4, "force_rms": 3e-5}
METHODS = {
    "lda": ("lda-rks", "LDA_X,LDA_C_PW"),
    "pbe": ("pbe-rks", "PBE"),
    "pbe0": ("pbe0-rks", "PBE0"),
    "r2scan": ("r2scan-rks", "R2SCAN"),
    "r2scan-3c": ("r2scan-3c-rks", "R2SCAN"),
    "wb97m-v": ("wb97m-v-rks", "WB97M_V"),
}
ARMS = ("gpu4pyscf", "standard", "fast")
NATIVE_ARMS = ("standard", "fast")
SIZES = (3, 6, 12, 24, 48, 96)


def problem_identity(method: str, atoms: Any) -> dict[str, Any]:
    """Match the Hamiltonian input, not the execution/numerical defaults."""
    if method not in METHODS:
        raise ValueError("unsupported default-comparison method")
    return {
        "method": method,
        "basis": "def2-mTZVPP" if method == "r2scan-3c" else "def2-svp",
        "composite_corrections": ["D4", "gCP"] if method == "r2scan-3c" else [],
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
    """Pass model/basis/device; let native select its implicit level-3 grid."""
    if method not in METHODS:
        raise ValueError("unknown native method")
    kwargs: dict[str, Any] = {
        "method": METHODS[method][0],
        "basis_representation": "spherical",
        "device": "cuda",
    }
    if method != "r2scan-3c":
        kwargs["basis"] = "def2-svp"
    # Canonical r2SCAN-3c selects its pinned def2-mTZVPP internally.
    # In #2626 an omitted grid resolves to v4 Treutler/Lebedev/NWChem level 3,
    # including PBE0 and WB97M-V. GridSpec() is still a LEGACY v1 grid.
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


def reference_r2scan3c_basis() -> dict[str, list[list[Any]]]:
    """Convert canonical def2-mTZVPP snapshot to independent PySCF basis."""
    from generativeqc import load_r2scan3c_basis
    from generativeqc.elements import SYMBOLS

    basis = load_r2scan3c_basis()
    return {
        SYMBOLS[element.atomic_number - 1]: [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *(float(column[i]) for column in shell.coefficients),
                    ]
                    for i, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in basis.elements
    }


def reference_r2scan3c_correction(atoms: Any) -> Any:
    """Use the existing independent, qualified CPU D4+gCP reference."""
    from benchmarks.readme_omol25 import reference_correction

    return reference_correction(atoms)


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
    dispersion = getattr(result, "dispersion", None)
    if method == "r2scan-3c" and not (
        dispersion is not None
        and getattr(dispersion, "ok", False)
        and getattr(dispersion.d4, "ok", False)
        and getattr(dispersion.gcp, "ok", False)
    ):
        raise RuntimeError("native r2SCAN-3c omitted D4/gCP corrections")

    # Fast is a public PrecisionCalculator wrapper; its internally selected
    # Calculator may have DIFFERENT default controls and may perform an FP64
    # reference audit. Read the result's actual status, never assume FP32.
    candidate = calculator if arm == "standard" else calculator._candidate
    validation = result.precision_validation
    if arm == "fast":
        if validation is None or validation.profile != "fast":
            raise RuntimeError("Fast result lacks its actual precision validation")
        validation_info: dict[str, Any] | None = validation.to_dict()
    else:
        validation_info = None
    grid = candidate.ks_options.grid
    if grid.version != 4 or grid.level != 3 or grid.pruning != "nwchem":
        raise RuntimeError(
            "default native Treutler/Lebedev level-3 grid is not in effect"
        )
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
            "rule": "native-implicit-Treutler-Lebedev-NWChem",
            "version": grid.version,
            "level": grid.level,
            "radial_rule": grid.radial_rule,
            "angular_rule": grid.angular_rule,
            "partition": grid.partition,
            "radial_points": grid.radial_points,
            "angular_polar": grid.angular_polar,
            "angular_azimuth": grid.angular_azimuth,
            "pruning": grid.pruning,
            "points": getattr(result.ks_diagnostic, "grid_points", None),
        },
        "grid_source": "implicit-native-default",
        "nonlocal_grid_inherits_xc": (
            candidate.ks_options.nonlocal_grid is None if method == "wb97m-v" else None
        ),
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
        "composite_corrections": (
            {
                "provider": "native-canonical-D4+gCP",
                "complete": True,
                "included_in_timing": True,
                "energy_hartree": float(dispersion.energy),
            }
            if method == "r2scan-3c"
            else None
        ),
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
    orbital_basis = reference_r2scan3c_basis() if method == "r2scan-3c" else "def2-svp"
    mol = gto.M(
        atom=atoms,
        basis=orbital_basis,
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
    # GPU4PySCF owns default electronic r2SCAN; independent CPU D4+gCP
    # provides the other two canonical 3c contributions, entirely IN time.
    correction_energy = None
    if method == "r2scan-3c":
        correction_energy, correction_gradient = reference_r2scan3c_correction(atoms)()
        energy += float(correction_energy)
        forces -= correction_gradient
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
        "composite_corrections": (
            {
                "provider": "independent-CPU-dftd4+gCP",
                "d4": {
                    "s6": 1.0,
                    "s8": 0.0,
                    "s9": 2.0,
                    "a1": 0.42,
                    "a2": 5.65,
                    "ga": 2.0,
                    "gc": 1.0,
                },
                "included_in_timing": True,
                "energy_hartree": float(correction_energy),
            }
            if method == "r2scan-3c"
            else None
        ),
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


def _positive_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _checked_protocol(record: dict[str, Any], engine: str) -> None:
    problem = record.get("problem")
    if not isinstance(problem, dict) or (
        not isinstance(problem.get("method"), str)
        or problem["method"] not in METHODS
        or problem.get("basis")
        != ("def2-mTZVPP" if problem["method"] == "r2scan-3c" else "def2-svp")
        or problem.get("composite_corrections")
        != (["D4", "gCP"] if problem["method"] == "r2scan-3c" else [])
        or problem.get("basis_representation") != "spherical"
        or type(problem.get("charge")) is not int
        or problem["charge"] != 0
        or type(problem.get("spin")) is not int
        or problem["spin"] != 0
        or problem.get("density_fitting") != "none"
        or type(problem.get("atoms")) is not int
        or problem["atoms"] <= 0
        or not _digest(problem.get("geometry_sha256"))
    ):
        raise ValueError("incomplete default scientific identity")
    if record.get("phase") != PHASE or record.get("endpoint") != ENDPOINT:
        raise ValueError("receipt is not a fresh-owner complete E+F endpoint")
    expected_controls = {
        "scf_overrides": False,
        "initial_guess_overrides": False,
        "reference_grid_overrides": False,
        "native_explicit_grid_overrides": False,
        "fast_default_policy": "fast" if engine == "fast" else None,
        "force_grid_response_override": False,
    }
    controls = record.get("controls")
    if not isinstance(controls, dict) or any(
        key not in controls
        or type(controls[key]) is not type(value)
        or controls[key] != value
        for key, value in expected_controls.items()
    ):
        raise ValueError("missing or overridden default controls")
    device = record.get("accelerator")
    if not isinstance(device, dict) or (
        not isinstance(device.get("name"), str)
        or not device["name"].strip()
        or not isinstance(device.get("compute_capability"), list)
        or len(device["compute_capability"]) != 2
        or any(
            type(value) is not int or value < 0
            for value in device["compute_capability"]
        )
        or any(
            type(device.get(key)) is not int or device[key] <= 0
            for key in ("total_global_memory_bytes", "multiprocessor_count")
        )
    ):
        raise ValueError("missing or incomplete GPU hardware")
    environment = record.get("environment")
    source = environment.get("git") if isinstance(environment, dict) else None
    if (
        not isinstance(source, dict)
        or not _digest(source.get("commit"), 40)
        or type(source.get("dirty")) is not bool
        or not _digest(record.get("source_sha256"))
        or (engine in NATIVE_ARMS and not _digest(record.get("native_library_sha256")))
    ):
        raise ValueError("incomplete source or native binary provenance")


def _checked_audit_errors(
    errors: Any, budget: dict[str, Any], *, successful: bool
) -> None:
    allowances = {
        ("energy", "absolute", "Eh"): budget["energy_abs"],
        ("forces", "max_abs", "Eh/bohr"): budget["force_max_abs"],
        ("forces", "rms", "Eh/bohr"): budget["force_rms"],
    }
    if not isinstance(errors, (list, tuple)) or len(errors) != len(allowances):
        raise ValueError("Fast observed audit lacks complete E/F norms")
    observed = set()
    met = []
    for error in errors:
        if not isinstance(error, dict) or any(
            type(error.get(key)) not in (int, float)
            or not math.isfinite(error[key])
            or error[key] < 0
            for key in ("difference", "reference_norm", "allowance")
        ):
            raise ValueError("Fast audit error evidence is incomplete or nonfinite")
        identity = (error.get("observable"), error.get("norm"), error.get("unit"))
        if (
            any(not isinstance(value, str) for value in identity)
            or identity not in allowances
        ):
            raise ValueError("Fast observed audit has an unknown E/F norm")
        if error["allowance"] != allowances[identity]:
            raise ValueError("Fast audit allowance differs from its default budget")
        observed.add(identity)
        met.append(error["difference"] <= error["allowance"])
    if observed != set(allowances) or all(met) != successful:
        raise ValueError("Fast audit norms disagree with the reported acceptance")


def _checked_fast_report(report: Any) -> None:
    if not isinstance(report, dict) or (
        report.get("profile") != "fast"
        or not isinstance(report.get("status"), str)
        or report.get("status")
        not in {
            "observed_met",
            "qualified_budget_fock",
            "qualified_mixed_gradient",
            "qualified_strict_final",
        }
        or type(report.get("solve_count")) is not int
        or report["solve_count"] < 1
        or not isinstance(report.get("solves"), (list, tuple))
        or len(report["solves"]) != report["solve_count"]
        or not isinstance(report.get("fallback_reasons"), (list, tuple))
        or any(not isinstance(reason, str) for reason in report["fallback_reasons"])
        or not isinstance(report.get("scope"), str)
        or not isinstance(report.get("gradient_admission"), str)
        or not isinstance(report.get("budget"), dict)
        or report["budget"] != DEFAULT_FAST_BUDGET
        or any(
            not _positive_number(report["budget"].get(key))
            for key in ("energy_abs", "force_max_abs", "force_rms")
        )
    ):
        raise ValueError("Fast execution/audit provenance is missing or unsuccessful")
    for solve in report["solves"]:
        if not isinstance(solve, dict) or (
            not isinstance(solve.get("role"), str)
            or solve.get("requested_precision") not in ("fp64", "auto")
            or solve.get("executed_backend") != "cuda"
            or not _positive_number(solve.get("seconds"))
            or any(
                not _positive_number(solve.get(key))
                for key in (
                    "energy_tolerance",
                    "density_tolerance",
                    "screening_tolerance",
                )
            )
            or "error" not in solve
            or (solve["error"] is not None and not isinstance(solve["error"], str))
            or (
                solve["error"] is None
                and (
                    type(solve.get("iterations")) is not int
                    or solve["iterations"] < 1
                    or solve.get("force_evaluations") != 1
                )
            )
        ):
            raise ValueError("incomplete Fast solve provenance")
    if report["solves"][-1]["error"] is not None:
        raise ValueError("Fast final solve failed")
    if report["status"] == "observed_met":
        solves = report["solves"]
        roles = [solve["role"] for solve in solves]
        rejected = report.get("rejected_errors")
        if not isinstance(rejected, (list, tuple)):
            raise ValueError("Fast observed audit lacks rejected-candidate provenance")
        verification = bool(rejected)
        if verification != (
            "candidate_exceeded_observable_budget" in report["fallback_reasons"]
        ):
            raise ValueError("Fast rejection and verification provenance disagree")
        expected = ["candidate", "fp64_audit"]
        if solves[0]["error"] is not None:
            expected.insert(1, "fp64_fallback")
        elif roles[0] == "fp64_fallback":
            # A temporary property-capable candidate may fail before a recorded
            # solve; the public controller still records its strict fallback.
            expected[0] = "fp64_fallback"
        if (
            "fp64_fallback" in expected
            and "budget_fock_failed_fp64_fallback" not in report["fallback_reasons"]
        ):
            raise ValueError("Fast strict fallback lacks its failure reason")
        if verification:
            expected.append("fp64_fallback_verification")
        if roles != expected or any(
            solve["error"] is not None or solve["requested_precision"] != "fp64"
            for solve in solves
            if solve["role"] != "candidate"
        ):
            raise ValueError(
                "Fast observed audit lacks an independent successful FP64 census"
            )
        _checked_audit_errors(report.get("errors"), report["budget"], successful=True)
        if verification:
            _checked_audit_errors(rejected, report["budget"], successful=False)


def _checked_samples(record: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        raise TypeError("invalid benchmark receipt")
    if record.get("schema") != SCHEMA or record.get("engine") != engine:
        raise ValueError("wrong engine or schema")
    if record.get("status") != "complete":
        raise ValueError("uncompleted default benchmark cannot be ranked")
    _checked_protocol(record, engine)
    samples = record.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("missing timed samples")
    if type(record.get("repeats")) is not int or len(samples) != record["repeats"]:
        raise ValueError("missing or extra repetitions")
    if any(not isinstance(sample, dict) for sample in samples) or any(
        type(sample.get("repeat")) is not int or sample["repeat"] != index
        for index, sample in enumerate(samples)
    ):
        raise ValueError("missing, duplicate or out-of-order repetition identity")
    atoms = record.get("problem", {}).get("atoms")
    if not isinstance(atoms, int) or atoms <= 0:
        raise ValueError("invalid atom count in measured scientific input")
    for sample in samples:
        settings = sample.get("settings")
        if (
            type(sample.get("grid_response")) is not bool
            or (engine in NATIVE_ARMS and sample["grid_response"] is not True)
            or type(sample.get("iterations")) is not int
            or sample["iterations"] <= 0
            or not isinstance(settings, dict)
            or not isinstance(settings.get("initial_guess"), str)
            or not isinstance(settings.get("xc_grid"), dict)
            or not isinstance(settings["xc_grid"].get("rule"), str)
            or not _positive_number(settings["xc_grid"].get("points"))
            or (
                engine in NATIVE_ARMS
                and (
                    settings["xc_grid"].get("version") != 4
                    or settings["xc_grid"].get("level") != 3
                    or settings["xc_grid"].get("pruning") != "nwchem"
                    or settings.get("grid_source") != "implicit-native-default"
                )
            )
            or (
                engine in NATIVE_ARMS
                and settings.get("selected_candidate_precision") not in ("fp64", "auto")
            )
        ):
            raise ValueError(
                "missing executed settings, iterations or force definition"
            )
        if engine == "fast":
            _checked_fast_report(sample.get("precision_validation"))
        if record["problem"]["method"] == "r2scan-3c":
            correction = (
                sample.get("composite_corrections")
                if engine in NATIVE_ARMS
                else settings.get("composite_corrections")
            )
            provider = (
                "native-canonical-D4+gCP"
                if engine in NATIVE_ARMS
                else "independent-CPU-dftd4+gCP"
            )
            if not isinstance(correction, dict) or (
                correction.get("provider") != provider
                or correction.get("included_in_timing") is not True
                or type(correction.get("energy_hartree")) not in (int, float)
                or not math.isfinite(correction["energy_hartree"])
                or (engine in NATIVE_ARMS and correction.get("complete") is not True)
            ):
                raise ValueError(
                    "missing or incomplete timed r2SCAN-3c D4/gCP evidence"
                )
            if engine == "gpu4pyscf" and correction.get("d4") != {
                "s6": 1.0,
                "s8": 0.0,
                "s9": 2.0,
                "a1": 0.42,
                "a2": 5.65,
                "ga": 2.0,
                "gc": 1.0,
            }:
                raise ValueError("r2SCAN-3c reference D4 parameters are not canonical")
        seconds = sample.get("seconds")
        energy = sample.get("energy")
        forces = sample.get("forces")
        if (
            sample.get("converged") is not True
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
            or type(energy) not in (int, float)
            or not math.isfinite(energy)
            or not isinstance(forces, list)
            or len(forces) != atoms
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
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
    for key in ("problem", "phase", "endpoint", "repeats", "source_sha256"):
        if (
            len({json.dumps(records[arm].get(key), sort_keys=True) for arm in ARMS})
            != 1
        ):
            raise ValueError(f"mismatched default comparison: {key}")
    if standard["native_library_sha256"] != fast["native_library_sha256"]:
        raise ValueError("Standard and Fast used different native binary images")
    hardware = [records[arm]["accelerator"] for arm in ARMS]
    for name in (
        "name",
        "compute_capability",
        "total_global_memory_bytes",
        "multiprocessor_count",
    ):
        if len({json.dumps(device[name]) for device in hardware}) != 1:
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
    if any(
        not _positive_number(value)
        for key, value in row.items()
        if key not in ("method", "atoms")
    ):
        raise ValueError("nonfinite or nonpositive derived performance ratio")
    return {
        "schema": COMPARISON_SCHEMA,
        "receipts": records,
        "problem": problem,
        "main_table": [row],
        "medians_seconds": medians,
        "iterations": {
            arm: [sample["iterations"] for sample in samples[arm]] for arm in ARMS
        },
        "fast_statuses": fast_status,
        "fast_internal_solve_counts": fast_solve_counts,
        "engine_settings": {arm: samples[arm][0]["settings"] for arm in ARMS},
        "same_gpu_hardware_verified": True,
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
    checked = []
    for record in comparisons:
        if (
            not isinstance(record, dict)
            or record.get("schema") != COMPARISON_SCHEMA
            or not isinstance(record.get("receipts"), dict)
            or set(record["receipts"]) != set(ARMS)
        ):
            raise ValueError(
                "cannot render an unverified/older comparison or incomplete receipts"
            )
        checked.append(compare(*(record["receipts"][arm] for arm in ARMS)))
    if not checked:
        raise ValueError("cannot render an empty comparison")
    headers = (
        "| Method | Atoms | GPU4PySCF default (s) | "
        "GQC Standard (s) | GQC Fast (s) | Standard speedup | Fast speedup |"
    )
    lines = [
        "Accuracy unverified: independent grid-converged E/F qualification is pending.",
    ]
    if any(not item["same_moving_grid_force_definition"] for item in checked):
        lines.append(
            "Stock force definitions differ: GPU4PySCF keeps its default grid response; "
            "GenerativeQC includes moving-grid response. Speedups compare end-user latency."
        )
    if any(item["problem"]["method"] == "r2scan-3c" for item in checked):
        lines.append(
            "For r2SCAN-3c rows, the GPU4PySCF column means GPU4PySCF r2SCAN + "
            "independent CPU D4/gCP, with canonical def2-mTZVPP and all corrections timed."
        )
    lines.extend(("", headers, "|---|---:|---:|---:|---:|---:|---:|"))
    rows = [row for item in checked for row in item["main_table"]]
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
            record.get("schema") != COMPARISON_SCHEMA
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
            "native_explicit_grid_overrides": False,
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
