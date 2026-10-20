"""Default-grid README endpoints, adapted from the #2623 comparison protocol.

Each process owns one method/size/DF/engine population: one fresh cold E+F and
three warm E+F calls from its own frozen cold density. Both engines retain their
default XC/VV10 grids and native convergence policies, with complete moving-grid
response. Common energy/force gates qualify the endpoints independently of the
engines' unlike residual definitions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks._retention import raw_output_path

SCHEMA = "generativeqc.readme-defaults.v2"
# A size-normalized energy gate keeps the water series comparable; force gates
# are intensive. These are declared before measurement and applied equally to
# Standard, Fast and GPU4PySCF against independent tighter references.
COMMON_GATES = {
    "energy_abs_per_atom_hartree": 1e-5,
    "force_max_abs_hartree_per_bohr": 1e-4,
    "force_rms_hartree_per_bohr": 3e-5,
}
METHODS = {
    "hf": ("rhf", None),
    "pbe": ("pbe-rks", "PBE"),
    "pbe0": ("pbe0-rks", "PBE0"),
    "b3lyp": ("b3lyp-rks", "B3LYP"),
    "r2scan": ("r2scan-rks", "R2SCAN"),
    "r2scan-3c": ("r2scan-3c", "R2SCAN"),
    "m06-2x": ("m06-2x-rks", "M06-2X"),
    "wb97m-v": ("wb97m-v-rks", "WB97M_V"),
}
# Retained receipts include an abandoned reference variant. New measurements
# and README plots use only the three requested profiles.
ARMS = ("standard", "fast", "gpu4pyscf", "gpu4pyscf-fast")
MEASUREMENT_ARMS = ("standard", "fast", "gpu4pyscf")
SIZES = (3, 12, 48, 96)
ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Hash large native images without holding a second copy in memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path: Path, value: dict) -> None:
    """Publish each completed endpoint atomically, retaining partial jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def orbital_basis(method: str) -> Any:
    return reference_r2scan3c_basis() if method == "r2scan-3c" else "def2-svp"


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


def prepare_inputs(destination: Path) -> None:
    """Freeze GPU4PySCF's default auxiliary tables without any SCF/oracle work.

    GPU4PySCF calls PySCF make_auxmol(mol, None). Export exactly those tables
    once as explicit scientific input for native DF, which requires a basis.
    Native timed endpoints never import PySCF or ask it to execute a solve.
    """
    import pyscf
    from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
    from pyscf import df, gto

    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()["water-3"].atoms
    destination.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        mol = gto.M(
            atom=atoms, unit="Bohr", basis=orbital_basis(method), cart=False, verbose=0
        )
        auxmol = df.addons.make_auxmol(mol, None)
        tables = auxmol._basis
        checksum = hashlib.sha256(
            json.dumps(tables, sort_keys=True).encode()
        ).hexdigest()
        elements = []
        for symbol, shells in tables.items():
            elements.append(
                ElementBasis(
                    gto.charge(symbol),
                    tuple(
                        BasisShell(
                            shell[0],
                            tuple(str(row[0]) for row in shell[1:]),
                            tuple(
                                tuple(str(row[c]) for row in shell[1:])
                                for c in range(1, len(shell[1]))
                            ),
                        )
                        for shell in shells
                    ),
                )
            )
        basis = BasisSet(
            "GPU4PySCF-default-" + method,
            tuple(elements),
            BasisProvenance(
                "pyscf.df.addons.make_auxmol(default)",
                pyscf.__version__,
                "Apache-2.0 (PySCF distribution)",
                checksum,
            ),
            representation="spherical",
        )
        save(destination / f"{method}-auxiliary.json", basis.to_payload())
        save(
            destination / f"{method}-reference.json",
            {
                "basis": tables,
                "basis_sha256": checksum,
                "selection": auxmol.basis,
                "nao_water3": auxmol.nao_nr(),
                "pyscf_version": pyscf.__version__,
            },
        )


def native_calculator(method: str, arm: str, mode: str, inputs: Path) -> Any:
    """Use the public Standard/Fast owner and implicit XC/VV10 defaults."""
    from generativeqc import Calculator, PrecisionCalculator

    options = {
        "method": METHODS[method][0],
        "device": "cuda",
        "basis_representation": "spherical",
    }
    if method != "r2scan-3c":
        options["basis"] = "def2-svp"
    if mode == "df":
        options.update(
            density_fitting="cuda", auxiliary_basis=inputs / f"{method}-auxiliary.json"
        )
    return (
        PrecisionCalculator(policy="fast", **options)
        if arm == "fast"
        else Calculator(**options)
    )


def grid_settings(grid: Any) -> dict | None:
    """Snapshot actual reference grid rules, including a distinct VV10 grid."""
    if grid is None:
        return None
    return {
        "level": int(grid.level),
        "radial": _function_name(grid.radi_method),
        "angular_pruning": _function_name(grid.prune),
        "radii_adjust": _function_name(grid.radii_adjust),
        "becke": _function_name(grid.becke_scheme),
        "points": _grid_point_count(grid),
    }


def native_settings(calculator: Any, result: Any) -> dict:
    """Retain resolved controls and work without altering the timed calculation."""
    ks = calculator.ks_options
    return {
        "energy_tolerance": calculator._energy_tolerance,
        "density_tolerance": calculator._density_tolerance,
        "screening_tolerance": calculator._screening_tolerance,
        "max_iterations": calculator._max_iterations,
        "diis_history": calculator._diis_history,
        "initial_guess": str(calculator.initial_guess),
        "xc_grid": dataclasses.asdict(ks.grid) if ks is not None else None,
        "xc_grid_points": getattr(result.ks_diagnostic, "grid_points", None),
        "nonlocal_grid": dataclasses.asdict(ks.nonlocal_grid)
        if ks is not None and ks.nonlocal_grid
        else None,
        "nonlocal_grid_inherits_xc": ks.nonlocal_grid is None
        if ks is not None
        else None,
        "force_grid_response": True if ks is not None else None,
    }


def run_native(args: Any, atoms: Any, record: dict, retain: Any) -> None:
    """Include constructor/preparation in cold; freeze its own seed for warm."""
    import cupy as cp

    from benchmarks.readme_method_endpoints import binary_identity

    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    calculator = native_calculator(args.method, args.arm, args.mode, args.inputs)
    owner = calculator.prepare_batch([atoms], warm_start=True)
    try:
        for repeat in range(args.warm_repeats + 1):
            if repeat:
                cp.cuda.runtime.deviceSynchronize()
                started = perf_counter()
            batch = owner.execute(strict=True, properties=("energy", "forces"))
            item = batch.items[0]
            cp.cuda.runtime.deviceSynchronize()
            seconds = perf_counter() - started
            if not item.converged or item.executed_backend != "cuda":
                raise RuntimeError("native CUDA endpoint did not converge")
            forces = _finite_endpoint(
                float(item.energy), item.forces, len(atoms), seconds
            )
            candidate = calculator if args.arm == "standard" else owner._calculator
            if not record.get("native_build"):
                record["native_build"] = binary_identity(candidate)
            dispersion = item.dispersion
            if args.method == "r2scan-3c" and not (
                dispersion and dispersion.ok and dispersion.d4.ok and dispersion.gcp.ok
            ):
                raise RuntimeError("incomplete native D4/gCP endpoint")
            validation = item.precision_validation
            if args.arm == "fast" and (
                validation is None or validation.profile != "fast"
            ):
                raise RuntimeError("missing actual Fast precision provenance")
            retain(
                {
                    "phase": "cold" if repeat == 0 else "warm",
                    "repeat": max(0, repeat - 1),
                    "seconds": seconds,
                    "energy": float(item.energy),
                    "forces": forces,
                    "converged": True,
                    "iterations": item.iterations,
                    "fock_builds": item.fock_builds,
                    "warm_start_used": item.warm_start_used,
                    "warm_start_fallback": item.warm_start_fallback,
                    "energy_change": item.energy_change,
                    "density_rms": item.density_rms,
                    "physical_residual_rms": item.physical_residual_rms,
                    "settings": native_settings(candidate, item),
                    "precision": item.precision,
                    "precision_validation": validation.to_dict()
                    if validation
                    else None,
                    "batch_precision_validation": batch.precision_validation,
                    "ks_diagnostic": item.ks_diagnostic.to_payload()
                    if item.ks_diagnostic
                    else None,
                    "initial_guess": item.initial_guess,
                    "correction_energy": float(dispersion.energy)
                    if dispersion
                    else None,
                }
            )
            if repeat == 0:
                owner.set_warm_start_updates(False)
    finally:
        owner.close()
        calculator.clear_cache()


def run_reference(args: Any, atoms: Any, record: dict, retain: Any) -> None:
    """Include grid response while preserving native reference convergence/SCF."""
    if args.arm != "gpu4pyscf":
        raise ValueError("only the default GPU4PySCF reference is measured")
    import cupy as cp
    from gpu4pyscf.dft import rks
    from gpu4pyscf.scf import hf
    from pyscf import gto

    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=orbital_basis(args.method),
        cart=False,
        charge=0,
        spin=0,
        verbose=0,
    )
    engine = hf.RHF(mol) if args.method == "hf" else rks.RKS(mol)
    if args.method != "hf":
        engine.xc = METHODS[args.method][1]
    if args.mode == "df":
        engine = engine.density_fit()
    seed = None
    # Count executed effective-potential builds without changing their arguments,
    # caching, precision, screening, or the stock initial guess.
    original_veff = engine.get_veff
    builds = 0

    def counted_veff(*values: Any, **kwargs: Any) -> Any:
        nonlocal builds
        builds += 1
        return original_veff(*values, **kwargs)

    engine.get_veff = counted_veff
    for repeat in range(args.warm_repeats + 1):
        builds = 0
        if repeat:
            cp.cuda.runtime.deviceSynchronize()
            started = perf_counter()
        # Copy inside the warm endpoint, preserving the same immutable cold seed.
        energy = float(engine.kernel(dm0=None if seed is None else seed.copy()))
        if not engine.converged:
            raise RuntimeError("GPU4PySCF SCF did not converge")
        gradient = engine.nuc_grad_method()
        if args.method != "hf":
            gradient.grid_response = True
        forces = -cp.asnumpy(gradient.kernel())
        correction_energy = None
        if args.method == "r2scan-3c":
            correction_energy, correction_gradient = reference_r2scan3c_correction(
                atoms
            )()
            energy += float(correction_energy)
            forces -= correction_gradient
        cp.cuda.runtime.deviceSynchronize()
        seconds = perf_counter() - started
        settings = {
            "energy_tolerance": float(engine.conv_tol),
            "orbital_gradient_tolerance": float(engine.conv_tol_grad)
            if engine.conv_tol_grad is not None
            else None,
            "max_iterations": int(engine.max_cycle),
            "diis_history": int(engine.diis_space),
            "initial_guess": str(engine.init_guess),
            "direct_scf": bool(engine.direct_scf),
            "direct_scf_tolerance": float(engine.direct_scf_tol),
            "xc_grid": grid_settings(getattr(engine, "grids", None)),
            "nonlocal_grid": grid_settings(getattr(engine, "nlcgrids", None))
            if args.method == "wb97m-v"
            else None,
            "force_grid_response": bool(getattr(gradient, "grid_response", False))
            if args.method != "hf"
            else None,
            "auxbasis_response": getattr(gradient, "auxbasis_response", None),
        }
        if args.mode == "df":
            actual = engine.with_df.auxmol._basis
            expected = json.loads(
                (args.inputs / f"{args.method}-reference.json").read_text()
            )
            if actual != expected["basis"]:
                raise RuntimeError(
                    "default GPU4PySCF auxiliary differs from frozen native input"
                )
            settings["auxiliary_basis_sha256"] = expected["basis_sha256"]
            settings["auxiliary_aos"] = engine.with_df.auxmol.nao_nr()
        retain(
            {
                "phase": "cold" if repeat == 0 else "warm",
                "repeat": max(0, repeat - 1),
                "seconds": seconds,
                "energy": energy,
                "forces": _finite_endpoint(energy, forces, len(atoms), seconds),
                "converged": True,
                "iterations": int(engine.cycles),
                "effective_potential_builds": builds,
                "warm_start_used": seed is not None,
                "settings": settings,
                "correction_energy": float(correction_energy)
                if correction_energy is not None
                else None,
            }
        )
        if repeat == 0:
            seed = engine.make_rdm1().copy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=(*MEASUREMENT_ARMS, "inputs"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--mode", choices=("direct", "df"), default="direct")
    parser.add_argument("--atoms", type=int, choices=SIZES, default=3)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-revision", required=False)
    parser.add_argument("--output", type=raw_output_path)
    args = parser.parse_args(argv)
    if args.arm == "inputs":
        prepare_inputs(args.inputs)
        return
    if not args.method or not args.output or not args.source_revision:
        parser.error("measured arms require --method, --output and --source-revision")
    if args.warm_repeats < 1:
        parser.error("warm repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU work requires a finite Slurm GPU allocation")
    if args.output.exists():
        raise FileExistsError(args.output)
    import cupy as cp

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    record = {
        "schema": SCHEMA,
        "status": "running",
        "arm": args.arm,
        "method": args.method,
        "mode": args.mode,
        "atoms": args.atoms,
        "geometry_bohr": atoms,
        "warm_repeats": args.warm_repeats,
        "orbital_basis": "def2-mTZVPP" if args.method == "r2scan-3c" else "def2-SVP",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
        "source_revision": args.source_revision,
        "runner_sha256": digest(Path(__file__)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "accelerator": cuda_accelerator_metadata(cp),
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                "cupy": ("cupy-cuda12x",),
                "numpy": ("numpy",),
            }
        ),
        "protocol": {
            "cold": "constructor+prepare+SCF+forces+host-export",
            "warm": "same-owner; frozen engine-local cold density",
            "grid_overrides": False,
            "force_response_overrides": True,
            "convergence": "native engine/profile defaults; common observable gates",
            "residual_definitions": {
                "generativeqc": "density RMS",
                "gpu4pyscf": "orbital gradient norm",
            },
            "moved_geometry": False,
            "accuracy_qualification": "unverified",
            "common_observable_gates": COMMON_GATES,
        },
        "samples": [],
    }
    if args.mode == "df":
        record["auxiliary_input_sha256"] = digest(
            args.inputs / f"{args.method}-auxiliary.json"
        )

    def retain(sample: dict) -> None:
        record["samples"].append(sample)
        save(args.output, record)
        print(
            json.dumps(
                {
                    key: sample[key]
                    for key in ("phase", "repeat", "seconds", "iterations")
                }
            ),
            flush=True,
        )

    save(args.output, record)
    try:
        (run_native if args.arm in ("standard", "fast") else run_reference)(
            args, atoms, record, retain
        )
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, record)


if __name__ == "__main__":
    main()
