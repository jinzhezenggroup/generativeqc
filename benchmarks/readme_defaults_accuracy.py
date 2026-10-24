"""Independent, untimed accuracy qualification for the default README campaign.

Fresh tighter direct calculations at two denser XC grids give one common target
for Direct and DF endpoints, with a separate VV10 grid-stability check.
They never supply production densities or timings.
Small CPU/PySCF spot checks independently validate the GPU oracle implementation.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks._retention import raw_output_path
from benchmarks.readme_defaults import (
    COMMON_GATES,
    METHODS,
    digest,
    grid_settings,
    orbital_basis,
    reference_r2scan3c_correction,
    save,
)
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_wb97mv import configure_reference_full_fock


def errors(energy: float, forces: Any, reference: dict, atoms: int) -> dict:
    """Apply the same declared intensive observable gates to every profile."""
    delta = np.asarray(forces) - np.asarray(reference["forces"])
    values = {
        "energy_abs_per_atom_hartree": abs(energy - reference["energy"]) / atoms,
        "force_max_abs_hartree_per_bohr": float(np.max(np.abs(delta))),
        "force_rms_hartree_per_bohr": float(np.sqrt(np.mean(delta * delta))),
    }
    return {
        **values,
        "passed": all(
            np.isfinite(v) and v <= COMMON_GATES[k] for k, v in values.items()
        ),
    }


def solve(method: str, atoms: Any, *, backend: str, level: int) -> dict:
    """Recompute a complete strict direct endpoint in a fresh reference owner."""
    from pyscf import dft, gto, scf

    cp = None
    if backend == "cuda":
        import cupy as cp
        from gpu4pyscf.dft import rks
        from gpu4pyscf.scf import hf

        cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    mol = gto.M(
        atom=atoms, unit="Bohr", basis=orbital_basis(method), cart=False, verbose=0
    )
    if backend == "cuda":
        engine = hf.RHF(mol) if method == "hf" else rks.RKS(mol)
    else:
        engine = scf.RHF(mol) if method == "hf" else dft.RKS(mol)
    engine.conv_tol = 1e-11
    engine.conv_tol_grad = 1e-7
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 150
    if method != "hf":
        engine.xc = METHODS[method][1]
        engine.grids.level = level
        # VV10 has its own quadrature. Check nonlocal levels 2/3 independently;
        # GPU4PySCF 1.8.1 already defaults to level 3, so these are not both
        # denser than the timed grid. These are untimed reference controls.
        engine.nlcgrids.level = level - 2
        engine.small_rho_cutoff = 0
    configure_reference_full_fock(engine)
    if backend == "cpu":
        # PySCF's range-separated get_k needs the direct-SCF optimizer even
        # when rebuilding full-density Focks. The wrapper still omits dm_last
        # and vhf_last, preventing delta-density reuse; this flag only restores
        # construction of the CPU integral prescreener (not GPU timing policy).
        engine.direct_scf = True
    energy = float(engine.kernel())
    if not engine.converged:
        raise RuntimeError(f"accuracy oracle did not converge ({engine.cycles} cycles)")
    gradient = engine.nuc_grad_method()
    if method != "hf":
        gradient.grid_response = True
    forces = -gradient.kernel()
    forces = cp.asnumpy(forces) if cp is not None else np.asarray(forces)
    if method == "r2scan-3c":
        correction, derivative = reference_r2scan3c_correction(atoms)()
        energy += float(correction)
        forces -= derivative
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("nonfinite reference endpoint")
    record = {
        "energy": energy,
        "forces": forces.tolist(),
        "converged": True,
        "iterations": int(engine.cycles),
        "seconds": perf_counter() - started,
        "level": None if method == "hf" else level,
        "xc_grid": grid_settings(getattr(engine, "grids", None)),
        "nonlocal_grid": grid_settings(getattr(engine, "nlcgrids", None))
        if method == "wb97m-v"
        else None,
        "controls": {
            "energy": engine.conv_tol,
            "orbital_gradient": engine.conv_tol_grad,
            "screening": engine.direct_scf_tol,
            "full_density_fock": True,
            "force_grid_response": None if method == "hf" else gradient.grid_response,
        },
    }
    del gradient, engine, mol
    gc.collect()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    return record


def main() -> None:
    """Retain partial oracle failures without promoting them to qualified data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--atoms", required=True, type=int, choices=(3, 12, 48, 96))
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True, type=raw_output_path)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("qualification requires a finite Slurm allocation")
    if args.output.exists():
        raise FileExistsError(args.output)
    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    from benchmarks._support import environment_metadata

    record = {
        "schema": "generativeqc.readme-defaults-oracle.v1",
        "method": args.method,
        "atoms": args.atoms,
        "backend": args.backend,
        "geometry_bohr": atoms,
        "driver_sha256": digest(Path(__file__)),
        "source_revision": args.source_revision,
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                "cupy": ("cupy-cuda12x",),
                "numpy": ("numpy",),
            }
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "common_observable_gates": COMMON_GATES,
        "grid_stability_gate_fraction": 0.25,
        "status": "running",
        "samples": [],
    }
    save(args.output, record)
    try:
        levels = (5,) if args.method == "hf" or args.backend == "cpu" else (4, 5)
        for level in levels:
            record["samples"].append(
                solve(args.method, atoms, backend=args.backend, level=level)
            )
            save(args.output, record)
        if len(record["samples"]) == 2:
            first, last = record["samples"]
            record["grid_stability"] = errors(
                first["energy"], first["forces"], last, args.atoms
            )
            record["grid_stability"]["passed"] = all(
                record["grid_stability"][key] <= 0.25 * threshold
                for key, threshold in COMMON_GATES.items()
            )
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, record)


if __name__ == "__main__":
    main()
