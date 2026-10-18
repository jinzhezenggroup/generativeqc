"""Reference-only split XC/VV10 grid feasibility, not a native speedup claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from generativeqc import GridSpec
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD

from benchmarks._readme_grid import parse_grid
from benchmarks._support import cuda_accelerator_metadata, environment_metadata
from benchmarks.compare_df_direct_endpoint import reference_work_counter
from benchmarks.compare_gpu4pyscf_batch import gpu_convergence_payload
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_omol25 import reference_energy_forces
from benchmarks.readme_wb97mv import (
    configure_reference_mixed_rsh_df,
    reference_engine,
    reference_vv10_domain,
)


def split_grid_work(
    atoms: int, xc_spec: GridSpec, nlc_spec: GridSpec
) -> dict[str, Any]:
    """Keep XC point counts separate from the pre-screening VV10 pair bound."""
    if type(atoms) is not int or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    xc_points = atoms * math.prod(
        (xc_spec.radial_points, xc_spec.angular_polar, xc_spec.angular_azimuth)
    )
    nlc_points = atoms * math.prod(
        (nlc_spec.radial_points, nlc_spec.angular_polar, nlc_spec.angular_azimuth)
    )
    return {
        "xc_grid": asdict(xc_spec),
        "nlc_grid": asdict(nlc_spec),
        "xc_points": xc_points,
        "vv10_points": nlc_points,
        "vv10_unscreened_ordered_pair_upper_bound": nlc_points**2,
        "pair_bound_is_executed_work": False,
    }


def compare_geometries(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    energy_tolerance: float = 1e-5,
    force_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Isolate NLC-grid error; missing, nonfinite or changed XC inputs cannot pass."""
    if any(
        not math.isfinite(value) or value <= 0
        for value in (energy_tolerance, force_tolerance)
    ):
        raise ValueError("comparison tolerances must be finite and positive")
    for field in ("atoms", "basis", "density_fitting", "xc_grid", "geometries_bohr"):
        if candidate[field] != baseline[field]:
            raise ValueError(f"split-grid comparison changes {field}")
    if candidate["nlc_grid"] == baseline["nlc_grid"]:
        raise ValueError("a baseline self-comparison cannot certify convergence")
    errors = []
    for geometry in (0, 1):
        rows = [
            next(
                (
                    row
                    for row in population["records"]
                    if row["geometry"] == geometry and row["phase"] in ("cold", "moved")
                ),
                None,
            )
            for population in (candidate, baseline)
        ]
        if any(row is None for row in rows):
            return {"gate": None, "reason": "missing geometry", "geometries": errors}
        measured, reference = rows
        actual = np.asarray(measured["forces"], dtype=float)
        expected = np.asarray(reference["forces"], dtype=float)
        valid = (
            measured["converged"]
            and reference["converged"]
            and actual.shape == expected.shape == (candidate["atoms"], 3)
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and math.isfinite(measured["energy"])
            and math.isfinite(reference["energy"])
        )
        energy = abs(measured["energy"] - reference["energy"]) if valid else None
        force = float(np.max(np.abs(actual - expected))) if valid else None
        errors.append(
            {
                "geometry": geometry,
                "energy_error": energy,
                "force_error": force,
                "gate": bool(
                    valid and energy <= energy_tolerance and force <= force_tolerance
                ),
            }
        )
    return {"gate": all(row["gate"] for row in errors), "geometries": errors}


def _engine(atoms: Any, xc_spec: GridSpec, nlc_spec: GridSpec, mode: str) -> Any:
    engine = reference_engine(atoms, "def2-svp", xc_spec, nlc_spec=nlc_spec)
    if mode == "cuda":
        engine = engine.density_fit(auxbasis="cc-pvdz-jkfit")
        configure_reference_mixed_rsh_df(engine)
    engine.conv_tol = 1e-12
    engine.conv_tol_grad = 1e-10
    engine.direct_scf_tol = 1e-14
    engine.max_cycle = 100
    return engine


def main() -> None:
    """Run every actual GPU operation inside the caller's finite Slurm allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(3, 12, 48, 96), required=True)
    parser.add_argument("--xc-grid", type=parse_grid, default=(48, 16, 32))
    parser.add_argument("--nlc-grid", type=parse_grid, required=True)
    parser.add_argument("--density-fitting", choices=("none", "cuda"), default="none")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--finite-difference-step", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.finite_difference_step is not None and (
        not math.isfinite(args.finite_difference_step)
        or args.finite_difference_step <= 0
    ):
        parser.error("finite-difference step must be finite and positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("use srun with an explicit finite --time")

    import cupy as cp

    xc_spec, nlc_spec = GridSpec(*args.xc_grid), GridSpec(*args.nlc_grid)
    original = scaling_cases()[f"water-{args.atoms}"].atoms
    moved = [
        (
            element,
            tuple(np.asarray(coords) + ((0, 0, 0.001) if index == 1 else (0, 0, 0))),
        )
        for index, (element, coords) in enumerate(original)
    ]
    result = {
        "schema": "generativeqc.reference-split-vv10.v1",
        "engine": "GPU4PySCF",
        "native_split_grid_supported": False,
        "atoms": args.atoms,
        "basis": "spherical def2-SVP",
        "density_fitting": args.density_fitting,
        "xc_grid": asdict(xc_spec),
        "nlc_grid": asdict(nlc_spec),
        "geometries_bohr": [original, moved],
        "grid_work": split_grid_work(args.atoms, xc_spec, nlc_spec),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/readme_split_vv10.py",
                "benchmarks/readme_wb97mv.py",
                "benchmarks/_gpu4pyscf_grid.py",
            )
        },
        "environment": environment_metadata(
            distributions={
                "gpu4pyscf": ("gpu4pyscf", "gpu4pyscf-cuda12x"),
                "pyscf": ("pyscf",),
            }
        ),
        "accelerator": cuda_accelerator_metadata(cp),
        "scheduler": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")
        },
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        result["stage"] = stage
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    density = None
    with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
        for geometry, atoms in enumerate((original, moved)):
            save(f"geometry-{geometry}/prepare")
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            engine = _engine(atoms, xc_spec, nlc_spec, args.density_fitting)
            cp.cuda.Stream.null.synchronize()
            prepare = perf_counter() - started
            for repeat in range(args.repeats + 1):
                phase = (
                    ("cold" if geometry == 0 else "moved")
                    if repeat == 0
                    else ("warm" if geometry == 0 else "moved-warm")
                )
                save(f"geometry-{geometry}/{phase}-{repeat}")
                with reference_work_counter(engine) as work:
                    energy, forces, seconds = reference_energy_forces(
                        engine,
                        density,
                        export=cp.asnumpy,
                        synchronize=cp.cuda.Stream.null.synchronize,
                    )
                    convergence = gpu_convergence_payload([engine], [work["tracker"]])[
                        0
                    ]
                if not np.isfinite(forces).all() or not math.isfinite(energy):
                    raise RuntimeError("nonfinite split-grid endpoint")
                result["records"].append(
                    {
                        "phase": phase,
                        "geometry": geometry,
                        "repeat": repeat,
                        "energy": energy,
                        "forces": forces.tolist(),
                        "converged": bool(engine.converged),
                        "iterations": convergence["iterations"],
                        "fock_builds": work["scf_jk_builds"],
                        "prepare_seconds": prepare if repeat == 0 else 0.0,
                        "complete_seconds": seconds + (prepare if repeat == 0 else 0.0),
                        "convergence": convergence,
                    }
                )
                if repeat == 0:
                    density = engine.make_rdm1().copy()
                save(f"geometry-{geometry}/{phase}-{repeat}/complete")
            if geometry == 0 and args.finite_difference_step is not None:
                analytic = result["records"][0]["forces"][1][2]
                differences = []
                for step in (
                    args.finite_difference_step,
                    args.finite_difference_step / 2,
                ):
                    energies = []
                    for direction in (-1, 1):
                        displaced = [
                            (
                                element,
                                tuple(
                                    np.asarray(coords)
                                    + (
                                        (0, 0, direction * step)
                                        if index == 1
                                        else (0, 0, 0)
                                    )
                                ),
                            )
                            for index, (element, coords) in enumerate(original)
                        ]
                        displaced_engine = _engine(
                            displaced, xc_spec, nlc_spec, args.density_fitting
                        )
                        energies.append(
                            float(displaced_engine.kernel(dm0=density.copy()))
                        )
                        cp.cuda.Stream.null.synchronize()
                        if not displaced_engine.converged or not math.isfinite(
                            energies[-1]
                        ):
                            raise RuntimeError("finite-difference SCF did not converge")
                    force = -(energies[1] - energies[0]) / (2 * step)
                    differences.append(
                        {
                            "step": step,
                            "energies": energies,
                            "force": force,
                            "analytic_force": analytic,
                            "error": abs(force - analytic),
                        }
                    )
                result["finite_difference"] = differences
                save("finite-difference/complete")
    save("complete")


if __name__ == "__main__":
    main()
