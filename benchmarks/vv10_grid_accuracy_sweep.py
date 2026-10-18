"""Accuracy-only GPU4PySCF grid sweep with independently reconverged densities.

Neighboring quadratures may reuse this reference engine's converged density,
never native outputs. This avoids repeating an atomic-guess SCF for every dense
grid, but it is deliberately not a cold/warm benchmark timing population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
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
from benchmarks.readme_split_vv10 import _engine
from benchmarks.readme_wb97mv import reference_vv10_domain


def parse_grid_pair(value: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Parse one explicit XC:NLC pair without conflating their point domains."""
    fields = value.split(":")
    if len(fields) != 2:
        raise ValueError(
            "grid pair must be XC_RADIAL,POLAR,AZIMUTH:NLC_RADIAL,POLAR,AZIMUTH"
        )
    return parse_grid(fields[0]), parse_grid(fields[1])


def save_population(path: Path, result: dict[str, Any]) -> None:
    """Keep incomplete endpoints inspectable without calling them successful."""
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Save each successful reconverged E+F pair independently under finite Slurm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, choices=(3, 12, 48, 96), required=True)
    parser.add_argument("--density-fitting", choices=("none", "cuda"), default="none")
    parser.add_argument("--grid-pairs", type=parse_grid_pair, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("use srun with an explicit finite --time")
    if len(set(args.grid_pairs)) != len(args.grid_pairs):
        parser.error("each XC:NLC pair must be unique")

    import cupy as cp

    original = scaling_cases()[f"water-{args.atoms}"].atoms
    moved = [
        (
            element,
            tuple(np.asarray(coords) + ((0, 0, 0.001) if index == 1 else (0, 0, 0))),
        )
        for index, (element, coords) in enumerate(original)
    ]
    densities: dict[int, Any] = {}
    previous_grids = {}
    args.output.mkdir(parents=True, exist_ok=True)
    with reference_vv10_domain(float(MOLECULAR_VV10_DENSITY_THRESHOLD)):
        for xc_dimensions, nlc_dimensions in args.grid_pairs:
            xc_spec, nlc_spec = GridSpec(*xc_dimensions), GridSpec(*nlc_dimensions)
            stem = (
                "xc-"
                + "x".join(map(str, xc_dimensions))
                + "-nlc-"
                + "x".join(map(str, nlc_dimensions))
            )
            path = args.output / f"{stem}.json"
            if path.exists():
                raise FileExistsError(
                    f"retain the earlier attempt instead of overwriting: {path}"
                )
            result = {
                "schema": "generativeqc.reference-vv10-grid-accuracy.v1",
                "engine": "GPU4PySCF",
                "purpose": "accuracy-only-not-a-latency-population",
                "native_split_grid_supported": False,
                "atoms": args.atoms,
                "basis": "spherical def2-SVP",
                "density_fitting": args.density_fitting,
                "xc_grid": asdict(xc_spec),
                "nlc_grid": asdict(nlc_spec),
                "geometries_bohr": [original, moved],
                "accuracy_protocol": {
                    "density_sources": "independent-GPU4PySCF-only",
                    "scf_tolerances": {
                        "energy": 1e-12,
                        "gradient": 1e-10,
                        "direct_scf": 1e-14,
                        "max_cycle": 100,
                    },
                    "vv10_density_threshold": float(MOLECULAR_VV10_DENSITY_THRESHOLD),
                    "full_grid_response": True,
                    "cold_timing_qualified": False,
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
                    for key in (
                        "SLURM_JOB_ID",
                        "SLURM_JOB_NODELIST",
                        "CUDA_VISIBLE_DEVICES",
                    )
                },
                "source_sha256": {
                    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in (
                        "benchmarks/vv10_grid_accuracy_sweep.py",
                        "benchmarks/readme_split_vv10.py",
                        "benchmarks/readme_wb97mv.py",
                        "benchmarks/_gpu4pyscf_grid.py",
                    )
                },
                "stage": "prepare",
                "records": [],
            }

            save_population(path, result)
            try:
                for geometry, atoms in enumerate((original, moved)):
                    result["stage"] = f"geometry-{geometry}/reconverge-and-forces"
                    save_population(path, result)
                    seed = densities.get(geometry)
                    initial = (
                        "previous-grid-same-geometry"
                        if seed is not None
                        else "reference-default-guess"
                    )
                    seed_grid = previous_grids.get(geometry)
                    if seed is None and geometry == 1 and 0 in densities:
                        seed = densities[0]
                        initial = "same-grid-original-geometry"
                        seed_grid = previous_grids[0]
                    engine = _engine(atoms, xc_spec, nlc_spec, args.density_fitting)
                    with reference_work_counter(engine) as work:
                        energy, forces, seconds = reference_energy_forces(
                            engine,
                            None if seed is None else seed.copy(),
                            export=cp.asnumpy,
                            synchronize=cp.cuda.Stream.null.synchronize,
                        )
                        convergence = gpu_convergence_payload(
                            [engine], [work["tracker"]]
                        )[0]
                    if (
                        not engine.converged
                        or not np.isfinite(energy)
                        or not np.isfinite(forces).all()
                    ):
                        raise RuntimeError(
                            "accuracy endpoint is unconverged or nonfinite"
                        )
                    result["records"].append(
                        {
                            "phase": "accuracy-original"
                            if geometry == 0
                            else "accuracy-moved",
                            "geometry": geometry,
                            "repeat": 0,
                            "energy": energy,
                            "forces": forces.tolist(),
                            "converged": True,
                            "convergence": convergence,
                            "iterations": convergence["iterations"],
                            "fock_builds": work["scf_jk_builds"],
                            "initial_density": initial,
                            "initial_density_grid": seed_grid,
                            "diagnostic_seconds_not_benchmark_latency": seconds,
                        }
                    )
                    densities[geometry] = engine.make_rdm1().copy()
                    previous_grids[geometry] = {
                        "xc": xc_dimensions,
                        "nlc": nlc_dimensions,
                    }
                    save_population(path, result)
                result["stage"] = "complete"
                save_population(path, result)
                outcome = {
                    "exit_code": 0,
                    "purpose": "accuracy-only-not-a-latency-population",
                }
            except BaseException as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["stage"] = "failed"
                save_population(path, result)
                outcome = {
                    "exit_code": 1,
                    "purpose": "accuracy-only-not-a-latency-population",
                }
                path.with_suffix(".outcome").write_text(json.dumps(outcome) + "\n")
                raise
            path.with_suffix(".outcome").write_text(json.dumps(outcome) + "\n")
            print(stem, result["stage"], flush=True)


if __name__ == "__main__":
    main()
