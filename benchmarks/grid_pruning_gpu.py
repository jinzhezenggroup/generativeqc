"""Qualify complete LDA/PBE GPU endpoints with independent production grids.

Native candidates use versioned GridPolicy prescriptions. GPU4PySCF candidates
use its own radial/Lebedev grids with NWChem pruning, never transplanted native
atomic rules. Dense independent grids qualify accuracy before ranking timings.
Run through a finite Slurm GPU allocation; all evidence is written as JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from benchmarks._retention import raw_output_path

if TYPE_CHECKING:
    from collections.abc import Callable


def accuracy_gates(method: str, atoms: int) -> dict[str, float]:
    """Retain the small-water error budgets per water unit and maximum force."""
    if method not in ("lda", "pbe"):
        raise ValueError("pruning qualification supports only LDA and PBE")
    return {
        "energy": (1e-6 if method == "lda" else 1.5e-6) * atoms / 3,
        "force": 5e-5 if method == "lda" else 3e-5,
    }


def errors(candidate: dict, reference: dict) -> dict[str, float]:
    """Compare complete total energies and maximum absolute analytic forces."""
    return {
        "energy": abs(candidate["energy"] - reference["energy"]),
        "force": float(
            np.max(np.abs(np.asarray(candidate["forces"]) - reference["forces"]))
        ),
    }


def qualification_errors(measured: dict, stability: dict) -> dict[str, float]:
    """Reserve margin for observed reference-grid variation, not a rigorous bound."""
    return {key: measured[key] + stability[key] for key in ("energy", "force")}


def reference_endpoint(method: str, atoms: tuple, level: int, pruning: bool) -> dict:
    """Time a fresh independent SCF/force owner, including preparation/cleanup."""
    import cupy as cp
    from gpu4pyscf.dft import gen_grid, rks
    from pyscf import gto

    from benchmarks.compare_df_direct_endpoint import reference_work_counter
    from benchmarks.compare_gpu4pyscf_batch import gpu_convergence_payload
    from benchmarks.readme_wb97mv import configure_reference_full_fock

    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    molecule = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", verbose=0)
    engine = rks.RKS(molecule)
    engine.xc = "LDA_X,LDA_C_PW" if method == "lda" else "PBE"
    engine.grids.level = level
    engine.grids.prune = gen_grid.nwchem_prune if pruning else None
    engine.small_rho_cutoff = 0
    engine.conv_tol = 1e-11
    engine.conv_tol_grad = 1e-8
    engine.direct_scf_tol = 1e-12
    engine.max_cycle = 120
    configure_reference_full_fock(engine)
    with reference_work_counter(engine) as work:
        energy = float(engine.kernel())
        if not engine.converged:
            raise RuntimeError("independent GPU4PySCF endpoint did not converge")
        gradient = engine.nuc_grad_method()
        gradient.grid_response = True
        forces = -cp.asnumpy(gradient.kernel())
    result = {
        "energy": energy,
        "forces": forces.tolist(),
        "grid_points": len(engine.grids.coords),
        "nonzero_weight_points": int(cp.count_nonzero(engine.grids.weights)),
        "work": {
            "scf_jk_builds": work["scf_jk_builds"],
            "convergence": gpu_convergence_payload(
                [engine], [work["tracker"]], warm_start_used=False
            )[0],
        },
    }
    del gradient, engine, molecule
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    result["seconds"] = perf_counter() - started
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("independent endpoint returned nonfinite observables")
    return result


def native_endpoint(method: str, atoms: tuple, pruning: str) -> dict:
    """Time the fresh public native energy-plus-force endpoint without oracle work."""
    import cupy as cp
    from generativeqc import Calculator, GridPolicy, KsOptions

    grid = GridPolicy(pruning=pruning).resolve(f"{method}-rks")
    cp.cuda.runtime.deviceSynchronize()
    started = perf_counter()
    calculator = Calculator(
        method=f"{method}-rks",
        basis="sto-3g",
        device="cuda",
        ks_options=None if pruning == "radial-v1" else KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    result = calculator.singlepoint(atoms, properties=("energy", "forces"))
    library_path = calculator._library._name
    del calculator
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    seconds = perf_counter() - started
    if (
        not result.converged
        or result.executed_backend != "cuda"
        or result.forces is None
    ):
        raise RuntimeError("native complete CUDA endpoint did not converge")
    return {
        "energy": result.energy,
        "forces": result.forces.tolist(),
        "seconds": seconds,
        "grid_points": result.ks_diagnostic.grid_points,
        "iterations": result.iterations,
        "work": result.ks_diagnostic.to_payload(),
        "grid": asdict(grid),
        "library_path": str(Path(library_path).resolve()),
    }


def file_sha256(path: Path) -> str:
    """Hash large native libraries without allocating a binary-sized buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_identity() -> dict:
    """Bind the selected library and actual installed/overlay Python source roots.

    Policy, generated-kernel and benchmark helper edits can change the endpoint
    without changing this driver or the native library. Capture their bytes at
    both ends of the run; this is an observed source identity, not a clean build.
    """
    import generativeqc
    import generativeqc_compiler
    from generativeqc import _native

    import benchmarks

    library = Path(_native.load_library(device="cuda", device_id=0)._name).resolve()
    sources = {Path(__file__).resolve()}
    for location in benchmarks.__path__:
        root = Path(location).resolve()
        sources.update(
            path
            for path in root.rglob("*.py")
            if "results" not in path.relative_to(root).parts
        )
    roots = set()
    for package in (generativeqc, generativeqc_compiler):
        roots.update(Path(root).resolve() for root in package.__path__)
    sources.update(path for root in roots for path in root.rglob("*.py"))
    return {
        "library_path": str(library),
        "library_sha256": file_sha256(library),
        "python_source_sha256": {
            str(path): file_sha256(path) for path in sorted(sources)
        },
    }


def verify_native_library(name: str, sample: dict, report: dict) -> None:
    """Reject a fresh owner selecting a different image from the captured identity."""
    if (
        name.startswith("native")
        and sample["library_path"] != report["execution_identity"]["library_path"]
    ):
        raise RuntimeError("native endpoint selected a different library")


def main(argv: list[str] | None = None) -> None:
    """Collect raw samples and rank only independently accuracy-qualified settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", choices=(3, 6, 12, 24, 48, 96), type=int, default=96)
    parser.add_argument(
        "--methods", choices=("lda", "pbe"), nargs="+", default=("lda", "pbe")
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--reference-levels", nargs=2, type=int, default=(8, 9))
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args(argv)
    if (
        args.repeats < 2
        or not 0 <= args.reference_levels[0] < args.reference_levels[1] <= 9
    ):
        parser.error(
            "require at least two repeats and increasing reference levels in [0,9]"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run through srun with an explicit finite --time")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects directories and dangling/existing symlinks.
    # Keep this descriptor: subsequent progress writes cannot follow a new symlink.
    with args.output.open("x", encoding="utf-8") as stream:
        report = {
            "schema": "generativeqc.production-pruning-qualification.v2",
            "status": "running",
        }

        def save() -> None:
            """Serialize before truncation so an encoding failure retains the receipt."""
            raw_output_path(args.output)
            text = json.dumps(report, indent=2, allow_nan=False) + "\n"
            stream.seek(0)
            stream.write(text)
            stream.truncate()
            stream.flush()

        save()
        try:
            collect(args, report, save)
            report["execution_identity_after"] = execution_identity()
            if report["execution_identity_after"] != report["execution_identity"]:
                raise RuntimeError(
                    "loaded library or Python sources changed during qualification"
                )
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            raise
        finally:
            save()


def collect(args: argparse.Namespace, report: dict, save: Callable[[], None]) -> None:
    """Collect into an exclusively owned receipt, retaining partial failures."""
    import cupy as cp
    import gpu4pyscf
    import pyscf

    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    device = cp.cuda.runtime.getDeviceProperties(0)
    report.update(
        {
            "schema": "generativeqc.production-pruning-qualification.v2",
            "status": "running",
            "atoms_bohr": atoms,
            "basis": "sto-3g",
            "host": platform.node(),
            "scheduler": {
                key: os.environ.get(key)
                for key in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
            },
            "gpu": device["name"].decode(),
            "versions": {
                "pyscf": pyscf.__version__,
                "gpu4pyscf": gpu4pyscf.__version__,
            },
            "execution_identity": execution_identity(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "endpoint": "fresh construction + preparation + SCF + analytic forces + cleanup; synchronized",
            "cache_policy": "one untimed complete warmup per candidate; no cross-sample SCF-state reuse",
            "methods": {},
        }
    )
    report["library_sha256"] = report["execution_identity"]["library_sha256"]
    save()
    for method in args.methods:
        entry = report["methods"][method] = {
            "gates": accuracy_gates(method, args.atoms)
        }
        references = entry["references"] = {}
        for level in args.reference_levels:
            references[str(level)] = reference_endpoint(method, atoms, level, True)
            save()
        stability = errors(*references.values())
        entry["reference_stability_errors"] = stability
        if stability["energy"] > 2e-8 * args.atoms / 3 or stability["force"] > 1e-6:
            raise RuntimeError(
                f"{method} independent reference stability failed: {stability}"
            )
        oracle = references[str(args.reference_levels[-1])]
        candidates = entry["candidates"] = {}
        calls = {
            "native-none": partial(native_endpoint, method, atoms, "none"),
            "native-radial-v1": partial(native_endpoint, method, atoms, "radial-v1"),
            "gpu4pyscf-none": partial(reference_endpoint, method, atoms, 3, False),
            "gpu4pyscf-nwchem": partial(reference_endpoint, method, atoms, 3, True),
        }
        for name, call in calls.items():
            print("WARMUP", method, name, flush=True)
            candidates[name] = {"warmup": call(), "samples": []}
            save()
            verify_native_library(name, candidates[name]["warmup"], report)
        for repeat in range(args.repeats):
            order = tuple(calls) if repeat % 2 == 0 else tuple(reversed(calls))
            for name in order:
                sample = calls[name]()
                sample["errors_vs_independent_reference"] = errors(sample, oracle)
                candidates[name]["samples"].append(sample)
                print(method, name, repeat + 1, sample["seconds"], flush=True)
                save()
                verify_native_library(name, sample, report)
        for candidate in candidates.values():
            samples = candidate["samples"]
            times = [sample["seconds"] for sample in samples]
            candidate["median_seconds"] = statistics.median(times)
            candidate["quartiles_seconds"] = np.quantile(times, [0.25, 0.75]).tolist()
            candidate["max_errors"] = {
                key: max(
                    sample["errors_vs_independent_reference"][key] for sample in samples
                )
                for key in ("energy", "force")
            }
            candidate["qualification_errors"] = qualification_errors(
                candidate["max_errors"], stability
            )
            candidate["qualified"] = all(
                candidate["qualification_errors"][key] <= value
                for key, value in entry["gates"].items()
            )
        entry["selected"] = {}
        for engine in ("native", "gpu4pyscf"):
            qualified = [
                name
                for name, result in candidates.items()
                if name.startswith(engine) and result["qualified"]
            ]
            entry["selected"][engine] = (
                min(qualified, key=lambda name: candidates[name]["median_seconds"])
                if qualified
                else None
            )
        if (
            not candidates["native-radial-v1"]["qualified"]
            or not candidates["gpu4pyscf-nwchem"]["qualified"]
        ):
            raise RuntimeError(
                f"{method} pruned production defaults failed accuracy qualification"
            )
        print("SELECTED", method, entry["selected"], flush=True)


if __name__ == "__main__":
    main()
