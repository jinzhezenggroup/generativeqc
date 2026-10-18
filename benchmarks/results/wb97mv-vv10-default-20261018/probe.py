"""Measure the real default, without an experiment-policy or imported-density knob."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO

import cupy as cp
import numpy as np

from generativeqc import Calculator, GridSpec, KsOptions

from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis, native_build_metadata
from benchmarks.readme_hf_scaling import scaling_cases
from tools.generate_build_identity import _inventory, _source_identity


def export_density(owner: Any, path: Path, destination: BinaryIO) -> dict:
    """Export after timing through the exclusively reserved density descriptor."""
    from generativeqc import _native
    from generativeqc.checkpoint import _descriptor, _pointer

    state = _descriptor()
    query = owner._library.generativeqc_batch_get_hf_warm_state
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    if not state.present or not 0 < state.density_count <= 768**2:
        raise RuntimeError("missing or unbounded final density")
    density = np.empty(state.density_count, dtype=np.float64)
    coordinates = np.empty(state.coordinate_count, dtype=np.float64)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(owner._library, query(owner._batch, 0, ctypes.byref(state)))
    np.save(destination, density)
    destination.flush()
    destination.seek(0)
    return {"path": path.name, "sha256": hashlib.sha256(destination.read()).hexdigest(),
            "count": density.size}


def main() -> None:
    """Authenticate source/library bytes before one fresh complete E+F endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", choices=(12, 96), type=int, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Real GPU probes require srun and its assigned visibility")
    for name in (
        "GENERATIVEQC_EXPERIMENT_VV10_INNER_POLICY",
        "GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED",
        "GENERATIVEQC_VV10_NATIVE_SEED",
    ):
        if name in os.environ:
            raise RuntimeError(f"Default qualification forbids an override: {name}")
    density_path = arguments.output.with_suffix(".density.npy")
    for path in (arguments.output, density_path):
        if os.path.lexists(path):
            raise FileExistsError(path)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    source = arguments.source.resolve()
    identity = _source_identity(
        source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    )
    library_path = Path(os.environ["GENERATIVEQC_LIBRARY"])
    library = ctypes.CDLL(str(library_path))
    library.generativeqc_get_source_identity.restype = ctypes.c_char_p
    if identity != library.generativeqc_get_source_identity().decode():
        raise RuntimeError("Native library does not match the declared production source")
    case = scaling_cases()[f"water-{arguments.atoms}"]
    basis, _ = load_comparison_basis(
        arguments.basis, case, role="orbital", compute_forces=True
    )
    record = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(arguments).items()},
        "native_source_identity": identity,
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "visibility": os.environ["CUDA_VISIBLE_DEVICES"],
        "basis_sha256": hashlib.sha256(arguments.basis.read_bytes()).hexdigest(),
        "basis_identity": basis.identity,
        "coordinates_bohr": [list(atom[1]) for atom in case.atoms],
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "density_fitting": "none",
        "grid": [48, 16, 32],
        "production_defaults_changed": True,
        "experimental_proposal_policy": None,
    }
    # Exclusive creation rejects existing files and symlinks, including races
    # after preflight. Keep both descriptors and failed attempts; never reopen
    # a receipt path or remove a reservation that another invocation may see.
    with (
        arguments.output.open("x", encoding="utf-8") as receipt,
        density_path.open("x+b") as density_destination,
    ):
        started = perf_counter()
        try:
            calculator = Calculator(
                method="wb97m-v-rks", basis=basis, device="cuda",
                basis_representation="spherical",
                ks_options=KsOptions(grid=GridSpec(48, 16, 32)),
                density_fitting="none", precision="fp64",
                energy_tolerance=1e-12, density_tolerance=1e-10,
                screening_tolerance=1e-12, max_iterations=100, diis_history=8,
            )
            record["construction_seconds"] = perf_counter() - started
            record["native_build"] = native_build_metadata(calculator)
            started = perf_counter()
            with calculator.prepare_batch([case.atoms], warm_start=True) as owner:
                cp.cuda.runtime.deviceSynchronize()
                record["prepare_seconds"] = perf_counter() - started
                execution_started = perf_counter()
                item = owner.execute(strict=False, properties=("energy", "forces")).items[0]
                cp.cuda.runtime.deviceSynchronize()
                record["execute_seconds"] = perf_counter() - execution_started
                record["cold_endpoint_seconds"] = perf_counter() - started
                record.update(
                    succeeded=item.succeeded, converged=item.converged,
                    energy=float(item.energy), iterations=int(item.iterations),
                    fock_builds=item.fock_builds, density_rms=float(item.density_rms),
                    physical_residual_rms=float(item.physical_residual_rms),
                    energy_change=float(item.energy_change), initial_guess=item.initial_guess,
                    warm_start_used=bool(item.warm_start_used),
                    warm_start_fallback=bool(item.warm_start_fallback), precision=item.precision,
                    ks_diagnostic=item.ks_diagnostic.to_payload(),
                    forces=None if item.forces is None else item.forces.tolist(),
                    finite_forces=None if item.forces is None else bool(np.isfinite(item.forces).all()),
                )
                if item.succeeded and item.converged:
                    record["final_density"] = export_density(
                        owner, density_path, density_destination
                    )
        except Exception:
            record["error"] = traceback.format_exc()
        receipt.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record.get(key) for key in (
        "iterations", "fock_builds", "energy", "density_rms", "physical_residual_rms",
        "cold_endpoint_seconds", "error",
    )}), flush=True)
    if "error" in record or not record.get("succeeded") or not record.get("converged"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
