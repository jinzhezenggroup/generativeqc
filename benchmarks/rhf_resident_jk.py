"""Slurm-only exact Direct action receipts for an endpoint's basis and geometry.

The native test seam uses CPU source construction only to normalize metadata;
all retained-value preparation, J/K replay and independent recomputation run
on CUDA. This is a diagnostic, not a substitute for complete-force timing or
the independent libcint/finite-difference acceptance tests.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.generativeqc_posthf.fixtures import source_arguments
from tools.generativeqc_posthf.sources import NativeSource


def read_input(path: Path, displacement: float) -> dict:
    """Read the basis-only native complete-force input, with no orbital oracle."""
    tokens = iter(path.read_text().split())
    atom_count, orbital_count, auxiliary_count, _ = (
        int(next(tokens)) for _ in range(4)
    )
    atoms = []
    coordinates = []
    for _ in range(atom_count):
        atoms.append(int(next(tokens)))
        coordinates.append([float(next(tokens)) for _ in range(3)])
    coordinates[0][0] += displacement

    def shells(count: int) -> list[dict]:
        records = []
        for _ in range(count):
            atom_index, angular, primitive_count = (int(next(tokens)) for _ in range(3))
            records.append(
                {
                    "atom_index": atom_index,
                    "angular_momentum": angular,
                    "primitives": [
                        [float(next(tokens)), float(next(tokens))]
                        for _ in range(primitive_count)
                    ],
                }
            )
        return records

    return {
        "inputs": {
            "atomic_numbers": atoms,
            "coordinates": coordinates,
            "shells": shells(orbital_count),
            "charge": 0,
            "basis_representation": "real_spherical",
        },
        "auxiliary_shells": shells(auxiliary_count),
    }


def main() -> None:
    """Emit reproducible paired host/device action timings, including cache setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-bytes", type=int, default=8 << 30)
    parser.add_argument("--density-count", type=int, default=3)
    parser.add_argument("--displacement", type=float, default=0.0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("requires a finite Slurm GPU allocation")
    if args.density_count <= 0 or args.maximum_bytes < 0:
        parser.error("density count must be positive and byte capacity nonnegative")
    library = ct.CDLL(str(args.probe.resolve()))
    call = library.rhf_resident_jk_probe
    double_pointer = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_void_p,
        double_pointer,
        ct.c_size_t,
        ct.c_bool,
        ct.c_size_t,
        double_pointer,
        ct.POINTER(ct.c_uint64),
        double_pointer,
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    metadata = read_input(args.input, args.displacement)
    with NativeSource(**source_arguments(metadata)) as source:
        native_path = Path(source._library._name).resolve()
        geometry_hash, basis_hash = source.geometry_hash, source.basis_hash
        dimension = source.nbf
        rng = np.random.default_rng(1972)
        densities = rng.normal(size=(args.density_count, dimension, dimension))
        densities = np.ascontiguousarray(densities + densities.swapaxes(-1, -2))
        output = np.full((args.density_count, 2, 2, dimension, dimension), np.nan)
        counts = np.zeros(8, dtype=np.uint64)
        timings = np.zeros(5)
        error = ct.create_string_buffer(2048)
        status = call(
            source._handle,
            densities.ctypes.data_as(double_pointer),
            args.density_count,
            False,
            args.maximum_bytes,
            output.ctypes.data_as(double_pointer),
            counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            timings.ctypes.data_as(double_pointer),
            error,
            len(error),
        )
    if status:
        raise RuntimeError(error.value.decode())
    np.testing.assert_allclose(output[:, 0], output[:, 1], atol=2e-10, rtol=2e-12)
    receipt = {
        "schema": "generativeqc.rhf_resident_jk.v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "native_library": str(native_path),
        "native_library_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "geometry_hash": geometry_hash,
        "basis_hash": basis_hash,
        "timing_scope": {
            "wall": "diagnostic action including input/output transfers and final fence",
            "device": "event interval around Direct action, excluding input/output transfers",
            "setup": "first resident-value preparation, excluding Direct plan construction",
        },
        "displacement_bohr": args.displacement,
        "nbf": dimension,
        "maximum_bytes": args.maximum_bytes,
        "required_bytes": int(counts[0]),
        "resident_bytes": int(counts[1]),
        "resident_values": int(counts[2]),
        "actions_per_route": int(counts[3]),
        "observed_route_quartet_visits": int(counts[4]),
        "observed_route_radial_evaluations": int(counts[5]),
        "admission_status": int(counts[6]),
        "observed_route_census_available": bool(counts[7]),
        "setup_seconds": float(timings[0]),
        "observed_route_wall_seconds_per_action": float(
            timings[1] / args.density_count
        ),
        "observed_route_device_seconds_per_action": float(
            timings[2] / args.density_count
        ),
        "uncached_wall_seconds_per_action": float(timings[3] / args.density_count),
        "uncached_device_seconds_per_action": float(timings[4] / args.density_count),
        "maximum_action_difference": float(np.max(np.abs(output[:, 0] - output[:, 1]))),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
