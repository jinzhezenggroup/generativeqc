"""Measure matching archived Python/native revisions, including complete cleanup.

Run the same frozen runner with each revision's Python package on PYTHONPATH and
its installed GENERATIVEQC_LIBRARY. This is a revision comparison, not a claim
that solving a small basis first is faster than an independent cold target.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
import typing
from pathlib import Path

import generativeqc
from _support import raw_output_path
from generativeqc import Calculator, projected_singlepoint
from generativeqc.autotune import source_identity


def file_hash(path: Path) -> str:
    """Record immutable source instrumentation and installed binary provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_record(item: typing.Any) -> dict[str, typing.Any]:
    """Retain physical publications and observed work, not just elapsed time."""
    return {
        "energy": item.energy,
        "forces": item.forces.tolist(),
        "iterations": item.iterations,
        "fock_builds": item.fock_builds,
        "energy_change": item.energy_change,
        "density_rms": item.density_rms,
        "converged": item.converged,
        "backend": item.executed_backend,
        "restart_origin": item.restart_origin,
        "warm_start_used": item.warm_start_used,
        "warm_start_fallback": item.warm_start_fallback,
        "precision": item.precision,
    }


def cases(molecules: list[str]) -> typing.Iterator[dict[str, typing.Any]]:
    """Original and genuinely displaced nuclei share every non-geometric control."""
    geometries = {
        "h2": [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))],
        "lih": [("Li", (0.0, 0.0, 0.0)), ("H", (0.1, 0.05, 3.0))],
        "water": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.43, 1.11)),
            ("H", (0.0, 1.43, 1.11)),
        ],
    }
    for molecule in molecules:
        for method in ("rhf", "uhf") if molecule == "h2" else ("rhf",):
            for fitted in (False, True):
                for moved in (False, True):
                    atoms = list(geometries[molecule])
                    if moved:
                        symbol, position = atoms[-1]
                        atoms[-1] = (
                            symbol,
                            tuple(
                                coordinate + displacement
                                for coordinate, displacement in zip(
                                    position, (0.07, -0.04, 0.08), strict=True
                                )
                            ),
                        )
                    yield {
                        "name": "-".join(
                            (
                                molecule,
                                method,
                                "df" if fitted else "direct",
                                "moved" if moved else "original",
                            )
                        ),
                        "atoms": atoms,
                        "charge": 1 if method == "uhf" else 0,
                        "multiplicity": 2 if method == "uhf" else 1,
                        "source_basis": "sto-3g",
                        "target_basis": "def2-svp",
                        "settings": {
                            "method": method,
                            "device": "cpu",
                            "energy_tolerance": 1e-12,
                            "density_tolerance": 1e-10,
                            "screening_tolerance": 1e-14,
                        },
                        "target_density_fitting": "cpu" if fitted else "none",
                    }


def endpoint(inputs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Time construction through source, projection, target, copies and teardown."""
    started = time.perf_counter()
    source = Calculator(basis=inputs["source_basis"], **inputs["settings"])
    target = Calculator(
        basis=inputs["target_basis"],
        density_fitting=inputs["target_density_fitting"],
        **inputs["settings"],
    )
    result = projected_singlepoint(
        target,
        source,
        inputs["atoms"],
        charge=inputs["charge"],
        multiplicity=inputs["multiplicity"],
    )
    publication = {
        "source": item_record(result.source),
        "target": item_record(result.target),
        "density": result.target_density.tolist(),
        "diagnostics": result.diagnostics,
    }
    del result, target, source
    publication["seconds"] = time.perf_counter() - started
    for name in ("source", "target"):
        if not publication[name]["converged"]:
            raise RuntimeError(f"{inputs['name']}: {name} did not converge")
        if publication[name]["backend"] != "cpu_reference":
            raise RuntimeError(f"{inputs['name']}: unexpected backend")
    if publication["target"]["restart_origin"] != "basis_projection":
        raise RuntimeError(f"{inputs['name']}: projection was not admitted")
    return publication


def main() -> None:
    """Write a fresh receipt; source hashing rejects mixed native/Python science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--molecule", choices=("h2", "lih", "water"), action="append")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.output.exists():
        parser.error(
            "positive repeats, nonnegative warmups and a fresh output required"
        )
    if len(args.source_sha) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_sha
    ):
        parser.error("source-sha must be the full archived Git revision")
    source_root = Path(generativeqc.__file__).resolve().parents[2]
    library_path = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    library = ctypes.CDLL(str(library_path))
    library.generativeqc_get_source_identity.restype = ctypes.c_char_p
    identity = source_identity(source_root)
    if library.generativeqc_get_source_identity().decode() != identity:
        raise RuntimeError("installed library does not match the archived source")
    rows = []
    for inputs in cases(args.molecule or ["h2"]):
        first = endpoint(inputs)
        for _ in range(args.warmups):
            endpoint(inputs)
        samples = [endpoint(inputs) for _ in range(args.repeats)]
        rows.append(
            {
                "inputs": inputs,
                "first_use": first,
                "samples": samples,
                "median_seconds": statistics.median(row["seconds"] for row in samples),
            }
        )
        print(inputs["name"], rows[-1]["median_seconds"], flush=True)
    receipt = {
        "schema": "generativeqc.basis_projection_revision_endpoint.v1",
        "source_sha": args.source_sha,
        "source_identity": identity,
        "binary_sha256": file_hash(library_path),
        "runner_sha256": file_hash(Path(__file__)),
        "python_source_root": str(source_root),
        "library": str(library_path),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
        "first_use_scope": "same process/compiler cache; not compiler-cold",
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
