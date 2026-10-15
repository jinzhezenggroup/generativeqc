"""Time complete CPU CPKS endpoints, retaining numerical and semantic-work receipts.

Run parent and candidate in separate processes with their matching Python trees
and native libraries. First-use samples share one process and a persistent
compiler cache, not independent process/compiler-cold trials. Every warm sample
reconstructs and closes all endpoint owners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, TextIO

try:
    from _retention import raw_output_path
except ModuleNotFoundError as error:
    if error.name != "_retention":
        raise
    from benchmarks._retention import raw_output_path


def endpoint(method: str, *, moved: bool, strategy: str) -> dict[str, Any]:
    """Include construction, SCF, physical response, true-residual checks and cleanup."""
    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc_compiler.dft import NativeAO

    from tools.generativeqc_response import (
        GMRESOptions,
        NativeRKSResponse,
        NativeUKSResponse,
        solve_many,
    )

    unrestricted = method.endswith("uks")
    atoms = (
        [("Li", (0.1, -0.2, 0.0)), ("H", (0.1, -0.2, 2.7))]
        if unrestricted
        else [("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5))]
    )
    if moved:
        symbol, position = atoms[-1]
        atoms[-1] = (
            symbol,
            (position[0] + 0.003, position[1] - 0.002, position[2] + 0.004),
        )
    charge, multiplicity = (1, 2) if unrestricted else (0, 1)
    started = time.perf_counter()
    calculator = Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    with (
        calculator.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(atoms, charge=charge, multiplicity=multiplicity) as basis,
    ):
        scf = batch.execute(strict=True).items[0]
        owner = NativeUKSResponse if unrestricted else NativeRKSResponse
        with owner.from_native(batch, basis, tile_points=257) as response:
            rhs = (
                np.random.default_rng(2227).normal(size=(response.dimension, 3)) * 0.02
            )
            rhs[:, 1] = -0.4 * rhs[:, 0]
            rhs[:, 2] = 0.0
            options = GMRESOptions(
                atol=1e-12, rtol=1e-11, restart=20, max_iterations=160
            )
            result = solve_many(
                response, rhs, strategy=strategy, options=options, raise_on_failure=True
            )
            residuals = [
                float(np.linalg.norm(rhs[:, column] - response.apply(item.solution)))
                for column, item in enumerate(result.results)
            ]
            for column, residual in enumerate(residuals):
                target = max(
                    options.atol, options.rtol * np.linalg.norm(rhs[:, column])
                )
                if not np.isfinite(residual) or residual > 1.01 * target:
                    raise RuntimeError(
                        "complete CPKS endpoint failed true-residual acceptance"
                    )
            receipt = {
                "atoms_bohr": atoms,
                "energy": scf.energy,
                "dimension": response.dimension,
                "solution": result.solution.tolist(),
                "true_residuals": residuals,
                "validation_actions": len(residuals),
                "solves": [
                    {
                        "converged": item.converged,
                        "reason": item.reason,
                        "iterations": item.iterations,
                        "operator_actions": item.operator_actions,
                        "preconditioner_actions": item.preconditioner_actions,
                        "residual_norm": item.residual_norm,
                        "workspace_bytes": item.workspace_bytes,
                    }
                    for item in result.results
                ],
            }
    receipt["seconds"] = time.perf_counter() - started
    return receipt


def main() -> None:
    """Validate retention before scientific imports, then record every measured endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--strategy", choices=("sequential", "recycled"), default="recycled"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"]
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    if not set(args.methods) <= {"lda-rks", "pbe-rks", "lda-uks", "pbe-uks"}:
        parser.error("unsupported CPU CPKS method")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        publication: dict[str, Any] = {
            "source_sha": args.source_sha,
            "status": "running",
            "cases": {},
            "active": {"phase": "initialization"},
        }
        _write_receipt(stream, publication)
        try:
            _run(args, publication)
            publication.pop("active", None)
            publication["status"] = "complete"
            _write_receipt(stream, publication)
        except BaseException as error:
            publication["status"] = "failed"
            publication["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            try:
                _write_receipt(stream, publication)
            except BaseException as journal_error:
                raise error from journal_error
            raise


def _write_receipt(stream: TextIO, publication: dict[str, Any]) -> None:
    """Write through the exclusive descriptor even if its pathname is replaced."""
    payload = json.dumps(publication, indent=2, allow_nan=False) + "\n"
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()


def _run(args: argparse.Namespace, publication: dict[str, Any]) -> None:
    """Preserve completed physical publications and the current failure boundary."""

    import generativeqc
    from generativeqc import response_solver

    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve(strict=True)
    publication.update(
        {
            "python_package": str(Path(generativeqc.__file__).resolve()),
            "solver_source": str(Path(response_solver.__file__).resolve()),
            "library": str(library),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "strategy": args.strategy,
            "first_use_scope": "per-case first endpoint; shared process and persistent compiler cache",
            "timing_scope": "construct+SCF+prepare-response+solve-many+residual-check+close",
            "cases": {},
        }
    )
    for method in args.methods:
        for moved in (False, True):
            key = f"{method}-{'moved' if moved else 'original'}"
            row: dict[str, Any] = {
                "warmups": args.warmups,
                "samples": [],
            }
            publication["cases"][key] = row
            publication["active"] = {"case": key, "phase": "first_use"}
            row["first_use"] = _validated_endpoint(
                method, moved=moved, strategy=args.strategy
            )
            for index in range(args.warmups):
                publication["active"].update(phase="warmup", index=index)
                _validated_endpoint(method, moved=moved, strategy=args.strategy)
            for index in range(args.repeats):
                publication["active"].update(phase="sample", index=index)
                row["samples"].append(
                    _validated_endpoint(method, moved=moved, strategy=args.strategy)
                )
            row["median_seconds"] = statistics.median(
                sample["seconds"] for sample in row["samples"]
            )


def _validated_endpoint(method: str, *, moved: bool, strategy: str) -> dict[str, Any]:
    """Reject invalid JSON before it can poison retained partial results."""
    result = endpoint(method, moved=moved, strategy=strategy)
    json.dumps(result, allow_nan=False)
    return result


if __name__ == "__main__":
    main()
