"""Measure the complete Direct Cartesian RKS Hessian/HVP endpoint (#1905)."""

import argparse
import ctypes
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import Calculator, _native_hessian
from generativeqc.rks_hessian import (
    RKSHessianResult,
    RKSHVPResult,
    rks_hessian,
    rks_hvp,
)
from generativeqc.rks_response import NativeRKSResponse
from generativeqc_compiler.dft import NativeAO
from test_semilocal_cartesian_hessian_cuda import (
    POSITIONS,
    SOLVER,
    SYMBOLS,
    _calculator,
    _direction,
)


def profile_native_response() -> None:
    """Bracket only the synchronous native response call, after the accepted SCF."""
    runtime = ctypes.CDLL("libcudart.so.12")
    start = runtime.cudaProfilerStart
    stop = runtime.cudaProfilerStop
    start.argtypes = stop.argtypes = []
    start.restype = stop.restype = ctypes.c_int
    unprofiled = _native_hessian.cuda_hessian_bridge

    def bridge(library: Any) -> Callable[..., int]:
        execute = unprofiled(library)

        def profiled(*args: Any) -> int:
            """Forward the caller's ctypes signature without narrowing ABI arguments."""
            execute.argtypes = profiled.argtypes
            execute.restype = profiled.restype
            if start():
                raise RuntimeError("cannot start the CUDA response capture")
            try:
                return execute(*args)
            finally:
                if stop():
                    raise RuntimeError("cannot stop the CUDA response capture")

        return profiled

    _native_hessian.cuda_hessian_bridge = bridge


def mixed_endpoint(
    calculator: Calculator,
    atoms: list[tuple[str, np.ndarray]],
    endpoint: str,
    direction: np.ndarray,
    cache: Path,
) -> tuple[RKSHVPResult | RKSHessianResult, dict[str, int]]:
    """Run the original mixed path, whose accepted primal is a CPU SCF."""
    options = {
        "cache": cache,
        "first_backend": "cuda",
        "second_backend": "cuda",
        "solver_options": SOLVER,
    }
    with (
        calculator.prepare_batch([atoms], warm_start=False) as batch,
        NativeAO(atoms, basis="sto-3g", representation="cartesian") as basis,
    ):
        primal = batch.execute(strict=True, properties=("energy",)).items[0]
        compiler = batch._stationary_cuda_compiler()
        options.update(first_compiler=compiler, second_compiler=compiler)
        operator = NativeRKSResponse.from_native(batch, basis)
        try:
            result = (
                rks_hvp(operator, direction, **options)
                if endpoint == "hvp"
                else rks_hessian(operator, block_size=1, **options)
            )
        finally:
            operator.close()
    return result, {
        "scf_iterations": primal.iterations,
        "scf_fock_builds": primal.fock_builds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("native", "mixed"), required=True)
    parser.add_argument("--method", choices=("lda-rks", "pbe-rks"), required=True)
    parser.add_argument("--endpoint", choices=("hvp", "dense"), required=True)
    parser.add_argument(
        "--fixture", choices=("water-dimer", "water"), default="water-dimer"
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--profile-response", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("a real GPU must be allocated through Slurm")
    if args.profile_response:
        if args.arm != "native":
            parser.error("response tracing requires the native arm")
        profile_native_response()

    args.directory.mkdir(parents=True, exist_ok=True)
    atoms_count = len(SYMBOLS) if args.fixture == "water-dimer" else 3
    symbols, positions = SYMBOLS[:atoms_count], POSITIONS[:atoms_count]

    def motion(seed: int) -> np.ndarray:
        if args.fixture == "water-dimer":
            return _direction(np.random.default_rng(seed))
        displacement = np.random.default_rng(seed).normal(size=positions.shape)
        displacement -= displacement.mean(axis=0)
        return displacement / np.linalg.norm(displacement)

    direction = motion(1905)
    changed = positions + 1.0e-2 * motion(42)
    phases = (("cold", positions), ("warm", positions), ("moved", changed))
    if args.profile_response:
        phases = phases[:1]
    for phase, geometry in phases:
        atoms = list(zip(symbols, geometry, strict=True))
        cache = args.directory / "cache"
        started = time.perf_counter()
        calculator = _calculator(args.method, "cuda" if args.arm == "native" else "cpu")
        if args.arm == "native":
            result = (
                calculator.hessian_vector_product(
                    atoms, direction, cache=cache, solver_options=SOLVER
                )
                if args.endpoint == "hvp"
                else calculator.hessian(
                    atoms, block_size=1, cache=cache, solver_options=SOLVER
                )
            )
            work = {}
        else:
            result, work = mixed_endpoint(
                calculator, atoms, args.endpoint, direction, cache
            )
        elapsed = time.perf_counter() - started
        value = result.value if args.endpoint == "hvp" else result.matrix
        np.save(args.directory / f"{phase}.npy", value, allow_pickle=False)
        receipt = {
            "arm": args.arm,
            "method": args.method,
            "endpoint": args.endpoint,
            "fixture": args.fixture,
            "phase": phase,
            "wall_seconds": elapsed,
            "job": os.environ["SLURM_JOB_ID"],
            "diagnostics": {**result.diagnostics, **work},
        }
        print(json.dumps(receipt, default=str), flush=True)


if __name__ == "__main__":
    main()
