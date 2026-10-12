"""Reproducible CPU-only public Hessian/HVP endpoint timing and parity receipt.

Use a fresh process for each *independent* process-first trial. The first call
within a process is NOT automatically a compiler-cold trial: the persistent JIT
cache may already contain artifacts. No sample is filtered from the receipt.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _fixture(case: str) -> tuple[list[tuple[str, tuple[float, ...]]], Any, np.ndarray]:
    """Give public Cartesian Bohr geometry, explicit basis and normalized direction."""
    if case == "h2":
        atoms = [
            ("H", (0.0, 0.0, -0.72)),
            ("H", (0.08, -0.03, 0.71)),
        ]
        basis: Any = "sto-3g"
        direction = np.asarray([[0.4, -0.1, 0.3], [-0.2, 0.5, -0.4]])
    elif case == "he13":
        from generativeqc import Primitive, Shell

        atoms = [("He", (0.13, -0.21, 0.17))]
        basis = (
            Shell(0, 0, (Primitive(1.5, 1.0),)),
            Shell(0, 2, (Primitive(0.8, 1.0),)),
            Shell(0, 2, (Primitive(0.35, 1.0),)),
        )
        direction = np.asarray([[0.31, -0.27, 0.19]])
    else:
        raise ValueError("unknown Hessian benchmark fixture")
    direction /= np.linalg.norm(direction)
    return atoms, basis, direction


def _moved(
    atoms: list[tuple[str, tuple[float, ...]]],
    direction: np.ndarray,
    displacement: float,
) -> list[tuple[str, tuple[float, float, float]]]:
    """Change the *molecular* geometry, without aliasing original positions."""
    if not np.isfinite(displacement) or displacement <= 0.0:
        raise ValueError("displacement must be positive and finite")
    if len(atoms) != len(direction):
        raise ValueError("direction must match atom count")
    return [
        (symbol, tuple((np.asarray(xyz) + displacement * dv).tolist()))
        for (symbol, xyz), dv in zip(atoms, direction, strict=True)
    ]


def _sample(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def _summary(samples: list[float]) -> dict[str, Any]:
    """Retain all observed timings, including slow first calls and outliers."""
    if not samples or not all(np.isfinite(s) and s >= 0.0 for s in samples):
        raise ValueError("timing samples must be nonempty, nonnegative and finite")
    return {
        "n": len(samples),
        "seconds": list(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _source_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _validate_pair(hvp: Any, hessian: Any, direction: np.ndarray) -> dict[str, float]:
    """Check raw analytic outputs without symmetrizing the reported Hessian."""
    vector = direction.ravel()
    hv = np.asarray(hvp.value, dtype=np.float64).ravel()
    matrix = np.asarray(hessian.matrix, dtype=np.float64)
    if hv.shape != vector.shape or matrix.shape != (len(vector), len(vector)):
        raise ValueError("public Hessian/HVP output layout mismatch")
    if not np.isfinite(hv).all() or not np.isfinite(matrix).all():
        raise FloatingPointError("nonfinite CPU Hessian/HVP")
    residual = float(np.max(np.abs(matrix @ vector - hv)))
    scale = float(max(1.0, np.max(np.abs(hv)), np.max(np.abs(matrix @ vector))))
    symmetry = float(np.max(np.abs(matrix - matrix.T)))
    if residual > 1e-6 * scale:
        raise ArithmeticError("complete public HVP and full Hessian disagree")
    return {
        "hvp_full_max_abs_error": residual,
        "hvp_full_relative_gate_scale": scale,
        "raw_symmetry_max_abs_error": symmetry,
        "raw_hessian_frobenius_norm": float(np.linalg.norm(matrix)),
        "posthoc_symmetrization": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run matched public CPU calls with one retained method and fixed controls."""
    from generativeqc import Calculator, GridSpec, KsOptions

    atoms, basis, direction = _fixture(args.case)
    moved = _moved(atoms, direction, args.displacement)
    calculator = Calculator(
        method=args.method,
        basis=basis,
        device="cpu",
        precision="fp64",
        ks_options=KsOptions(
            grid=GridSpec(
                radial_points=args.radial_points,
                angular_polar=args.angular_polar,
                angular_azimuth=args.angular_azimuth,
            )
        ),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    if not calculator.capabilities.supported_second_order:
        raise NotImplementedError("selected CPU Hessian context is not qualified")

    def hvp(geometry: Any) -> Any:
        return calculator.hessian_vector_product(
            geometry,
            direction,
            cache=args.cache,
            integral_budget_bytes=args.integral_budget_bytes,
        )

    def hessian(geometry: Any) -> Any:
        return calculator.hessian(
            geometry,
            cache=args.cache,
            integral_budget_bytes=args.integral_budget_bytes,
            output_budget_bytes=args.output_budget_bytes,
            strategy=args.strategy,
        )

    # Both timed public calls include a fresh SCF and live response preparation.
    first_hvp, t_first_hvp = _sample(lambda: hvp(atoms))
    warm_hvp_times = []
    for _ in range(args.repeats):
        _, elapsed = _sample(lambda: hvp(atoms))
        warm_hvp_times.append(elapsed)
    changed_hvp, t_changed_hvp = _sample(lambda: hvp(moved))

    first_hessian, t_first_hessian = _sample(lambda: hessian(atoms))
    warm_hessian_times = []
    for _ in range(args.repeats):
        _, elapsed = _sample(lambda: hessian(atoms))
        warm_hessian_times.append(elapsed)
    changed_hessian, t_changed_hessian = _sample(lambda: hessian(moved))

    receipt = {
        "schema": "generativeqc.cpu-hessian-endpoint.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _source_sha(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "precision": "fp64",
        },
        "configuration": {
            "method": args.method,
            "case": args.case,
            "atoms_bohr": atoms,
            "direction": direction.tolist(),
            "displacement_bohr": args.displacement,
            "grid": {
                "radial_points": args.radial_points,
                "angular_polar": args.angular_polar,
                "angular_azimuth": args.angular_azimuth,
            },
            "integral_budget_bytes": args.integral_budget_bytes,
            "output_budget_bytes": args.output_budget_bytes,
            "cache": str(Path(args.cache).resolve()),
            "strategy": args.strategy,
            "repeats": args.repeats,
        },
        "timing_scope": (
            "complete public Calculator calls including fresh SCF, CPKS, "
            "integral HVP and publication; process-first may have a warm "
            "persistent JIT artifact cache"
        ),
        "timings": {
            "hvp": {
                "process_first": _summary([t_first_hvp]),
                "same_geometry": _summary(warm_hvp_times),
                "changed_geometry": _summary([t_changed_hvp]),
            },
            "full_hessian": {
                "process_first": _summary([t_first_hessian]),
                "same_geometry": _summary(warm_hessian_times),
                "changed_geometry": _summary([t_changed_hessian]),
            },
        },
        "validation": {
            "original": _validate_pair(first_hvp, first_hessian, direction),
            "changed_geometry": _validate_pair(
                changed_hvp, changed_hessian, direction
            ),
        },
        "notes": [
            "No CUDA or GPU acceptance is used.",
            "Same geometry calls rebuild live SCF/response, not a persistent CPKS.",
            "No timing samples are filtered or marked compiler-cold.",
            "This script does not replace independent gradient-difference oracles.",
        ],
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lda-rks", "pbe-rks"), default="pbe-rks")
    parser.add_argument("--case", choices=("h2", "he13"), default="h2")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--displacement", type=float, default=0.02)
    parser.add_argument("--radial-points", type=int, default=10)
    parser.add_argument("--angular-polar", type=int, default=4)
    parser.add_argument("--angular-azimuth", type=int, default=8)
    parser.add_argument("--integral-budget-bytes", type=int, default=64 << 20)
    parser.add_argument("--output-budget-bytes", type=int, default=64 << 20)
    parser.add_argument(
        "--strategy", choices=("sequential", "blocked", "recycled"), default="recycled"
    )
    parser.add_argument("--cache", type=Path, default=Path(".artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    payload = json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
