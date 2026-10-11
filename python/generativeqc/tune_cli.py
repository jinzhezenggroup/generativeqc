"""Explicit source/compiler autotuning CLI; no runtime command dispatch.

This developer tool is intentionally separate from the Python-free native
`generativeqc` executable. It never acts as a fallback for runtime commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    """Parse the explicit compiler search without shadowing native commands."""
    root = argparse.ArgumentParser(
        prog="generativeqc-tune",
        description="Compile and qualify workload-specific CUDA kernel candidates",
    )
    modes = root.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick", action="store_true", help="bounded hotspot search (default)"
    )
    modes.add_argument(
        "--full", action="store_true", help="broader class and schedule coverage"
    )
    root.add_argument("input", type=Path, help="representative XYZ workload")
    root.add_argument("--basis", default="def2-svp")
    root.add_argument("--method", choices=("rhf", "uhf"), default="rhf")
    root.add_argument("--charge", type=int, default=0)
    root.add_argument("--multiplicity", type=int, default=1)
    root.add_argument("--units", choices=("angstrom", "bohr"), default="angstrom")
    root.add_argument(
        "--representation", choices=("cartesian", "spherical"), default="cartesian"
    )
    root.add_argument("--batch", type=int, default=1)
    root.add_argument("--coverage", type=float)
    root.add_argument("--max-classes", type=int)
    root.add_argument("--max-candidates", type=int)
    root.add_argument("--compile-jobs", type=int, default=2)
    root.add_argument("--budget-seconds", type=int, default=7200)
    root.add_argument("--repeats", type=int, default=6)
    root.add_argument(
        "--retune", action="store_true", help="include already qualified consumers"
    )
    root.add_argument(
        "--portable-baseline",
        action="store_true",
        help="compare against generic CUDA even when the device has official tuning",
    )
    root.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="matching GenerativeQC source checkout for native/codegen compilation",
    )
    root.add_argument("--nvcc", type=Path)
    root.add_argument("--export", type=Path, help="export an accepted evidence bundle")
    root.set_defaults(device_id=0)
    return root


def main(argv: list[str] | None = None) -> int:
    """Run the original evidence-gated tuner, without importing runtime CLI code."""
    arguments = parser()
    args = arguments.parse_args(argv)
    args.coverage = (
        args.coverage if args.coverage is not None else (1.0 if args.full else 0.97)
    )
    args.max_classes = (
        args.max_classes if args.max_classes is not None else (55 if args.full else 8)
    )
    args.max_candidates = (
        args.max_candidates
        if args.max_candidates is not None
        else (16 if args.full else 4)
    )
    if (
        not 0 < args.coverage <= 1
        or min(
            args.batch,
            args.max_classes,
            args.max_candidates,
            args.compile_jobs,
            args.budget_seconds,
            args.multiplicity,
        )
        < 1
        or args.repeats < 4
    ):
        arguments.error(
            "positive budgets/counts, coverage in (0,1], and at least four repeats are required"
        )
    try:
        from .autotune import run

        report = run(args)
        print(
            json.dumps(
                {
                    "installed": report["installed"],
                    "report": report["directory"] + "/report.json",
                    "failure": report.get("failure"),
                    "reason": report.get("reason"),
                },
                indent=2,
            )
        )
        return int("failure" in report)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"generativeqc-tune: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
