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
    root.add_argument(
        "--component",
        choices=("xc", "jk", "df"),
        help="DFT XC tile search (default for KS), or isolated native J/K policy recipes",
    )
    from ._dft_autotune import DFT_FUNCTIONALS

    root.add_argument(
        "--method", choices=("rhf", "uhf", *DFT_FUNCTIONALS), default="rhf"
    )
    references = root.add_mutually_exclusive_group()
    references.add_argument(
        "--reference-engine",
        choices=("pyscf", "gpu4pyscf"),
        default="pyscf",
        help="independent DFT oracle; GPU4PySCF avoids a CPU SCF/gradient solve",
    )
    references.add_argument(
        "--reference-file",
        type=Path,
        help="reuse a model-matched independent DFT reference document, not its timings",
    )
    root.add_argument("--reference-sha256", help="required SHA256 of --reference-file")
    root.add_argument(
        "--auxiliary-basis", help="fixed DF auxiliary basis (defaults to orbital basis)"
    )
    root.add_argument(
        "--df-memory-mib", type=int, help="fixed DF workspace hint in MiB (default 64)"
    )
    root.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="DFT Treutler grid level (0..9; fixed during tuning)",
    )
    root.add_argument(
        "--xc-tiles",
        type=int,
        nargs="+",
        help="DFT device-resident XC point-tile candidates (1..4096)",
    )
    root.add_argument("--charge", type=int, default=0)
    root.add_argument("--multiplicity", type=int, default=1)
    root.add_argument("--units", choices=("angstrom", "bohr"), default="angstrom")
    root.add_argument(
        "--representation", choices=("cartesian", "spherical"), default="cartesian"
    )
    root.add_argument("--batch", type=int, default=1)
    root.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA-visible GPU ordinal used by every tuning and validation stage",
    )
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
    return root


def main(argv: list[str] | None = None) -> int:
    """Run the original evidence-gated tuner, without importing runtime CLI code."""
    arguments = parser()
    args = arguments.parse_args(argv)
    is_dft = args.method not in ("rhf", "uhf")
    if args.reference_engine == "gpu4pyscf" and not is_dft:
        arguments.error("--reference-engine gpu4pyscf requires a DFT method")
    if bool(args.reference_file) != bool(args.reference_sha256):
        arguments.error("--reference-file and --reference-sha256 are required together")
    if args.reference_file and not is_dft:
        arguments.error("--reference-file requires a DFT method")
    if args.component in ("xc", "jk") and not is_dft:
        arguments.error("--component requires a DFT method")
    if args.component == "df" and args.method != "rhf":
        arguments.error(
            "DF policy recipes currently require RHF; UHF lacks complete Fock counts and KS/DF resource admission is not qualified"
        )
    if args.component != "df" and (
        args.auxiliary_basis is not None or args.df_memory_mib is not None
    ):
        arguments.error("--auxiliary-basis and --df-memory-mib require --component df")
    if args.component == "df":
        args.df_memory_mib = 64 if args.df_memory_mib is None else args.df_memory_mib
        if not 1 <= args.df_memory_mib <= 2**32 - 1:
            arguments.error("--df-memory-mib must be a positive uint32 MiB bound")
    if args.component in ("jk", "df") and (
        args.xc_tiles is not None or args.portable_baseline or args.export
    ):
        arguments.error(
            "native policy recipes do not use --xc-tiles, --portable-baseline or --export"
        )
    if not 0 <= args.grid_level <= 9:
        arguments.error("--grid-level must be in [0,9]")
    if not is_dft and (args.xc_tiles is not None or args.grid_level != 3):
        arguments.error("--xc-tiles and --grid-level require a DFT method")
    if args.xc_tiles is not None and any(
        not 1 <= tile <= 4096 for tile in args.xc_tiles
    ):
        arguments.error("--xc-tiles must contain integers in [1,4096]")
    if (is_dft or args.component == "df") and args.repeats < 5:
        arguments.error("DFT tuning requires at least five paired repeats")
    if not 0 <= args.device_id <= 2**31 - 1:
        arguments.error("--device-id must be a nonnegative int32 CUDA-visible ordinal")
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
                    "qualified_recipe": report.get("qualified_recipe"),
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
