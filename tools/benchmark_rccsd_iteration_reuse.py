"""Retain cache-backed, interleaved full solve_cpu timings for opt-in reuse.

This constructs a synthetic dense Problem before timing. It measures every native
solver allocation, iteration, trial, DIIS step, independent replay and result
copy; it does not measure an SCF-to-correlated molecular endpoint or source setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, timeout: int = 300) -> str:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\n"
            + result.stdout
            + result.stderr
        )
    return result.stdout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, object]:
    base = _run(["git", "rev-parse", "HEAD"]).strip()
    paths = set(_run(["git", "diff", "--name-only", "HEAD"]).splitlines())
    paths.update(
        _run(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    )
    dirty = {
        name: _sha(ROOT / name) if (ROOT / name).is_file() else "deleted"
        for name in sorted(paths)
    }
    identity = {"base_commit": base, "dirty_files_sha256": dirty}
    identity["working_tree_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--diis-size", type=int, default=6)
    parser.add_argument("--shapes", nargs="+", default=["2,4", "4,8", "6,12", "8,16"])
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--launcher", help="Verified sccache or ccache path")
    parser.add_argument("--fixed-work-control", action="store_true")
    parser.add_argument("--with-fixed-work-control", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30 or args.diis_size not in (0, *range(2, 21)):
        parser.error("repeats must be 1..30 and DIIS size 0 or 2..20")
    try:
        shapes = [tuple(map(int, shape.split(","))) for shape in args.shapes]
        if any(
            len(shape) != 2 or not (1 <= shape[0] <= 12 and 1 <= shape[1] <= 32)
            for shape in shapes
        ):
            raise ValueError
    except ValueError:
        parser.error("shapes must be occupied,virtual pairs bounded by 12,32")
    if args.fixed_work_control and len(shapes) != 1:
        parser.error("fixed-work control requires exactly one --shapes pair")
    if args.fixed_work_control and args.with_fixed_work_control:
        parser.error("choose one fixed-work control mode")
    launcher = args.launcher or shutil.which("sccache") or shutil.which("ccache")
    compiler = shutil.which(args.compiler)
    if launcher is None or compiler is None:
        parser.error("a C++ compiler and verified sccache or ccache are required")
    launcher = str(Path(launcher).resolve())
    if Path(launcher).name not in ("sccache", "ccache"):
        parser.error("launcher must be sccache or ccache")
    launcher_version = _run([launcher, "--version"])
    compiler_version = _run([compiler, "--version"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache-before.txt").write_text(_run([launcher, "--show-stats"]))
    source_identity = _source_identity()
    # Generate under -S to preserve the standalone compiler ownership boundary.
    generation = """
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root / 'python')]
from tools import generate_rccsd_native as dense, generate_df_ccsd_native as actions
from tools import generate_df_ccsd_core as core, generate_df_ccsd_hoisted as hoisted
for name, module in [('rccsd', dense), ('df_ccsd', actions), ('df_ccsd_core', core), ('df_ccsd_hoisted', hoisted)]:
    (target / ('generated_' + name + '_cpu.hpp')).write_text(module.cpu_header())
"""
    _run([sys.executable, "-S", "-c", generation, str(ROOT), str(output)])
    sources = [
        ROOT / "src/cc/solver.cpp",
        ROOT / "benchmarks/rccsd_iteration_reuse.cpp",
    ]
    flags = [
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-DGENERATIVEQC_HAS_CUDA=0",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "include"),
        "-I" + str(output),
    ]
    commands, objects = [], []
    for index, source in enumerate(sources):
        obj = output / f"benchmark-{index}.o"
        command = [launcher, compiler, *flags, "-c", str(source), "-o", str(obj)]
        commands.append(command)
        _run(command)
        objects.append(str(obj))
    binary = output / "benchmark"
    link = [compiler, *objects, "-o", str(binary)]
    commands.append(link)
    _run(link)
    if _source_identity() != source_identity:
        raise RuntimeError(
            "source tree changed during generation/compilation; rerun a stable tree"
        )
    run = (
        [str(binary), "--fixed-work-control", str(args.repeats)]
        if args.fixed_work_control
        else [str(binary), str(args.repeats), str(args.diis_size)]
    )
    run.extend(str(extent) for shape in shapes for extent in shape)
    raw = _run(run)
    control_command = None
    control_raw = None
    if args.with_fixed_work_control:
        largest = max(shapes, key=lambda shape: shape[0] * shape[1])
        control_command = [str(binary), "--fixed-work-control", str(args.repeats)]
        control_command.extend(str(extent) for extent in largest)
        control_raw = _run(control_command)
    if _source_identity() != source_identity:
        raise RuntimeError("source tree changed during execution; rerun a stable tree")
    # Each pair is retained in execution order; never replace raw repeats with
    # an unqualified speedup claim. The binary checks exact final-state/count parity.
    rows = [json.loads(line) for line in raw.splitlines()]
    (output / "raw.jsonl").write_text(raw)
    if control_raw is not None:
        (output / "raw-control.jsonl").write_text(control_raw)
    (output / "cache-after.txt").write_text(_run([launcher, "--show-stats"]))
    provenance = {
        "scope": __doc__,
        "fixed_work_control": args.fixed_work_control,
        "fixed_work_scope": (
            "Nine changing-amplitude evaluations; original full versus pinned reprepare "
            "every call versus pinned prepare once. Reprepare and reuse use identical "
            "buffer addresses. Allocation, input construction, reference construction "
            "and bitwise checks are outside reported time. No solver, DIIS or replay."
            if args.fixed_work_control or args.with_fixed_work_control
            else None
        ),
        **source_identity,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": compiler_version,
        "launcher": launcher_version,
        "compile_commands": commands,
        "run_command": run,
        "control_command": control_command,
        "control_raw_sha256": (
            _sha(output / "raw-control.jsonl") if control_raw is not None else None
        ),
        "cache_environment": {
            key: os.environ[key]
            for key in ("CCACHE_DIR", "CCACHE_BASEDIR", "SCCACHE_DIR")
            if key in os.environ
        },
        "binary_sha256": _sha(binary),
        "benchmark_source_sha256": _sha(sources[1]),
        "generated_headers_sha256": {
            path.name: _sha(path) for path in sorted(output.glob("generated_*_cpu.hpp"))
        },
        "raw_sha256": _sha(output / "raw.jsonl"),
        "shapes": shapes,
        "diis_size": args.diis_size,
        "repeats": args.repeats,
        "cases": len(rows),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Retained {len(rows)} shapes and {args.repeats} interleaved pairs in {output}"
    )


if __name__ == "__main__":
    main()
