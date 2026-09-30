"""Compile the exact native grid/XC source and publish complete PTXAS pressure.

This is compile-only evidence. It never initializes a GPU and cannot promote a
DFT schedule by itself. The report is usable by the shared profitability path
only after all active device-fused region scopes are present.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc.autotune import source_identity
from generativeqc.profiles import atomic_json
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_resources import KernelResources, parse_resources
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.ao_cuda import emit_grid_source
from generativeqc_compiler.dft.xc_compiled_resources import (
    GridXcCompiledResourceShape,
    native_grid_xc_compiled_region_evidence,
)


def _demangler(nvcc: Path) -> Path:
    adjacent = nvcc.resolve().parent / "cu++filt"
    if adjacent.is_file():
        return adjacent
    for name in ("cu++filt", "c++filt"):
        candidate = shutil.which(name)
        if candidate:
            return Path(candidate)
    raise RuntimeError("CUDA resource classification requires cu++filt or c++filt")


def _demangle(
    resources: tuple[KernelResources, ...],
    *,
    nvcc: Path,
) -> tuple[KernelResources, ...]:
    if not resources:
        raise ValueError("PTXAS emitted no native grid/XC resource records")
    names = [resource.function for resource in resources]
    run = subprocess.run(
        [str(_demangler(nvcc))],
        input="\n".join(names) + "\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if run.returncode:
        raise RuntimeError(f"CUDA symbol demangling failed: {run.stderr}")
    demangled = run.stdout.splitlines()
    if len(demangled) != len(resources):
        raise RuntimeError("CUDA symbol demangler returned an incomplete resource list")
    return tuple(
        replace(resource, function=name)
        for resource, name in zip(resources, demangled, strict=True)
    )


def run(args: argparse.Namespace) -> dict:
    generated_dir = args.generated_dir.resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)
    source, grid_identity, _ = emit_grid_source(native_ks=True)
    cmake_source = (
        args.cmake_source.resolve()
        if args.cmake_source is not None
        else generated_dir / "generated_grid_policy.cu"
    )
    if not cmake_source.is_file():
        raise FileNotFoundError(
            "native grid/XC resource qualification requires the CMake-generated source"
        )
    if cmake_source.read_text() != source:
        raise ValueError("CMake native grid/XC source differs from compiler emission")

    args.output.mkdir(parents=True, exist_ok=True)
    source_path = args.output / "generated_grid_policy.cu"
    object_path = args.output / "generated_grid_policy.o"
    source_path.write_text(source)

    target = cuda_target_info(args.architecture)
    compiler = CudaCompilerAdapter(args.nvcc.resolve(), target, args.compile_timeout)
    result = compiler.compile(
        source_path,
        object_path,
        includes=(
            ROOT / "src/dft",
            ROOT / "src",
            ROOT / "include",
            generated_dir,
        ),
        options=(
            "--fmad=false",
            "-DNDEBUG",
            "-DGENERATIVEQC_HAS_CUDA=1",
        ),
        standard="c++20",
    )
    (args.output / "compiler.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            "native grid/XC resource compilation failed: "
            + result.stdout
            + result.stderr
        )

    resources = _demangle(parse_resources(result.stderr), nvcc=args.nvcc)
    shape = GridXcCompiledResourceShape(
        npoint=args.npoint,
        tile_points=args.tile_points,
        nao=args.nao,
        spins=args.spins,
    )
    evidence = native_grid_xc_compiled_region_evidence(
        resources,
        shape=shape,
        functional=args.functional,
        target=target,
        source_identity=source_identity(ROOT),
        object_bytes=object_path.stat().st_size,
        compile_seconds=result.duration_seconds,
    )
    report = {
        "schema": "generativeqc.dft.grid-xc-resource-report.v1",
        "generated_grid_identity": grid_identity,
        "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cmake_source": str(cmake_source),
        "target": target.to_payload(),
        "compile": {
            "seconds": result.duration_seconds,
            "object_bytes": object_path.stat().st_size,
            "fmad": False,
            "ndebug": True,
            "generativeqc_has_cuda": True,
        },
        "compiled_region": evidence.to_payload(),
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--cmake-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", choices=("LDA_XC_PW", "PBE"), default="PBE")
    parser.add_argument("--npoint", type=int, default=4096)
    parser.add_argument("--tile-points", type=int, default=256)
    parser.add_argument("--nao", type=int, default=96)
    parser.add_argument("--spins", type=int, choices=(1, 2), default=2)
    parser.add_argument("--compile-timeout", type=float, default=300.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
