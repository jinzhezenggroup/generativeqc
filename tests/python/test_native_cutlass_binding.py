"""Optional AOT provider qualification with real build and semantic provenance."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
    toolchain_identity,
)
from generativeqc_compiler.tensor.cuda_cutlass import CUTLASS_FAMILY


def test_native_cutlass_execution(tmp_path: Path) -> None:
    """Compile the fixed SIMT family and compare every layout to affine sums."""
    if os.environ.get("GENERATIVEQC_CUTLASS_CUDA_TEST") != "1":
        pytest.skip("requires optional CUTLASS and finite Slurm real-GPU allocation")
    assert os.environ.get("SLURM_JOB_ID")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    sdk = Path(os.environ["GENERATIVEQC_CUTLASS_ROOT"]).resolve() / "include"
    assert (sdk / "cutlass/gemm/device/gemm_batched.h").is_file()
    # Actual external header content is part of this artifact. Neither an
    # installation path nor a version string alone proves the compiled inputs.
    identity = {
        "toolchain": toolchain_identity(Path(compiler)),
        "sources": source_hashes(
            "common",
            "tensor",
            assets=(
                "tests/native/test_native_cutlass.cu",
                "cmake/GenerativeQCCutlass.cmake",
                *sorted(
                    path.relative_to(root).as_posix()
                    for directory in (root / "src/tensor", root / "src/runtime")
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix in (".hpp", ".cuh")
                ),
            ),
        ),
        "cutlass_headers": {
            path.relative_to(sdk).as_posix(): file_hash(path)
            for path in sorted(sdk.rglob("*"))
            if path.is_file()
        },
    }
    # Exercise the same optional SDK/macro propagation as the library. A tiny
    # consumer keeps qualification independent of unrelated method builds.
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CutlassQualification LANGUAGES CXX CUDA)\n"
        "set(CMAKE_CXX_STANDARD 20)\n"
        "set(CMAKE_CUDA_STANDARD 20)\n"
        "set(CMAKE_CUDA_STANDARD_REQUIRED ON)\n"
        "set(GENERATIVEQC_ENABLE_CUDA ON)\n"
        "set(GENERATIVEQC_CUDA_PROVIDER nvidia)\n"
        "set(GENERATIVEQC_ENABLE_CUTLASS ON)\n"
        f'include("{root / "cmake/GenerativeQCCutlass.cmake"}")\n'
        f'add_executable(cutlass "{root / "tests/native/test_native_cutlass.cu"}")\n'
        f'target_include_directories(cutlass PRIVATE "{root / "src"}")\n'
        "target_compile_definitions(cutlass PRIVATE GENERATIVEQC_TEST_HOOKS=1)\n"
        "generativeqc_configure_cutlass(cutlass)\n"
    )
    build = tmp_path / "build"
    executable = build / "cutlass"
    for command in (
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_CUDA_COMPILER={compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={cache}",
            f"-DCMAKE_CUDA_COMPILER_LAUNCHER={cache}",
            "-DCMAKE_CUDA_ARCHITECTURES=120-real",
            "-DCMAKE_CUDA_FLAGS=-O2",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DGENERATIVEQC_CUTLASS_ROOT={sdk.parent}",
        ],
        ["cmake", "--build", str(build), "--verbose", "-j", "2"],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=360,
            check=False,
            env={**os.environ, "CCACHE_BASEDIR": str(root)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout, end="")
    assert cache in result.stdout, "the generated compilation must use ccache"
    commands = json.loads((build / "compile_commands.json").read_text())
    assert len(commands) == 1
    assert "GENERATIVEQC_HAS_CUTLASS=1" in commands[0]["command"]
    # Hash effective generated commands and the linked bytes, rather than a
    # hand-maintained approximation of the CMake compiler flags. Paths are
    # provenance labels; SDK/source contents remain explicitly hashed above.
    encoded_commands = json.dumps(commands, sort_keys=True)
    for path, label in ((sdk, "<cutlass>"), (tmp_path, "<build>"), (root, "<source>")):
        encoded_commands = encoded_commands.replace(str(path), label)
    identity["compile_commands"] = json.loads(encoded_commands)
    identity["executable_sha256"] = file_hash(executable)
    artifact = canonical_hash(identity)
    atomic_json(tmp_path / "artifact.json", {"identity": identity, "key": artifact})
    result = subprocess.run(
        [str(executable), artifact],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")
    assert CUTLASS_FAMILY in result.stdout
    atomic_json(
        tmp_path / "qualified.json",
        {"artifact": artifact, "executable_sha256": file_hash(executable)},
    )
