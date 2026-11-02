"""Build concurrency policy keeps generated AOT work independently tunable."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_compile_pools_are_opt_in_and_aot_can_split() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda = (ROOT / "cmake" / "GenerativeQCCuda.cmake").read_text(encoding="utf-8")

    assert 'set(GENERATIVEQC_CUDA_COMPILE_JOBS "" CACHE STRING' in cmake
    assert 'set(GENERATIVEQC_AOT_COMPILE_JOBS "" CACHE STRING' in cmake
    assert 'set(_generativeqc_cuda_compile_pool "")' in cmake
    assert 'set(_generativeqc_aot_compile_pool "")' in cmake
    assert "generativeqc_cuda_compile=${GENERATIVEQC_CUDA_COMPILE_JOBS}" in cmake
    assert "generativeqc_aot_compile=${GENERATIVEQC_AOT_COMPILE_JOBS}" in cmake
    assert "set(_generativeqc_aot_compile_pool generativeqc_cuda_compile)" in cmake
    assert "set(_generativeqc_aot_compile_pool generativeqc_aot_compile)" in cmake
    assert "function(generativeqc_set_cuda_compile_pool target pool)" in cuda

    native = cuda.split("if(GENERATIVEQC_ENABLE_AOT_SHELLS)", 1)[0]
    aot = cuda.split("if(GENERATIVEQC_ENABLE_AOT_SHELLS)", 1)[1]
    assert '"${_generativeqc_cuda_compile_pool}"' in native
    assert '"${_generativeqc_aot_compile_pool}"' in aot
    assert "JOB_POOL_COMPILE generativeqc_cuda_compile" not in cuda


@pytest.mark.parametrize("backend", ["NVIDIA", "Clang"])
@pytest.mark.parametrize("pool", ["", "native_compile", "aot_compile"])
def test_shared_cuda_compile_owner_obeys_selected_pool(
    tmp_path: Path, native_cxx: Any, backend: str, pool: str
) -> None:
    """Ninja schedules source compilation on the returned owner, including fallback.

    Host sources make the ownership/pool contract executable without requiring a
    CUDA toolkit. Device closure and registration have separate compiled tests.
    """
    (tmp_path / "owner.cpp").write_text("int exported_value() { return 42; }\n")
    (tmp_path / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(CompilePoolContract LANGUAGES CXX)
include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CUDA_COMPILER_ID {backend})
set(GENERATIVEQC_PRUNE_CUDA_ARCHIVES OFF)
set_property(GLOBAL PROPERTY JOB_POOLS native_compile=3 aot_compile=1)
generativeqc_add_cuda_shared_library(plugin compile_owner owner.cpp)
generativeqc_set_cuda_compile_pool(${{compile_owner}} "{pool}")
''')
    directory = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(directory),
            "-G",
            "Ninja",
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    ninja = (directory / "build.ninja").read_text()
    edges = [
        block
        for block in ninja.split("\n\n")
        if block.startswith("build ") and "/owner.cpp.o:" in block
    ]
    assert len(edges) == 1
    scheduled = [line.strip() for line in edges[0].splitlines() if "pool =" in line]
    assert scheduled == ([f"pool = {pool}"] if pool else [])


def test_cuda_presets_follow_ninja_parallelism() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    for name in ("cuda-dev-fast", "cuda-release"):
        preset = next(
            item for item in presets["configurePresets"] if item["name"] == name
        )
        variables = preset["cacheVariables"]
        assert "GENERATIVEQC_CUDA_COMPILE_JOBS" not in variables
        assert "GENERATIVEQC_AOT_COMPILE_JOBS" not in variables


def test_release_presets_are_architecture_neutral_and_auto_tuned() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    configured = {preset["name"]: preset for preset in presets["configurePresets"]}
    expected = {"cuda-dev-fast", "cuda-release", "cuda-release-cli"}
    assert set(configured) == expected
    assert {preset["name"] for preset in presets["buildPresets"]} == expected
    assert (
        configured["cuda-release"]["cacheVariables"]["GENERATIVEQC_AOT_PROFILE"]
        == "auto"
    )
    assert configured["cuda-release-cli"]["inherits"] == "cuda-release"
    for preset in configured.values():
        options = preset["cacheVariables"]
        assert "CMAKE_CUDA_ARCHITECTURES" not in options
        assert "GENERATIVEQC_CUDA_ARCHITECTURES" not in options
        assert "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES" not in options


def test_host_pch_is_opt_in_and_cxx_only() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    fast = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-dev-fast"
    )
    assert "GENERATIVEQC_ENABLE_CXX_PCH" not in fast["cacheVariables"]

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "option(GENERATIVEQC_ENABLE_CXX_PCH" in cmake
    assert "target_precompile_headers(generativeqc PRIVATE" in cmake
    assert "$<$<COMPILE_LANGUAGE:CXX>:" in cmake

    pch = (ROOT / "src" / "pch.hpp").read_text(encoding="utf-8")
    assert '#include "' not in pch


def test_fast_cuda_preset_uses_bounded_aot_split_compile() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    fast = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-dev-fast"
    )
    assert fast["cacheVariables"]["GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS"] == "2"

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'set(GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS "1" CACHE STRING' in cmake

    cuda = (ROOT / "cmake" / "GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    aot = cuda.split("if(GENERATIVEQC_ENABLE_AOT_SHELLS)", 1)[1]
    assert "--split-compile=${GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS}" in aot
    assert "$<CUDA_COMPILER_ID:NVIDIA>" in aot


def test_release_preset_keeps_aot_split_compile_disabled() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    release = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-release"
    )
    assert "GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS" not in release["cacheVariables"]


def test_fock_benchmark_records_aot_split_compile_identity() -> None:
    benchmark = (ROOT / "tools" / "benchmark_fock_strategies.py").read_text(
        encoding="utf-8"
    )
    assert '"GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS"' in benchmark


def test_dft_test_support_reuses_object_targets() -> None:
    tests = (ROOT / "cmake" / "GenerativeQCTests.cmake").read_text(encoding="utf-8")

    assert "add_library(generativeqc_dft_grid_test_objects OBJECT" in tests
    assert "add_library(generativeqc_dft_xc_test_objects OBJECT" in tests
    assert tests.count("$<TARGET_OBJECTS:generativeqc_dft_grid_test_objects>") == 3
    assert tests.count("$<TARGET_OBJECTS:generativeqc_dft_xc_test_objects>") == 2

    # Three grid/basis sources formerly compiled in three executables and two
    # XC/density sources in two executables. The object targets reduce those
    # 13 compile actions to five without changing test-only compile definitions.
    for source in (
        "src/dft/ao_grid.cpp",
        "src/dft/grid.cpp",
        "src/molecule/basis.cpp",
    ):
        assert tests.count(source) == 1
    for source in ("src/scf/density_factor.cpp", "src/dft/xc.cpp"):
        assert tests.count(source) == 1
