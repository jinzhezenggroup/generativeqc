"""Mandatory NVIDIA fixture compile/link wiring, inspectable without nvcc."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TARGET = "generativeqc_ordered_history_cuda_compile_check"


@pytest.mark.parametrize("compile_pool", (None, "", "generativeqc_cuda_compile"))
@pytest.mark.parametrize(
    "enabled,provider,archive,expected",
    (
        (True, "nvidia", True, True),
        (True, "cumetal", True, False),
        (True, "nvidia", False, False),
        (False, "nvidia", True, False),
    ),
)
def test_optional_device_fixture_is_a_mandatory_nvidia_build_dependency(
    tmp_path: Path,
    enabled: bool,
    provider: str,
    archive: bool,
    expected: bool,
    compile_pool: str | None,
) -> None:
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    cache = shutil.which(os.environ.get("CCACHE", "ccache"))
    if cmake is None or ninja is None or cache is None:
        pytest.skip("CMake, Ninja and a verified compiler cache are required")
    subprocess.run([cache, "--version"], check=True, capture_output=True, timeout=10)
    for directory in ("src", "include", "tests"):
        (tmp_path / directory).symlink_to(ROOT / directory, target_is_directory=True)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(HistoryCudaBuildGraph LANGUAGES CXX)\n"
        "enable_testing()\n"
        f'include("{ROOT / "cmake/GenerativeQCTests.cmake"}")\n'
        f"set(GENERATIVEQC_ENABLE_CUDA {'ON' if enabled else 'OFF'})\n"
        f'set(GENERATIVEQC_CUDA_PROVIDER "{provider}")\n'
        'set(CMAKE_CUDA_ARCHITECTURES "70;80")\n'
        f'set(CMAKE_CUDA_COMPILER_LAUNCHER "{cache}")\n'
        + (
            f'set(_generativeqc_cuda_compile_pool "{compile_pool}")\n'
            if compile_pool is not None
            else ""
        )
        + (
            f"set_property(GLOBAL PROPERTY JOB_POOLS {compile_pool}=2)\n"
            if compile_pool
            else ""
        )
        # Only external targets are stubbed. The target under review, its
        # sources, includes, flags, guards and dependency edges are production.
        + "add_library(CUDA::cudart INTERFACE IMPORTED)\n"
        "add_custom_target(generativeqc_ordered_history_codegen)\n"
        "add_custom_target(generativeqc_cuda_runtime_tests)\n"
        + ("add_library(generativeqc_gfn2_cuda INTERFACE)\n" if archive else "")
        + "generativeqc_add_ordered_history_cuda_compile_check()\n"
        f"if(TARGET {TARGET})\n"
        f"  get_target_property(fixture_sources {TARGET} SOURCES)\n"
        # Relabeling allows graph generation; this test never compiles CUDA
        # with a host compiler or claims device execution.
        "  set_source_files_properties(${fixture_sources} PROPERTIES LANGUAGE CXX)\n"
        '  file(WRITE "${CMAKE_BINARY_DIR}/fixture-properties.txt" "")\n'
        "  foreach(property TYPE SOURCES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS\n"
        "                   LINK_LIBRARIES LINK_OPTIONS CUDA_STANDARD CUDA_STANDARD_REQUIRED\n"
        "                   CUDA_ARCHITECTURES CUDA_COMPILER_LAUNCHER CUDA_SEPARABLE_COMPILATION\n"
        "                   CUDA_RESOLVE_DEVICE_SYMBOLS POSITION_INDEPENDENT_CODE JOB_POOL_COMPILE)\n"
        f"    get_target_property(value {TARGET} ${{property}})\n"
        '    file(APPEND "${CMAKE_BINARY_DIR}/fixture-properties.txt" "${property}=${value}\\n")\n'
        "  endforeach()\n"
        "endif()\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={cache}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    query = subprocess.check_output(
        [ninja, "-C", str(build), "-t", "query", "generativeqc_cuda_runtime_tests"],
        text=True,
        timeout=30,
    )
    properties = build / "fixture-properties.txt"
    if not expected:
        assert TARGET not in query
        assert not properties.exists()
        return
    assert TARGET in query
    values = dict(line.split("=", 1) for line in properties.read_text().splitlines())
    assert values["TYPE"] == "SHARED_LIBRARY"
    assert set(values["SOURCES"].split(";")) == {
        "tests/native/test_ordered_history_consumers.cu",
        "src/xtb/native/src/backends/cuda/gfn2_scc_mixer.cu",
    }
    includes = set(values["INCLUDE_DIRECTORIES"].split(";"))
    assert {
        str(tmp_path / "src/xtb/native/src"),
        str(tmp_path / "src"),
        str(build / "generated"),
    } <= includes
    assert values["COMPILE_DEFINITIONS"] == "GENERATIVEQC_XTB_HAS_CUDA=1"
    assert values["COMPILE_OPTIONS"].endswith("-NOTFOUND")
    assert values["LINK_LIBRARIES"] == "CUDA::cudart"
    if os.name == "posix" and os.uname().sysname == "Linux":
        assert values["LINK_OPTIONS"] == "LINKER:--no-undefined"
    assert values["CUDA_STANDARD"] == "20"
    assert values["CUDA_ARCHITECTURES"] == "70;80"
    assert values["CUDA_COMPILER_LAUNCHER"] == cache
    if compile_pool:
        assert values["JOB_POOL_COMPILE"] == compile_pool
    else:
        assert values["JOB_POOL_COMPILE"].endswith("-NOTFOUND")
    for property_name in (
        "CUDA_STANDARD_REQUIRED",
        "CUDA_SEPARABLE_COMPILATION",
        "CUDA_RESOLVE_DEVICE_SYMBOLS",
        "POSITION_INDEPENDENT_CODE",
    ):
        assert values[property_name] == "ON"
    order = subprocess.check_output(
        [
            ninja,
            "-C",
            str(build),
            "-t",
            "query",
            f"cmake_object_order_depends_target_{TARGET}",
        ],
        text=True,
        timeout=30,
    )
    assert "generativeqc_ordered_history_codegen" in order
    commands = subprocess.check_output(
        [ninja, "-C", str(build), "-t", "commands", "generativeqc_cuda_runtime_tests"],
        text=True,
        timeout=30,
    )
    assert "test_ordered_history_consumers.cu" in commands
    assert "gfn2_scc_mixer.cu" in commands
    assert "-shared" in commands
    assert "fmad" not in commands and "ffp-contract" not in commands
    ctest = subprocess.check_output(
        [cmake, "-E", "env", "ctest", "--test-dir", str(build), "--show-only=json-v1"],
        text=True,
        timeout=30,
    )
    assert json.loads(ctest)["tests"] == []


def test_normal_native_cuda_suite_invokes_the_compile_check() -> None:
    source = (ROOT / "cmake/GenerativeQCTests.cmake").read_text()
    registration = source.index(
        "generativeqc_native_test(generativeqc_cuda_runtime_tests"
    )
    invocation = source.index(
        "generativeqc_add_ordered_history_cuda_compile_check()", registration
    )
    assert (
        registration
        < invocation
        < source.index(
            "generativeqc_native_test(generativeqc_df_eigensystem_tests", registration
        )
    )
