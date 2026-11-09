"""Build the isolated native archive's generated policy prerequisite without CUDA."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_isolated_direct_archive_generates_schedule(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Exercise the production CMake target with device bodies excluded from this host probe."""
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (tmp_path / "include").symlink_to(ROOT / "include", target_is_directory=True)
    (tmp_path / "probe.cpp").write_text(
        '#include "generated_direct_resident_psss_schedule.cuh"\n'
        "static_assert(generativeqc::scf::cuda_execution::kPersistentFockAngularOrderCount == 6);\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(DirectScheduleDependency LANGUAGES CXX)
include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")
# Stationary providers are outside this isolated Direct-J/K target probe.
function(generativeqc_select_stationary_aot_profiles output)
  set(${{output}} "" PARENT_SCOPE)
endfunction()
set(CMAKE_CUDA_COMPILER_ID NVIDIA)
set(GENERATIVEQC_CUDA_DIRECT_DEVICE_LINK ON)
set(GENERATIVEQC_CUDA_SEPARABLE_COMPILATION OFF)
set(GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS 1)
foreach(provider cudart cublas cusolver)
  add_library(CUDA::${{provider}} INTERFACE IMPORTED)
endforeach()
set(GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER
    "${{CMAKE_BINARY_DIR}}/generated/generated_direct_resident_psss_schedule.cuh")
add_custom_command(OUTPUT "${{GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}}"
  COMMAND "{sys.executable}" "{ROOT / "tools/generate_direct_resident_schedule.py"}"
          --output "${{GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}}")
add_library(probe STATIC probe.cpp)
generativeqc_configure_cuda_backend(probe)
# Keep the production target/dependency graph, but compile only a host header consumer.
get_target_property(device_sources generativeqc_direct_native SOURCES)
set_source_files_properties(${{device_sources}} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(src/scf/cuda/direct_angular_force.cu PROPERTIES HEADER_FILE_ONLY TRUE)
set_target_properties(generativeqc_direct_angular_force PROPERTIES LINKER_LANGUAGE CXX)
target_sources(generativeqc_direct_native PRIVATE probe.cpp)
''')
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={required_native_cxx.cache}",
            f"-DCMAKE_CXX_COMPILER={required_native_cxx.compiler}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(tmp_path / "build"),
            "--target",
            "generativeqc_direct_native",
            "-j",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
