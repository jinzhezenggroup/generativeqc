"""Inspect the production GFN2 CUDA build graph without requiring NVCC."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not all(shutil.which(command) for command in ("cmake", "ninja", "ccache")),
    reason="requires the CI build tools",
)
def test_gfn2_cuda_objects_wait_for_method_parameters(tmp_path: Path) -> None:
    """A dependency on the consuming library cannot order its CUDA prerequisite."""
    subprocess.run(["ccache", "--version"], check=True, capture_output=True, timeout=10)
    for name in ("tools", "python", "src", "include", "manifests", "upstream", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "dummy.cpp").write_text("int fixture;\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(Gfn2CudaDependencies LANGUAGES CXX)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCGeneratedSources.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCGfn2Runtime.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "set(GENERATIVEQC_ENABLE_CUDA ON)\n"
        "set(GENERATIVEQC_CUDA_PROVIDER nvidia)\n"
        # Only unrelated CUDA driver and SDQ setup are stubbed. The runtime
        # target and its method-parameter generator use production declarations.
        "function(generativeqc_attach_cuda_driver_implib target)\n"
        "endfunction()\n"
        "add_custom_target(generativeqc_gfn2_sdq_cuda_codegen)\n"
        "add_library(generativeqc STATIC dummy.cpp)\n"
        "generativeqc_add_gfn2_runtime(generativeqc)\n"
        "generativeqc_register_host_generated_sources(generativeqc)\n"
        # CXX labeling permits Ninja graph generation, not CUDA compilation.
        "get_target_property(cuda_sources generativeqc_gfn2_cuda SOURCES)\n"
        "set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CXX)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    order = subprocess.check_output(
        [
            "ninja",
            "-C",
            str(build),
            "-t",
            "query",
            "cmake_object_order_depends_target_generativeqc_gfn2_cuda",
        ],
        text=True,
        timeout=30,
    )
    assert "generativeqc_method_parameters_codegen" in order
    commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "generativeqc_gfn2_cuda"],
        text=True,
        timeout=30,
    )
    assert "generate_method_parameters.py" in commands
    assert "generated_method_parameters.hpp" in commands
    assert not list(build.glob("generated/*.hpp"))
    assert not list(build.glob("*.a"))
