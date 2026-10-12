"""Execute the CUDA build graph and link its selected stationary host boundary."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("python_aot", (False, True))
def test_independent_stationary_options_keep_linkable_binding(
    native: bool, python_aot: bool, tmp_path: Path, native_cxx: NativeCxx
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to execute the CUDA target graph")
    # Record target declarations without requiring a CUDA compiler. The real
    # policy and complete backend macro run unchanged; registry generation uses
    # the actual compiler entry point. Numerical CUDA wrappers are not compiled.
    script = tmp_path / "graph.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(CMAKE_CURRENT_BINARY_DIR "{tmp_path.as_posix()}")\n'
        f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        "set(GENERATIVEQC_ENABLE_AOT_SHELLS OFF)\n"
        "set(GENERATIVEQC_CUDA_SPLIT_COMPILE_THREADS 1)\n"
        "set(GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS 1)\n"
        "set(GENERATIVEQC_CUDA_PROVIDER nvidia)\n"
        "set(CMAKE_CUDA_COMPILER_ID NVIDIA)\n"
        "set(CMAKE_CUDA_ARCHITECTURES 120)\n"
        "set(_generativeqc_cuda_compile_architectures 120)\n"
        'set(GENERATIVEQC_STATIONARY_AOT_PROFILES pbe_rks CACHE STRING "")\n'
        f"set(GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA {'ON' if native else 'OFF'})\n"
        f"set(GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT {'ON' if python_aot else 'OFF'})\n"
        "function(target_sources target visibility)\n"
        "  set_property(GLOBAL APPEND PROPERTY ${target}_sources ${ARGN})\n"
        "endfunction()\n"
        "function(add_library target kind)\n"
        "  set_property(GLOBAL APPEND PROPERTY ${target}_sources ${ARGN})\n"
        "  set_property(GLOBAL APPEND PROPERTY libraries ${target}:${kind})\n"
        "endfunction()\n"
        "function(generativeqc_register_generated_sources)\n"
        '  cmake_parse_arguments(G "" "GENERATOR" "ARGS;OUTPUTS;NAME;DEPENDS;COMMENT" ${ARGN})\n'
        '  if("--registry" IN_LIST G_ARGS)\n'
        "    execute_process(COMMAND ${Python3_EXECUTABLE} ${G_GENERATOR} ${G_ARGS}\n"
        "      COMMAND_ERROR_IS_FATAL ANY)\n"
        "  endif()\n"
        "endfunction()\n"
        + "".join(
            f"function({name})\nendfunction()\n"
            for name in (
                "add_dependencies",
                "target_include_directories",
                "target_compile_definitions",
                "target_compile_options",
                "set_target_properties",
                "target_link_libraries",
                "add_custom_target",
                "install",
            )
        )
        + f'include("{ROOT.as_posix()}/cmake/GenerativeQCStationaryProfiles.cmake")\n'
        + f'include("{ROOT.as_posix()}/cmake/GenerativeQCCuda.cmake")\n'
        + "generativeqc_configure_cuda_backend(core)\n"
        + "get_property(sources GLOBAL PROPERTY core_sources)\n"
        + "get_property(libraries GLOBAL PROPERTY libraries)\n"
        + "get_property(native_sources GLOBAL PROPERTY generativeqc_native_stationary_sources)\n"
        + f'file(WRITE "{tmp_path.as_posix()}/native_sources.txt" "${{native_sources}}")\n'
        + f'file(WRITE "{tmp_path.as_posix()}/sources.txt" "${{sources}}")\n'
        + f'file(WRITE "{tmp_path.as_posix()}/libraries.txt" "${{libraries}}")\n'
    )
    subprocess.run(
        [cmake, "-P", str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    libraries = (tmp_path / "libraries.txt").read_text().split(";")
    assert any(entry.endswith(":SHARED") for entry in libraries) == python_aot
    native_sources = (tmp_path / "native_sources.txt").read_text().split(";")
    assert native_sources.count("src/dft/native_stationary_shared_kernels.cu") == int(
        native
    )
    # Link the actual CMake-selected implementations of this host ABI. Other
    # CUDA-core sources and device objects belong to hosted CUDA integration.
    selected = []
    for entry in (tmp_path / "sources.txt").read_text().split(";"):
        source = Path(entry)
        if not source.is_absolute():
            source = ROOT / source
        if (
            source.suffix == ".cpp"
            and '"dft/cuda_stationary_binding.hpp"' in source.read_text()
        ):
            selected.append(source)
    probe = tmp_path / "probe.cpp"
    probe.write_text(
        '#include "dft/cuda_stationary_binding.hpp"\n'
        "namespace generativeqc::dft {\n"
        "const CudaStationaryBinding& native_stationary_profile_pbe_rks() {\n"
        '  static const CudaStationaryBinding binding{.profile="pbe_rks"};\n'
        "  return binding;\n}\n}\n"
        "int main() {\n"
        "  using namespace generativeqc::dft;\n"
        f"  if(cuda_stationary_bindings().size() != {int(native)}) return 1;\n"
        f'  if(bool(cuda_stationary_binding("pbe_rks")) != {int(native)}) return 2;\n'
        f"  if(bool(cuda_stationary_binding(CudaStationaryModel{{}})) != {int(native)}) return 3;\n"
        '  if(cuda_stationary_binding("unavailable")) return 4;\n'
        "}\n"
    )
    executable = native_cxx.build_executable(
        [probe, *selected],
        tmp_path / "binding-probe",
        compile_args=("-std=c++20", "-I", str(ROOT / "src")),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
