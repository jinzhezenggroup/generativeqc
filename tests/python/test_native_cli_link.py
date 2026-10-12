"""Exercise the final CLI link graph with host stand-ins and real lazy imports.

These probes qualify archive extraction/registration, not CUDA compilation or
scientific execution. No CUDA toolkit or driver is loaded by the executables.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"},
    reason="Native distribution and lazy CUDA imports target Linux ELF",
)


def _run(*args: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=60, check=False, **kwargs
    )


@pytest.fixture
def link_probe(tmp_path: Path, native_cxx: object) -> object:
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("link probe requires CMake >= 3.24 and Ninja")
    # The production driver-import helper owns generation and archive placement.
    # Its repository-relative tools/templates are exposed in this tiny project.
    for directory in ("tools", "cmake"):
        (tmp_path / directory).symlink_to(ROOT / directory, target_is_directory=True)
    (tmp_path / "entry.cpp").write_text(
        "#ifdef WITH_CUDA\n"
        "int registrations = 0;\n"
        "extern int cuda_compute();\n"
        "int core() { return cuda_compute(); }\n"
        "#else\n"
        "int registrations = 1;\n"
        "int core() { return 42; }\n"
        "#endif\n"
    )
    (tmp_path / "schema.cpp").write_text("int host_schema() { return 41; }\n")
    (tmp_path / "cuda.cpp").write_text(
        'extern "C" int cuGetErrorString(int, const char**);\n'
        'extern "C" int cuMemGetAddressRange_v2(void*, void*, unsigned long long);\n'
        "auto driver_error = &cuGetErrorString;\n"
        "auto driver_range = &cuMemGetAddressRange_v2;\n"
        "extern int host_schema();\n"
        "int cuda_compute() { return driver_error && driver_range ? host_schema() + 1 : 0; }\n"
    )
    (tmp_path / "registration.cpp").write_text(
        "extern int registrations;\n"
        "struct Registration { Registration() { ++registrations; } };\n"
        "Registration registration;\n"
    )
    (tmp_path / "main.cpp").write_text(
        "extern int core();\n"
        "extern int registrations;\n"
        "int main() { return core() == 42 && registrations == 1 ? 0 : 1; }\n"
    )

    def build(mode: str) -> tuple[Path, subprocess.CompletedProcess[str]]:
        monolithic = mode != "shared"
        cuda = mode != "cpu"
        # Negative controls remove the repair or the linker's retention behavior,
        # without inspecting or rewriting production implementation spelling.
        cli_link = (
            "target_link_libraries(generativeqc_cli PRIVATE generativeqc)\n"
            if mode == "old-order"
            else "generativeqc_link_cli_core(generativeqc_cli)\n"
        )
        retention = (
            'set(CMAKE_CXX_LINK_LIBRARY_USING_WHOLE_ARCHIVE "<LINK_ITEM>")\n'
            "set(CMAKE_CXX_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)\n"
            if mode == "missing-whole"
            else ""
        )
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(NativeCliLinkProbe LANGUAGES C CXX ASM)\n"
            f'set(Python3_EXECUTABLE "{sys.executable}")\n'
            'include("cmake/GenerativeQCGfn2Runtime.cmake")\n'
            + retention
            + f"set(GENERATIVEQC_MONOLITHIC_CLI {'ON' if monolithic else 'OFF'})\n"
            f"add_library(generativeqc {'STATIC' if monolithic else 'SHARED'} "
            "entry.cpp schema.cpp)\n"
            "set_target_properties(generativeqc PROPERTIES POSITION_INDEPENDENT_CODE ON)\n"
            'target_link_options(generativeqc PRIVATE "LINKER:-z,defs")\n'
            + (
                "add_library(generativeqc_gfn2_cuda STATIC cuda.cpp registration.cpp)\n"
                "set_target_properties(generativeqc_gfn2_cuda PROPERTIES "
                "POSITION_INDEPENDENT_CODE ON)\n"
                "target_compile_definitions(generativeqc PRIVATE WITH_CUDA)\n"
                'target_link_libraries(generativeqc PRIVATE "'
                '$<LINK_LIBRARY:WHOLE_ARCHIVE,generativeqc_gfn2_cuda>")\n'
                + (
                    'include("cmake/GenerativeQCCudaImplib.cmake")\n'
                    "generativeqc_attach_cuda_driver_implib(generativeqc)\n"
                    if mode != "missing-imports"
                    else ""
                )
                if cuda
                else ""
            )
            + "add_executable(generativeqc_cli main.cpp)\n"
            + cli_link
            + 'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/cli-link.txt" '
            'CONTENT "$<TARGET_PROPERTY:generativeqc_cli,LINK_LIBRARIES>")\n'
            'set_target_properties(generativeqc_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")\n'
            "install(TARGETS generativeqc generativeqc_cli "
            "RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)\n"
        )
        build_dir = tmp_path / "build"
        env = {**os.environ, "CCACHE_BASEDIR": str(tmp_path)}
        configured = _run(
            cmake,
            "-S",
            str(tmp_path),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_C_COMPILER_LAUNCHER={native_cxx.cache}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
            f"-DCMAKE_ASM_COMPILER_LAUNCHER={native_cxx.cache}",
            env=env,
        )
        assert configured.returncode == 0, configured.stdout + configured.stderr
        compiled = _run(cmake, "--build", str(build_dir), "--verbose", env=env)
        assert native_cxx.cache in compiled.stdout
        return build_dir, compiled

    return build


@pytest.mark.parametrize("mode", ["monolithic", "shared", "cpu"])
def test_native_cli_link_and_installed_execution(link_probe: object, mode: str) -> None:
    build_dir, result = link_probe(mode)
    assert result.returncode == 0, result.stdout + result.stderr
    libraries = (build_dir / "cli-link.txt").read_text()
    assert ("LINK_GROUP:RESCAN" in libraries) == (mode == "monolithic")
    if mode == "monolithic":
        assert "--whole-archive" in result.stdout
        assert "--start-group" in result.stdout
    assert _run(str(build_dir / "generativeqc_cli")).returncode == 0
    prefix = build_dir / "install"
    installed = _run("cmake", "--install", str(build_dir), "--prefix", str(prefix))
    assert installed.returncode == 0, installed.stdout + installed.stderr
    execution = _run(
        str(prefix / "bin/generativeqc_cli"),
        env={"PATH": "/no-runtime-tools", "LD_LIBRARY_PATH": ""},
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    # Actual lazy trampolines satisfy the references without a driver dependency.
    if shutil.which("readelf"):
        dynamic = _run("readelf", "-d", str(prefix / "bin/generativeqc_cli"))
        assert "libcuda.so" not in dynamic.stdout
        assert ("libgenerativeqc.so" in dynamic.stdout) == (mode == "shared")
        if mode == "shared":
            core_dynamic = _run("readelf", "-d", str(prefix / "lib/libgenerativeqc.so"))
            assert core_dynamic.returncode == 0, core_dynamic.stderr
            assert "libcuda.so" not in core_dynamic.stdout


def test_old_archive_order_reproduces_both_failure_families(link_probe: object) -> None:
    _build_dir, result = link_probe("old-order")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "host_schema" in output
    assert "cuGetErrorString" in output
    assert "cuMemGetAddressRange_v2" in output


def test_rescan_does_not_replace_required_driver_imports(link_probe: object) -> None:
    _build_dir, result = link_probe("missing-imports")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "cuGetErrorString" in output
    assert "cuMemGetAddressRange_v2" in output
    assert "undefined reference to `host_schema" not in output


def test_rescan_does_not_replace_cuda_whole_archive_registration(
    link_probe: object,
) -> None:
    build_dir, result = link_probe("missing-whole")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _run(str(build_dir / "generativeqc_cli")).returncode == 1
