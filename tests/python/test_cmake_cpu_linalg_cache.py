"""Integration tests for the OpenBLAS capability-probe cache boundary."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _make_openblas_provider(path: Path, *, version: str, local_threads: bool) -> None:
    """Provide linkable header-only OpenBLAS stubs without system dependencies."""
    includes = path / "include"
    includes.mkdir(parents=True, exist_ok=True)
    (path / "OpenBLASConfig.cmake").write_text(
        "set(OpenBLAS_FOUND TRUE)\n"
        f'set(OpenBLAS_VERSION "{version}")\n'
        "if(NOT TARGET OpenBLAS::OpenBLAS)\n"
        "  add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)\n"
        "endif()\n"
        'set(OpenBLAS_LIBRARIES OpenBLAS::OpenBLAS)\n'
        'set(OpenBLAS_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")\n'
    )
    (includes / "cblas.h").write_text(
        "#pragma once\n"
        + (
            "#ifndef OPENBLAS_DISABLE_LOCAL_THREADS\n"
            "inline int openblas_set_num_threads_local(int) { return 0; }\n"
            "#endif\n"
            if local_threads
            else ""
        )
        + "inline int openblas_get_num_threads() { return 1; }\n"
        + "inline void openblas_set_num_threads(int) {}\n"
    )
    (includes / "lapacke.h").write_text(
        "#pragma once\n"
        "#define LAPACK_ROW_MAJOR 101\n"
        "inline int LAPACKE_dpotrf(int, char, int, double*, int) { return 0; }\n"
        "inline int LAPACKE_dsyevd(int, char, char, int, double*, "
        "int, double*) { return 0; }\n"
    )


def _capability(cache: Path, name: str) -> bool:
    result = re.search(rf"^{name}:INTERNAL=(.*)$", cache.read_text(), re.MULTILINE)
    assert result is not None, name
    return result.group(1) == "1"


def test_openblas_capabilities_reuse_and_invalidate(tmp_path: Path) -> None:
    """No-op configures reuse both successes and failures; input changes re-probe."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "probe.cpp").write_text("int probe() { return 0; }\n")
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(OpenBLASCacheProbe LANGUAGES CXX)\n"
        # Keep this test independent of the host's pkg-config installation.
        "set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig TRUE)\n"
        f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        "add_library(probe STATIC probe.cpp)\n"
        "generativeqc_configure_cpu_linalg(probe)\n"
    )
    first_provider = tmp_path / "openblas-1"
    second_provider = tmp_path / "openblas-2"
    _make_openblas_provider(first_provider, version="1.0", local_threads=True)
    _make_openblas_provider(second_provider, version="1.0", local_threads=False)
    build = tmp_path / "build"
    cache = build / "CMakeCache.txt"

    def configure(provider: Path, *, cxx_flags: str = "") -> str:
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(src),
                "-B",
                str(build),
                "-DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas",
                f"-DOpenBLAS_DIR={provider}",
                f"-DCMAKE_CXX_FLAGS={cxx_flags}",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout + result.stderr

    def probe_count(output: str) -> int:
        return len(
            re.findall(
                r"^-- Performing Test GENERATIVEQC_OPENBLAS_HAS_[A-Z_]+$",
                output,
                re.MULTILINE,
            )
        )

    assert probe_count(configure(first_provider)) == 3
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LAPACKE")
    assert probe_count(configure(first_provider)) == 0

    # The imported target name is identical; only its include path changes.
    assert probe_count(configure(second_provider)) == 3
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert probe_count(configure(second_provider)) == 0

    # An in-place header replacement must not reuse a stale negative result.
    _make_openblas_provider(second_provider, version="1.0", local_threads=True)
    assert probe_count(configure(second_provider)) == 3
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")

    _make_openblas_provider(second_provider, version="2.0", local_threads=True)
    assert probe_count(configure(second_provider)) == 3
    assert probe_count(configure(second_provider)) == 0

    # Compiler options also belong to the probe identity.
    no_local_threads = "-DOPENBLAS_DISABLE_LOCAL_THREADS"
    assert probe_count(configure(second_provider, cxx_flags=no_local_threads)) == 3
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert probe_count(configure(second_provider, cxx_flags=no_local_threads)) == 0
    assert probe_count(configure(second_provider)) == 3
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
