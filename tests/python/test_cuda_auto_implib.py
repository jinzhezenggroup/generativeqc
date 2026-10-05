"""Provider-free CUDA wheel imports are discovered from strict link failures."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.link_cuda_implib import (
    link_with_auto_implib,
    provider_for_symbol,
    unresolved_provider_symbols,
)

ROOT = Path(__file__).resolve().parents[2]


def _implib_target() -> str:
    machine = platform.machine().lower()
    targets = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    if platform.system() != "Linux" or machine not in targets:
        pytest.skip("provider-free CUDA trampolines require supported Linux ELF")
    return targets[machine]


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )


def test_linker_diagnostics_classify_only_supported_provider_families() -> None:
    diagnostic = """
/usr/bin/ld: a.o: undefined reference to `cudaDeviceSynchronize'
ld.lld: error: undefined symbol: cublasDgemm_v2
/usr/bin/ld: b.o: undefined reference to `cusolverDnCreate'
/usr/bin/ld: c.o: undefined reference to `cublasLtMatmul'
/usr/bin/ld: d.o: undefined reference to `ordinary_missing_symbol'
"""
    assert unresolved_provider_symbols(diagnostic) == {
        "cudart": {"cudaDeviceSynchronize"},
        "cublas": {"cublasDgemm_v2"},
        "cusolver": {"cusolverDnCreate"},
    }
    cudart = provider_for_symbol("__cudaRegisterFunction")
    assert cudart is not None and cudart.name == "cudart"
    assert provider_for_symbol("cublasLtMatmul") is None
    assert provider_for_symbol("ordinary_missing_symbol") is None


def test_linker_driven_implib_retries_without_provider_dependency(tmp_path: Path) -> None:
    target = _implib_target()
    cc, cxx, readelf = (shutil.which(name) for name in ("cc", "c++", "readelf"))
    if cc is None or cxx is None or readelf is None:
        pytest.skip("C/C++ compiler and ELF inspector required")

    consumer = tmp_path / "consumer.cpp"
    consumer.write_text(
        'extern "C" int cudaDeviceSynchronize();\n'
        'extern "C" int probe() { return cudaDeviceSynchronize(); }\n'
    )
    consumer_object = tmp_path / "consumer.o"
    library = tmp_path / "consumer.so"
    _run(cxx, "-fPIC", "-c", str(consumer), "-o", str(consumer_object))

    status = link_with_auto_implib(
        command=[
            cxx,
            "-shared",
            str(consumer_object),
            "-Wl,-z,defs",
            "-ldl",
            "-o",
            str(library),
        ],
        cc=cc,
        implib_root=ROOT / "cmake/3rdparty/implib",
        work_dir=tmp_path / "implib",
        target=target,
    )
    assert status == 0
    dynamic = _run(readelf, "-d", str(library)).stdout
    assert "Shared library: [libcudart" not in dynamic

    provider_source = tmp_path / "provider.cpp"
    provider_source.write_text(
        'extern "C" int cudaDeviceSynchronize() { return 73; }\n'
    )
    available = tmp_path / "available-provider.so"
    provider = tmp_path / "libcudart.so.12"
    _run(cxx, "-shared", "-fPIC", str(provider_source), "-o", str(available))

    script = """
import ctypes
import pathlib
import sys
library, available, provider = map(pathlib.Path, sys.argv[1:])
assert not provider.exists()
loaded = ctypes.CDLL(str(library))
loaded.probe.argtypes = []
loaded.probe.restype = ctypes.c_int
available.rename(provider)
assert loaded.probe() == 73
assert loaded.probe() == 73
"""
    environment = {**os.environ, "LD_LIBRARY_PATH": str(tmp_path)}
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(library),
            str(available),
            str(provider),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cmake_launcher_discovers_runtime_import(tmp_path: Path) -> None:
    _implib_target()
    cmake = shutil.which("cmake")
    readelf = shutil.which("readelf")
    if cmake is None or readelf is None:
        pytest.skip("CMake and ELF inspector required")

    (tmp_path / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    (tmp_path / "cmake").symlink_to(ROOT / "cmake", target_is_directory=True)
    (tmp_path / "probe.cpp").write_text(
        'extern "C" int cudaDeviceSynchronize();\n'
        'extern "C" int probe() { return cudaDeviceSynchronize(); }\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(AutoImplibProbe LANGUAGES C CXX ASM)\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'include("{ROOT / "cmake/GenerativeQCCudaImplib.cmake"}")\n'
        "add_library(probe SHARED probe.cpp)\n"
        "generativeqc_attach_cuda_implib(probe)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [cmake, "--build", str(build), "-j2"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    dynamic = _run(readelf, "-d", str(build / "libprobe.so")).stdout
    assert "Shared library: [libcudart" not in dynamic


def test_unrelated_undefined_symbol_remains_fatal(tmp_path: Path) -> None:
    target = _implib_target()
    cc, cxx = (shutil.which(name) for name in ("cc", "c++"))
    if cc is None or cxx is None:
        pytest.skip("C/C++ compiler required")

    source = tmp_path / "bad.cpp"
    source.write_text(
        'extern "C" int ordinary_missing_symbol();\n'
        'extern "C" int probe() { return ordinary_missing_symbol(); }\n'
    )
    object_path = tmp_path / "bad.o"
    library = tmp_path / "bad.so"
    _run(cxx, "-fPIC", "-c", str(source), "-o", str(object_path))
    status = link_with_auto_implib(
        command=[
            cxx,
            "-shared",
            str(object_path),
            "-Wl,-z,defs",
            "-ldl",
            "-o",
            str(library),
        ],
        cc=cc,
        implib_root=ROOT / "cmake/3rdparty/implib",
        work_dir=tmp_path / "implib",
        target=target,
    )
    assert status != 0
