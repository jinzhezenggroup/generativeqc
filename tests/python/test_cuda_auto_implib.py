"""Provider-free CUDA wheel imports are discovered from strict link failures."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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


def test_truncated_diagnostics_discover_more_than_three_batches() -> None:
    required = {f"cudaProbe{index:03d}" for index in range(61)}
    compiled: set[str] = set()

    def compile_trampolines(*, symbols: dict[str, set[str]], **_: object) -> list[str]:
        compiled.update(symbols["cudart"])
        return ["generated-trampolines.o"]

    def link(command: list[str]) -> subprocess.CompletedProcess[str]:
        missing = sorted(required - compiled)
        diagnostics = "".join(
            f"ld.lld: error: undefined symbol: {symbol}\n" for symbol in missing[:20]
        )
        if len(missing) > 20:
            diagnostics += "ld.lld: error: too many errors emitted, stopping now\n"
        return subprocess.CompletedProcess(command, int(bool(missing)), "", diagnostics)

    with (
        patch("tools.link_cuda_implib._run", side_effect=link) as run,
        patch(
            "tools.link_cuda_implib._compile_trampolines",
            side_effect=compile_trampolines,
        ),
    ):
        status = link_with_auto_implib(
            command=["c++"],
            cc="cc",
            implib_root=ROOT / "cmake/3rdparty/implib",
            work_dir=Path("unused"),
            target="x86_64",
        )

    assert status == 0
    assert compiled == required
    assert run.call_count == 5


def test_symbol_discovery_stops_when_link_failure_makes_no_progress() -> None:
    failure = subprocess.CompletedProcess(
        ["c++"], 1, "", "ld.lld: error: undefined symbol: cudaDeviceSynchronize\n"
    )
    with (
        patch("tools.link_cuda_implib._run", return_value=failure) as run,
        patch(
            "tools.link_cuda_implib._compile_trampolines",
            return_value=["generated-trampolines.o"],
        ) as compile_trampolines,
    ):
        status = link_with_auto_implib(
            command=["c++"],
            cc="cc",
            implib_root=ROOT / "cmake/3rdparty/implib",
            work_dir=Path("unused"),
            target="x86_64",
        )

    assert status == 1
    assert run.call_count == 2
    assert compile_trampolines.call_count == 1


@pytest.mark.parametrize(
    ("signature", "arguments", "provider_body"),
    [
        ("cudaDeviceSynchronize()", "", "return 73;"),
        (
            "cudaStreamWaitEvent(void* stream, void* event, unsigned int flags)",
            (
                "reinterpret_cast<void*>(0x123456789abcULL), "
                "reinterpret_cast<void*>(0x23456789abcdULL), 0xa5a51234U"
            ),
            (
                "return stream == reinterpret_cast<void*>(0x123456789abcULL) && "
                "event == reinterpret_cast<void*>(0x23456789abcdULL) && "
                "flags == 0xa5a51234U ? 73 : -1;"
            ),
        ),
    ],
    ids=("device-synchronize", "stream-wait-event"),
)
def test_linker_driven_implib_retries_without_provider_dependency(
    tmp_path: Path, signature: str, arguments: str, provider_body: str
) -> None:
    target = _implib_target()
    cc, cxx, readelf = (shutil.which(name) for name in ("cc", "c++", "readelf"))
    if cc is None or cxx is None or readelf is None:
        pytest.skip("C/C++ compiler and ELF inspector required")

    consumer = tmp_path / "consumer.cpp"
    symbol = signature.split("(", 1)[0]
    # Non-dereferenced pointer and flags sentinels verify every host ABI argument
    # of cudaStreamWaitEvent; this is a mock provider, not a real CUDA operation.
    consumer.write_text(
        f'extern "C" int {signature};\n'
        f'extern "C" int probe() {{ return {symbol}({arguments}); }}\n'
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
    provider_source.write_text(f'extern "C" int {signature} {{ {provider_body} }}\n')
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


def test_cuda_only_target_wraps_host_link_and_keeps_device_link(tmp_path: Path) -> None:
    """Inspect generated commands with an NVIDIA toolchain model, without CUDA."""
    _implib_target()
    cmake, ninja, cxx = (shutil.which(name) for name in ("cmake", "ninja", "c++"))
    if cmake is None or ninja is None or cxx is None:
        pytest.skip("CMake, Ninja, and a host C++ compiler required")

    (tmp_path / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    (tmp_path / "cmake").symlink_to(ROOT / "cmake", target_is_directory=True)
    (tmp_path / "probe.cu").write_text("int probe() { return 0; }\n")
    # Supply compiler-identification results so CMake can generate the genuine
    # NVIDIA CUDA link rules without an installed toolkit. Never execute the
    # generated CUDA compilation/device-link commands in this host-only test.
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CudaLinkProbe LANGUAGES C CXX)\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        f'set(CMAKE_CUDA_COMPILER "{cxx}")\n'
        "set(CMAKE_CUDA_COMPILER_ID NVIDIA)\n"
        "set(CMAKE_CUDA_COMPILER_VERSION 12.9.0)\n"
        "set(CMAKE_CUDA_COMPILER_WORKS TRUE)\n"
        "set(CMAKE_CUDA_COMPILER_FORCED TRUE)\n"
        "set(CMAKE_CUDA_COMPILER_ID_RUN TRUE)\n"
        "set(CMAKE_CUDA_ARCHITECTURES 80)\n"
        "set(CMAKE_CUDA_RUNTIME_LIBRARY None)\n"
        'set(CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT "#$ LIBRARIES=-lcudart\\n'
        '${CMAKE_CXX_COMPILER} dummy.o -lcudart\\n")\n'
        "enable_language(CUDA)\n"
        f'include("{ROOT / "cmake/GenerativeQCCudaImplib.cmake"}")\n'
        "add_library(probe SHARED probe.cu)\n"
        "set_target_properties(probe PROPERTIES\n"
        "  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)\n"
        "generativeqc_attach_cuda_implib(probe)\n"
    )
    build = tmp_path / "build"
    _run(
        cmake,
        "-G",
        "Ninja",
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
    )
    rules = (build / "CMakeFiles/rules.ninja").read_text()
    assert "rule CXX_SHARED_LIBRARY_LINKER" in rules
    assert "rule CXX_SHARED_LIBRARY_DEVICE_LINKER" in rules
    host_rule = rules.split("rule CXX_SHARED_LIBRARY_LINKER", 1)[1].split("\n\n", 1)[0]
    device_rule = rules.split("rule CXX_SHARED_LIBRARY_DEVICE_LINKER", 1)[1].split(
        "\n\n", 1
    )[0]
    assert "tools/link_cuda_implib.py" in host_rule
    assert " -dlink " in device_rule
    assert "tools/link_cuda_implib.py" not in device_rule
    graph = (build / "build.ninja").read_text()
    host_edge = next(
        line for line in graph.splitlines() if line.startswith("build libprobe.so:")
    )
    assert "CMakeFiles/probe.dir/probe.cu.o" in host_edge
    assert "CMakeFiles/probe.dir/cmake_device_link.o" in host_edge


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
