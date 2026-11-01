"""Provider-free CUDA wheel imports are discovered from strict link failures."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
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
    commands = _run(ninja, "-C", str(build), "-t", "commands").stdout.splitlines()
    host_commands = [line for line in commands if "tools/link_cuda_implib.py" in line]
    device_commands = [line for line in commands if " -dlink " in line]
    assert host_commands and device_commands
    if _implib_target() == "x86_64":
        assert all("cuda_wheel_payload.ld" in line for line in host_commands)
    assert all("cuda_wheel_payload.ld" not in line for line in device_commands)


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


@pytest.mark.parametrize("large_section", [None, ".nv_fatbin", "__nv_relfatbin"])
def test_wheel_loads_with_retained_images_and_reachable_host_sections(
    tmp_path: Path, native_cxx: Any, large_section: str | None
) -> None:
    """Keep opaque images out of host-relative reach, without huge physical I/O.

    NOBITS reserves virtual space; a middle-placement negative control reproduces
    the real relocation overflow. Small PROGBITS images check retained bytes.
    """
    if _implib_target() != "x86_64":
        pytest.skip("the payload layout targets Linux x86-64 ELF")
    for tool in ("cmake", "ninja", "readelf"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required for the executable ELF contract")
    # Copy the build helpers so editing the script can exercise relink tracking
    # without modifying the checkout used by other tests.
    shutil.copytree(ROOT / "cmake", tmp_path / "cmake")
    (tmp_path / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    (tmp_path / "host.cpp").write_text(
        "#include <cstdint>\n#include <stdexcept>\n"
        'extern "C" {\n'
        "extern const unsigned char image0[], image0_end[], image1[], image1_end[];\n"
        "extern const void* wrappers[];\n"
        'extern const char host_label[] = "host-rodata-oracle";\n'
        "int cudaDeviceSynchronize();\n"
        "int lazy_cuda_probe() { return cudaDeviceSynchronize(); }\n"
        "static thread_local int counter;\n"
        "int probe() {\n"
        "  try { throw std::runtime_error(host_label); }\n"
        "  catch (const std::runtime_error& error) {\n"
        "    return error.what()[0] == 'h' ? ++counter + 6 : -1;\n"
        "  }\n}\n"
        "int image_contract() {\n"
        "  return wrappers[0] == image0 && wrappers[1] == image1\n"
        "    && reinterpret_cast<uintptr_t>(wrappers) < reinterpret_cast<uintptr_t>(image0)\n"
        "    && reinterpret_cast<uintptr_t>(wrappers) < reinterpret_cast<uintptr_t>(image1);\n"
        "}\n}\n"
    )
    assembly = []
    sizes = []
    for index, section in enumerate((".nv_fatbin", "__nv_relfatbin")):
        large = section == large_section
        size = 0x80001000 if large else 32
        sizes.append(size)
        assembly.extend(
            [
                # Keep both sections NOBITS in the large probe: a following
                # PROGBITS section can make ld materialize the virtual gap.
                f'.section {section},"a",@{"nobits" if large_section else "progbits"}',
                ".balign 8",
                f".global image{index}, image{index}_end",
                f"image{index}:",
                f".skip {size}" if large_section else f".fill {size},1,{65 + index}",
                f"image{index}_end:",
            ]
        )
    assembly.extend(
        [
            '.section .nvFatBinSegment,"aw",@progbits',
            ".balign 8",
            ".global wrappers",
            "wrappers: .quad image0, image1",
            '.section .note.GNU-stack,"",@progbits',
        ]
    )
    (tmp_path / "images.S").write_text("\n".join(assembly) + "\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(WheelLayoutProbe LANGUAGES C CXX ASM)\n"
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        'include("${PROJECT_SOURCE_DIR}/cmake/GenerativeQCCudaImplib.cmake")\n'
        "add_library(probe SHARED host.cpp images.S)\n"
        "generativeqc_attach_cuda_implib(probe)\n"
    )
    build = tmp_path / "build"
    _run(
        "cmake",
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
        f"-DCMAKE_C_COMPILER_LAUNCHER={native_cxx.cache}",
        f"-DCMAKE_ASM_COMPILER_LAUNCHER={native_cxx.cache}",
    )
    _run("cmake", "--build", str(build), "--parallel", "2")
    library = build / "libprobe.so"
    dynamic = _run("readelf", "-d", str(library)).stdout
    assert "Shared library: [libcudart" not in dynamic
    # The regression must stay cheap even though its mapped span exceeds 2 GiB.
    assert library.stat().st_size < 1_000_000
    _run(
        sys.executable,
        "-c",
        """
import ctypes
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
lib = ctypes.CDLL(sys.argv[1])
assert [lib.probe(), lib.probe()] == [7, 8]
with ThreadPoolExecutor(max_workers=1) as pool:
    assert pool.submit(lib.probe).result() == 7
assert lib.probe() == 9
assert lib.image_contract() == 1
for index, expected in enumerate(map(int, sys.argv[2:])):
    start = ctypes.addressof(ctypes.c_ubyte.in_dll(lib, f'image{index}'))
    end = ctypes.addressof(ctypes.c_ubyte.in_dll(lib, f'image{index}_end'))
    assert end - start == expected
    value = 0 if max(map(int, sys.argv[2:])) > 32 else 65 + index
    assert ctypes.c_ubyte.from_address(start).value == value
    assert ctypes.c_ubyte.from_address(end - 1).value == value
    # GNU ld may merge synthetic NOBITS into the writable BSS reservation.
    # The real image representation is PROGBITS; require its pages read-only.
    for address in ((start, end - 1) if max(map(int, sys.argv[2:])) == 32 else ()):
        mapping = next(line.split() for line in Path('/proc/self/maps').read_text().splitlines()
                       if int(line.split('-')[0], 16) <= address
                       < int(line.split()[0].split('-')[1], 16))
        assert mapping[1].startswith('r--'), mapping
""",
        str(library),
        *map(str, sizes),
    )

    if large_section:
        # NOBITS normally sorts late. Force the faulty placement so the negative
        # control exercises the same host relocation range as real huge fatbins.
        middle = tmp_path / "middle.ld"
        middle.write_text(
            f"SECTIONS {{ {large_section} : {{ *({large_section}) }} }} "
            "INSERT AFTER .rodata;\n"
        )
        bad = subprocess.run(
            [
                native_cxx.compiler,
                "-shared",
                str(build / "CMakeFiles/probe.dir/host.cpp.o"),
                str(build / "CMakeFiles/probe.dir/images.S.o"),
                f"-Wl,-T,{middle}",
                "-o",
                str(tmp_path / "bad.so"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert bad.returncode != 0
        assert any(
            message in bad.stderr
            for message in (
                "relocation truncated",
                "out of range",
                "failed to convert GOTPCREL",
            )
        ), bad.stderr
    else:
        before = library.stat().st_mtime_ns
        script = tmp_path / "cmake/cuda_wheel_payload.ld"
        script.write_text(script.read_text() + "\n/* Relink dependency probe. */\n")
        _run("cmake", "--build", str(build), "--parallel", "2")
        assert library.stat().st_mtime_ns > before
