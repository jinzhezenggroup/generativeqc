"""Compile CUDA archive packaging contracts without probing or running a GPU.

The original archive must remain device-linkable. The host copy must retain
every executable image and its registration, including virtual-only targets.
Real-device registration/execution is qualified separately under Slurm.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="host-archive pruning targets Linux NVIDIA ELF"
)


def run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Bound each compiler/tool invocation and retain its diagnostic output."""
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=120, **kwargs
    )


@pytest.fixture
def archive_probe(tmp_path: Path, native_cxx: Any) -> Any:
    compiler = os.environ.get("GENERATIVEQC_NVCC")
    if not compiler:
        pytest.skip("set GENERATIVEQC_NVCC for compiled CUDA archive contracts")
    toolkit = Path(compiler).parent
    if not (toolkit / "nvprune").is_file():
        pytest.skip("CUDA toolkit has no nvprune")
    for name in ("cmake", "tools", "src"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "callee.cu").write_text(
        "__device__ int add_one(int value) { return value + 1; }\n"
    )
    (tmp_path / "owner.cu").write_text(
        "#include <cuda_runtime.h>\n"
        "__device__ int add_one(int);\n"
        "__global__ void fill(int* value) { *value = add_one(41); }\n"
        'extern "C" int compute() {\n'
        " int* device = nullptr; int value = 0;\n"
        " if (cudaMalloc(&device, sizeof(int))) return -1;\n"
        " fill<<<1, 1>>>(device);\n"
        " auto status = cudaMemcpy(&value, device, sizeof(int), cudaMemcpyDeviceToHost);\n"
        " cudaFree(device); return status == cudaSuccess ? value : -2;\n"
        "}\n"
    )
    (tmp_path / "main.cpp").write_text(
        'extern "C" int compute();\nint main() { return compute() == 42 ? 0 : 1; }\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.24)
project(ArchiveProbe LANGUAGES C CXX CUDA)
find_package(CUDAToolkit REQUIRED)
include("{ROOT / "cmake/GenerativeQCCudaCompression.cmake"}")
generativeqc_configure_cuda_compression()
include("{ROOT / "cmake/GenerativeQCCudaArchive.cmake"}")
add_library(original STATIC owner.cu callee.cu)
set_target_properties(original PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(original INTERFACE CUDA::cudart)
generativeqc_cuda_host_archive(original selected)
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/selected-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:${{selected}}>")
add_executable(probe main.cpp)
target_link_libraries(probe PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${{selected}}>")
generativeqc_add_cuda_shared_library(plugin plugin_owner owner.cu callee.cu)
if(WHEEL_PROBE)
  set(Python3_EXECUTABLE "{sys.executable}")
  include("{ROOT / "cmake/GenerativeQCCudaImplib.cmake"}")
  set_target_properties(${{plugin_owner}} plugin PROPERTIES CUDA_RUNTIME_LIBRARY None)
  generativeqc_attach_cuda_implib(plugin)
  target_link_libraries(${{plugin_owner}} PRIVATE generativeqc_cuda_wheel_imports)
else()
  target_link_libraries(${{plugin_owner}} PRIVATE CUDA::cudart)
endif()
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/plugin-owner-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:${{plugin_owner}}>")
add_executable(plugin_probe main.cpp)
target_link_libraries(plugin_probe PRIVATE plugin)
""")

    def build(
        architectures: str,
        enabled: bool = True,
        multi: bool = False,
        compression: bool = True,
        wheel: bool = False,
    ) -> Path:
        directory = tmp_path / ("multi" if multi else "build")
        launcher = shutil.which("sccache") or native_cxx.cache
        run(launcher, "--version")
        run(
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(directory),
            "-G",
            "Ninja Multi-Config" if multi else "Ninja",
            f"-DCMAKE_CUDA_COMPILER={compiler}",
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CUDA_HOST_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}",
            f"-DCMAKE_C_COMPILER_LAUNCHER={launcher}",
            f"-DCMAKE_CUDA_COMPILER_LAUNCHER={launcher}",
            f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
            f"-DGENERATIVEQC_PRUNE_CUDA_ARCHIVES={'ON' if enabled else 'OFF'}",
            f"-DGENERATIVEQC_COMPRESS_CUDA_FATBINS={'ON' if compression else 'OFF'}",
            f"-DWHEEL_PROBE={'ON' if wheel else 'OFF'}",
            "-DCMAKE_BUILD_TYPE=Release",
        )
        run(
            "cmake", "--build", str(directory), "--config", "Release", "--parallel", "2"
        )
        return directory

    return build, tmp_path, toolkit


def test_shared_archive_keeps_provider_free_final_link(archive_probe: Any) -> None:
    """Wheel import synthesis belongs to the final ELF, not its static owner."""
    build, _, _ = archive_probe
    directory = build("120-real", wheel=True)
    dynamic = run("readelf", "-d", str(directory / "libplugin.so")).stdout
    assert "Shared library: [libcudart" not in dynamic
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") == "1":
        assert os.environ.get("SLURM_JOB_ID"), "CUDA execution requires finite srun"
        run(str(directory / "plugin_probe"))


def executable_images(toolkit: Path, archive: Path, directory: Path) -> list[str]:
    """Compare decoded executable payloads, independent of archive layout/names."""
    directory.mkdir()
    for kind in ("elf", "ptx"):
        run(
            str(toolkit / "cuobjdump"),
            "-all",
            f"--extract-{kind}",
            "all",
            str(archive),
            cwd=directory,
        )
    images = []
    for path in directory.iterdir():
        data = path.read_bytes()
        # cuobjdump extracts relocatable inputs too when reading an archive.
        # They are deliberately removed; compare executable ELF and all PTX.
        if data.startswith(b"\x7fELF") and int.from_bytes(data[16:18], "little") == 1:
            continue
        images.append(hashlib.sha256(data).hexdigest())
    return sorted(images)


@pytest.mark.parametrize(
    "architectures,multi",
    [
        ("120-real", False),
        ("120-virtual", False),
        ("120", False),
        ("80-real;120", False),
        ("120-real", True),
    ],
)
def test_closed_archive_preserves_executable_images(
    archive_probe: Any,
    architectures: str,
    multi: bool,
) -> None:
    build, root, toolkit = archive_probe
    directory = build(architectures, multi=multi)
    original = directory / ("Release/liboriginal.a" if multi else "liboriginal.a")
    selected = Path((directory / "selected-Release.txt").read_text())
    assert selected != original
    assert selected.stat().st_size <= original.stat().st_size
    if any(not value.endswith("-virtual") for value in architectures.split(";")):
        assert selected.stat().st_size < original.stat().st_size
    before = executable_images(toolkit, original, root / "original-images")
    assert before
    assert executable_images(toolkit, selected, root / "host-images") == before
    plugin = directory / ("Release/libplugin.so" if multi else "libplugin.so")
    plugin_owner = Path((directory / "plugin-owner-Release.txt").read_text())
    # The shared artifact retains the same device program and a linkable host
    # export after its private archive has been pruned and wholly retained.
    plugin_images = executable_images(toolkit, plugin, root / "plugin-images")
    owned_images = executable_images(
        toolkit, plugin_owner, root / "plugin-owner-images"
    )
    # A PTX-only host link can also retain CUDA's device-runtime images. Every
    # program image owned by this archive must survive byte-for-byte, including
    # multiplicity; the runtime provider is validated by actual execution below.
    assert Counter(owned_images) <= Counter(plugin_images)
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") == "1":
        assert os.environ.get("SLURM_JOB_ID"), "CUDA execution requires finite srun"
        run(str(directory / ("Release/probe" if multi else "probe")))
        run(str(directory / ("Release/plugin_probe" if multi else "plugin_probe")))
        if "120" in architectures.split(";"):
            # Forward-compatible PTX must still support a forced driver JIT,
            # including the cross-translation-unit device call.
            run(
                str(directory / ("Release/probe" if multi else "probe")),
                env={**os.environ, "CUDA_FORCE_PTX_JIT": "1"},
            )
            run(
                str(directory / ("Release/plugin_probe" if multi else "plugin_probe")),
                env={**os.environ, "CUDA_FORCE_PTX_JIT": "1"},
            )
    original_bytes = original.read_bytes()
    host_bytes = selected.read_bytes()
    # No-op builds preserve both artifacts. An input edit updates the host copy
    # while preserving a separate, complete source archive.
    run("cmake", "--build", str(directory), "--config", "Release")
    assert original.read_bytes() == original_bytes
    assert selected.read_bytes() == host_bytes
    source = root / "callee.cu"
    source.write_text(source.read_text().replace("value + 1", "value + 2"))
    run("cmake", "--build", str(directory), "--config", "Release", "--parallel", "2")
    assert original.read_bytes() != original_bytes
    assert selected.read_bytes() != host_bytes
    assert executable_images(
        toolkit, original, root / "updated-original"
    ) == executable_images(toolkit, selected, root / "updated-host")


def test_archive_pruning_can_be_disabled(archive_probe: Any) -> None:
    build, root, toolkit = archive_probe
    directory = build("120-real", enabled=False)
    assert (
        Path((directory / "selected-Release.txt").read_text())
        == directory / "liboriginal.a"
    )
    assert executable_images(
        toolkit, directory / "libplugin.so", root / "unpruned-plugin"
    ) == executable_images(
        toolkit,
        Path((directory / "plugin-owner-Release.txt").read_text()),
        root / "unpruned-owner",
    )
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") == "1":
        assert os.environ.get("SLURM_JOB_ID"), "CUDA execution requires finite srun"
        run(str(directory / "plugin_probe"))


@pytest.mark.parametrize("architectures", ["120-real", "120-virtual", "80-real;120"])
def test_compression_preserves_device_linked_images(
    archive_probe: Any, architectures: str
) -> None:
    """Compare decoded executables after rebuilding the same RDC archive.

    The OFF arm is the toolchain's original storage policy. Compression applies
    to both translation units and the final device link, without losing PTX.
    """
    build, root, toolkit = archive_probe
    directory = build(architectures, compression=False)
    original = directory / "liboriginal.a"
    before = executable_images(toolkit, original, root / "uncompressed-images")
    original_size = original.stat().st_size
    selected = Path((directory / "selected-Release.txt").read_text())
    host_size = selected.stat().st_size
    build(architectures, compression=True)
    assert executable_images(toolkit, original, root / "compressed-images") == before
    # PTX-only images may already use the same compression under the toolchain
    # default. Real cubins exercise both ordinary compilation and device linking.
    assert original.stat().st_size <= original_size
    assert selected.stat().st_size <= host_size
    if any(not value.endswith("-virtual") for value in architectures.split(";")):
        assert original.stat().st_size < original_size
        assert selected.stat().st_size < host_size
    if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") == "1":
        assert os.environ.get("SLURM_JOB_ID"), "CUDA execution requires finite srun"
        run(str(directory / "probe"))
        if "120" in architectures.split(";"):
            run(
                str(directory / "probe"),
                env={**os.environ, "CUDA_FORCE_PTX_JIT": "1"},
            )


@pytest.mark.parametrize(
    "mode", ["off", "other-backend", "missing-tool", "unresolved", "broken-tool"]
)
def test_archive_fallback_and_invalid_closure(
    tmp_path: Path,
    native_cxx: Any,
    mode: str,
) -> None:
    """Exercise configuration failure/fallback with a host-only archive owner."""
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    if mode == "broken-tool":
        tool = toolkit / "nvprune"
        tool.write_text("#!/bin/sh\nexit 1\n")
        tool.chmod(0o755)
    (tmp_path / "owner.cpp").write_text("int owner() { return 42; }\n")
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.24)
project(ArchiveFallback LANGUAGES CXX)
include("{ROOT / "cmake/GenerativeQCCudaArchive.cmake"}")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CUDA_COMPILER "{toolkit / "nvcc"}")
set(CMAKE_CUDA_COMPILER_ID {"Clang" if mode == "other-backend" else "NVIDIA"})
set(GENERATIVEQC_PRUNE_CUDA_ARCHIVES {"OFF" if mode == "off" else "ON"})
add_library(original STATIC owner.cpp)
set_target_properties(original PROPERTIES CUDA_ARCHITECTURES 120-real
  CUDA_RESOLVE_DEVICE_SYMBOLS {"OFF" if mode == "unresolved" else "ON"})
generativeqc_cuda_host_archive(original selected)
file(WRITE "${{CMAKE_BINARY_DIR}}/selection.txt" "${{selected}}")
""")
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DCMAKE_CXX_COMPILER={native_cxx.compiler}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={native_cxx.cache}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if mode in {"unresolved", "broken-tool"}:
        assert result.returncode != 0
        assert (
            "device-linked static library"
            if mode == "unresolved"
            else "Cannot execute selected nvprune"
        ) in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "build/selection.txt").read_text() == "original"
