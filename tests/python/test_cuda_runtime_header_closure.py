"""Installed JIT headers must be complete and independently includable."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]


def test_installed_tensor_runtime_has_complete_local_header_closure() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assets = config["tool"]["scikit-build"]["wheel"]["force-include"]
    pending = [ROOT / "src/tensor/cuda_runtime.cuh"]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(ROOT).as_posix()
        assert relative in assets, f"missing installed JIT dependency: {relative}"
        for dependency in re.findall(
            r'^\s*#include\s+"([^"]+)"', path.read_text(), re.MULTILINE
        ):
            child = path.parent / dependency
            assert child.is_file(), (
                f"header requires an undeclared include root: {dependency}"
            )
            pending.append(child)


def test_transitive_resources_participate_in_all_tensor_consumer_identities() -> None:
    # These consumers directly include the shared Tensor CUDA runtime. Their
    # source/header inventories must invalidate artifacts when ownership changes.
    modules = (
        "dft/ao_cuda.py",
        "tensor/cuda_execute.py",
        "xc/cuda_emit.py",
        "integral/second_derivatives_execute.py",
        "integral/weighted_eri_execute.py",
        "integral/first_directional_execute.py",
        "integral/first_gradient_execute.py",
        "method/stationary_cuda.py",
    )
    for module in modules:
        text = (ROOT / "python/generativeqc_compiler" / module).read_text()
        for header in (
            "bounded_workspace.hpp",
            "cuda_resources.cuh",
            "resource_cuda.cuh",
            "resource_ledger.hpp",
            "residency_boundaries.hpp",
            "residency_observer.hpp",
        ):
            assert f'"src/runtime/{header}"' in text, (module, header)


@pytest.mark.parametrize(
    ("header", "installed"),
    [
        ("residency_boundaries.hpp", False),
        ("residency_cuda.cuh", False),
        ("cuda_resources.cuh", False),
        ("residency_boundaries.hpp", True),
        ("cuda_resources.cuh", True),
    ],
)
def test_residency_headers_compile_with_narrow_jit_include_roots(
    header: str, installed: bool, native_cxx: NativeCxx, tmp_path: Path
) -> None:
    """Compile the real Tensor resource include chain without an implicit src root.

    CUDA declarations are host-only stubs; this tests header portability, not
    device execution. The installed case copies only declared wheel assets.
    """
    root = ROOT
    if installed:
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assets = config["tool"]["scikit-build"]["wheel"]["force-include"]
        root = tmp_path / "wheel/generativeqc_compiler/assets"
        for source, destination in assets.items():
            if source.startswith("src/runtime/"):
                target = tmp_path / "wheel" / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / source, target)
        (root / "src/tensor").mkdir(parents=True)
    (tmp_path / "cuda_runtime_api.h").write_text(
        """#pragma once
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;
constexpr int cudaErrorInvalidValue = 3, cudaErrorInvalidDevice = 4;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDefault = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice };
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMallocAsync(void**, std::size_t, cudaStream_t);
cudaError_t cudaFreeAsync(void*, cudaStream_t);
cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
"""
    )
    source = tmp_path / "probe.cpp"
    source.write_text(f'#include "../runtime/{header}"\n')
    native_cxx.compile_object(
        source,
        tmp_path / "probe.o",
        args=(
            "-std=c++17",
            f"-I{tmp_path}",
            f"-I{root / 'src/tensor'}",
            f"-I{root / 'src/runtime'}",
        ),
    )


@pytest.mark.parametrize(
    "header", ["residency_boundaries.hpp", "residency_observer.hpp"]
)
def test_xc_residency_dependencies_invalidate_generated_identity(
    header: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from generativeqc_compiler.common import paths
    from generativeqc_compiler.xc import build_program, cuda_emit, functional

    program = build_program(functional("PBE"), order=0)
    _, before, dependencies = cuda_emit.emit_cuda(program)
    relative = f"src/runtime/{header}"
    original = paths.asset_path
    assert original(relative) in dependencies
    assert relative in before["generator_sources"]
    changed = tmp_path / header
    changed.write_text(original(relative).read_text() + "\n// dependency change\n")

    def asset_path(name: str) -> Path:
        return changed if name == relative else original(name)

    monkeypatch.setattr(paths, "asset_path", asset_path)
    monkeypatch.setattr(cuda_emit, "asset_path", asset_path)
    _, after, dependencies = cuda_emit.emit_cuda(program)
    assert changed in dependencies
    assert before["generator_sources"][relative] != after["generator_sources"][relative]
    assert before["identity"] != after["identity"]
