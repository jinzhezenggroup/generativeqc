"""Compile the native provider-initialization admission with no CUDA provider."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_native_provider_initialization(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    (tmp_path / "cuda_runtime_api.h").write_text(
        """#pragma once
#include <cstddef>
using cudaStream_t=void*;
enum cudaError_t {cudaSuccess,cudaErrorMemoryAllocation,cudaErrorInvalidValue};
enum cudaMemcpyKind {cudaMemcpyDeviceToHost};
inline const char* cudaGetErrorString(cudaError_t) {return "unused CUDA interface";}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {return cudaErrorInvalidValue;}
inline cudaError_t cudaMemGetInfo(std::size_t*,std::size_t*) {return cudaErrorInvalidValue;}
inline cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t) {
 return cudaErrorInvalidValue;
}
""",
        encoding="utf-8",
    )
    try:
        executable = required_native_cxx.build_executable(
            [ROOT / "tests/native/test_provider_initialization.cpp"],
            tmp_path / "provider_initialization",
            compile_args=[
                "-std=c++20",
                "-O2",
                f"-I{tmp_path}",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
            ],
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
