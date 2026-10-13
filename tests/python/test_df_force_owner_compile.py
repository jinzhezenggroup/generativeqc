"""Compile the complete DF force orchestration owner with its production headers.

This is a host C++ contract gate, not a linked endpoint or GPU qualification.
The owner contains no CUDA kernels. Only CUDA runtime declarations are supplied
so CPU-only CI can type-check every owner call, including recursive companions
and replay/retry paths, without substituting scientific implementations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

# Declaration-only CUDA runtime boundary. Preserve distinct opaque handle types,
# enum values, and function signatures; there are deliberately no definitions to
# turn this compile check into an executable mock of the scientific endpoint.
CUDA_RUNTIME_DECLARATIONS = r"""
#pragma once
#include <cstddef>
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidDevice = 101
};
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDefault = 0;
extern "C" {
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMallocAsync(void**, std::size_t, cudaStream_t);
cudaError_t cudaFreeAsync(void*, cudaStream_t);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
}
"""


def test_complete_df_force_owner_compiles(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    (tmp_path / "cuda_runtime_api.h").write_text(
        CUDA_RUNTIME_DECLARATIONS, encoding="utf-8"
    )
    output = tmp_path / "df_ccsdt_force.o"
    # Compile the actual .cu file, not an extracted prefix or reconstructed body.
    # -x c++ makes the host compiler parse it instead of ignoring the .cu suffix.
    try:
        native_cxx.compile_object(
            ROOT / "src/methods/df_ccsdt_force.cu",
            output,
            args=(
                "-x",
                "c++",
                "-std=c++20",
                "-DGENERATIVEQC_HAS_CUDA=1",
                f"-I{tmp_path}",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
            ),
            timeout=120,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    assert output.is_file() and output.stat().st_size > 0
