"""Exercise production pending-owner attachment with HOST allocation faults.

The owner, resource, graph and eigen-handle implementations are compiled intact.
CUDA/vendor seams abort if reached: this transition must remain host metadata.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/native/fixtures/shared_eigen_provider"


@pytest.fixture(scope="module")
def attach_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("pending-provider-attach")
    (folder / "cuda_runtime_api.h").write_text(CUDA_API)
    (folder / "cuda_runtime.h").write_text(
        '#include "cuda_runtime_api.h"\nstruct dim3 { unsigned x, y, z; };\n'
    )
    # Execute the driver's actual rollback while isolating SCF numerics.
    driver = (ROOT / "src/scf/cuda/rhf_bucket.cpp").read_text()
    start = driver.index("      try {\n        owner->attach_bucket(raw);")
    end = driver.index("      *plan = raw;", start)
    (folder / "attach_rollback.inc").write_text(driver[start:end])
    return required_native_cxx.build_executable(
        [
            ROOT / "src/scf/cuda/rhf_provider_owner.cpp",
            ROOT / "src/scf/cuda/resources.cpp",
            ROOT / "src/scf/cuda/rhf_graph.cpp",
            ROOT / "src/solver/cuda/symmetric_eigen_handles.cpp",
            ROOT / "tests/native/test_pending_provider_attach.cpp",
        ],
        folder / "probe",
        compile_args=[
            "-std=c++20",
            "-O0",
            "-I" + str(folder),
            "-I" + str(FIXTURES),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        link_args=["-pthread"],
    )


def test_every_attach_allocation_failure_preserves_ordinary_cleanup(
    attach_probe: Path,
) -> None:
    result = subprocess.run(
        [str(attach_probe)], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


CUDA_API = r"""#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorUnknown = 999;
constexpr int cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2;
constexpr int cudaErrorInvalidDevice = 101, cudaErrorInvalidResourceHandle = 400;
constexpr int cudaEventDisableTiming = 2, cudaStreamCaptureModeThreadLocal = 1;
constexpr unsigned long long cudaGraphInstantiateFlagDeviceLaunch = 4;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                      cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMalloc(void**, std::size_t);
cudaError_t cudaMallocAsync(void**, std::size_t, cudaStream_t);
cudaError_t cudaFree(void*);
cudaError_t cudaFreeAsync(void*, cudaStream_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, int);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
"""
