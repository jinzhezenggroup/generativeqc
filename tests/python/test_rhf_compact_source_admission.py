"""Exercise actual compact handoff allocation/admission with host CUDA shims.

The shims model allocation, copying and synchronization, not numerical ERIs or
CUDA execution. Production headers and the complete handoff translation unit
are compiled; unrelated CUDA owner destructors and tile execution are stubbed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_compact_source_admission(tmp_path: Path) -> None:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    (tmp_path / "cuda_runtime_api.h").write_text(CUDA_API)
    (tmp_path / "cuda_runtime.h").write_text(
        '#pragma once\n#include "cuda_runtime_api.h"\nstruct dim3 { unsigned x=1,y=1,z=1; };\n'
    )
    (tmp_path / "cublas_v2.h").write_text("#pragma once\nusing cublasHandle_t=void*;\n")
    (tmp_path / "cusolverDn.h").write_text(SOLVER_API)
    objects = []
    for name, source in (
        ("handoff", ROOT / "src/scf/cuda/rhf_source_handoff.cpp"),
        ("probe", ROOT / "tests/native/test_rhf_compact_source_admission.cpp"),
    ):
        obj = tmp_path / f"{name}.o"
        result = subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O0",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DGENERATIVEQC_HAS_CUDA=1",
                f"-I{tmp_path}",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(obj))
    exe = tmp_path / "probe"
    subprocess.run(
        [compiler, *objects, "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


CUDA_API = r"""
#pragma once
#include <cstddef>
using cudaStream_t=void*; using cudaGraph_t=void*; using cudaGraphExec_t=void*;
using cudaError_t=int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2;
constexpr int cudaErrorInvalidDevice=10;
constexpr unsigned cudaStreamNonBlocking=1;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
 cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMallocAsync(void**,std::size_t,cudaStream_t);
cudaError_t cudaFreeAsync(void*,cudaStream_t);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
"""

SOLVER_API = r"""
#pragma once
using cusolverDnHandle_t=void*; using cusolverDnParams_t=void*; using syevjInfo_t=void*;
using cusolverStatus_t=int; using cusolverEigMode_t=int; using cublasFillMode_t=int;
using cudaDataType=int;
int cusolverDnXsyevBatched(...); int cusolverDnXsyevBatched_bufferSize(...);
"""
