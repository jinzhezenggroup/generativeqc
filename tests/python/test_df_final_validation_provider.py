"""Execute the complete production DF translation unit with host vendor injection.

This proves submission mapping and exceptional draining, not device qualification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tools.check_scf_structure import audit_scf_structure

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_df_provider_dependency_is_an_exact_leaf(tmp_path: Path) -> None:
    source = tmp_path / "src"
    owner = source / "scf/cuda/df_final_validation.cpp"
    owner.parent.mkdir(parents=True)
    tensor = source / "tensor"
    tensor.mkdir()
    for name in (
        "cuda_square_linalg.hpp",
        "cuda_square_linalg.hpp_impl.hpp",
        "cuda_runtime.cuh",
    ):
        (tensor / name).write_text("\n")
        owner.write_text(f'#include "tensor/{name}"\n')
        errors = audit_scf_structure(tmp_path)["errors"]
        if name == "cuda_square_linalg.hpp":
            assert not errors
        else:
            assert len(errors) == 1
            assert f"forbidden cuda_df_runtime dependency on tensor/{name}" in errors[0]


CUDA = r"""
#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using cudaError_t = int;
enum cudaDataType { CUDA_R_32F = 0 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2;
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaPeekAtLastError();
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
const char* cudaGetErrorString(cudaError_t);
"""

PROBE = r"""
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include "scf/cuda/df_final_validation.cpp"

static int submitted{}, cleared{}, drained{}, vendor_failure{}, zero_failure{};
static const double *expected_left{}, *expected_right{};
static double* expected_output{};
static void* stream = reinterpret_cast<void*>(23);
static cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(17);
static int dimension, reduction;
static bool transpose_left, transpose_right;
static double expected_scale;

extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t ta,
 cublasOperation_t tb, int m, int n, int k, const double* alpha, const double* a,
 int lda, const double* b, int ldb, const double* beta, double* c, int ldc) {
  ++submitted;
  assert(h == handle && a == expected_left && b == expected_right && c == expected_output);
  assert(m == dimension && n == dimension && k == reduction);
  assert(lda == dimension && ldb == dimension && ldc == dimension);
  assert(ta == (transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N));
  assert(tb == (transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N));
  assert(*alpha == expected_scale && *beta == 0);
  if (vendor_failure) return vendor_failure;
  for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) {
    long double sum = 0;
    for (int p=0; p<k; ++p) {
      sum += static_cast<long double>(a[ta == CUBLAS_OP_N ? i+p*lda : p+i*lda]) *
                                     b[tb == CUBLAS_OP_N ? p+j*ldb : j+p*ldb];
    }
    c[i+j*ldc] = static_cast<double>(*alpha * sum);
  }
  return CUBLAS_STATUS_SUCCESS;
}
cudaError_t cudaStreamSynchronize(cudaStream_t s) {
  assert(s == stream); ++drained; return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void* p, int v, std::size_t bytes, cudaStream_t s) {
  assert(p == expected_output && v == 0 && s == stream);
  assert(bytes == dimension*dimension*sizeof(double));
  ++cleared;
  if (zero_failure) return zero_failure;
  std::memset(p, v, bytes); return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA error"; }
namespace generativeqc::runtime::cuda_trace {
void trace_counter(const char*, std::uint64_t) noexcept {}
}
namespace generativeqc::scf::cuda_df {
std::uint64_t next_factor_basis_identity() noexcept { return 1; }
}
namespace generativeqc::solver::cuda {
PreparedSymmetricEigenHandles::~PreparedSymmetricEigenHandles() {
  assert(!view().solver && !view().parameters && !view().jacobi);
}
}

int main(int argc, char** argv) {
  assert(argc == 6);
  dimension = std::stoi(argv[1]);
  const int rank = std::stoi(argv[2]);
  reduction = rank < 0 ? dimension : rank;
  transpose_left = std::stoi(argv[3]); transpose_right = std::stoi(argv[4]);
  expected_scale = std::stod(argv[5]);
  std::vector<double> a(dimension*dimension), b(a.size()), c(a.size(), -999);
  for (std::size_t i=0; i<a.size(); ++i) { a[i] = int(i%11)-4; b[i] = int(i%7)-3; }
  expected_left=a.data(); expected_right=b.data(); expected_output=c.data();
  generativeqc::scf::CudaDensityFittingJkPlan plan;
  plan.blas=handle; plan.stream=stream; plan.nbf=dimension;
  using namespace generativeqc::scf::cuda_df;
  {
    Drain drain{stream};
    gemm(plan, a.data(), b.data(), c.data(), transpose_left, transpose_right, rank, expected_scale);
    drain.finish();
  }
  assert(drained == 1);
  assert(submitted == (rank != 0) && cleared == (rank == 0));
  std::cout << std::setprecision(17);
  for (double v : c) std::cout << v << ' ';
  std::cout << '\n';
  if (rank == 0) {
    zero_failure = cudaErrorMemoryAllocation;
    bool oom=false;
    try { Drain drain{stream}; gemm(plan,a.data(),b.data(),c.data(),false,false,0); }
    catch (const std::bad_alloc&) { oom=true; }
    assert(oom && drained == 2 && submitted == 0);
  } else {
    // Preserve the original runtime_error mapping, including ALLOC_FAILED.
    for (int failure : {3, 7, 13, 14}) {
      vendor_failure = failure;
      bool rejected=false;
      try {
        Drain drain{stream};
        gemm(plan,a.data(),b.data(),c.data(),transpose_left,transpose_right,rank,expected_scale);
      } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()) ==
            "cuBLAS final-validation product failed: " + std::to_string(failure);
      }
      assert(rejected);
    }
    assert(drained == 5 && plan.blas == handle && plan.stream == stream);
    // Fault-only alias captures prove that the boundary forwards pointers and
    // does not manufacture packing/scratch. Aliased GEMM success is not admitted.
    expected_output=a.data();
    try { gemm(plan,a.data(),b.data(),a.data(),transpose_left,transpose_right,rank,expected_scale); }
    catch (const std::runtime_error&) {}
  }
  // Avoid the plan's unrelated runtime-owned resource destructors in this probe.
  std::cout.flush();
  std::_Exit(0);
}
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx) -> Path:
    path = tmp_path_factory.mktemp("df-validation-provider")
    (path / "cuda_runtime_api.h").write_text(CUDA)
    (path / "cuda_runtime.h").write_text('#include "cuda_runtime_api.h"\n')
    (path / "library_types.h").write_text(
        "#pragma once\nenum libraryPropertyType { MAJOR_VERSION=0 };\n"
    )
    resource = path / "runtime/resource_cuda.cuh"
    resource.parent.mkdir()
    resource.write_text(
        "#pragma once\nnamespace generativeqc::runtime {\n"
        "inline int resource_cuda_free(void*) { return 0; }\n}\n"
    )
    source = path / "probe.cpp"
    source.write_text(PROBE)
    return native_cxx.build_executable(
        [source],
        path / "probe",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-flto",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{path}",
            f"-I{ROOT / 'src/runtime/nvidia_host_api'}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ),
        link_args=("-flto", "-Wl,--gc-sections"),
    )


@pytest.mark.parametrize(
    "n,rank", [(0, 0), (1, -1), (2, 0), (3, 1), (7, 6), (17, 17), (65, -1)]
)
@pytest.mark.parametrize(
    "ta,tb", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("scale", [1.0, 0.5, -2.0])
def test_submission_and_drain(
    probe: Path, n: int, rank: int, ta: bool, tb: bool, scale: float
) -> None:
    result = subprocess.run(
        [str(probe), str(n), str(rank), str(int(ta)), str(int(tb)), str(scale)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    a = (np.arange(n * n) % 11 - 4).astype(np.longdouble).reshape((n, n), order="F")
    b = (np.arange(n * n) % 7 - 3).astype(np.longdouble).reshape((n, n), order="F")
    k = n if rank < 0 else rank
    oracle = scale * (a.T if ta else a)[:, :k] @ (b.T if tb else b)[:k, :]
    actual = np.fromstring(result.stdout.strip(), sep=" ").reshape((n, n), order="F")
    np.testing.assert_array_equal(actual, oracle)
