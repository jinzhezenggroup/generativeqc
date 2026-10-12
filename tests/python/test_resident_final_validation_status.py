"""Host fault controls for the resident CUDA final-proof status boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, start: str) -> str:
    begin = source.index(start)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_resident_proof_preserves_driver_and_allocation_status(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute production status conversion and public catch ordering on the host."""
    validation = (ROOT / "src/scf/cuda/resident_final_validation.cpp").read_text()
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    policy = "\n".join(
        _definition(validation, symbol)
        for symbol in ("void check(cudaError_t", "void check_blas(cublasStatus_t")
    )
    boundary = _definition(
        ks, "generativeqc_status CudaKsPlan::prepare_resident_final_stationary_weights("
    )
    source = tmp_path / "resident_proof_status.cpp"
    source.write_text(PREFIX + policy + OWNER + boundary + MAIN)
    binary = tmp_path / "resident_proof_status"
    native_cxx.build_executable(
        [source],
        binary,
        compile_args=["-std=c++20", "-O2", "-I", str(ROOT / "include")],
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include "generativeqc/generativeqc.hpp"
using generativeqc::Error;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorLaunchFailure };
enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED
};
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA status"; }
"""

OWNER = r"""
struct CudaKsFinalStateToken {};
struct CudaKsResidentStationaryWeightsBinding { unsigned ready{}; };
struct KsFinalStateDiagnostic { unsigned accepted{}; };
struct Impl {
  unsigned failure{};
  KsFinalStateDiagnostic prepare_resident_weights(const CudaKsFinalStateToken&, std::string&) {
    switch (failure) {
      case 1: check(cudaErrorMemoryAllocation); break;
      case 2: check(cudaErrorLaunchFailure); break;
      case 3: check_blas(CUBLAS_STATUS_ALLOC_FAILED); break;
      case 4: check_blas(CUBLAS_STATUS_EXECUTION_FAILED); break;
      case 5: throw std::runtime_error("nonfinite final-state products");
      case 6: throw std::invalid_argument("stale final-state token");
      case 7: throw std::bad_alloc();
      default: check(cudaSuccess); check_blas(CUBLAS_STATUS_SUCCESS);
    }
    return {42};
  }
};
class CudaKsPlan {
 public:
  Impl* impl_;
  generativeqc_status resident_final_stationary_weights(
      const CudaKsFinalStateToken&, CudaKsResidentStationaryWeightsBinding& output,
      std::string&) {
    output.ready = 1;
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  generativeqc_status prepare_resident_final_stationary_weights(
      const CudaKsFinalStateToken&, CudaKsResidentStationaryWeightsBinding&,
      KsFinalStateDiagnostic&, std::string&);
};
"""

MAIN = r"""
int main() {
  Impl impl;
  CudaKsPlan plan{&impl};
  const generativeqc_status expected[]{
      GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
      GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
      GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
      GENERATIVEQC_STATUS_INVALID_ARGUMENT, GENERATIVEQC_STATUS_OUT_OF_MEMORY};
  for (unsigned mode = 0; mode < sizeof(expected) / sizeof(expected[0]); ++mode) {
    impl.failure = mode;
    CudaKsResidentStationaryWeightsBinding weights{99};
    KsFinalStateDiagnostic diagnostic{99};
    std::string detail = "stale diagnostic";
    const auto status = plan.prepare_resident_final_stationary_weights(
        {}, weights, diagnostic, detail);
    if (status != expected[mode]) return 10 + mode;
    if (!mode) {
      if (weights.ready != 1 || diagnostic.accepted != 42 || !detail.empty()) return 20;
    } else {
      if (weights.ready || diagnostic.accepted || detail.empty()) return 30 + mode;
      impl.failure = 0;
      if (plan.prepare_resident_final_stationary_weights({}, weights, diagnostic, detail) !=
              GENERATIVEQC_STATUS_SUCCESS ||
          weights.ready != 1 || diagnostic.accepted != 42) return 40 + mode;
    }
  }
}
"""
