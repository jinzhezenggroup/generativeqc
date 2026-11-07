"""Execute combined-source admission with host doubles, without CUDA numerics."""

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner
from test_rhf_resident_retry import _definition

ROOT = Path(__file__).resolve().parents[2]


def test_compact_mask_refusal_preserves_raw_source(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    direct = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    wrapper = _definition(
        direct, "generativeqc_status prepare_cuda_direct_jk_combined_range_values("
    )
    source = tmp_path / "combined_admission.cpp"
    source.write_text(PREFIX + wrapper + MAIN)
    executable = tmp_path / "combined_admission"
    compile_owner(compiler, tmp_path, [source], executable)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
enum generativeqc_status { GENERATIVEQC_STATUS_SUCCESS,
  GENERATIVEQC_STATUS_INVALID_ARGUMENT, GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_NUMERICAL_FAILURE };
enum class FockOperator { FullRange };
struct CudaDirectJkPlan {
  double resident_full_exchange_scale{};
  std::uint32_t resident_bucket_mask{};
  struct { std::size_t resident_values_submitted{11}, resident_values_completed{11}; }
    diagnostic;
};
static int calls{};
static bool mutate{};
static generativeqc_status outcome = GENERATIVEQC_STATUS_SUCCESS;
static void require(bool v) { if (!v) throw std::runtime_error("admission invariant"); }
static generativeqc_status prepare_cuda_direct_jk_range_resident_values_impl(
    CudaDirectJkPlan* plan, std::size_t, std::uint32_t mask, FockOperator,
    double, std::string&, double scale) {
  ++calls;
  // Model the existing producer's immutable layout requirement.
  if (mask != plan->resident_bucket_mask) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (mutate) plan->resident_full_exchange_scale = scale;
  return outcome;
}
"""

MAIN = r"""
int main() {
  std::string detail;
  // Complete all-s and through-d selections need not include empty buckets.
  for (const std::uint32_t mask : {1U, (1U << 15) - 1}) {
    CudaDirectJkPlan plan;
    plan.resident_bucket_mask = mask;
    require(prepare_cuda_direct_jk_combined_range_values(&plan, 0.5, 4096, detail)
            == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    require(calls == 0 && plan.resident_full_exchange_scale == 0.0);
    require(plan.diagnostic.resident_values_submitted == 0 &&
            plan.diagnostic.resident_values_completed == 0);
  }
  CudaDirectJkPlan full;
  full.resident_bucket_mask = (1U << 28) - 1;
  require(prepare_cuda_direct_jk_combined_range_values(&full, 0.5, 4096, detail)
          == GENERATIVEQC_STATUS_SUCCESS);
  require(calls == 1);
  outcome = GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  require(prepare_cuda_direct_jk_combined_range_values(&full, 0.5, 4096, detail)
          == GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  mutate = true;
  require(prepare_cuda_direct_jk_combined_range_values(&full, 0.5, 4096, detail)
          == GENERATIVEQC_STATUS_NUMERICAL_FAILURE);
  const auto previous_calls = calls;
  require(prepare_cuda_direct_jk_combined_range_values(&full, 1.0, 4096, detail)
          == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require(calls == previous_calls);
}
"""
