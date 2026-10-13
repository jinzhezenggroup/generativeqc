"""Execute canonical force fallback/status contracts without CUDA or ERI stand-ins."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition, cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_force_fallback_preserves_channels_and_failures(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Run actual wrappers; fake transfers carry labeled channels, not integrals.

    The separate real-GPU Libcint tests own numerical qualification. This probe
    covers resource refusal, ABI shape, coefficient forwarding, validation,
    finite-result rejection, and transactional output on provider failures.
    """
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    helpers = [cpp_record_definition(source, "DirectJkFailure") + ";"]
    helpers += [
        cpp_function_definition(source, name)
        for name in ("direct_jk_check", "direct_jk_require", "direct_jk_finite_result")
    ]
    helpers += [cpp_record_definition(source, "DirectJkDownloadFence") + ";"]
    helpers += [
        cpp_function_definition(source, "direct_jk_guard", include_template=True),
        cpp_function_definition(
            source, "execute_cuda_direct_rsh_energy_derivatives_device"
        ),
        cpp_function_definition(
            source, "execute_cuda_direct_shell_full_range_derivatives_device"
        ),
    ]
    cpp, binary = tmp_path / "fallback.cpp", tmp_path / "fallback"
    cpp.write_text(PREFIX + "\n".join(helpers) + MAIN)
    native_cxx.build_executable(
        [cpp], binary, compile_args=["-std=c++17", "-O0", "-D_GLIBCXX_ASSERTIONS"]
    )
    subprocess.run([binary], check=True, timeout=10)


PREFIX = r"""
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
enum generativeqc_status {
  GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_INVALID_ARGUMENT,
  GENERATIVEQC_STATUS_NOT_IMPLEMENTED, GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_CUDA_ERROR
};
enum cudaError_t { cudaSuccess, cudaErrorUnknown };
using cudaStream_t = void*;
constexpr int cudaMemcpyDeviceToHost = 1;
enum class FockSpin { Restricted, Unrestricted };
struct Exchange { bool force_capability{}; };
struct Diagnostic { unsigned derivative_order{1}; std::size_t batch_size{1}, nbf{2}; };
struct CudaDirectJkPlan {
  int* canonical_pairs{};
  Exchange* generated_exchange{};
  Diagnostic diagnostic;
  std::size_t coordinates_per_item{3}, coordinate_elements{3};
  double* derivative{};
  int device_id{};
  cudaStream_t stream{reinterpret_cast<void*>(1)};
};
struct Download { void* destination; const void* source; std::size_t bytes; };
std::vector<Download> pending;
int fail_download = -1, copies{}, mode{};
bool canonical_used{}, shell_used{}, expected_unrestricted{};
double expected_j{}, expected_k{};
const double* expected_density{};
const double* expected_beta{};
generativeqc_status source_cuda_status(cudaError_t) { return GENERATIVEQC_STATUS_CUDA_ERROR; }
const char* cudaGetErrorString(cudaError_t) { return "injected transfer failure"; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t size, cudaStream_t) {
  std::memset(dst, value, size); return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t size, int, cudaStream_t) {
  if (copies++ == fail_download) return cudaErrorUnknown;
  pending.push_back({dst, src, size}); return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  for (const auto& copy : pending) std::memcpy(copy.destination, copy.source, copy.bytes);
  pending.clear(); return cudaSuccess;
}
// Labeled output only: this is an ABI/provider test, not an integral oracle.
void direct_rsh_derivatives(CudaDirectJkPlan* p, std::size_t item, double j,
    double k, double lr, bool unrestricted, double omega, const double* density,
    const double* beta) {
  canonical_used = true;
  assert(item == 0 && j == expected_j && k == expected_k && lr == 0 && omega == 0);
  assert(unrestricted == expected_unrestricted);
  assert(density == expected_density && beta == expected_beta);
  for (std::size_t i = 0; i < p->coordinates_per_item; ++i) {
    p->derivative[i] = j * double(i + 1);
    p->derivative[p->coordinate_elements + i] = k * double(i + 11);
  }
  if (mode == 1) p->derivative[0] = std::numeric_limits<double>::infinity();
  if (mode == 2) {
    p->derivative[0] = std::numeric_limits<double>::max();
    p->derivative[p->coordinate_elements] = std::numeric_limits<double>::max();
  }
}
namespace cuda_execution {
cudaError_t execute_generated_full_range_energy_derivatives(
    Exchange&, bool, const double*, const double*, double, double,
    std::vector<double>& out, bool separate, bool) {
  shell_used = true; out.assign(separate ? 6 : 3, 42); return cudaSuccess;
}
}
"""

MAIN = r"""
int main() {
  int pair = 1;
  double scratch[9]{}, density[4]{}, beta[4]{};
  CudaDirectJkPlan plan;
  plan.canonical_pairs = &pair; plan.derivative = scratch;
  Exchange unavailable, available{true};
  std::vector<double> output;
  std::string detail;
  auto invoke = [&](CudaDirectJkPlan* selected, bool unrestricted, double j,
                    double k, bool separate, const double* d, const double* b,
                    std::size_t count = 4) {
    output = {123}; detail = "old detail"; pending.clear(); copies = 0;
    canonical_used = shell_used = false;
    expected_j = j; expected_k = k; expected_unrestricted = unrestricted;
    expected_density = d; expected_beta = b;
    return execute_cuda_direct_shell_full_range_derivatives_device(
        selected, unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
        j, k, d, b, count, output, detail, separate, true);
  };
  for (auto* owner : {static_cast<Exchange*>(nullptr), &unavailable}) {
    plan.generated_exchange = owner;
    for (bool unrestricted : {false, true})
      for (bool separate : {false, true})
        for (double j : {0.0, 1.0, -0.7})
          for (double k : {0.0, -0.125, 0.25}) {
            assert(invoke(&plan, unrestricted, j, k, separate, density,
                          unrestricted ? beta : nullptr) == GENERATIVEQC_STATUS_SUCCESS);
            assert(detail.empty() && pending.empty() && !shell_used);
            assert(canonical_used == (j != 0 || k != 0));
            assert(output.size() == (separate ? 6U : 3U));
            for (std::size_t i = 0; i < 3; ++i) {
              assert(output[i] == j * double(i + 1) + (separate ? 0 : k * double(i + 11)));
              if (separate) assert(output[3 + i] == k * double(i + 11));
            }
          }
  }
  auto unchanged = [&](generativeqc_status actual, generativeqc_status expected) {
    assert(actual == expected && !detail.empty() && output == std::vector<double>{123});
    assert(pending.empty());
  };
  unchanged(invoke(nullptr, false, 1, 0, false, density, nullptr),
            GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  plan.canonical_pairs = nullptr;
  unchanged(invoke(&plan, false, 1, 0, false, density, nullptr),
            GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  assert(!canonical_used && !shell_used);
  plan.canonical_pairs = &pair; plan.diagnostic.derivative_order = 0;
  unchanged(invoke(&plan, false, 1, 0, false, density, nullptr),
            GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
  assert(!canonical_used && !shell_used);
  plan.diagnostic.derivative_order = 1; plan.diagnostic.batch_size = 2;
  unchanged(invoke(&plan, false, 1, 0, false, density, nullptr),
            GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  plan.diagnostic.batch_size = 1;
  for (bool separate : {false, true}) {
    unchanged(invoke(&plan, false, 1, 0, separate, nullptr, nullptr),
              GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    unchanged(invoke(&plan, false, 1, 0, separate, density, nullptr, 3),
              GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    unchanged(invoke(&plan, false, 1, 0, separate, density, beta),
              GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    unchanged(invoke(&plan, true, 1, 0, separate, density, nullptr),
              GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    for (double invalid : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
      unchanged(invoke(&plan, false, invalid, 0, separate, density, nullptr),
                GENERATIVEQC_STATUS_INVALID_ARGUMENT);
      unchanged(invoke(&plan, false, 1, invalid, separate, density, nullptr),
                GENERATIVEQC_STATUS_INVALID_ARGUMENT);
      assert(!canonical_used && !shell_used);
    }
    for (int failure : {0, 1, 2}) {
      fail_download = failure;
      unchanged(invoke(&plan, false, 1, -0.125, separate, density, nullptr),
                GENERATIVEQC_STATUS_CUDA_ERROR);
    }
    fail_download = -1; mode = 1;
    unchanged(invoke(&plan, false, 1, -0.125, separate, density, nullptr),
              GENERATIVEQC_STATUS_NUMERICAL_FAILURE);
  }
  mode = 2;
  assert(invoke(&plan, false, 1, 1, true, density, nullptr) == GENERATIVEQC_STATUS_SUCCESS);
  unchanged(invoke(&plan, false, 1, 1, false, density, nullptr),
            GENERATIVEQC_STATUS_NUMERICAL_FAILURE);
  mode = 0; plan.generated_exchange = &available;
  assert(invoke(&plan, false, 1, 0, true, density, nullptr) == GENERATIVEQC_STATUS_SUCCESS);
  assert(shell_used && !canonical_used && output == std::vector<double>(6, 42));
}
"""
