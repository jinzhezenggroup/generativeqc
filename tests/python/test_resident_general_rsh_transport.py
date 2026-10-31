"""Execute resident RSH transport/status contracts with labeled source buffers.

Native CUDA composition tests own the independent integral qualification. This
probe covers borrowed rows, route admission, capture and callback failure drains.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _producer_definitions() -> str:
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    helpers = cpp_record_definition(source, "DirectJkFailure") + ";\n"
    for name in ["direct_jk_check", "direct_jk_require"]:
        helpers += cpp_function_definition(source, name) + "\n"
    helpers += cpp_function_definition(source, "direct_jk_guard", include_template=True)
    for name in [
        "consume_cuda_direct_rsh_energy_derivatives_device",
        "consume_cuda_direct_shell_rsh_energy_derivatives_device",
    ]:
        helpers += "\n" + cpp_function_definition(source, name)
    return helpers


def test_general_rsh_borrowing_routing_and_failures(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    cpp, binary = tmp_path / "resident.cpp", tmp_path / "resident"
    cpp.write_text(PREFIX + _producer_definitions() + MAIN)
    native_cxx.build_executable(
        [cpp], binary, compile_args=["-std=c++20", "-O0", "-I" + str(ROOT / "src")]
    )
    subprocess.run([binary], check=True, timeout=10)


@pytest.mark.parametrize("spin_blocks", (1, 2))
def test_general_rsh_forwards_into_actual_stationary_consumer(
    tmp_path: Path, native_cxx: NativeCxx, spin_blocks: int
) -> None:
    """Join the production producer and receiver; double only CUDA side effects."""
    transport = (ROOT / "src/dft/native_stationary_runtime.cuh").read_text()
    executor = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    receiver = cpp_record_definition(transport, "NativeOwner") + ";\n"
    receiver += cpp_function_definition(executor, "guarded", include_template=True)
    for name in ("native_drain_integrals", "native_consume_integrals"):
        receiver += "\n" + cpp_function_definition(transport, name)
    cpp, binary = tmp_path / "stationary.cpp", tmp_path / "stationary"
    cpp.write_text(
        f"#define TEST_SPIN_BLOCKS {spin_blocks}\n"
        + PREFIX
        + STATIONARY_PREFIX
        + receiver
        + _producer_definitions()
        + STATIONARY_MAIN
    )
    native_cxx.build_executable(
        [cpp], binary, compile_args=["-std=c++20", "-O0", "-I" + str(ROOT / "src")]
    )
    subprocess.run([binary], check=True, timeout=10)


PREFIX = r"""
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include "scf/cuda_derivative_sources.hpp"
using namespace generativeqc::scf;
enum generativeqc_status { GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_INVALID_ARGUMENT,
  GENERATIVEQC_STATUS_NOT_IMPLEMENTED, GENERATIVEQC_STATUS_CUDA_ERROR,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_NUMERICAL_FAILURE };
namespace generativeqc {
class Error : public std::runtime_error {
  generativeqc_status code_;
 public:
  Error(generativeqc_status code, const char* text) : std::runtime_error(text), code_(code) {}
  generativeqc_status status() const { return code_; }
};
}
enum cudaError_t { cudaSuccess, cudaErrorUnknown };
using cudaStream_t = void*;
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive };
enum class FockSpin { Restricted, Unrestricted };
struct Exchange { bool force_capability{true}; };
struct Diagnostic { std::size_t batch_size{1}, nbf{2}; unsigned derivative_order{1}; };
struct CudaDirectJkPlan {
  int device_id{2};
  cudaStream_t stream{reinterpret_cast<void*>(5)};
  Diagnostic diagnostic;
  Exchange* generated_exchange{};
  double* derivative{};
  std::size_t coordinates_per_item{3}, coordinate_elements{3};
};
cudaStreamCaptureStatus capture_state = cudaStreamCaptureStatusNone;
unsigned kernels{}, shell_calls{}, clears{}, drains{}, callbacks{}, pending{};
cudaError_t device_error = cudaSuccess, capture_error = cudaSuccess;
const double* expected_alpha{};
const double* expected_beta{};
double expected_omega{};
generativeqc_status source_cuda_status(cudaError_t) { return GENERATIVEQC_STATUS_CUDA_ERROR; }
const char* cudaGetErrorString(cudaError_t) { return "injected driver failure"; }
cudaError_t cudaSetDevice(int device) { assert(device == 2); return device_error; }
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* state) {
  *state = capture_state; return capture_error;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) { ++drains; pending = 0; return cudaSuccess; }
cudaError_t cudaMemsetAsync(void* values, int value, std::size_t bytes, cudaStream_t) {
  ++clears; std::memset(values, value, bytes); return cudaSuccess;
}
namespace runtime::cuda_trace {
struct Shape { std::size_t batch, nbf, naux; bool derivative, streamed; };
struct TraceOperation { TraceOperation(const char*, cudaStream_t, Shape) {} };
void trace_counter(const char*, std::size_t) {}
}
void direct_rsh_derivatives(CudaDirectJkPlan* plan, std::size_t item, double cj,
    double cs, double cl, bool unrestricted, double omega, const double* alpha, const double* beta) {
  assert(item == 0 && omega == expected_omega && alpha == expected_alpha && beta == expected_beta);
  assert(unrestricted ? beta != nullptr : beta == nullptr);
  ++kernels;
  const double coefficients[]{cj, cs, cl};
  for (unsigned source = 0; source < 3; ++source)
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate)
      plan->derivative[3*source + coordinate] = coefficients[source] * (coordinate + 1);
}
namespace cuda_execution {
cudaError_t consume_generated_rsh_energy_derivatives(Exchange&, bool, const double*,
    const double*, double cj, double cs, double, double omega,
    const CudaDerivativeSourceConsumer& consumer) {
  assert(cj == 0 && cs == 0 && omega == 0.3);
  ++shell_calls;
  static double force[9]{};
  consumer({CudaDerivativeSourceKind::RangeSeparatedJk, CudaDerivativeConvention::Force,
            2, reinterpret_cast<void*>(5), force, 3, 3});
  return cudaSuccess;
}
}
"""

MAIN = r"""
int main() {
  Exchange exchange;
  double density[4]{}, beta[4]{}, scratch[9]{};
  CudaDirectJkPlan plan; plan.generated_exchange = &exchange; plan.derivative = scratch;
  std::string detail;
  auto receive = [&](const CudaDerivativeSourceView& view) {
    assert(view.kind == CudaDerivativeSourceKind::RangeSeparatedJk && view.channels == 3);
    assert(view.device == 2 && view.producer_stream == plan.stream && view.coordinates == 3);
    ++callbacks;
  };
  for (bool unrestricted : {false, true})
    for (double omega : {0.3, 0.4, 0.5}) {
      expected_alpha = density; expected_beta = unrestricted ? beta : nullptr; expected_omega = omega;
      const auto old_drains = drains;
      const auto old_kernels = kernels;
      assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
          &plan, unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
          1, -0.1, -0.25, omega, density, unrestricted ? beta : nullptr, 4,
          [&](const auto& view) {
            receive(view);
            assert(view.values == scratch && view.convention == CudaDerivativeConvention::EnergyGradient);
            assert(view.values[0] == 1 && view.values[3] == -0.1 && view.values[6] == -0.25);
            pending = 1;
          }, detail) == GENERATIVEQC_STATUS_SUCCESS);
      assert(kernels == old_kernels + 1 && !shell_calls && drains == old_drains && pending);
      cudaStreamSynchronize(plan.stream);
    }
  assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Restricted, 0, 0, -0.25, 0.3, density, nullptr, 4,
      [&](const auto& view) { receive(view); assert(view.convention == CudaDerivativeConvention::Force); },
      detail) == GENERATIVEQC_STATUS_SUCCESS && shell_calls == 1);
  plan.generated_exchange = nullptr; // General fallback needs no optional shell allocation.
  expected_alpha = density; expected_beta = nullptr; expected_omega = 0.4;
  const auto old_kernels = kernels;
  assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Restricted, 0, 0, 0, 0.4, density, nullptr, 4,
      [&](const auto& view) { receive(view); for (unsigned i=0;i<9;++i) assert(view.values[i] == 0); },
      detail) == GENERATIVEQC_STATUS_SUCCESS && kernels == old_kernels);
  for (auto status : {GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                     GENERATIVEQC_STATUS_NUMERICAL_FAILURE, GENERATIVEQC_STATUS_INVALID_ARGUMENT}) {
    const auto old_drains = drains;
    assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
        &plan, FockSpin::Restricted, 1, -0.1, -0.25, 0.4, density, nullptr, 4,
        [&](const auto& view) { receive(view); pending = 1; throw generativeqc::Error(status,"typed failure"); },
        detail) == status);
    assert(drains == old_drains + 1 && !pending);
  }
  const auto before_capture = std::array{kernels, clears, callbacks, drains};
  capture_state = cudaStreamCaptureStatusActive;
  assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Restricted, 1, -0.1, -0.25, 0.4, density, nullptr, 4, receive, detail) ==
      GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  assert((before_capture == std::array{kernels, clears, callbacks, drains}));
  capture_state = cudaStreamCaptureStatusNone;
  const auto before_invalid = std::array{kernels, clears, callbacks};
  assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Unrestricted, 1, -0.1, -0.25, 0.4, density, nullptr, 4, receive, detail) ==
      GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  assert((before_invalid == std::array{kernels, clears, callbacks}));
  assert(consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Restricted, 1, -0.1, -0.25, 0.4, density, nullptr, 4, receive, detail) ==
      GENERATIVEQC_STATUS_SUCCESS); // Recovery after capture/input/callback refusals.
}
"""


STATIONARY_PREFIX = r"""
#include <cstdio>
#include <cstdint>
struct DeviceAllocationError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DeviceRuntimeError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Owner {
  struct { int device{2}; int* error{}; } context;
  bool failed{};
  std::size_t atoms{1}, launches{};
  double* sources{};
};
constexpr std::size_t stationary_spin_blocks = TEST_SPIN_BLOCKS;
constexpr std::size_t native_one_electron_slot = 0, native_overlap_slot = 1,
                      native_coulomb_slot = 2, native_exchange_storage_slot = 3;
constexpr int native_exchange_slot = -1, native_short_exchange_slot = 3,
              native_long_exchange_slot = 4;
constexpr uint64_t native_exchange_mask = 0;
constexpr double native_short_exchange = 0.1, native_long_exchange = 0.35;
void check(Owner& owner) { if (owner.failed) throw std::runtime_error("failed owner"); }
void finished(Owner&, cudaStream_t stream) { cudaStreamSynchronize(stream); }
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw DeviceRuntimeError("injected device failure");
}
void error_text(char* output, std::size_t size, const char* message) {
  if (output && size) std::snprintf(output, size, "%s", message);
}
namespace generativeqc::dft {
cudaError_t launch_native_stationary_source_transport(
    const double* input, std::size_t coordinates, std::size_t channels, double sign,
    double* first, double* second, int*, cudaStream_t stream) noexcept {
  assert(stream == reinterpret_cast<void*>(5));
  for (std::size_t i = 0; i < coordinates; ++i) {
    first[i] = sign * input[i];
    if (channels == 2 && second) second[i] = sign * input[coordinates + i];
  }
  return cudaSuccess;
}
cudaError_t launch_native_stationary_range_correction(
    const double*, std::size_t, double, double, double*, double*, int*, cudaStream_t) noexcept {
  std::abort(); // This fixture forwards complete RSH sources, not a correction.
}
}
"""


STATIONARY_MAIN = r"""
int main() {
  constexpr bool unrestricted = TEST_SPIN_BLOCKS == 2;
  constexpr double coefficients[]{1, -0.1, -0.25};
  double density[4]{}, beta[4]{}, scratch[9]{};
  CudaDirectJkPlan plan; plan.derivative = scratch;
  expected_alpha = density;
  expected_beta = unrestricted ? beta : nullptr;
  for (double omega : {0.3, 0.4, 0.5}) {
    expected_omega = omega;
    double output[15]{};
    Owner scientific; scientific.sources = output;
    NativeOwner owner; owner.scientific = &scientific;
    char error[128]{};
    std::string detail;
    const auto status = consume_cuda_direct_shell_rsh_energy_derivatives_device(
        &plan, unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
        coefficients[0], coefficients[1], coefficients[2], omega,
        density, expected_beta, 4, [&](const auto& view) {
          const int result = native_consume_integrals(&owner, &view, error, sizeof(error));
          if (result) throw generativeqc::Error(GENERATIVEQC_STATUS_INVALID_ARGUMENT, error);
        }, detail);
    assert(status == GENERATIVEQC_STATUS_SUCCESS);
    assert(owner.integral_mask == ((1U << 2) | (1U << 3) | (1U << 4)));
    assert(owner.integral_pending && owner.integral_stream == plan.stream);
    assert(scientific.launches == 3);
    for (unsigned channel = 0; channel < 3; ++channel)
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        assert(output[(2 + channel)*3 + coordinate] == coefficients[channel]*(coordinate + 1));
        scratch[channel*3 + coordinate] = 42; // Provider may reuse its borrowed rows.
      }
    assert(output[6] == 1 && output[9] == -0.1 && output[12] == -0.25);
    const auto before_drain = drains;
    native_drain_integrals(owner);
    assert(drains == before_drain + 1 && !owner.integral_pending);
  }
  // Existing force-sign producers stay valid, and invalid metadata fails before
  // any source is published. Neither path may silently flip positive gradients.
  for (bool force : {false, true}) {
    double output[15]{};
    Owner scientific; scientific.sources = output;
    NativeOwner owner; owner.scientific = &scientific;
    char error[128]{};
    for (unsigned i = 0; i < 9; ++i) scratch[i] = i + 1;
    CudaDerivativeSourceView view{
        CudaDerivativeSourceKind::RangeSeparatedJk,
        force ? CudaDerivativeConvention::Force : CudaDerivativeConvention::EnergyGradient,
        2, plan.stream, scratch, 3, 3};
    assert(native_consume_integrals(&owner, &view, error, sizeof(error)) == 0);
    for (unsigned i = 0; i < 9; ++i)
      assert(output[6 + i] == (force ? -1.0 : 1.0) * scratch[i]);
    assert(native_consume_integrals(&owner, &view, error, sizeof(error)) != 0);
  }
  for (bool invalid_channels : {false, true}) {
    double output[15]{};
    Owner scientific; scientific.sources = output;
    NativeOwner owner; owner.scientific = &scientific;
    char error[128]{};
    const CudaDerivativeSourceView view{
        CudaDerivativeSourceKind::RangeSeparatedJk,
        invalid_channels ? CudaDerivativeConvention::EnergyGradient
                         : static_cast<CudaDerivativeConvention>(99),
        2, plan.stream, scratch, 3, invalid_channels ? 2U : 3U};
    assert(native_consume_integrals(&owner, &view, error, sizeof(error)) != 0);
    assert(!owner.integral_mask && !owner.integral_pending && !scientific.launches);
    for (double value : output) assert(value == 0);
  }
}
"""
