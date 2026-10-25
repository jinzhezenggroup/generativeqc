"""Execute the resident RSH adapter's capture and typed callback boundaries."""

import subprocess
from pathlib import Path

import pytest
from _cpp_source_support import cpp_function_definition
from test_coulomb_optional_allocation import compile_cached_probe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rsh_adapter_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    definitions = "template <class Function>\n" + "\n".join(
        cpp_function_definition(source, name)
        for name in (
            "direct_jk_guard",
            "consume_cuda_direct_shell_rsh_energy_derivatives_device",
        )
    )
    directory = tmp_path_factory.mktemp("rsh-callback")
    cpp, binary = directory / "contract.cpp", directory / "contract"
    cpp.write_text(
        r"""
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive,
                               cudaStreamCaptureStatusInvalidated };
enum generativeqc_status { GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
    GENERATIVEQC_STATUS_INVALID_ARGUMENT, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
    GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_NUMERICAL_FAILURE };
namespace generativeqc {
class Error : public std::runtime_error {
  generativeqc_status status_;
 public:
  Error(generativeqc_status status, const char* text) : std::runtime_error(text), status_(status) {}
  generativeqc_status status() const { return status_; }
};
}
struct DirectJkFailure { generativeqc_status status; std::string detail; };
int submissions{}, callbacks{}, drains{};
cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
cudaError_t capture_error = cudaSuccess;
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* result) {
  *result = capture; return capture_error;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) { ++drains; return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
generativeqc_status source_cuda_status(cudaError_t error) {
  return error == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                            : GENERATIVEQC_STATUS_CUDA_ERROR;
}
void direct_jk_check(cudaError_t error) { if (error != cudaSuccess) throw error; }
void direct_jk_require(bool valid, const char* text) {
  if (!valid) throw DirectJkFailure{GENERATIVEQC_STATUS_INVALID_ARGUMENT, text};
}
enum class FockSpin { Restricted, Unrestricted };
using CudaDerivativeSourceConsumer = std::function<void(int)>;
struct GeneratedExchangePlan { bool force_capability{true}; };
struct CudaDirectJkPlan {
  std::unique_ptr<GeneratedExchangePlan> generated_exchange{new GeneratedExchangePlan};
  struct { std::size_t batch_size{1}, derivative_order{1}, nbf{1}; } diagnostic;
  int device_id{};
  cudaStream_t stream{reinterpret_cast<void*>(1)};
};
namespace cuda_execution {
cudaError_t consume_generated_rsh_energy_derivatives(
    GeneratedExchangePlan&, bool, const double*, const double*, double, double, double, double,
    const CudaDerivativeSourceConsumer& consumer) {
  ++submissions;
  consumer(1);
  return cudaSuccess;
}
}
"""
        + definitions
        + r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const int scenario = std::atoi(argv[1]);
  CudaDirectJkPlan plan;
  double density = 1;
  std::string detail;
  if (scenario == 1) capture = cudaStreamCaptureStatusActive;
  if (scenario == 2) capture = cudaStreamCaptureStatusInvalidated;
  if (scenario == 3) capture_error = cudaErrorUnknown;
  const auto consumer = [&](int) {
    ++callbacks;
    if (scenario == 4) throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, "typed driver");
    if (scenario == 5) throw generativeqc::Error(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "typed numeric");
    if (scenario == 6) throw std::bad_alloc();
    if (scenario == 7) throw std::invalid_argument("bad source");
  };
  const auto status = consume_cuda_direct_shell_rsh_energy_derivatives_device(
      &plan, FockSpin::Restricted, 1.0, 0.2, 1.0, 0.3, &density, nullptr, 1, consumer, detail);
  if (scenario >= 1 && scenario <= 3) {
    assert(status == (scenario == 3 ? GENERATIVEQC_STATUS_CUDA_ERROR
                                    : GENERATIVEQC_STATUS_INVALID_ARGUMENT));
    assert(submissions == 0 && callbacks == 0 && drains == 0);
  } else if (scenario == 0) {
    assert(status == GENERATIVEQC_STATUS_SUCCESS && submissions == 1 && callbacks == 1);
    assert(detail.empty());
  } else {
    const auto expected = scenario == 4 ? GENERATIVEQC_STATUS_CUDA_ERROR
                        : scenario == 5 ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
                        : scenario == 6 ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                        : GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    assert(status == expected && callbacks == 1 && submissions == 1 && drains > 0);
    if (scenario == 4) assert(detail == "typed driver");
    if (scenario == 5) assert(detail == "typed numeric");
  }
}
"""
    )
    compile_cached_probe(cpp, binary)
    return binary


@pytest.mark.parametrize("scenario", range(8))
def test_resident_rsh_callback_status_and_capture(
    rsh_adapter_probe: Path, scenario: int
) -> None:
    subprocess.run([str(rsh_adapter_probe), str(scenario)], check=True, timeout=10)
