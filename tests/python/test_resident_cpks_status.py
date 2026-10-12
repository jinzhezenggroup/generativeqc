"""Host fault controls for native resident CPKS status and stream lifetimes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, start: str) -> str:
    begin = source.index(start)
    opening, parentheses = begin, 0
    while True:
        character = source[opening]
        parentheses += (character == "(") - (character == ")")
        if character == "{" and not parentheses:
            break
        opening += 1
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.mark.parametrize(
    "mutant", ("none", "cuda", "blas", "handoff", "missing_drain")
)
def test_resident_cpks_fault_status_and_cleanup(
    tmp_path: Path, native_cxx: NativeCxx, mutant: str
) -> None:
    """Execute production creation, iteration and cross-stream callback boundaries."""
    fock = (ROOT / "src/api/c_api_fock.cpp").read_text()
    snapshot = (ROOT / "src/api/c_api_ks_snapshot.cpp").read_text()
    resources = (ROOT / "src/runtime/cuda_resources.cuh").read_text()
    errors = (ROOT / "src/api/error.cpp").read_text()
    methods = (ROOT / "src/methods/method.hpp").read_text()
    policy = "\n".join(
        _definition(fock, symbol)
        for symbol in (
            "void resident_cuda(",
            "void resident_blas(",
            "std::size_t resident_product(",
            "std::size_t resident_sum(",
            "double* resident_slot(",
            "template <class Function>\ngenerativeqc_status resident_guard(",
            "void resident_sync(",
        )
    )
    if mutant == "cuda":
        policy = policy.replace(
            _definition(fock, "void resident_cuda("),
            "void resident_cuda(cudaError_t status) { "
            "if (status != cudaSuccess) throw std::runtime_error(\"old CUDA flattening\"); }",
        )
    if mutant == "blas":
        policy = policy.replace(
            _definition(fock, "void resident_blas("),
            "void resident_blas(cublasStatus_t status) { "
            "if (status != CUBLAS_STATUS_SUCCESS) "
            "throw std::runtime_error(\"old BLAS flattening\"); }",
        )
    handoff_policy = _definition(snapshot, "void check_cpks_cuda(")
    if mutant == "handoff":
        handoff_policy = (
            "void check_cpks_cuda(cudaError_t status) { cuda_resource_check(status); }"
        )
    start = snapshot.index(
        "    generativeqc::runtime::CudaDeviceScope device(response->device, check_cpks_cuda);"
    )
    end = snapshot.index(
        "    generativeqc_rhf_response_resident* raw_owner = nullptr;", start
    )
    handoff = snapshot[start:end]
    if mutant == "missing_drain":
        handoff = handoff.replace("response->stream.synchronize();", "(void)0;")
    source = (
        PREFIX
        + "namespace generativeqc::methods {\n"
        + _definition(methods, "class MethodError final")
        + ";\n}\nnamespace generativeqc::api {\n"
        + _definition(errors, "generativeqc_status map_exception(")
        + "\n}\n"
        + FAKE_PROVIDER
        + _definition(fock, "struct generativeqc_rhf_response_resident")
        + ";\n"
        + policy
        + "\n"
        + _definition(fock, "static generativeqc_status create_response_resident(")
        + "\n"
        + _definition(
            fock, 'extern "C" void generativeqc_rhf_response_resident_destroy('
        )
        + "\n"
        + _definition(
            fock, 'extern "C" generativeqc_status generativeqc_rhf_response_resident_norm('
        )
        + "\nnamespace generativeqc::runtime {\n"
        + _definition(resources, "inline void cuda_resource_check(")
        + "\n"
        + _definition(resources, "class CudaDeviceScope")
        + ";\n"
        + _definition(resources, "class OwnedCudaEvent")
        + ";\n}\nusing generativeqc::runtime::cuda_resource_check;\n"
        + handoff_policy
        + CALLBACK_OWNER
        + "\nconst double* handoff(CallbackOwner* response, Batch* batch) {\n"
        + handoff
        + "return density_response(response->density, reinterpret_cast<void*>(1));\n}\n"
        + MAIN
    )
    path = tmp_path / "resident_cpks_status.cpp"
    path.write_text(source)
    binary = tmp_path / "resident_cpks_status"
    native_cxx.build_executable(
        [path],
        binary,
        compile_args=["-std=c++20", "-O0", "-I", str(ROOT / "include")],
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    if mutant == "none":
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, f"{mutant} escaped the executable controls"


PREFIX = r"""

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "generativeqc/generativeqc.hpp"
#define GENERATIVEQC_HAS_CUDA 1
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED
};
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cublasHandle_t = void*;
constexpr unsigned cudaEventDefault = 0, cudaEventDisableTiming = 2;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
constexpr int CUBLAS_POINTER_MODE_HOST = 0;
enum Site {
  none, get_device, set_device, allocate, upload, stream_sync, blas_create,
  blas_stream, blas_pointer, blas_norm, event_create, event_record, event_sync,
  event_elapsed, stream_wait
};
Site fault_site = none;
cudaError_t cuda_failure = cudaErrorUnknown;
cublasStatus_t blas_failure = CUBLAS_STATUS_EXECUTION_FAILED;
int current_device = 3, live_allocations = 0, live_blas = 0, live_events = 0;
int fences = 0, releases = 0;
bool bad_norm = false;
cudaError_t cuda_status(Site site) {
  if (fault_site != site) return cudaSuccess;
  fault_site = none;
  return cuda_failure;
}
cublasStatus_t blas_status(Site site) {
  if (fault_site != site) return CUBLAS_STATUS_SUCCESS;
  fault_site = none;
  return blas_failure;
}
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
cudaError_t cudaGetDevice(int* device) {
  auto status = cuda_status(get_device);
  if (status == cudaSuccess) *device = current_device;
  return status;
}
cudaError_t cudaSetDevice(int device) {
  auto status = cuda_status(set_device);
  if (status == cudaSuccess) current_device = device;
  return status;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++fences;
  return cuda_status(stream_sync);
}
cudaError_t cudaMemcpyAsync(void* output, const void* input, std::size_t count, int,
                            cudaStream_t) {
  auto status = cuda_status(upload);
  if (status == cudaSuccess) std::memcpy(output, input, count);
  return status;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
  auto status = cuda_status(event_create);
  if (status == cudaSuccess) { *event = new int(0); ++live_events; }
  return status;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
  delete static_cast<int*>(event);
  --live_events;
  return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cuda_status(event_record); }
cudaError_t cudaEventSynchronize(cudaEvent_t) { return cuda_status(event_sync); }
cudaError_t cudaEventElapsedTime(float* value, cudaEvent_t, cudaEvent_t) {
  *value = 1;
  return cuda_status(event_elapsed);
}
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) {
  return cuda_status(stream_wait);
}
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
  auto status = blas_status(blas_create);
  if (status == CUBLAS_STATUS_SUCCESS) { *handle = new int(0); ++live_blas; }
  return status;
}
cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  delete static_cast<int*>(handle);
  --live_blas;
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) {
  return blas_status(blas_stream);
}
cublasStatus_t cublasSetPointerMode(cublasHandle_t, int) {
  return blas_status(blas_pointer);
}
cublasStatus_t cublasDnrm2(cublasHandle_t, int, const double*, int, double* value) {
  *value = bad_norm ? std::numeric_limits<double>::infinity() : 2;
  return blas_status(blas_norm);
}
namespace generativeqc::runtime {
cudaError_t resource_cuda_malloc(void** output, std::size_t bytes) {
  auto status = cuda_status(allocate);
  if (status == cudaSuccess) { *output = std::calloc(1, bytes); ++live_allocations; }
  return status;
}
cudaError_t resource_cuda_free(void* pointer) {
  std::free(pointer);
  --live_allocations;
  ++releases;
  return cudaSuccess;
}
}
"""

FAKE_PROVIDER = r"""

namespace generativeqc::scf {
enum class FockBackend { Cuda };
enum class FockSpin { Restricted };
enum class FockApproximation { Exact };
enum class FockOperator { FullRange };
struct Term {
  bool present = true;
  double coefficient = 1, omega = 0;
  FockApproximation approximation = FockApproximation::Exact;
  FockOperator op = FockOperator::FullRange;
};
struct FockBuildSpec {
  FockSpin spin = FockSpin::Restricted;
  Term coulomb{}, exchange{false, 0};
  unsigned derivative_order = 0;
};
struct Strategy {
  FockBackend backend = FockBackend::Cuda;
  FockBuildSpec spec;
  double screening_tolerance = 0;
};
struct CudaDirectJkPlan {};
struct PreparedFockPlan {
  Strategy options;
  CudaDirectJkPlan direct;
  const Strategy& strategy() const { return options; }
  CudaDirectJkPlan* cuda_direct_source() const {
    return const_cast<CudaDirectJkPlan*>(&direct);
  }
  struct OneElectron { std::size_t nbf = 3; };
  OneElectron one_electron() const { return {}; }
};
int cuda_direct_jk_device(CudaDirectJkPlan*) { return 0; }
cudaStream_t cuda_direct_jk_stream(CudaDirectJkPlan*) {
  return reinterpret_cast<void*>(1);
}
}
namespace generativeqc::api {
using CudaDensityResponseAction = std::function<const double*(const double*, void*)>;
}
struct generativeqc_fock_plan {
  std::unique_ptr<generativeqc::scf::PreparedFockPlan> source =
      std::make_unique<generativeqc::scf::PreparedFockPlan>();
  std::string detail;
};
void require(bool condition, const char* detail) {
  if (!condition) throw std::invalid_argument(detail);
}
"""

CALLBACK_OWNER = r"""

bool borrowed = false;
int callback_mode = 0, callback_drains = 0;
bool cleanup_failure = false, stale = false;
struct Batch {};
generativeqc_status check_current(Batch&, std::size_t, int) {
  return stale ? GENERATIVEQC_STATUS_INVALID_ARGUMENT : GENERATIVEQC_STATUS_SUCCESS;
}
struct CallbackStream {
  cudaStream_t get() const { return reinterpret_cast<void*>(2); }
  void synchronize() const {
    ++callback_drains;
    borrowed = false;
    if (cleanup_failure) throw std::runtime_error("secondary cleanup failure");
  }
};
struct CallbackPlan {
  struct Scalars { int error{}; };
  struct View { const double* potential{}; };
  double output[9]{};
  unsigned transfers() const { return 0; }
  void enqueue_response(const double*, const double*, std::size_t, std::uint64_t) {
    borrowed = true;
    if (callback_mode == 1)
      throw generativeqc::methods::MethodError(GENERATIVEQC_STATUS_CUDA_ERROR,
                                               "XC enqueue failure after borrowing");
    if (callback_mode == 2) throw std::bad_alloc();
    if (callback_mode == 3) throw std::runtime_error("scientific numerical failure");
  }
  Scalars read_scalars(std::uint64_t) {
    if (callback_mode == 5)
      throw generativeqc::methods::MethodError(GENERATIVEQC_STATUS_CUDA_ERROR,
                                               "XC readback failure");
    borrowed = false;
    return {callback_mode == 4 ? 1 : 0};
  }
  View view(std::uint64_t) { return {output}; }
};
struct CallbackOwner {
  int device = 0, token = 1;
  std::size_t index = 0, count = 9;
  std::uint64_t generation = 0;
  double values[9]{};
  const double* density = values;
  CallbackStream stream;
  std::unique_ptr<CallbackPlan> plan = std::make_unique<CallbackPlan>();
};
"""

MAIN = r"""

template <class Function>
generativeqc_status mapped(Function function) {
  try { function(); return GENERATIVEQC_STATUS_SUCCESS; }
  catch (...) {
    std::string detail;
    return generativeqc::api::map_exception(&detail);
  }
}
generativeqc_status create(generativeqc_fock_plan& plan,
                           generativeqc_rhf_response_resident** output) {
  const double coefficients[9]{1,0,0,0,1,0,0,0,1}, energies[3]{-1,1,2};
  return create_response_resident(
      &plan, coefficients, 9, energies, 3, 1, 12, 1U << 20, output,
      [](const double* density, void*) { return density; });
}
int main() {
  generativeqc_fock_plan plan;
  for (auto site : {set_device, allocate, upload, stream_sync}) {
    for (auto failure : {cudaErrorMemoryAllocation, cudaErrorUnknown}) {
      fault_site = site;
      cuda_failure = failure;
      generativeqc_rhf_response_resident* owner = nullptr;
      const auto expected = failure == cudaErrorMemoryAllocation
          ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
      if (create(plan, &owner) != expected) return 10 + site;
      if (owner || live_allocations || live_blas) return 30 + site;
    }
  }
  for (auto site : {blas_create, blas_stream, blas_pointer}) {
    for (auto failure : {CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED}) {
      fault_site = site;
      blas_failure = failure;
      generativeqc_rhf_response_resident* owner = nullptr;
      const auto expected = failure == CUBLAS_STATUS_ALLOC_FAILED
          ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
      if (create(plan, &owner) != expected) return 50 + site;
      if (owner || live_allocations || live_blas) return 70 + site;
    }
  }
  generativeqc_rhf_response_resident* owner = nullptr;
  if (create(plan, &owner) != GENERATIVEQC_STATUS_SUCCESS || !owner) return 90;
  double norm = -1;
  for (auto failure : {CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_EXECUTION_FAILED}) {
    fault_site = blas_norm;
    blas_failure = failure;
    const auto expected = failure == CUBLAS_STATUS_ALLOC_FAILED
        ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
    if (generativeqc_rhf_response_resident_norm(owner, 0, &norm) != expected) return 91;
    if (generativeqc_rhf_response_resident_norm(owner, 0, &norm) !=
            GENERATIVEQC_STATUS_SUCCESS || norm != 2) return 92;
  }
  for (auto failure : {cudaErrorMemoryAllocation, cudaErrorUnknown}) {
    fault_site = stream_sync;
    cuda_failure = failure;
    const auto expected = failure == cudaErrorMemoryAllocation
        ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
    if (generativeqc_rhf_response_resident_norm(owner, 0, &norm) != expected) return 93;
  }
  bad_norm = true;
  if (generativeqc_rhf_response_resident_norm(owner, 0, &norm) !=
      GENERATIVEQC_STATUS_NUMERICAL_FAILURE) return 94;
  bad_norm = false;
  if (generativeqc_rhf_response_resident_norm(owner, 99, &norm) !=
      GENERATIVEQC_STATUS_INVALID_ARGUMENT) return 95;
  const auto before = fences;
  generativeqc_rhf_response_resident_destroy(owner);
  if (live_allocations || live_blas || fences <= before || !releases) return 96;

  CallbackOwner response;
  Batch batch;
  current_device = 3;
  for (auto site : {get_device, set_device, event_create, event_record, stream_wait}) {
    for (auto failure : {cudaErrorMemoryAllocation, cudaErrorUnknown}) {
      fault_site = site;
      cuda_failure = failure;
      const double* published = nullptr;
      const auto status = mapped([&] { published = handoff(&response, &batch); });
      const auto expected = failure == cudaErrorMemoryAllocation
          ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
      if (status != expected || published || live_events || current_device != 3)
        return 100 + site;
      if (handoff(&response, &batch) != response.plan->output || live_events) return 120;
    }
  }
  const generativeqc_status expected[]{
      GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_CUDA_ERROR,
      GENERATIVEQC_STATUS_OUT_OF_MEMORY, GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
      GENERATIVEQC_STATUS_INVALID_ARGUMENT, GENERATIVEQC_STATUS_CUDA_ERROR};
  for (int mode = 1; mode <= 5; ++mode) {
    for (bool secondary : {false, true}) {
      callback_mode = mode;
      cleanup_failure = secondary;
      const auto drains = callback_drains;
      const double* published = nullptr;
      const auto status = mapped([&] { published = handoff(&response, &batch); });
      if (status != expected[mode] || published || borrowed || live_events ||
          callback_drains <= drains || current_device != 3) return 130 + mode;
      callback_mode = 0;
      cleanup_failure = false;
      if (handoff(&response, &batch) != response.plan->output || borrowed || live_events)
        return 140 + mode;
    }
  }
  stale = true;
  if (mapped([&] { handoff(&response, &batch); }) != GENERATIVEQC_STATUS_INVALID_ARGUMENT ||
      borrowed || live_events) return 150;
  stale = false;
  {
    generativeqc::runtime::OwnedCudaEvent first(0, cudaEventDisableTiming, check_cpks_cuda);
    generativeqc::runtime::OwnedCudaEvent second(std::move(first)), third;
    third = std::move(second);
    fault_site = event_record;
    cuda_failure = cudaErrorUnknown;
    if (mapped([&] { third.record(nullptr); }) != GENERATIVEQC_STATUS_CUDA_ERROR) return 151;
    fault_site = event_sync;
    if (mapped([&] { third.synchronize(); }) != GENERATIVEQC_STATUS_CUDA_ERROR) return 152;
    fault_site = event_elapsed;
    if (mapped([&] { third.elapsed_since(third); }) != GENERATIVEQC_STATUS_CUDA_ERROR) return 153;
  }
  if (live_events || current_device != 3) return 154;
  {
    generativeqc::runtime::OwnedCudaEvent unchanged(0);
    fault_site = event_record;
    if (mapped([&] { unchanged.record(nullptr); }) != GENERATIVEQC_STATUS_NUMERICAL_FAILURE)
      return 155;
  }
  if (live_events || live_blas || live_allocations) return 156;
}
"""
