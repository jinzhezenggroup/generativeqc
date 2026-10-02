"""Host-check the production rollback/fence boundary; native tests execute J/K."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, marker: str) -> str:
    begin = source.index(marker)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


STUBS = r"""
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorUnknown };
enum generativeqc_status { GENERATIVEQC_STATUS_SUCCESS, GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                          GENERATIVEQC_STATUS_CUDA_ERROR, GENERATIVEQC_STATUS_NUMERICAL_FAILURE };
struct DirectJkFailure { generativeqc_status status; std::string detail; };
generativeqc_status source_cuda_status(cudaError_t e) {
  return e == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                       : GENERATIVEQC_STATUS_CUDA_ERROR;
}
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA error"; }
struct Copy {
  void* destination; const void* source; std::size_t bytes; bool device_destination = true;
};
std::vector<Copy> pending;
std::unordered_map<void*, std::size_t> live;
cudaError_t sticky = cudaSuccess, synchronization_error = cudaSuccess;
unsigned fences = 0;
cudaError_t cudaStreamSynchronize(cudaStream_t) {
  ++fences;
  for (auto copy : pending) {
    assert(!copy.device_destination || live.count(copy.destination));
    std::memcpy(copy.destination, copy.source, copy.bytes);
  }
  pending.clear();
  return synchronization_error;
}
cudaError_t cudaGetLastError() {
  const auto error = sticky;
  sticky = cudaSuccess;
  return error;
}
namespace runtime {
cudaError_t resource_cuda_free(void* pointer) {
  // Device buffers must survive every queued upload and use until the fence.
  assert(pending.empty() && live.erase(pointer) == 1);
  std::free(pointer);
  return cudaSuccess;
}
}
struct CudaDirectJkPlan {
  cudaStream_t stream = reinterpret_cast<void*>(1);
  std::vector<void*> allocations;
  std::size_t device_bytes = 0;
  ~CudaDirectJkPlan() {
    cudaStreamSynchronize(stream);
    for (void* pointer : allocations) runtime::resource_cuda_free(pointer);
  }
};
void* allocate(CudaDirectJkPlan& plan, std::size_t bytes) {
  void* pointer = std::calloc(bytes, 1);
  assert(pointer);
  live.emplace(pointer, bytes);
  plan.allocations.push_back(pointer);
  plan.device_bytes += bytes;
  return pointer;
}
void fault(int kind) {
  switch (kind) {
    case 0: throw std::bad_alloc();
    case 1:
      sticky = cudaErrorMemoryAllocation;
      throw DirectJkFailure{GENERATIVEQC_STATUS_OUT_OF_MEMORY, "allocation"};
    case 2:
      sticky = cudaErrorMemoryAllocation;
      throw cudaErrorMemoryAllocation;
    case 3: throw DirectJkFailure{GENERATIVEQC_STATUS_CUDA_ERROR, "execution"};
    case 4: throw cudaErrorUnknown;
    case 5: throw std::invalid_argument("invalid input");
    case 6: throw DirectJkFailure{GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "numerical"};
  }
  assert(false);
}
"""

DRIVER = r"""
void exercise(bool optional, int stage, int kind) {
  assert(live.empty());
  sticky = synchronization_error = cudaSuccess;
  fences = 0;
  {
    CudaDirectJkPlan plan;
    auto* retained = static_cast<double*>(allocate(plan, sizeof(double)));
    *retained = 37.0;
    void* availability = nullptr;
    std::string detail = "failed allocation detail";
    bool restored = false, propagated = false;
    try {
      direct_jk_optional_storage(plan, optional, detail, [&] {
        // Match preparation's staging lifetime: fence dies before its source.
        std::vector<double> staging(4, 19.0);
        DirectJkDownloadFence staging_fence{plan.stream};
        for (int allocation = 0; allocation != 4; ++allocation) {
          if (stage == allocation) fault(kind);
          availability = allocate(plan, staging.size() * sizeof(double));
          pending.push_back({availability, staging.data(), staging.size() * sizeof(double)});
        }
        if (stage == 4) fault(kind);
        staging_fence.complete();
      }, [&] { availability = nullptr; restored = true; });
    } catch (...) {
      propagated = true;
    }
    const bool recover = optional && kind < 3;
    assert(propagated == !recover);
    assert(restored == recover);
    assert(*retained == 37.0 && pending.empty() && fences > 0);
    if (recover) {
      assert(availability == nullptr && detail.empty());
      assert(plan.allocations.size() == 1 && plan.device_bytes == sizeof(double));
      assert(live.size() == 1 && sticky == cudaSuccess);
      // A recovered owner remains usable and accepts another optional group.
      direct_jk_optional_storage(plan, true, detail, [&] {
        availability = allocate(plan, sizeof(double));
      }, [&] { assert(false); });
      assert(availability && live.size() == 2);
    }
  }
  assert(live.empty());
}
void execution_failure_during_rollback() {
  for (bool at_fence : {false, true}) {
    assert(live.empty());
    sticky = synchronization_error = cudaSuccess;
    CudaDirectJkPlan plan;
    allocate(plan, 8);
    std::string detail;
    bool propagated = false;
    try {
      direct_jk_optional_storage(plan, true, detail, [&] {
        allocate(plan, 16);
        if (at_fence) synchronization_error = cudaErrorUnknown;
        else sticky = cudaErrorUnknown;
        throw std::bad_alloc();
      }, [] {});
    } catch (const DirectJkFailure& failure) {
      propagated = failure.status == GENERATIVEQC_STATUS_CUDA_ERROR;
    }
    assert(propagated);
    synchronization_error = sticky = cudaSuccess;
  }
  assert(live.empty());
}
void numerical_download_before_oom() {
  for (int device_failure : {0, 1}) {
    assert(live.empty());
    CudaDirectJkPlan plan;
    allocate(plan, 8);
    int numerical_failure = 0;
    std::string detail;
    bool propagated = false, restored = false;
    try {
      direct_jk_optional_storage(plan, true, detail, [&] {
        DirectJkDownloadFence staging_fence{plan.stream};
        allocate(plan, 16);
        pending.push_back({&numerical_failure, &device_failure, sizeof(int), false});
        throw std::bad_alloc();
      }, [&] { restored = true; }, &numerical_failure);
    } catch (const DirectJkFailure& failure) {
      propagated = failure.status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    assert(numerical_failure == device_failure && propagated == bool(device_failure));
    assert(restored == !device_failure && pending.empty());
  }
  assert(live.empty());
}
int main() {
  for (bool optional : {false, true})
    for (int stage = 0; stage != 5; ++stage)
      for (int kind = 0; kind != 7; ++kind) exercise(optional, stage, kind);
  execution_failure_during_rollback();
  numerical_download_before_oom();
}
"""


def test_production_optional_rollback_preserves_owner_and_failure_classification(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    definitions = [
        _definition(source, "void direct_jk_check("),
        _definition(source, "struct DirectJkDownloadFence") + ";",
        _definition(source, "template <class Prepare, class Restore>"),
    ]
    cpp, binary = tmp_path / "probe.cpp", tmp_path / "probe"
    cpp.write_text(STUBS + "\n".join(definitions) + DRIVER)
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
