// Host dependencies for compiling the ordinary generated ABI unchanged.
// Only control passthrough programs are admitted by the tests. This fixture
// exercises generated entry points, not real Context allocation or CUDA Graphs.
#pragma once
#include <cstdio>
#include <memory>

#include "cuda_runtime.cuh"

#define __device__
using I = std::int64_t;

namespace generativeqc::runtime {
struct GraphMetrics {};
}  // namespace generativeqc::runtime

namespace generativeqc_tensor {
struct DeviceAllocationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct GraphContext : Context {
  struct {
    generativeqc::runtime::GraphMetrics metrics;
    std::string reason;
  } graph;
  GraphContext() : Context(sizeof(int), 0) {}
  void prepare(int selected_device, size_t, size_t, size_t bytes, size_t error_offset, size_t,
               size_t, size_t, bool) {
    device = selected_device;
    storage.resize(bytes);
    arena = storage.data();
    if (error_offset % alignof(int) || error_offset > bytes || sizeof(int) > bytes - error_offset)
      throw std::runtime_error("invalid status reservation");
    error = new (arena + error_offset) int(0);
    metrics.owned_device_bytes = bytes;
    for (auto& range : device_ranges)
      if (range.owner == this)
        range = {this, reinterpret_cast<std::uintptr_t>(arena), bytes, error};
  }
  template <class F>
  void submit_region(int, F action) {
    action();
  }
  void configure_graph(bool enabled, const char*) {
    if (enabled) throw std::runtime_error("host fixture cannot capture CUDA Graphs");
  }
};
inline void blas_check(int status) { cuda_check(status); }
}  // namespace generativeqc_tensor

// Deterministic dependency responses allow tensor_probe's ABI/JSON to execute;
// they are never presented as observed hardware or a device qualification.
struct cudaDeviceProp {
  struct {
    char bytes[16]{};
  } uuid;
  int major = 8, minor = 0, multiProcessorCount = 1;
};
inline int cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  *p = {};
  return 0;
}
inline int cudaDriverGetVersion(int* p) {
  *p = 12000;
  return 0;
}
inline int cudaRuntimeGetVersion(int* p) { return cudaDriverGetVersion(p); }
enum { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL };
inline int cublasGetProperty(int, int* p) {
  *p = 0;
  return 0;
}
