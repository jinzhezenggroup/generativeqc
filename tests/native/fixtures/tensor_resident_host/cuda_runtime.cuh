// Host-only CUDA/Context dependency seam. Never evaluates a TensorIR kernel.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "metrics.hpp"

enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
struct PendingCopy {
  void* to;
  const void* from;
  size_t bytes;
};
inline std::vector<PendingCopy> pending;
inline int failure = 0, synchronizations = 0;
inline int cudaMemcpyAsync(void* to, const void* from, size_t bytes, int, int stream) {
  if (stream != 73) throw std::runtime_error("wrong owner stream");
  if (failure == 1) return 7;
  pending.push_back({to, from, bytes});
  return 0;
}
inline int cudaStreamSynchronize(int stream) {
  if (stream != 73) throw std::runtime_error("wrong owner stream");
  ++synchronizations;
  if (failure == 2) {
    pending.clear();
    return 7;
  }
  for (auto copy : pending) std::memcpy(copy.to, copy.from, copy.bytes);
  pending.clear();
  return 0;
}
inline int cudaMemsetAsync(void* p, int value, size_t bytes, int) {
  std::memset(p, value, bytes);
  return 0;
}
inline int cudaEventRecord(int, int) { return 0; }
inline int cudaEventSynchronize(int) { return cudaStreamSynchronize(73); }
inline int cudaEventElapsedTime(float* time, int, int) {
  *time = 0;
  return 0;
}

namespace generativeqc_tensor {
inline void error_text(char* p, size_t size, const char* message) {
  if (p && size) {
    std::strncpy(p, message, size - 1);
    p[size - 1] = 0;
  }
}
inline void cuda_check(int status) {
  if (status) throw std::runtime_error("injected CUDA failure");
}
struct Context {
  std::mutex mutex;
  std::vector<unsigned char> storage;
  unsigned char* arena;
  int arithmetic_error = 0;
  int* error = &arithmetic_error;
  int device = 0, stream = 73, begin = 0, end = 0;
  bool static_ready = true;
  Metrics metrics;
  explicit Context(size_t bytes) : storage(bytes, 0xa5), arena(storage.data()) {
    metrics.owned_device_bytes = bytes;
  }
  void check_device() {
    if (failure == 3) throw std::runtime_error("injected device mismatch");
  }
  size_t device_delta() { return 0; }
  template <class F>
  void section(int, double&, F f) {
    f();
  }
};
}  // namespace generativeqc_tensor
using namespace generativeqc_tensor;

extern "C" void* host_owner(size_t bytes) { return new Context(bytes); }
extern "C" void host_destroy(void* p) { delete static_cast<Context*>(p); }
extern "C" void* host_arena(void* p) { return static_cast<Context*>(p)->arena; }
extern "C" void host_capacity(void* p, size_t bytes) {
  static_cast<Context*>(p)->metrics.owned_device_bytes = bytes;
}
extern "C" void host_static_ready(void* p, int ready) {
  static_cast<Context*>(p)->static_ready = ready;
}
extern "C" void host_failure(int mode) { failure = mode; }
extern "C" int host_synchronizations() { return synchronizations; }
