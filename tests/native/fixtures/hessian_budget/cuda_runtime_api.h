#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

using cudaStream_t = void*;
using cudaGraphExec_t = void*;
struct dim3 {};
enum cudaError_t {
  cudaSuccess,
  cudaErrorInvalidValue,
  cudaErrorMemoryAllocation,
  cudaErrorNotSupported,
  cudaErrorUnknown,
  cudaErrorInvalidResourceHandle
};
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone };
inline constexpr int cudaMemcpyHostToDevice = 1;
namespace stub {
inline std::unordered_map<void*, std::size_t> allocations;
inline std::size_t live_bytes{}, peak_bytes{}, allocation_attempts{}, fail_allocation{};
}  // namespace stub
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA mock error"; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* p) {
  *p = cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void** p, std::size_t bytes) {
  if (++stub::allocation_attempts == stub::fail_allocation) return cudaErrorMemoryAllocation;
  *p = std::malloc(bytes);
  if (!*p) return cudaErrorMemoryAllocation;
  stub::allocations[*p] = bytes;
  stub::live_bytes += bytes;
  if (stub::live_bytes > stub::peak_bytes) stub::peak_bytes = stub::live_bytes;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
  if (p) {
    assert(stub::allocations.contains(p));
    stub::live_bytes -= stub::allocations.at(p);
    stub::allocations.erase(p);
    std::free(p);
  }
  return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int value, std::size_t bytes, cudaStream_t) {
  std::memset(p, value, bytes);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, int,
                                   cudaStream_t) {
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}
