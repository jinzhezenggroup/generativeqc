// Host control seam for the production device scope; no GPU is accessed.
#pragma once
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidDevice = 101;
inline thread_local int host_current_device = 0;
inline int host_failed_device = -1;
inline cudaError_t cudaGetDevice(int* device) {
  *device = host_current_device;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) {
  if (device < 0 || device == host_failed_device) return cudaErrorInvalidDevice;
  host_current_device = device;
  return cudaSuccess;
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "injected device selection failure"; }
