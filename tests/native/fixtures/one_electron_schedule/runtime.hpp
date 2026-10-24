#pragma once
#include "cuda_runtime_api.h"
using cudaGraph_t = void*;
using cudaEvent_t = void*;
inline constexpr auto cudaErrorStreamCaptureUnsupported = cudaErrorNotSupported;
inline constexpr int cudaStreamNonBlocking = 1, cudaEventDisableTiming = 1,
                     cudaStreamCaptureModeThreadLocal = 1;
namespace stub {
inline std::size_t streams{}, events{};
inline cudaError_t graph_error = cudaSuccess;
}  // namespace stub
inline cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, int) {
  *stream = reinterpret_cast<void*>(++stub::streams + 1);
  return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) {
  --stub::streams;
  return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, int) {
  *event = reinterpret_cast<void*>(++stub::events + 1);
  return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t) {
  --stub::events;
  return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, int) { return stub::graph_error; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) {
  *graph = reinterpret_cast<void*>(1);
  return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* replay, cudaGraph_t, void*, void*, int) {
  *replay = reinterpret_cast<void*>(1);
  return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, int) { return cudaSuccess; }
namespace generativeqc::integrals {
inline unsigned blocks(std::size_t) { return 1; }
inline bool disjoint(const void* first, std::size_t first_bytes, const void* second,
                     std::size_t second_bytes) {
  auto a = reinterpret_cast<std::uintptr_t>(first), b = reinterpret_cast<std::uintptr_t>(second);
  return a <= b ? b - a >= first_bytes : a - b >= second_bytes;
}
inline cudaError_t device_buffer(const void*, int) { return cudaSuccess; }
inline cudaError_t gate(int*, cudaStream_t) { return cudaSuccess; }
inline cudaError_t drain(cudaError_t error, cudaStream_t) { return error; }
template <class... Args>
void build_records(Args...) {}
template <class... Args>
void project(Args...) {}
template <class... Args>
void finite_values(Args...) {}
template <class... Args>
void validate_basis(Args...) {}
}  // namespace generativeqc::integrals
