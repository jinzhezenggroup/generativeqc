#pragma once
#include <cstddef>
using cudaStream_t = void*;
using cudaError_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaSuccess = 0;
constexpr int cudaStreamCaptureStatusNone = 0;
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaMemGetInfo(std::size_t*, std::size_t*);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
