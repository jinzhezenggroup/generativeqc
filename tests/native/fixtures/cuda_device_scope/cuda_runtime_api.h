#pragma once

/* Host-only CUDA control double; no device operations are implemented. */
typedef enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidDevice = 101,
  cudaErrorInvalidResourceHandle = 400
} cudaError_t;

typedef struct CUstream_st* cudaStream_t;
typedef enum {
  cudaMemoryTypeUnregistered = 0,
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeManaged = 3
} cudaMemoryType;
typedef struct {
  cudaMemoryType type;
  int device;
  void* hostPointer;
  void* devicePointer;
} cudaPointerAttributes;
typedef enum { cudaStreamCaptureStatusNone = 0 } cudaStreamCaptureStatus;
enum { cudaHostRegisterReadOnly = 8 };

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t status);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* pointer);
cudaError_t cudaHostGetFlags(unsigned int* flags, void* pointer);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status);
cudaError_t cudaStreamGetDevice(cudaStream_t stream, int* device);
#ifdef __cplusplus
}
#endif
