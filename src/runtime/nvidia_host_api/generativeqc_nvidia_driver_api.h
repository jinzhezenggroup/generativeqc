#pragma once

#include <stddef.h>

/* ABI-stable CUDA driver facts needed for borrowed allocation admission.
 * Provider SDK enum declarations stay in separate translation units. These
 * declarations do not load the driver or relax descriptor device ownership.
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NOT_FOUND = 500
} CUresult;

#if defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
#error "the provider-free CUDA driver ABI requires a 64-bit ELF target"
#endif

CUresult cuGetErrorString(CUresult error, const char** detail);
CUresult cuMemGetAddressRange_v2(CUdeviceptr* base, size_t* bytes, CUdeviceptr pointer);
#define cuMemGetAddressRange cuMemGetAddressRange_v2

#ifdef __cplusplus
}
#endif
