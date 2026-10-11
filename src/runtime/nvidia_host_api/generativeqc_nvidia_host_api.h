#pragma once

/*
 * Minimal NVIDIA host-library ABI declarations used by GenerativeQC's provider-free
 * Linux wheels. This GenerativeQC-owned interface header declares only opaque types,
 * stable enum values, and generated C function signatures for APIs GenerativeQC actually calls.
 *
 * Native SDK builds continue to use NVIDIA's cublas_v2.h/cusolverDn.h. Wheel
 * builds put this directory first on the include path so nvcc needs only its
 * compiler/runtime headers; Implib.so trampolines resolve the deployed CUDA-12
 * providers lazily at runtime.
 */

#include <cuda_runtime_api.h>
#include <library_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext* cublasHandle_t;
typedef uint32_t cublasStatus_t;
enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16,
};

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1,
  CUBLAS_FILL_MODE_FULL = 2,
} cublasFillMode_t;
typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;
typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2 } cublasMath_t;

typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef struct cusolverDnParams* cusolverDnParams_t;
typedef struct syevjInfo* syevjInfo_t;
typedef uint32_t cusolverStatus_t;
enum {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_ARCH_MISMATCH = 4,
  CUSOLVER_STATUS_MAPPING_ERROR = 5,
  CUSOLVER_STATUS_EXECUTION_FAILED = 6,
  CUSOLVER_STATUS_INTERNAL_ERROR = 7,
  CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  CUSOLVER_STATUS_NOT_SUPPORTED = 9,
};
typedef enum { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 } cusolverEigMode_t;

// Declarations and _v2 aliases are generated from NVIDIA SDK headers.
#include "generativeqc_nvidia_host_functions.h"

#ifdef __cplusplus
}  // extern "C"
#endif
