#pragma once

#include <cusolverDn.h>

struct cublasContext;
using cublasHandle_t = cublasContext*;
enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_EXECUTION_FAILED = 13
};
enum cublasSideMode_t { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };
enum cublasDiagType_t { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 };
enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2 };

cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, std::size_t);
cublasStatus_t cublasSetPointerMode(cublasHandle_t, cublasPointerMode_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasDtrsmBatched(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
                                  cublasOperation_t, cublasDiagType_t, int, int, const double*,
                                  const double* const*, int, double* const*, int, int);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t, int,
                                         int, int, const double*, const double*, int, long long,
                                         const double*, int, long long, const double*, double*, int,
                                         long long, int);
