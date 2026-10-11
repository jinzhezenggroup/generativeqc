#pragma once
#include "cuda_runtime_api.h"
using cublasHandle_t = void*;
using cublasStatus_t = int;
using cublasOperation_t = int;
constexpr int CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_ALLOC_FAILED = 3;
constexpr int CUBLAS_OP_N = 0, CUBLAS_OP_T = 1;
constexpr int CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_PEDANTIC_MATH = 2;
cublasStatus_t cublasCreate(cublasHandle_t*);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetPointerMode(cublasHandle_t, int);
cublasStatus_t cublasSetMathMode(cublasHandle_t, int);
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, std::size_t);
cublasStatus_t cublasDgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                           const double*, const double*, int, const double*, int, const double*,
                           double*, int);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t, int,
                                         int, int, const double*, const double*, int, long long,
                                         const double*, int, long long, const double*, double*, int,
                                         long long, int);
