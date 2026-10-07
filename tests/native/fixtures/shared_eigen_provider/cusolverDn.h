#pragma once

// Host-only provider tracing ABI. Distinct opaque pointers and enum status match
// the official CUDA headers; void* aliases would hide missing production casts.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

struct cusolverDnContext;
struct cusolverDnParams;
struct syevjInfo;
using cusolverDnHandle_t = cusolverDnContext*;
using cusolverDnParams_t = cusolverDnParams*;
using syevjInfo_t = syevjInfo*;
enum cusolverStatus_t {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_ARCH_MISMATCH = 4,
  CUSOLVER_STATUS_MAPPING_ERROR = 5,
  CUSOLVER_STATUS_EXECUTION_FAILED = 6,
  CUSOLVER_STATUS_INTERNAL_ERROR = 7,
  CUSOLVER_STATUS_NOT_SUPPORTED = 9
};
enum cusolverEigMode_t { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 };
enum cublasFillMode_t { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1 };
enum cudaDataType { CUDA_R_32F = 0, CUDA_R_64F = 1 };

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t*);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t, cudaStream_t);
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t*);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t);
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t*);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t, double);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t, int);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t, int);

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t, cusolverEigMode_t,
                                                    cublasFillMode_t, int, const double*, int,
                                                    const double*, int*, syevjInfo_t, int);
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t,
                                         int, double*, int, double*, double*, int, int*,
                                         syevjInfo_t, int);
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t, cusolverDnParams_t,
                                             cusolverEigMode_t, cublasFillMode_t, std::int64_t,
                                             cudaDataType, const void*, std::int64_t, cudaDataType,
                                             const void*, cudaDataType, std::size_t*, std::size_t*);
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                                  cublasFillMode_t, std::int64_t, cudaDataType, void*, std::int64_t,
                                  cudaDataType, void*, cudaDataType, void*, std::size_t, void*,
                                  std::size_t, int*);

#if defined(TEST_CUMETAL_STRIDED)
// CuMetal's explicit-stride-only API must resolve through the production shim.
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t, cusolverDnParams_t,
                                                   cusolverEigMode_t, cublasFillMode_t,
                                                   std::int64_t, cudaDataType, const void*,
                                                   std::int64_t, std::int64_t, cudaDataType,
                                                   const void*, std::int64_t, cudaDataType,
                                                   std::int64_t, std::size_t*, std::size_t*);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                                        cublasFillMode_t, std::int64_t, cudaDataType, void*,
                                        std::int64_t, std::int64_t, cudaDataType, void*,
                                        std::int64_t, cudaDataType, std::int64_t, void*,
                                        std::size_t, void*, std::size_t, int*);
#else
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t, cusolverDnParams_t,
                                                   cusolverEigMode_t, cublasFillMode_t,
                                                   std::int64_t, cudaDataType, const void*,
                                                   std::int64_t, cudaDataType, const void*,
                                                   cudaDataType, std::size_t*, std::size_t*,
                                                   std::int64_t);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t,
                                        cublasFillMode_t, std::int64_t, cudaDataType, void*,
                                        std::int64_t, cudaDataType, void*, cudaDataType, void*,
                                        std::size_t, void*, std::size_t, int*, std::int64_t);
#endif
