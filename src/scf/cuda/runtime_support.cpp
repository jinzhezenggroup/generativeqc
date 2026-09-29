#include "scf/cuda/runtime_support.hpp"

namespace generativeqc::scf::cuda_execution {

generativeqc_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return GENERATIVEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                             : GENERATIVEQC_STATUS_CUDA_ERROR;
}

generativeqc_status solver_status(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return GENERATIVEQC_STATUS_SUCCESS;
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                : GENERATIVEQC_STATUS_CUDA_ERROR;
}

generativeqc_status blas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return GENERATIVEQC_STATUS_SUCCESS;
  return status == CUBLAS_STATUS_ALLOC_FAILED ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                              : GENERATIVEQC_STATUS_CUDA_ERROR;
}

generativeqc_status copy_to_device(void* destination, const void* source, std::size_t bytes,
                                   cudaStream_t stream) {
  if (bytes == 0) return GENERATIVEQC_STATUS_SUCCESS;
  return cuda_status(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
}

}  // namespace generativeqc::scf::cuda_execution
