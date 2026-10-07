#include "scf/cuda/resources.hpp"

#include <cstdlib>

#include "runtime/allocation_measurement.hpp"
#include "runtime/resource_cuda.cuh"

namespace generativeqc::scf::cuda_execution {

CudaResources::~CudaResources() {
  std::lock_guard<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
  if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
  eigen_handles_.reset();
  if (blas_ != nullptr) (void)cublasDestroy(blas_);
  if (stream_ != nullptr) {
    // Numeric allocations come from CUDA's stream-ordered device pool. Queue
    // their release on the owning bucket stream so destroying one plan does
    // not impose a device-wide synchronization on unrelated workloads.
    if (reference_eri_ != nullptr) {
      (void)runtime::resource_cuda_free_async(reference_eri_, stream_);
    }
    if (reference_fock_correction_ != nullptr)
      (void)runtime::resource_cuda_free_async(reference_fock_correction_, stream_);
    if (solver_workspace_ != nullptr) {
      (void)runtime::resource_cuda_free_async(solver_workspace_, stream_);
    }
    if (direct_tile_validation_ != nullptr) {
      (void)runtime::resource_cuda_free_async(direct_tile_validation_, stream_);
    }
    if (arena_ != nullptr) (void)runtime::resource_cuda_free_async(arena_, stream_);
    (void)cudaStreamSynchronize(stream_);
    (void)cudaStreamDestroy(stream_);
  }
  std::free(solver_host_workspace_);
}

EigensolverResources CudaResources::eigensolver_view() const {
  return {stream_,
          static_cast<cusolverDnHandle_t>(eigen_handles_.view().solver),
          static_cast<cusolverDnParams_t>(eigen_handles_.view().parameters),
          static_cast<syevjInfo_t>(eigen_handles_.view().jacobi),
          solver_workspace_,
          solver_workspace_bytes_,
          solver_host_workspace_,
          solver_host_workspace_bytes_};
}

}  // namespace generativeqc::scf::cuda_execution
