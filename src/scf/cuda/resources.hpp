#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>

#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/matrix_library.hpp"
#include "solver/cuda/symmetric_eigen_handles.hpp"

namespace generativeqc::scf::cuda_execution {

struct DirectTileValidationRecord;

/** Own a bucket's stream, library workspaces and numeric arena.
 * Teardown preserves stream-ordered allocation release and selects the owning
 * device. RHF iteration Graphs have a separate host-control owner; borrowed
 * eigensolver/matrix views never acquire lifetime ownership.
 */
class CudaResources {
 public:
  ~CudaResources();

  /** Borrow library state while retaining ownership in the prepared bucket. */
  EigensolverResources eigensolver_view() const;

  /** Borrow only the stream and BLAS handle; matrix execution owns no bucket state. */
  MatrixLibraryResources matrix_view() const { return {stream_, blas_}; }

  int device_id_{-1};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  ::generativeqc::solver::cuda::PreparedSymmetricEigenHandles eigen_handles_;
  void* arena_{};
  DirectTileValidationRecord* direct_tile_validation_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
  /** Optional unscreened reference ERIs; fallback arena remains available. */
  double* reference_eri_{};
  std::size_t reference_eri_bytes_{};
  /** Numeric reference peak includes the observed provider allocations. */
  std::size_t reference_peak_bytes_{};
  std::size_t provider_retained_bytes_{};
};

}  // namespace generativeqc::scf::cuda_execution
