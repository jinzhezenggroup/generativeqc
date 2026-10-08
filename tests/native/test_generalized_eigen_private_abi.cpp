// Compile the real GFN private ABI with the phase interface, independently of
// the official-header lowering TU. Including vendor declarations here regresses
// the actual GFN CUDA build even if the official-header host probes still pass.
#include <type_traits>

#include "runtime/nvidia_host_api.h"
#include "solver/cuda/generalized_eigen.hpp"

namespace eigen = generativeqc::solver;
static_assert(std::is_same_v<cublasStatus_t, std::uint32_t>);
static_assert(CUBLAS_STATUS_SUCCESS == 0 && CUBLAS_STATUS_INVALID_VALUE == 7);

std::uint32_t private_generalized_phase(eigen::GeneralizedEigenBasis basis, bool recovery,
                                        eigen::GeneralizedEigenDomain domain, void* handle,
                                        eigen::GeneralizedEigenMatrices matrices,
                                        eigen::cuda::GeneralizedEigenPointerMatrices pointers) {
  const cublasHandle_t blas = static_cast<cublasHandle_t>(handle);
  const auto lowering = basis == eigen::GeneralizedEigenBasis::lower_cholesky
                            ? eigen::cuda::GeneralizedEigenLowering{domain, blas, pointers}
                            : eigen::cuda::GeneralizedEigenLowering{domain, blas, matrices};
  const cublasStatus_t status = recovery ? eigen::recover_generalized_eigen(basis, lowering)
                                         : eigen::reduce_generalized_eigen(basis, lowering);
  return status;
}
