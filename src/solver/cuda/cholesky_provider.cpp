#include "solver/cuda/cholesky_provider.hpp"

#include <cusolverDn.h>

namespace generativeqc::solver::cuda {
static_assert(sizeof(cusolverStatus_t) == sizeof(std::uint32_t));
static_assert(CUSOLVER_STATUS_SUCCESS == 0);

std::uint32_t query_lower_cholesky(void* solver, int n, double* matrix,
                                   int& workspace_elements) noexcept {
  return static_cast<std::uint32_t>(
      cusolverDnDpotrf_bufferSize(static_cast<cusolverDnHandle_t>(solver), CUBLAS_FILL_MODE_LOWER,
                                  n, matrix, n, &workspace_elements));
}

std::uint32_t launch_lower_cholesky(void* solver, int n, double* matrix, double* workspace,
                                    int workspace_elements, int* info) noexcept {
  return static_cast<std::uint32_t>(cusolverDnDpotrf(static_cast<cusolverDnHandle_t>(solver),
                                                     CUBLAS_FILL_MODE_LOWER, n, matrix, n,
                                                     workspace, workspace_elements, info));
}

}  // namespace generativeqc::solver::cuda
