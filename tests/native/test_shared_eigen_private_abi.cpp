// Compile the real private uint32_t ABI independently from the official-header
// provider TU. Only the ABI-free borrowed-resource boundary crosses that seam.
#include <type_traits>

#include "runtime/nvidia_host_api.h"
#include "solver/cuda/symmetric_eigen_provider.hpp"

namespace shared = generativeqc::solver::cuda;
static_assert(std::is_same_v<cusolverStatus_t, std::uint32_t>);
static_assert(CUSOLVER_STATUS_SUCCESS == 0 && CUSOLVER_STATUS_ALLOC_FAILED == 2 &&
              CUSOLVER_STATUS_INVALID_VALUE == 3);

std::uint32_t private_eigen_query(const shared::SymmetricEigenResources& input,
                                  shared::SymmetricEigenFamily family,
                                  const shared::SymmetricEigenProblem& problem,
                                  const double* matrix, const double* values,
                                  shared::SymmetricEigenWorkspace& workspace) {
  const cusolverDnHandle_t solver = static_cast<cusolverDnHandle_t>(input.solver);
  const cusolverDnParams_t parameters = static_cast<cusolverDnParams_t>(input.parameters);
  const syevjInfo_t jacobi = static_cast<syevjInfo_t>(input.jacobi);
  const shared::SymmetricEigenResources borrowed{solver, parameters, jacobi};
  const cusolverStatus_t status =
      shared::query_symmetric_eigen(borrowed, family, problem, matrix, values, workspace);
  return status;
}

std::uint32_t private_eigen_launch(const shared::SymmetricEigenResources& input,
                                   shared::SymmetricEigenFamily family,
                                   const shared::SymmetricEigenProblem& problem, double* matrix,
                                   double* values, int* info, int lwork) {
  const cusolverDnHandle_t solver = static_cast<cusolverDnHandle_t>(input.solver);
  const cusolverDnParams_t parameters = static_cast<cusolverDnParams_t>(input.parameters);
  const syevjInfo_t jacobi = static_cast<syevjInfo_t>(input.jacobi);
  const shared::SymmetricEigenResources borrowed{solver,
                                                 parameters,
                                                 jacobi,
                                                 input.device_workspace,
                                                 input.device_workspace_bytes,
                                                 input.host_workspace,
                                                 input.host_workspace_bytes};
  const cusolverStatus_t status =
      shared::launch_symmetric_eigen(borrowed, family, problem, matrix, values, info, lwork);
  return status;
}
