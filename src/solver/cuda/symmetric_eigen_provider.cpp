#include "solver/cuda/symmetric_eigen_provider.hpp"

#include <cusolverDn.h>

#include "solver/cuda/cusolver_compat.hpp"

namespace generativeqc::solver::cuda {
namespace {
static_assert(sizeof(cusolverStatus_t) == sizeof(std::uint32_t));
static_assert(CUSOLVER_STATUS_SUCCESS == 0);

cusolverEigMode_t vector_mode(Eigenvectors vectors) noexcept {
  return vectors == Eigenvectors::values_only ? CUSOLVER_EIG_MODE_NOVECTOR
                                              : CUSOLVER_EIG_MODE_VECTOR;
}
}  // namespace

std::uint32_t query_symmetric_eigen(const SymmetricEigenResources& resources,
                                    SymmetricEigenFamily family,
                                    const SymmetricEigenProblem& problem, const double* matrix,
                                    const double* values,
                                    SymmetricEigenWorkspace& workspace) noexcept {
  const auto solver = static_cast<cusolverDnHandle_t>(resources.solver);
  const auto parameters = static_cast<cusolverDnParams_t>(resources.parameters);
  const auto jacobi = static_cast<syevjInfo_t>(resources.jacobi);
  const auto mode = vector_mode(problem.vectors);
  if (family == SymmetricEigenFamily::jacobi_batched) {
    return static_cast<std::uint32_t>(cusolverDnDsyevjBatched_bufferSize(
        solver, mode, CUBLAS_FILL_MODE_LOWER, static_cast<int>(problem.n), matrix,
        static_cast<int>(problem.n), values, &workspace.jacobi_elements, jacobi,
        static_cast<int>(problem.batch)));
  }
  if (family == SymmetricEigenFamily::xsyev_batched) {
    return static_cast<std::uint32_t>(cuda_compat::xsyev_batched_buffer_size(
        solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, problem.n, CUDA_R_64F, matrix, problem.n,
        CUDA_R_64F, values, CUDA_R_64F, &workspace.device_bytes, &workspace.host_bytes,
        problem.batch));
  }
  if (family != SymmetricEigenFamily::xsyevd)
    return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  return static_cast<std::uint32_t>(cusolverDnXsyevd_bufferSize(
      solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, problem.n, CUDA_R_64F, matrix, problem.n,
      CUDA_R_64F, values, CUDA_R_64F, &workspace.device_bytes, &workspace.host_bytes));
}

std::uint32_t launch_symmetric_eigen(const SymmetricEigenResources& resources,
                                     SymmetricEigenFamily family,
                                     const SymmetricEigenProblem& problem, double* matrix,
                                     double* values, int* info, int jacobi_elements) noexcept {
  const auto solver = static_cast<cusolverDnHandle_t>(resources.solver);
  const auto parameters = static_cast<cusolverDnParams_t>(resources.parameters);
  const auto jacobi = static_cast<syevjInfo_t>(resources.jacobi);
  const auto mode = vector_mode(problem.vectors);
  if (family == SymmetricEigenFamily::jacobi_batched) {
    return static_cast<std::uint32_t>(cusolverDnDsyevjBatched(
        solver, mode, CUBLAS_FILL_MODE_LOWER, static_cast<int>(problem.n), matrix,
        static_cast<int>(problem.n), values, static_cast<double*>(resources.device_workspace),
        jacobi_elements, info, jacobi, static_cast<int>(problem.batch)));
  }
  if (family == SymmetricEigenFamily::xsyev_batched) {
    return static_cast<std::uint32_t>(cuda_compat::xsyev_batched(
        solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, problem.n, CUDA_R_64F, matrix, problem.n,
        CUDA_R_64F, values, CUDA_R_64F, resources.device_workspace,
        resources.device_workspace_bytes, resources.host_workspace, resources.host_workspace_bytes,
        info, problem.batch));
  }
  if (family != SymmetricEigenFamily::xsyevd)
    return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  const std::size_t matrix_elements =
      static_cast<std::size_t>(problem.n) * static_cast<std::size_t>(problem.n);
  for (std::int64_t system = 0; system < problem.batch; ++system) {
    const auto status = cusolverDnXsyevd(
        solver, parameters, mode, CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(problem.n),
        CUDA_R_64F, matrix + static_cast<std::size_t>(system) * matrix_elements,
        static_cast<std::int64_t>(problem.n), CUDA_R_64F,
        values + static_cast<std::size_t>(system) * problem.n, CUDA_R_64F,
        resources.device_workspace, resources.device_workspace_bytes, resources.host_workspace,
        resources.host_workspace_bytes, info + system);
    if (status != CUSOLVER_STATUS_SUCCESS) return static_cast<std::uint32_t>(status);
  }
  return static_cast<std::uint32_t>(CUSOLVER_STATUS_SUCCESS);
}
}  // namespace generativeqc::solver::cuda
