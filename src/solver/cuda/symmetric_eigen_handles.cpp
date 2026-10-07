// Provider lifecycle consolidates GenerativeQC and adapted xTBloom ownership.
// xTBloom portions: GPL-3.0-or-later; original scoped CUDA/MKL additional
// permission: LICENSES/xtbloom-CUDA_MKL_LINKING_EXCEPTION.txt.
#include "solver/cuda/symmetric_eigen_handles.hpp"

#include <cusolverDn.h>

#include <utility>

namespace generativeqc::solver::cuda {
namespace {
static_assert(sizeof(cusolverStatus_t) == sizeof(std::uint32_t));
static_assert(CUSOLVER_STATUS_SUCCESS == 0);
}  // namespace

PreparedSymmetricEigenHandles::~PreparedSymmetricEigenHandles() { reset(); }

PreparedSymmetricEigenHandles::PreparedSymmetricEigenHandles(
    PreparedSymmetricEigenHandles&& other) noexcept
    : solver_(std::exchange(other.solver_, nullptr)),
      parameters_(std::exchange(other.parameters_, nullptr)),
      jacobi_(std::exchange(other.jacobi_, nullptr)) {}

PreparedSymmetricEigenHandles& PreparedSymmetricEigenHandles::operator=(
    PreparedSymmetricEigenHandles&& other) noexcept {
  if (this != &other) {
    reset();
    solver_ = std::exchange(other.solver_, nullptr);
    parameters_ = std::exchange(other.parameters_, nullptr);
    jacobi_ = std::exchange(other.jacobi_, nullptr);
  }
  return *this;
}

std::uint32_t PreparedSymmetricEigenHandles::create() noexcept {
  if (solver_) return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  cusolverDnHandle_t handle{};
  const auto status = cusolverDnCreate(&handle);
  solver_ = handle;
  return static_cast<std::uint32_t>(status);
}

std::uint32_t PreparedSymmetricEigenHandles::bind_stream(void* stream) noexcept {
  if (!solver_) return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  return static_cast<std::uint32_t>(cusolverDnSetStream(static_cast<cusolverDnHandle_t>(solver_),
                                                        static_cast<cudaStream_t>(stream)));
}

std::uint32_t PreparedSymmetricEigenHandles::create_parameters() noexcept {
  if (!solver_ || parameters_) return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  cusolverDnParams_t parameters{};
  const auto status = cusolverDnCreateParams(&parameters);
  parameters_ = parameters;
  return static_cast<std::uint32_t>(status);
}

std::uint32_t PreparedSymmetricEigenHandles::configure_jacobi(double tolerance, int max_sweeps,
                                                              int sort) noexcept {
  if (!solver_ || jacobi_) return static_cast<std::uint32_t>(CUSOLVER_STATUS_INVALID_VALUE);
  syevjInfo_t jacobi{};
  auto status = cusolverDnCreateSyevjInfo(&jacobi);
  jacobi_ = jacobi;
  if (status == CUSOLVER_STATUS_SUCCESS) status = cusolverDnXsyevjSetTolerance(jacobi, tolerance);
  if (status == CUSOLVER_STATUS_SUCCESS) status = cusolverDnXsyevjSetMaxSweeps(jacobi, max_sweeps);
  if (status == CUSOLVER_STATUS_SUCCESS) status = cusolverDnXsyevjSetSortEig(jacobi, sort);
  return static_cast<std::uint32_t>(status);
}

void PreparedSymmetricEigenHandles::reset() noexcept {
  if (jacobi_) (void)cusolverDnDestroySyevjInfo(static_cast<syevjInfo_t>(jacobi_));
  if (parameters_) (void)cusolverDnDestroyParams(static_cast<cusolverDnParams_t>(parameters_));
  if (solver_) (void)cusolverDnDestroy(static_cast<cusolverDnHandle_t>(solver_));
  jacobi_ = nullptr;
  parameters_ = nullptr;
  solver_ = nullptr;
}

}  // namespace generativeqc::solver::cuda
