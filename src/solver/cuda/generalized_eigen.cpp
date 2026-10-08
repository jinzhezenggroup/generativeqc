#include "solver/cuda/generalized_eigen.hpp"

#include "tensor/cuda_square_linalg.hpp"

namespace generativeqc::solver::cuda {

static_assert(sizeof(cublasStatus_t) == sizeof(std::uint32_t));
static_assert(CUBLAS_STATUS_SUCCESS == 0);

std::uint32_t GeneralizedEigenLowering::invalid() noexcept {
  return static_cast<std::uint32_t>(CUBLAS_STATUS_INVALID_VALUE);
}
bool GeneralizedEigenLowering::success(std::uint32_t status) noexcept {
  return status == CUBLAS_STATUS_SUCCESS;
}
std::uint32_t GeneralizedEigenLowering::identity(bool) noexcept {
  return static_cast<std::uint32_t>(CUBLAS_STATUS_SUCCESS);
}
std::uint32_t GeneralizedEigenLowering::multiply(bool transpose, GeneralizedEigenOperand left,
                                                 GeneralizedEigenOperand right,
                                                 GeneralizedEigenOperand output) const {
  return static_cast<std::uint32_t>(tensor::cuda::square_gemm(
      static_cast<cublasHandle_t>(handle_), transpose, static_cast<int>(domain_.order),
      static_cast<int>(domain_.solves), read(left), read(right), write(output)));
}
std::uint32_t GeneralizedEigenLowering::triangular(bool right, bool transpose) const {
  return static_cast<std::uint32_t>(tensor::cuda::square_lower_solve(
      static_cast<cublasHandle_t>(handle_), right, transpose, static_cast<int>(domain_.order),
      static_cast<int>(domain_.solves), pointers_.factors, pointers_.matrices));
}

}  // namespace generativeqc::solver::cuda
