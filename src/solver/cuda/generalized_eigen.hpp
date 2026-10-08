#pragma once

#include "solver/generalized_eigen.hpp"
#include "tensor/cuda_square_linalg.hpp"

namespace generativeqc::solver::cuda {

/** Pointer tables are device-resident. Their entries were prepared/validated by
 * the owner; only the borrowed table extents are inspectable on the host. */
struct GeneralizedEigenPointerMatrices {
  double** factors{};
  double** matrices{};
  std::size_t factor_capacity{}, matrix_capacity{};
};

class GeneralizedEigenLowering {
 public:
  GeneralizedEigenLowering(GeneralizedEigenDomain domain, cublasHandle_t handle,
                           GeneralizedEigenMatrices matrices)
      : domain_(domain), handle_(handle), matrices_(matrices) {}
  GeneralizedEigenLowering(GeneralizedEigenDomain domain, cublasHandle_t handle,
                           GeneralizedEigenPointerMatrices pointers)
      : domain_(domain), handle_(handle), pointers_(pointers) {}

  bool valid(GeneralizedEigenBasis basis, bool recovery) const noexcept {
    if (!domain_.valid() || !handle_ || domain_.layout != GeneralizedEigenLayout::column_major)
      return false;
    if (basis == GeneralizedEigenBasis::lower_cholesky)
      return pointers_.factors && pointers_.matrices &&
             pointers_.factor_capacity >= domain_.solves &&
             pointers_.matrix_capacity >= domain_.solves;
    if (!generalized_eigen_detail::validate(domain_, basis, matrices_, recovery)) return false;
    // Identity is explicit aliasing of an existing device eigenframe. Owners
    // retain any necessary upload, layout conversion or device copy.
    return basis != GeneralizedEigenBasis::identity ||
           (recovery ? matrices_.coefficients == matrices_.reduced
                     : matrices_.input == matrices_.reduced);
  }
  static cublasStatus_t invalid() noexcept { return CUBLAS_STATUS_INVALID_VALUE; }
  static bool success(cublasStatus_t status) noexcept { return status == CUBLAS_STATUS_SUCCESS; }
  static cublasStatus_t identity(bool) noexcept { return CUBLAS_STATUS_SUCCESS; }
  cublasStatus_t multiply(bool transpose, GeneralizedEigenOperand left,
                          GeneralizedEigenOperand right, GeneralizedEigenOperand output) const {
    return tensor::cuda::square_gemm(handle_, transpose, static_cast<int>(domain_.order),
                                     static_cast<int>(domain_.solves), read(left), read(right),
                                     write(output));
  }
  cublasStatus_t triangular(bool right, bool transpose) const {
    return tensor::cuda::square_lower_solve(
        handle_, right, transpose, static_cast<int>(domain_.order),
        static_cast<int>(domain_.solves), pointers_.factors, pointers_.matrices);
  }

 private:
  const double* read(GeneralizedEigenOperand operand) const {
    if (operand == GeneralizedEigenOperand::input) return matrices_.input;
    if (operand == GeneralizedEigenOperand::basis) return matrices_.basis;
    if (operand == GeneralizedEigenOperand::temporary) return matrices_.temporary;
    return matrices_.reduced;
  }
  double* write(GeneralizedEigenOperand operand) const {
    if (operand == GeneralizedEigenOperand::temporary) return matrices_.temporary;
    if (operand == GeneralizedEigenOperand::coefficients) return matrices_.coefficients;
    return matrices_.reduced;
  }
  GeneralizedEigenDomain domain_;
  cublasHandle_t handle_{};
  GeneralizedEigenMatrices matrices_;
  GeneralizedEigenPointerMatrices pointers_;
};
}  // namespace generativeqc::solver::cuda
