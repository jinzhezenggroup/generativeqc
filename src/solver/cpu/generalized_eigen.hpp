#pragma once

#include <algorithm>

#include "solver/generalized_eigen.hpp"
#include "tensor/cpu/lp64_provider.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::solver::cpu {

/** Synchronous primitive lowering. The CPU plan/provider and its surrounding
 * thread scope belong to the caller; borrowing never admits another provider. */
class GeneralizedEigenLowering {
 public:
  GeneralizedEigenLowering(GeneralizedEigenDomain domain, GeneralizedEigenMatrices matrices,
                           const tensor::CpuLinalgPlan& plan)
      : domain_(domain), matrices_(matrices), plan_(&plan) {}
  GeneralizedEigenLowering(GeneralizedEigenDomain domain, GeneralizedEigenMatrices matrices,
                           const tensor::cpu::CpuLinearAlgebraBackend& backend)
      : domain_(domain), matrices_(matrices), backend_(&backend) {}

  bool valid(GeneralizedEigenBasis basis, bool recovery) const noexcept {
    return domain_.solves == 1 &&
           ((plan_ && domain_.layout == GeneralizedEigenLayout::row_major) ||
            (backend_ && backend_->ready() &&
             domain_.layout == GeneralizedEigenLayout::column_major)) &&
           generalized_eigen_detail::validate(domain_, basis, matrices_, recovery);
  }
  static int invalid() noexcept { return -1; }
  static bool success(int status) noexcept { return status == 0; }
  int identity(bool recovery) const {
    const auto* input = recovery ? matrices_.reduced : matrices_.input;
    auto* output = recovery ? matrices_.coefficients : matrices_.reduced;
    if (input != output) std::copy_n(input, domain_.matrix_extent(), output);
    return 0;
  }
  int multiply(bool transpose, GeneralizedEigenOperand left, GeneralizedEigenOperand right,
               GeneralizedEigenOperand output) const {
    const auto n = domain_.order;
    if (plan_)
      tensor::cpu_gemm(transpose ? 'T' : 'N', 'N', n, n, n, read(left), read(right), write(output),
                       1.0, 0.0, *plan_);
    else
      tensor::cpu::bind_gemm (*backend_)(102, transpose ? 112 : 111, 111, n, n, n, 1.0, read(left),
                                         n, read(right), n, 0.0, write(output), n);
    return 0;
  }
  int triangular(bool right, bool transpose) const {
    const auto n = domain_.order;
    if (plan_)
      tensor::cpu_trsm(right ? 'R' : 'L', 'L', transpose ? 'T' : 'N', 'N', n, n, matrices_.basis,
                       matrices_.reduced, 1.0, *plan_);
    else
      tensor::cpu::solve_lower_triangular(
          *backend_, right ? tensor::cpu::TriangularSide::right : tensor::cpu::TriangularSide::left,
          transpose ? tensor::cpu::Transpose::transpose : tensor::cpu::Transpose::none,
          static_cast<tensor::cpu::LapackInt>(n), matrices_.basis, matrices_.reduced);
    return 0;
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
  GeneralizedEigenMatrices matrices_;
  const tensor::CpuLinalgPlan* plan_{};
  const tensor::cpu::CpuLinearAlgebraBackend* backend_{};
};
}  // namespace generativeqc::solver::cpu
