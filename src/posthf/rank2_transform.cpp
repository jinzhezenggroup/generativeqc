#include "posthf/rank2_transform.hpp"

#include <stdexcept>

#include "posthf/capacity.hpp"

namespace generativeqc::posthf {
namespace {

std::size_t matrix_elements(std::size_t n) {
  if (!n) throw std::invalid_argument("rank-2 basis transform requires a nonzero dimension");
  return checked_mul(n, n);
}

void validate(std::span<const double> coefficients, std::span<const double> matrix,
              std::size_t n) {
  const auto elements = matrix_elements(n);
  if (coefficients.size() != elements || matrix.size() != elements)
    throw std::invalid_argument("rank-2 basis transform matrix shape mismatch");
}

}  // namespace

std::size_t rank2_transform_workspace_bytes(std::size_t n) {
  if (!n) return 0;
  return checked_mul(matrix_elements(n), sizeof(double));
}

std::vector<double> rank2_ao_to_mo(std::span<const double> coefficients,
                                   std::span<const double> ao, std::size_t n,
                                   const tensor::CpuLinalgPlan& plan) {
  validate(coefficients, ao, n);
  const auto elements = matrix_elements(n);
  std::vector<double> temporary(elements), result(elements);
  tensor::cpu_gemm('N', 'N', n, n, n, ao.data(), coefficients.data(), temporary.data(), 1.0, 0.0,
                   plan);
  tensor::cpu_gemm('T', 'N', n, n, n, coefficients.data(), temporary.data(), result.data(), 1.0,
                   0.0, plan);
  return result;
}

std::vector<double> rank2_mo_to_ao(std::span<const double> coefficients,
                                   std::span<const double> mo, std::size_t n,
                                   const tensor::CpuLinalgPlan& plan) {
  validate(coefficients, mo, n);
  const auto elements = matrix_elements(n);
  std::vector<double> temporary(elements), result(elements);
  tensor::cpu_gemm('N', 'N', n, n, n, coefficients.data(), mo.data(), temporary.data(), 1.0, 0.0,
                   plan);
  tensor::cpu_gemm('N', 'T', n, n, n, temporary.data(), coefficients.data(), result.data(), 1.0,
                   0.0, plan);
  return result;
}

}  // namespace generativeqc::posthf
