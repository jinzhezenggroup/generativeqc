#include "scf/solver/cpu_target_eigen.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "scf/solver/eigen_frame.hpp"
#include "solver/cpu/generalized_eigen.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::scf::solver {
std::size_t cpu_target_eigen_workspace_bytes(std::size_t n) {
  constexpr auto limit = std::numeric_limits<std::size_t>::max();
  constexpr auto value_bytes = sizeof(double), index_bytes = sizeof(std::size_t);
  if (n == 0 || n > limit / n || n * n > limit / (4 * value_bytes) ||
      n > limit / (value_bytes + index_bytes))
    throw std::overflow_error("CPU target eigen workspace dimensions overflow");
  const auto matrix_bytes = n * n * value_bytes;
  const auto vectors_bytes = n * (value_bytes + index_bytes);
  if (4 * matrix_bytes > limit - vectors_bytes)
    throw std::overflow_error("CPU target eigen workspace dimensions overflow");
  // Scalar sorting: input + rotations + sorted C + values + sort indices.
  // Validation: C + SC + transpose(C) + Gram + values; FC is not yet allocated.
  return std::max(3 * matrix_bytes + vectors_bytes, 4 * matrix_bytes + n * value_bytes);
}

reference::EigenResult cpu_target_eigen(const reference::Matrix& matrix,
                                        const reference::Matrix* overlap,
                                        const reference::Matrix* orthogonalizer, std::size_t n) {
  (void)cpu_target_eigen_workspace_bytes(n);  // Check all products before allocation.
  const auto valid = [n](const auto& values) {
    return values.size() == n * n &&
           std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
  };
  if (!valid(matrix) || bool(overlap) != bool(orthogonalizer) ||
      (overlap && (!valid(*overlap) || !valid(*orthogonalizer))))
    throw std::invalid_argument("CPU target eigen inputs require matching finite F/S/X matrices");
  constexpr tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::scalar,
                                       tensor::CpuLinalgThreadOwnership::task_parallel, 1};
  namespace shared = ::generativeqc::solver;
  const shared::GeneralizedEigenDomain domain{n, 1, 1, shared::GeneralizedEigenLayout::row_major};
  const auto basis = orthogonalizer ? shared::GeneralizedEigenBasis::canonical_x
                                    : shared::GeneralizedEigenBasis::identity;
  reference::Matrix transformed(n * n);
  {
    // The transform workspace is released before entering the scalar leaf.
    reference::Matrix workspace(orthogonalizer ? n * n : 0);
    const shared::GeneralizedEigenMatrices matrices{
        matrix.data(),
        orthogonalizer ? orthogonalizer->data() : nullptr,
        workspace.data(),
        transformed.data(),
        nullptr,
        matrix.size(),
        orthogonalizer ? orthogonalizer->size() : 0,
        workspace.size(),
        transformed.size(),
        0};
    if (shared::reduce_generalized_eigen(
            basis, shared::cpu::GeneralizedEigenLowering{domain, matrices, plan}) != 0)
      throw std::invalid_argument("CPU target eigen reduction binding is invalid");
  }
  auto frame = tensor::cpu_symmetric_eigen(std::move(transformed), n, plan, 1e-14);
  if (orthogonalizer) {
    reference::Matrix coefficients(n * n);
    const shared::GeneralizedEigenMatrices matrices{
        nullptr, orthogonalizer->data(), nullptr, frame.vectors.data(), coefficients.data(),
        0,       orthogonalizer->size(), 0,       frame.vectors.size(), coefficients.size()};
    if (shared::recover_generalized_eigen(
            basis, shared::cpu::GeneralizedEigenLowering{domain, matrices, plan}) != 0)
      throw std::invalid_argument("CPU target eigen recovery binding is invalid");
    frame.vectors = std::move(coefficients);
  }
  EigenFrameDiagnostic diagnostic;
  std::string detail;
  if (!validate_eigen_frame(matrix, overlap, frame.values, frame.vectors, n, diagnostic, detail))
    throw std::runtime_error(detail);
  return {std::move(frame.values), std::move(frame.vectors)};
}
}  // namespace generativeqc::scf::solver
