#include "scf/solver/cpu_target_eigen.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "scf/solver/eigen_frame.hpp"
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
  reference::Matrix transformed(n * n);
  if (orthogonalizer) {
    // The transform workspace is released before entering the scalar leaf.
    reference::Matrix workspace(n * n);
    tensor::cpu_congruence('T', n, orthogonalizer->data(), matrix.data(), transformed.data(),
                           workspace.data(), plan);
  } else {
    std::copy(matrix.begin(), matrix.end(), transformed.begin());
  }
  auto frame = tensor::cpu_symmetric_eigen(std::move(transformed), n, plan, 1e-14);
  if (orthogonalizer) {
    reference::Matrix coefficients(n * n);
    tensor::cpu_gemm('N', 'N', n, n, n, orthogonalizer->data(), frame.vectors.data(),
                     coefficients.data(), 1.0, 0.0, plan);
    frame.vectors = std::move(coefficients);
  }
  EigenFrameDiagnostic diagnostic;
  std::string detail;
  if (!validate_eigen_frame(matrix, overlap, frame.values, frame.vectors, n, diagnostic, detail))
    throw std::runtime_error(detail);
  return {std::move(frame.values), std::move(frame.vectors)};
}
}  // namespace generativeqc::scf::solver
