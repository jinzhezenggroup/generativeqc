#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "generated_weighted_gram_native.hpp"

namespace generativeqc::tensor::weighted_gram {

// Only the primitive LP64 provider binding crosses this boundary. Shapes and
// borrowed storage have already been validated by the calling prepared plan.
using CblasDgemmLp64 = void (*)(int layout, int transpose_left, int transpose_right,
                                std::int32_t rows, std::int32_t columns, std::int32_t inner,
                                double alpha, const double* left, std::int32_t leading_left,
                                const double* right, std::int32_t leading_right, double beta,
                                double* result, std::int32_t leading_result);

// This transform deliberately remains separate from any thermodynamic reduction:
// callers retain the original order and number of occupation*eigenvalue products.
inline bool energy_weights_inplace(std::int32_t n, const double* eigenvalues, double* weights) {
  const std::size_t dimension = static_cast<std::size_t>(n);
  for (std::size_t orbital = 0u; orbital < dimension; ++orbital) {
    if (!generated::energy_weight(weights[orbital], eigenvalues[orbital], weights[orbital])) {
      return false;
    }
  }
  return true;
}

// C diag(w) C^T, using the caller's existing column-major scaling panel. Checked
// scalar scaling precedes the sole DGEMM call; a failed element leaves the output
// matrix untouched and the already-computed panel prefix available to the caller.
template <class Algorithm = generated::CpuExecution, class Provider = CblasDgemmLp64>
inline bool execute_column_major(Provider dgemm, std::int32_t n, const double* coefficients,
                                 const double* weights, double* weighted_coefficients,
                                 double* density) {
  static_assert(std::is_same_v<Algorithm, generated::ColumnScaleDgemmLp64>,
                "selected weighted-Gram candidate has no native executor");
  const std::size_t dimension = static_cast<std::size_t>(n);
  for (std::size_t orbital = 0u; orbital < dimension; ++orbital) {
    for (std::size_t row = 0u; row < dimension; ++row) {
      const std::size_t index = row + orbital * dimension;
      if (!generated::weighted_coefficient(coefficients[index], weights[orbital],
                                           weighted_coefficients[index])) {
        return false;
      }
    }
  }
  constexpr int kCblasColMajor = 102;
  constexpr int kCblasNoTrans = 111;
  constexpr int kCblasTrans = 112;
  dgemm(kCblasColMajor, kCblasNoTrans, kCblasTrans, n, n, n, 1.0, weighted_coefficients, n,
        coefficients, n, 0.0, density, n);
  return true;
}

}  // namespace generativeqc::tensor::weighted_gram
