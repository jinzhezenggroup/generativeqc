#pragma once

#include <cmath>
#include <cstddef>

namespace generativeqc::dft::user_expression {

// Runtime transport and failure boundaries only. The compiler supplies the
// canonical scalar mathematics; no functional, derivative or SCF algebra lives
// here. Output is caller-owned scratch and must be discarded on nonzero status.
template <std::size_t OutputCount, std::size_t MaxPoints, class Scalar>
int points(const double* input, std::size_t input_count, double* output, std::size_t output_count,
           std::size_t npoint, Scalar scalar) noexcept {
  static_assert(OutputCount == 1 || OutputCount == 3 || OutputCount == 6);
  static_assert(MaxPoints <= 65536);
  if (npoint > MaxPoints || input_count != 2 * npoint || output_count != OutputCount * npoint)
    return -1;
  if (!npoint) return 0;
  if (!input || !output) return -1;
  // Validate both declared inputs before any output, including constant DAGs.
  for (std::size_t i = 0; i < input_count; ++i)
    if (!std::isfinite(input[i]) || input[i] < 0.0) return -2;
  for (std::size_t point = 0; point < npoint; ++point) {
    const double rho[2] = {input[point], input[npoint + point]};
    double values[OutputCount];
    scalar(rho, values);
    for (std::size_t row = 0; row < OutputCount; ++row)
      if (!std::isfinite(values[row])) return static_cast<int>(row + 1);
    for (std::size_t row = 0; row < OutputCount; ++row) output[row * npoint + point] = values[row];
  }
  return 0;
}

}  // namespace generativeqc::dft::user_expression
