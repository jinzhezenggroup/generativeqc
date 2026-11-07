#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "dft/vv10_rebase_history.hpp"

namespace generativeqc::tests::vv10_rebase {
constexpr unsigned n = 3, history = 4;
constexpr std::size_t m = n * n;
using Matrix = std::array<double, m>;
constexpr std::size_t at(unsigned i, unsigned j) { return j * n + i; }

struct Case {
  Matrix overlap{}, old_potential{}, new_potential{};
  std::array<Matrix, history> density{}, fock{};
};
inline Case fixture(bool zero_delta = false) {
  Case value;
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < n; ++i) {
      const auto k = at(i, j);
      // SPD, nonorthogonal metric; F and D are intentionally nonsymmetric.
      value.overlap[k] = i == j ? 1.5 + 0.2 * i : 0.05 * (1 + i + j);
      value.old_potential[k] = 0.1 * (int(i) - int(j)) + (i == j ? 0.3 : 0.0);
      value.new_potential[k] =
          zero_delta ? value.old_potential[k] : 0.04 * (1 + int(i) - 2 * int(j));
      for (unsigned h = 0; h < history; ++h) {
        value.density[h][k] =
            ((i + j + h) % 2 ? -1.0 : 1.0) * (0.2 + 0.017 * (1 + i + 3 * j + h * h));
        value.fock[h][k] =
            0.03 * (1 + int(i) - 3 * int(j)) * (1 + h) + (i == (j + h) % n ? 0.8 + 0.13 * h : -0.2);
      }
    }
  return value;
}

/** Independent definition: directly contract F_i,k D_k,l S_l,j minus
 * S_i,k D_k,l F_l,j in long double. It does not use the DeltaV update. */
inline Matrix residual(const Matrix& f, const Matrix& d, const Matrix& s) {
  Matrix output{};
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < n; ++i) {
      long double value = 0;
      for (unsigned k = 0; k < n; ++k)
        for (unsigned l = 0; l < n; ++l)
          value += static_cast<long double>(f[at(i, k)]) * d[at(k, l)] * s[at(l, j)] -
                   static_cast<long double>(s[at(i, k)]) * d[at(k, l)] * f[at(l, j)];
      output[at(i, j)] = static_cast<double>(value);
    }
  return output;
}
inline Matrix product(const Matrix& a, const Matrix& b) {
  Matrix output{};
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < n; ++i)
      for (unsigned k = 0; k < n; ++k) output[at(i, j)] += a[at(i, k)] * b[at(k, j)];
  return output;
}
inline Matrix shifted_fock(const Case& value, unsigned slot) {
  Matrix f{};
  for (std::size_t i = 0; i < m; ++i)
    f[i] = value.fock[slot][i] + (value.new_potential[i] - value.old_potential[i]);
  return f;
}
inline void close(const Matrix& actual, const Matrix& expected) {
  for (std::size_t i = 0; i < m; ++i)
    assert(std::isfinite(actual[i]) &&
           std::abs(actual[i] - expected[i]) <= 3e-13 * (1 + std::abs(expected[i])));
}
inline double dot(const Matrix& a, const Matrix& b) {
  long double sum = 0;
  for (std::size_t i = 0; i < m; ++i) sum += static_cast<long double>(a[i]) * b[i];
  return static_cast<double>(sum);
}
}  // namespace generativeqc::tests::vv10_rebase
