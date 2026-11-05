#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace generativeqc::scf {
/** Preserve the Hermitian AO-Fock invariant before storing Pulay history.
 * Independently accumulated J/K triangles differ by roundoff. Large but valid
 * DIIS coefficients can amplify that harmless skew beyond the eigensolver's
 * input tolerance. Validate with its existing gate, then store identical
 * triangles; never loosen the gate or repair a materially asymmetric source. */
inline void preserve_fock_symmetry(std::span<double> matrix, std::size_t nbf) {
  if (!nbf || matrix.size() / nbf != nbf || matrix.size() % nbf ||
      !std::all_of(matrix.begin(), matrix.end(), [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("DF Fock requires a finite square AO matrix");
  for (std::size_t row = 0; row < nbf; ++row)
    for (std::size_t column = row + 1; column < nbf; ++column) {
      auto& upper = matrix[row * nbf + column];
      auto& lower = matrix[column * nbf + row];
      if (std::abs(upper - lower) > 1e-12 * std::max({1.0, std::abs(upper), std::abs(lower)}))
        throw std::invalid_argument("DF Fock source violates AO symmetry before DIIS");
      upper = lower = 0.5 * upper + 0.5 * lower;
    }
}
}  // namespace generativeqc::scf
