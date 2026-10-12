#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/electronic_reference.hpp"

namespace generativeqc::cc {

/** One synchronous, borrowed restricted-reference input to the RCCSD problem
 * builder. The producing method owner must remain alive until this call returns.
 * No orbital arrays, source handles or SCF-private solver objects are retained.
 * This adapter does not authorize correlated KS or unrestricted RCCSD methods. */
struct CanonicalReferenceInput {
  std::size_t basis_functions{}, occupied{}, virtuals{};
  std::span<const double> orbital_energies;
  double reference_energy{};
  // Identical row-major C^T F C convention to the previous RCCSD builder.
  std::vector<double> fock_mo;
};

/** Reject malformed/unsupported references before MO allocation or source
 * reads. Input spans are borrowed; the transformed Fock output is owned. */
[[nodiscard]] inline CanonicalReferenceInput prepare_canonical_reference_input(
    const core::ElectronicReferenceView& reference) {
  core::validate_electronic_reference_shape(reference);
  if (!reference.restricted())
    throw std::invalid_argument("RCCSD canonical input requires one restricted spin channel");
  const auto n = reference.basis_functions;
  const auto& orbital = reference.channels[0];
  if (!orbital.occupied || orbital.occupied >= n)
    throw std::invalid_argument("RCCSD canonical input requires occupied and virtual orbitals");
  const auto finite = [](std::span<const double> values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (!std::isfinite(reference.energy) || !finite(reference.overlap) || !finite(reference.hcore) ||
      !finite(orbital.coefficients) || !finite(orbital.orbital_energies) ||
      !finite(orbital.density) || !finite(orbital.fock) || !finite(orbital.weighted_density))
    throw std::invalid_argument("nonfinite RCCSD canonical reference input");

  CanonicalReferenceInput input;
  input.basis_functions = n;
  input.occupied = orbital.occupied;
  input.virtuals = n - orbital.occupied;
  input.orbital_energies = orbital.orbital_energies;
  input.reference_energy = reference.energy;

  std::vector<double> scratch(n * n);
  input.fock_mo.resize(n * n);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu)
        value += orbital.fock[mu * n + nu] * orbital.coefficients[nu * n + q];
      scratch[mu * n + q] = value;
    }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < n; ++mu)
        value += orbital.coefficients[mu * n + p] * scratch[mu * n + q];
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD MO Fock matrix");
      input.fock_mo[p * n + q] = value;
    }
  return input;
}

}  // namespace generativeqc::cc
