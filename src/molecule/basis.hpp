#ifndef GENERATIVEQC_MOLECULE_BASIS_HPP
#define GENERATIVEQC_MOLECULE_BASIS_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::molecule {

/** Cartesian exponent triple in CCA order (xx, xy, xz, yy, yz, zz for d). */
using CartesianComponent = std::array<unsigned, 3>;

struct CartesianExpansionTerm {
  CartesianComponent component{};
  // Coefficient multiplying a normalized Cartesian AO of the same shell.
  double coefficient{};
};

using AoExpansion = std::vector<CartesianExpansionTerm>;

/** Maximum terms in the legacy through-f CUDA topology (CPU uses vectors). */
inline constexpr std::size_t kMaximumAoExpansionTerms = 3;
// Auxiliary spherical g has up to six Cartesian terms. Consumers must opt in
// with matching metadata stride; the established s/p/d/f packing stays at three.
inline constexpr std::size_t kMaximumAuxiliaryAoExpansionTerms = 6;

/** Number of Cartesian functions in a shell of angular momentum `l`. */
[[nodiscard]] constexpr std::size_t cartesian_count(unsigned l) noexcept {
  const std::uint64_t n = l;
  const std::uint64_t count = n % 2 == 0 ? (n + 1) * ((n + 2) / 2) : ((n + 1) / 2) * (n + 2);
  return count > std::numeric_limits<std::size_t>::max() ? 0 : static_cast<std::size_t>(count);
}

/** Public shell dimension, shared by normalized systems and shape-only plans.
 * Zero reports an unrepresentable dimension rather than wrapped size_t data. */
[[nodiscard]] constexpr std::size_t shell_ao_count(
    unsigned angular_momentum, generativeqc_basis_representation representation) noexcept {
  if (representation != GENERATIVEQC_BASIS_SPHERICAL) return cartesian_count(angular_momentum);
  const std::uint64_t count = 2 * static_cast<std::uint64_t>(angular_momentum) + 1;
  return count > std::numeric_limits<std::size_t>::max() ? 0 : static_cast<std::size_t>(count);
}

/** Generate Cartesian functions in the CCA ordering used by libcint/PySCF. */
[[nodiscard]] std::vector<CartesianComponent> cartesian_components(unsigned l);

/**
 * Expand each public AO into normalized Cartesian components.
 *
 * Cartesian mode returns one unit-coefficient term per AO. Spherical mode
 * returns real solid harmonics in the PySCF/libcint ordering used by the
 * independent numerical oracle.
 */
[[nodiscard]] std::vector<AoExpansion> ao_expansions(
    unsigned l, generativeqc_basis_representation representation);

/** Total Cartesian AO count represented by a system's shells. */
[[nodiscard]] std::size_t ao_count(const core::System& system) noexcept;

[[nodiscard]] std::size_t cartesian_ao_count(const core::System& system) noexcept;

/**
 * Cartesian normalization factor not contained in a shell's radial factor.
 *
 * The validator folds `(2a/pi)^(3/4) (4a)^(l/2)` and contraction
 * normalization into each primitive coefficient.  This factor supplies the
 * remaining `1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)` for an AO component.
 */
[[nodiscard]] double cartesian_component_normalization(
    const CartesianComponent& component) noexcept;

/** Validate and radially normalize contracted Cartesian Gaussian shells. */
generativeqc_status validate_and_normalize(core::System& system, std::string& detail);

}  // namespace generativeqc::molecule

#endif
