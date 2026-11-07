#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

#include "core/types.hpp"
#include "generated_nuclear_repulsion.hpp"

namespace generativeqc::molecule {
/** Add ionic nuclear repulsion once to an electronic gradient. This small
 * pairwise host assembly is shared by native post-HF force owners; it performs
 * no electronic integral or reference-oracle work. Scalar Coulomb derivatives
 * are compiler emitted; this adapter validates geometry and scatters atoms. */
inline void add_nuclear_repulsion_gradient(const core::System& system, std::span<double> gradient) {
  if (gradient.size() != 3 * system.atoms.size())
    throw std::invalid_argument("nuclear gradient shape mismatch");
  for (std::size_t a = 0; a < system.atoms.size(); ++a)
    for (std::size_t b = 0; b < a; ++b) {
      double distance2 = 0.0;
      std::array<double, 3> displacement{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        displacement[axis] = system.atoms[a].position[axis] - system.atoms[b].position[axis];
        distance2 += displacement[axis] * displacement[axis];
      }
      if (!(distance2 > 0.0) || !std::isfinite(distance2))
        throw std::invalid_argument("nuclear repulsion derivative has coincident atoms");
      std::array<double, 3> pair_gradient{};
      ::generativeqc::generated::nuclear_repulsion::gradient(
          static_cast<double>(system.atoms[a].ionic_charge()) * system.atoms[b].ionic_charge(),
          displacement[0], displacement[1], displacement[2], pair_gradient.data());
      for (std::size_t axis = 0; axis < 3; ++axis) {
        gradient[3 * a + axis] += pair_gradient[axis];
        gradient[3 * b + axis] -= pair_gradient[axis];
      }
    }
}
}  // namespace generativeqc::molecule
