#pragma once

#include <array>
#include <cstddef>

#include "hf/reference.hpp"
#include "integrals/electron_interaction_source.hpp"

namespace generativeqc::mp2 {

struct UnrestrictedEnergy {
  std::array<double, 3> channels{};  // alpha-alpha, beta-beta, alpha-beta
  double minimum_denominator{};
  std::size_t numeric_capacity_bytes{};
  std::size_t tiles{};
  std::array<const char*, 3> equation_hashes{};
};

UnrestrictedEnergy conventional_unrestricted_energy(
    const hf::UnrestrictedPhysicalReference& reference,
    const integrals::ElectronInteractionSource& source, std::size_t budget,
    double denominator_threshold);

}  // namespace generativeqc::mp2
