#pragma once

#include <array>
#include <cstddef>

#include "hf/reference.hpp"
#include "integrals/electron_interaction_source.hpp"

namespace generativeqc::posthf {
class NativeBlockProvider;
}

namespace generativeqc::mp2 {

struct UnrestrictedEnergy {
  std::array<double, 3> channels{};  // alpha-alpha, beta-beta, alpha-beta
  double minimum_denominator{};
  std::size_t numeric_capacity_bytes{};
  std::size_t tiles{};
  std::array<const char*, 3> equation_hashes{};
};

/** Validate source identity and every active denominator before any source read. */
double validate_unrestricted_energy_input(const hf::UnrestrictedPhysicalReference& reference,
                                          const integrals::ElectronInteractionSource& source,
                                          double denominator_threshold);

/** Peak provider and host-feed payload for active occupied/virtual CUDA channels.
 * This shape-only query neither reads AO integrals nor touches a CUDA device.
 * Provider output storage is charged by the provider, not again as a feed. */
std::size_t unrestricted_cuda_energy_host_capacity(
    const hf::UnrestrictedPhysicalReference& reference,
    const posthf::NativeBlockProvider& provider);

UnrestrictedEnergy conventional_unrestricted_energy(
    const hf::UnrestrictedPhysicalReference& reference,
    const integrals::ElectronInteractionSource& source, std::size_t budget,
    double denominator_threshold, bool cuda = false, int device = 0);

}  // namespace generativeqc::mp2
