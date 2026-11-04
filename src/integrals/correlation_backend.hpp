#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "integrals/s_integrals.hpp"

namespace generativeqc::integrals {

/** Method-composed synchronous integral adapters for correlation consumers.
 * Values use the public AO representation, and gradients are positive electronic
 * derivatives without nuclear repulsion. Borrowed spans remain live until each
 * adapter drains its backend work. Missing adapters never permit CPU fallback. */
struct CorrelationIntegralBackend {
  DensityFittingIntegralData (*density_fitting)(int device, const core::System& orbital,
                                                const core::System& auxiliary, bool derivatives){};
  std::vector<double> (*one_electron_gradient)(int device, const core::System& system,
                                               std::span<const double> overlap,
                                               std::span<const double> hcore, std::size_t budget){};
};

}  // namespace generativeqc::integrals
