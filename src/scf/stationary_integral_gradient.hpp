#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace generativeqc::core {
struct System;
}

namespace generativeqc::scf {
class PreparedFockPlan;

/** Preserve the existing public Direct s/p/d force work ceilings: two million
 * primitive records, one million grid points and one hundred million Becke
 * pair visits. The conservative primitive bound retains independent J/K pass
 * charges even though the native owner shares their recurrence. Zero grid
 * points applies only the integral-source admission gate. */
bool cpu_stationary_direct_force_work_admitted(const core::System& system, bool pbe0,
                                               std::size_t grid_points,
                                               std::size_t tile_points) noexcept;

/** Compact hcore, Pulay, Coulomb and exchange positive derivatives, in that order.
 * Capacity covers owned numeric publication/weight buffers, not recurrence
 * scratch, allocator overhead or the retained SCF owner. Semantic counts are
 * actual shell submissions and shared primitive evaluations, not AO-rank powers.
 */
struct CpuStationaryIntegralGradient {
  std::vector<double> sources;
  std::size_t numeric_capacity{};
  std::uint64_t shell_quartets{}, primitive_evaluations{};
};

/** Consume occupation-weighted restricted D/W using the prepared exact provider's
 * immutable system and compiler-emitted stationary AD weights. The method owner
 * authenticates the PBE/PBE0 composition and live final-state token. This helper
 * validates the exact full-range source and contracts bounded shell-local weights
 * into derivatives; it neither owns an SCF solver nor publishes a final force.
 */
CpuStationaryIntegralGradient cpu_stationary_integral_gradient(
    const PreparedFockPlan& target, std::span<const double> density,
    std::span<const double> weighted_density, bool pbe0, std::size_t maximum_numeric_bytes);
}  // namespace generativeqc::scf
