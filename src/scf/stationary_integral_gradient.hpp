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
struct ResolvedFockBuild;

/** Validate the packaged Direct s/p/d inventory and checked work counters.
 * Source pages have bounded storage independently of the molecular size.
 * Actual memory admission belongs to the owning resource ledger; historical
 * test sizes and conservative primitive counts are not capability limits. */
bool cpu_stationary_direct_force_work_admitted(const core::System& system, bool hybrid,
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

/** Consume prepared full-range Fock exchange weights, never a named method selector.
 * The prepared calculation independently verifies physical/capability admission. */
/** Consume occupation-weighted RKS D/W or separately occupied UKS alpha/beta
 * D/W using the prepared exact provider's
 * immutable system and compiler-emitted stationary AD weights. The method owner
 * authenticates the MethodIR/Fock composition and the live final-state token. This helper
 * validates the exact full-range source and contracts bounded shell-local weights
 * into derivatives; it neither owns an SCF solver nor publishes a final force.
 * UKS requires both beta blocks, including explicit zero matrices for an empty
 * beta population. Compact flattening serves the compiler-owned spin layout;
 * its live capacities are charged, without copying any density equations.
 */
CpuStationaryIntegralGradient cpu_stationary_integral_gradient(
    const PreparedFockPlan& target, std::span<const double> density,
    std::span<const double> weighted_density, std::size_t maximum_numeric_bytes,
    std::span<const double> beta_density = {}, std::span<const double> beta_weighted_density = {});

/** Add the operator-matched LR correction to the exchange source of a full-range
 * baseline. Uses the compiler's shared CPU/CUDA Cartesian recurrence, with
 * bounded public-AO batches and no rank-four derivative tensor. */
std::vector<double> cpu_stationary_range_correction(const PreparedFockPlan& target,
                                                    const ResolvedFockBuild& correction,
                                                    std::span<const double> density,
                                                    std::span<const double> beta_density,
                                                    std::size_t maximum_bytes);
}  // namespace generativeqc::scf
