#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "integrals/ecp.hpp"

namespace generativeqc::integrals {

/** Explicit endpoint bounds, including retained outputs and primitive staging. */
struct EcpGaussianLimits {
  std::size_t workspace_bytes{512ULL * 1024 * 1024};
  std::uint64_t kernel_blocks{64ULL * 1024 * 1024};
};

/** Semantic work, not timings or cache-hit estimates. No oracle work is included. */
struct EcpGaussianWork {
  std::uint64_t shell_pairs{}, primitive_groups{}, radial_frontiers{}, kernel_blocks{};
  std::uint64_t cartesian_values{}, kernel_block_bound{};
  std::size_t workspace_bound_bytes{};
};

/** Fixed-density derivatives of sum_ab weights_ab <a|V_ECP|b>.
 * Weights are arbitrary ordered real AO weights, without an occupancy factor.
 * Only the final coordinate Hessian is retained: no coordinate-squared AO tensor.
 */
struct EcpWeightedJet {
  double value{};
  std::vector<double> gradient, hessian;
};

struct EcpWeightedSecondOrder {
  /** Local residual first, then the nonlocal projector sum. */
  std::array<EcpWeightedJet, 2> parts;
  EcpGaussianWork work;
};

/** Frozen ordered-AO weights: [value, left, right, mixed] for each residual part.
 * Local and nonlocal parts stay separate, so channel cancellation cannot hide
 * an invalid result. This is not a relaxed electronic or stationary Hessian.
 */
struct EcpWeightedMixed {
  std::array<std::array<double, 4>, 2> parts{};
  EcpGaussianWork work;
};

/** Exact structural specialization: power two, local/s/p/d projectors, s--f AOs.
 * Other legal residual operators retain the independently checked grid path.
 */
bool gaussian_ecp_supported(const core::System& system);

/** Evaluate the same analytic operator for the primal and its center derivatives. */
EcpData gaussian_ecp_integrals(const core::System& system, bool derivatives,
                               EcpGaussianWork* work = nullptr, EcpGaussianLimits limits = {});

/** Contract second derivatives before physical-atom scatter and global storage. */
EcpWeightedSecondOrder gaussian_ecp_weighted_second_order(const core::System& system,
                                                          std::span<const double> weights,
                                                          EcpGaussianLimits limits = {});

/** Query a complete order-two source pass retaining only eight scalar outputs.
 * This analytic specialization refuses unsupported operators; it does not
 * silently substitute an unqualified general-grid or finite-difference source.
 */
EcpGaussianLimits gaussian_ecp_weighted_mixed_plan(const core::System& system);

/** Consume the same normalized analytic pair jets once for both nuclear legs.
 * weights is an arbitrary finite ordered AO matrix, without occupancy factors;
 * left/right use atom/xyz order. Independent A/C and B/C slots remain distinct
 * until compiler-owned physical-motion projection. No coordinate Hessian or
 * coordinate/AO tensor is retained. Both residual parts publish by return only
 * after final owner validation. Mandatory limits include complete-source work.
 * The synchronous validator authenticates immutable system/weights/directions;
 * these borrowed inputs outlive the call. This supplies neither ionic tails,
 * moving density nor energy-weighted-density response, and does not certify SCF.
 */
EcpWeightedMixed gaussian_ecp_weighted_mixed(const core::System& system,
                                             std::span<const double> weights,
                                             std::span<const double> left,
                                             std::span<const double> right,
                                             EcpGaussianLimits limits,
                                             const std::function<void()>& validate_current);

/** Source-driven column provider over the existing order-two pair authority.
 * Borrows stable ordered weights/right weights and nuclear direction; their
 * lifetime/content validation belongs to the checked consumer. This raw provider
 * plan excludes that consumer's input snapshots. No SCF response is certified.
 */
EcpDirectionPlan gaussian_ecp_curvature_plan(const core::System& system);
EcpCurvatureData gaussian_ecp_curvature(const core::System& system, std::span<const double> weights,
                                        std::span<const double> weights_right,
                                        std::span<const double> right, EcpDirectionLimits limits,
                                        const std::function<void()>& validate_current);

/** Conservative owned workspace including immutable output publication. */
std::size_t gaussian_ecp_workspace_capacity(const core::System& system, bool derivatives);

/** Directional retention over the same analytic primitive/pair-jet authority. */
EcpDirectionPlan gaussian_ecp_direction_plan(const core::System& system);
EcpDirectionData gaussian_ecp_direction(const core::System& system,
                                        std::span<const double> direction,
                                        EcpDirectionLimits limits,
                                        const std::function<void()>& validate_current);

}  // namespace generativeqc::integrals
