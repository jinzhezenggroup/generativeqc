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

/** Conservative owned workspace including immutable output publication. */
std::size_t gaussian_ecp_workspace_capacity(const core::System& system, bool derivatives);

}  // namespace generativeqc::integrals
