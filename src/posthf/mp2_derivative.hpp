#pragma once

#include <vector>

#include "posthf/mp2_gradient.hpp"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::hf {
struct PhysicalReference;
}

namespace generativeqc::mp2 {

/** Contract relaxed canonical-MO Lagrangian weights with CPU derivatives.
 *
 * Returns positive total-energy derivatives. One-electron weights are pulled
 * back into public AO matrices. Four-index weights are transformed one shell
 * at a time and immediately consumed, so no molecular AO-rank-four cotangent
 * or coordinate-major derivative tensor exists.
 */
std::vector<double> conventional_derivative_cpu(const core::System& system,
                                                const hf::PhysicalReference& reference,
                                                const LagrangianWeights& weights);

/** Contract public-AO RI Lagrangian weights through the bounded #143 CPU
 * derivative consumers. Nuclear repulsion is added exactly once.
 */
std::vector<double> density_fitted_derivative_cpu(const core::System& orbital,
                                                  const core::System& auxiliary,
                                                  const DensityFittedLagrangianWeights& weights,
                                                  std::size_t stage_budget);

std::vector<double> conventional_derivative_cuda(const core::System& system,
                                                 const hf::PhysicalReference& reference,
                                                 const LagrangianWeights& weights, int device_id,
                                                 std::size_t stage_budget);

}  // namespace generativeqc::mp2
