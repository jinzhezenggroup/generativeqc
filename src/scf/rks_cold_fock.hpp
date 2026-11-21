#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "core/work_budget.hpp"

namespace generativeqc::scf {

/** Fixed prefix of cold RKS orbital proposals, included in the normal SCF
 * iteration budget. The first stores an undamped proposal; subsequent prefix
 * iterations mix in the previous proposal. DIIS starts after this prefix. */
struct RksColdFockDamping {
  unsigned prefix_iterations;
  double previous_weight;
  std::size_t maximum_bytes;

  /** Admission is shared by preparation and execution. Only the extra retained
   * AO-square proposal is charged here, not the ordinary SCF workspace. */
  [[nodiscard]] std::size_t proposal_bytes(std::size_t basis, unsigned max_iterations) const {
    if (prefix_iterations < 2 || prefix_iterations >= max_iterations ||
        !std::isfinite(previous_weight) || previous_weight <= 0 || previous_weight >= 1 ||
        !maximum_bytes || !basis)
      throw std::invalid_argument("invalid cold RKS Fock proposal controls");
    if (basis > std::numeric_limits<std::size_t>::max() / sizeof(double) / basis ||
        basis * basis * sizeof(double) > maximum_bytes)
      throw core::WorkLimitError("cold RKS Fock proposal exceeds its numerical workspace cap");
    return basis * basis * sizeof(double);
  }
};

}  // namespace generativeqc::scf
