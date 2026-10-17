#pragma once

#include <cstdint>

namespace generativeqc::scf::cuda_execution {

/** Reserved per-item update states. Full-energy refinement is sticky until
 * the next solve resets the controller, including when cadence is disabled. */
constexpr std::uint32_t kIncrementalDirectJkUnanchored = 0xffffffffU;
constexpr std::uint32_t kIncrementalDirectJkFullEnergyRefinement = 0xfffffffeU;

}  // namespace generativeqc::scf::cuda_execution
