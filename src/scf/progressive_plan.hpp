#pragma once

#include <cmath>
#include <cstdint>
#include <initializer_list>

#include "generativeqc/generativeqc.h"

namespace generativeqc::scf::progressive {

/** Numerical admission for the fixed source/target HF plan. Typed frontends
 * still validate graph/model identities; this decision never certifies target
 * accuracy or changes caller-selected solver controls. */
inline bool valid_plan(const generativeqc_hf_plan_descriptor_v1& input) noexcept {
  if (input.flags >> 3 || input.reserved || !input.source_iterations || !input.target_iterations ||
      !input.source_iteration_limit || !input.total_iteration_limit)
    return false;
  for (const auto value : {input.source_energy_tolerance, input.source_density_tolerance,
                           input.target_energy_tolerance, input.target_density_tolerance})
    if (!std::isfinite(value) || value <= 0.0) return false;
  const double costs[] = {input.source_cost, input.target_cost, input.cost_limit};
  for (unsigned bit = 0; bit < 3; ++bit)
    if ((input.flags & (1u << bit)) && (!std::isfinite(costs[bit]) || costs[bit] <= 0.0))
      return false;
  return true;
}

/** Report all unmet planning gates. Subtraction avoids overflow of the sum of
 * two full-width iteration counts. Cost addition deliberately preserves the
 * caller's binary64 two-stage estimate, including an overflowing estimate. */
inline std::uint32_t plan_failures(const generativeqc_hf_plan_descriptor_v1& input) noexcept {
  std::uint32_t failures{};
  if (input.source_energy_tolerance < input.target_energy_tolerance ||
      input.source_density_tolerance < input.target_density_tolerance)
    failures |= GENERATIVEQC_HF_PLAN_STRICTER_SOURCE;
  if (input.source_iterations > input.source_iteration_limit)
    failures |= GENERATIVEQC_HF_PLAN_SOURCE_ITERATIONS;
  if (input.source_iterations > input.total_iteration_limit ||
      input.target_iterations > input.total_iteration_limit - input.source_iterations)
    failures |= GENERATIVEQC_HF_PLAN_TOTAL_ITERATIONS;
  if (input.flags & GENERATIVEQC_HF_PLAN_COST_LIMIT_PRESENT) {
    if ((input.flags & 3u) != 3u)
      failures |= GENERATIVEQC_HF_PLAN_MISSING_COST;
    else if (input.source_cost + input.target_cost > input.cost_limit)
      failures |= GENERATIVEQC_HF_PLAN_TOTAL_COST;
  }
  return failures;
}

}  // namespace generativeqc::scf::progressive
