#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "generativeqc/precision_admission.h"

namespace generativeqc::runtime {

/** Preserve small positive contributions without platform-dependent NumPy
 * pairwise brackets. Inputs are normalized squares in [0,1]; storage is fixed. */
struct ObservableSquares {
  double sum{}, correction{};

  void add(double value) noexcept {
    const auto next = sum + value;
    correction += sum >= value ? (sum - next) + value : (value - next) + sum;
    sum = next;
  }

  double rms(double maximum, std::size_t count) const noexcept {
    // RMS cannot exceed its largest component, including rounded mean overshoot.
    const auto mean = std::min(1.0, (sum + correction) / static_cast<double>(count));
    return maximum * std::sqrt(mean);
  }
};

/** Pure numerical owner, with no model admission or electronic execution. The
 * caller publishes only after this routine has read every borrowed input. */
inline generativeqc_status compare_observables(
    const generativeqc_observable_comparison_context_v1& context, const double* forces,
    const double* reference, generativeqc_observable_comparison_v1& result) noexcept {
  const auto mask = context.requested_mask;
  if (!mask || mask > 7 || context.reserved) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const bool force_requested = mask & 6u;
  if (force_requested) {
    const auto limit = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double) / 3;
    if (!forces || !reference || !context.atom_count || context.atom_count > limit)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } else if (forces || reference || context.atom_count) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  for (unsigned slot = 0; slot != 3; ++slot) {
    const auto absolute = context.absolute[slot], relative = context.relative[slot];
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0 || relative < 0 ||
        ((mask & (1u << slot)) ? (absolute == 0 && relative == 0)
                               : (absolute != 0 || relative != 0)))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  result = {sizeof(result), GENERATIVEQC_ABI_VERSION, 1, mask, 0, 0, {}, {}, {}};
  if (mask & 1u) {
    if (!std::isfinite(context.energy) || !std::isfinite(context.reference_energy))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    result.differences[0] = std::abs(context.energy - context.reference_energy);
    result.reference_norms[0] = std::abs(context.reference_energy);
  } else if (context.energy != 0 || context.reference_energy != 0) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  double maximum_difference{}, maximum_reference{};
  const auto count = static_cast<std::size_t>(context.atom_count) * 3;
  for (std::size_t component = 0; component != count; ++component) {
    if (!std::isfinite(forces[component]) || !std::isfinite(reference[component]))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto difference = std::abs(forces[component] - reference[component]);
    if (!std::isfinite(difference)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    maximum_difference = std::max(maximum_difference, difference);
    maximum_reference = std::max(maximum_reference, std::abs(reference[component]));
  }
  if (mask & 2u) {
    result.differences[1] = maximum_difference;
    result.reference_norms[1] = maximum_reference;
  }
  if (mask & 4u) {
    ObservableSquares differences, references;
    for (std::size_t component = 0; component != count; ++component) {
      if (maximum_difference) {
        const auto normalized = (forces[component] - reference[component]) / maximum_difference;
        differences.add(normalized * normalized);
      }
      if (maximum_reference) {
        const auto normalized = reference[component] / maximum_reference;
        references.add(normalized * normalized);
      }
    }
    result.differences[2] = differences.rms(maximum_difference, count);
    result.reference_norms[2] = references.rms(maximum_reference, count);
  }
  for (unsigned slot = 0; slot != 3; ++slot) {
    if (!(mask & (1u << slot))) continue;
    // Keep the documented multiply-then-add rounding even on FMA-capable hosts.
    volatile double product = context.relative[slot] * result.reference_norms[slot];
    result.allowances[slot] = context.absolute[slot] + product;
    if (!std::isfinite(result.differences[slot]) || !std::isfinite(result.allowances[slot]))
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    if (result.differences[slot] <= result.allowances[slot]) result.met_mask |= 1u << slot;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::runtime
