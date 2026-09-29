#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

#include "posthf/capacity.hpp"

namespace generativeqc::mp2::detail {
/** Extra live storage outside the retained MP2 weight/derivative plan.
 *
 * block_peak(shape) reports one provider call's peak above its retained state,
 * including its returned block and any host/device transform staging. Previously
 * returned blocks belong to the caller and must be added before the next call.
 * No integral or response action is performed by this admission helper.
 */
template <class BlockPeak>
std::size_t streamed_force_workspace_bytes(std::size_t n, std::size_t occupied,
                                          BlockPeak&& block_peak) {
  if (!n || !occupied || occupied >= n)
    throw std::invalid_argument("invalid MP2 streamed workspace dimensions");
  const auto virtuals = n - occupied;
  const auto n2 = posthf::checked_mul(n, n);
  const auto n3 = posthf::checked_mul(n2, n);
  const auto rotations = posthf::checked_mul(occupied, virtuals);
  const auto amplitudes = posthf::checked_mul(rotations, rotations);
  auto bytes = [](std::size_t elements) {
    return posthf::checked_mul(elements, sizeof(double));
  };
  // The final adjoint is already retained in the component plan; raw and
  // reordered inputs coexist with it during canonical_energy_adjoint().
  auto peak = bytes(posthf::checked_mul(2, amplitudes));
  auto include = [&](std::array<std::size_t, 4> shape, std::size_t previous_elements) {
    std::size_t elements = 1;
    for (const auto extent : shape) elements = posthf::checked_mul(elements, extent);
    const auto call_peak = block_peak(shape);
    if (call_peak < bytes(elements))
      throw std::logic_error("MP2 provider peak omits its returned block");
    peak = std::max(peak, posthf::checked_add(bytes(previous_elements), call_peak));
  };
  include({occupied, virtuals, occupied, virtuals}, 0);
  // streamed_fock(): two N^2 blocks coexist.
  include({n, n, 1, 1}, 0);
  include({n, 1, 1, n}, n2);
  // rotation_gradient_streamed(): all four N^3 blocks coexist. Each axis can
  // have a different transform peak; do not infer one orientation from another.
  include({1, n, n, n}, 0);
  include({n, 1, n, n}, n3);
  include({n, n, 1, n}, posthf::checked_mul(2, n3));
  include({n, n, n, 1}, posthf::checked_mul(3, n3));
  // response_problem(): three occupied-virtual blocks coexist per action.
  include({1, 1, virtuals, occupied}, 0);
  include({1, virtuals, 1, occupied}, rotations);
  include({1, occupied, 1, virtuals}, posthf::checked_mul(2, rotations));
  // Also reserve the active rotation result and bounded caller-owned orbital
  // index lists, including temporary MOSlots copies. This is a conservative
  // allowance, not measured allocation telemetry or an extra dense N^4 weight.
  peak = posthf::checked_add(peak, bytes(n2));
  return posthf::checked_add(
      peak, posthf::checked_mul(posthf::checked_mul(64, n), sizeof(std::size_t)));
}

inline std::size_t remaining_force_budget(std::size_t budget, std::size_t workspace) {
  if (workspace >= budget)
    throw std::length_error("MP2 streamed workspace exceeds endpoint memory budget");
  return budget - workspace;
}
}  // namespace generativeqc::mp2::detail
