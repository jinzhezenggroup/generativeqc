#pragma once

#include <string_view>

namespace generativeqc::scf::cuda_execution {

/** Capture execution preference at preparation, without encoding DFT policy.
 * Automatic needs an authenticated method-owned eligibility hint at execution;
 * an explicit request still cannot bypass the scheduler's resource guards. */
enum class DirectSpdForceSchedule : unsigned char {
  Incumbent,
  Automatic,
  SpatialReuse,
  RelativeReuse,
  DensityReuse
};

/** Bounded experimental queue sizes; unknown requests retain the incumbent.
 * This is task ownership only: each admitted task still uses all 256 lanes of
 * the same generated derivative. Capture the preference with the geometry. */
inline constexpr unsigned select_direct_order_seven_page_size(std::string_view selector) noexcept {
  return selector == "8" ? 8U : selector == "16" ? 16U : 32U;
}

/** Unknown selectors fail closed; zero overrides even an eligible method hint.
 * Relative recurrence reuse is an explicit experiment, not an automatic default. */
inline constexpr DirectSpdForceSchedule select_direct_spd_force_schedule(
    std::string_view selector) noexcept {
  if (selector.empty()) return DirectSpdForceSchedule::Automatic;
  if (selector == "1" || selector == "spatial-reuse") return DirectSpdForceSchedule::SpatialReuse;
  if (selector == "relative-reuse") return DirectSpdForceSchedule::RelativeReuse;
  if (selector == "relative-density") return DirectSpdForceSchedule::DensityReuse;
  return DirectSpdForceSchedule::Incumbent;
}

/** Generic/HF/RSH consumers supply no hint and keep their original default. */
inline constexpr DirectSpdForceSchedule resolve_direct_spd_force_schedule(
    DirectSpdForceSchedule captured, bool qualified_default) noexcept {
  if (captured == DirectSpdForceSchedule::Automatic)
    return qualified_default ? DirectSpdForceSchedule::SpatialReuse
                             : DirectSpdForceSchedule::Incumbent;
  return captured;
}
}  // namespace generativeqc::scf::cuda_execution
