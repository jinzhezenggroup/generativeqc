#pragma once

#include <string_view>

namespace generativeqc::scf::cuda_execution {

/** Capture execution preference at preparation, without encoding DFT policy.
 * Automatic needs an authenticated method-owned eligibility hint at execution;
 * an explicit request still cannot bypass the scheduler's resource guards. */
enum class DirectSpdForceSchedule : unsigned char { Incumbent, Automatic, SpatialReuse };

/** Unknown selectors fail closed; zero overrides even an eligible method hint. */
inline constexpr DirectSpdForceSchedule select_direct_spd_force_schedule(
    std::string_view selector) noexcept {
  if (selector.empty()) return DirectSpdForceSchedule::Automatic;
  if (selector == "1" || selector == "spatial-reuse") return DirectSpdForceSchedule::SpatialReuse;
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
