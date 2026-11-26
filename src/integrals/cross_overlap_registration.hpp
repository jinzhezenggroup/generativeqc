#pragma once

#include <array>
#include <span>

namespace generativeqc::core {
struct System;
}

namespace generativeqc::integrals {
/** Compute <target|source> with the optional target displacement subtracted
 * from centre coordinates only. Systems and normalized shells stay borrowed. */
void cross_overlap_registered(const core::System& target, const core::System& source,
                              std::span<double> output,
                              const std::array<double, 3>* subtract_from_target);
}  // namespace generativeqc::integrals
