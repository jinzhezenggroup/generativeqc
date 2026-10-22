#pragma once
#include <cstddef>

namespace generativeqc::cc::detail {
/** Expanded replay uses scalar Q when optional preparation/allocation left the
 * stored batch at zero. Zero remaining Q is absence, not a scalar operation.
 * Share this rule with read-only admission reporting to avoid sentinel drift. */
constexpr std::size_t replay_tile_width(std::size_t remaining, std::size_t prepared_batch,
                                        bool physical_replay) noexcept {
  if (!remaining) return 0;
  return physical_replay && prepared_batch
             ? (prepared_batch < remaining ? prepared_batch : remaining)
             : std::size_t{1};
}
}  // namespace generativeqc::cc::detail
