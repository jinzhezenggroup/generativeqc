#pragma once

#include <cstdint>

namespace generativeqc::core {

/** Execution choice independent of the physical convergence allowance.
 * Legacy preserves existing callers. Mixed requests bounded auxiliary warmup;
 * the solver still publishes only after its selected FP64 solve and replay.
 * Resource admission may refuse warmup without changing the final controls. */
enum class DfPreconvergence : std::uint32_t { Legacy = 0, Fp64 = 1, Mixed = 2, Automatic = 3 };

}  // namespace generativeqc::core
