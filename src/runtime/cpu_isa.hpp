#pragma once

#include <cstdint>

namespace generativeqc::runtime {

/** Internal kernel identity, not an automatic CPU-feature selection policy.
 * Consumers must only bind compiled and independently admitted kernels. The
 * production molecular CPU runtime continues to select the baseline explicitly.
 */
enum class CpuIsa : std::uint8_t { kBaseline, kAvx2Fma };

}  // namespace generativeqc::runtime
