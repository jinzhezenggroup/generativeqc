#pragma once

namespace generativeqc::tensor {

/** Optional device-side work receipt for actual newly refreshed Gram entries.
 * These count arithmetic vector work, not cache/solve traffic or endpoint time.
 * The caller owns one zero-initialized record per independent history ring. */
struct RingGramWork {
  unsigned long long dots{};
  unsigned long long vector_elements{};
};

}  // namespace generativeqc::tensor
