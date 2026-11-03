#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define GENERATIVEQC_FORCE_QUEUE_HD __host__ __device__
#else
#define GENERATIVEQC_FORCE_QUEUE_HD
#endif

namespace generativeqc::scf::cuda_execution {

inline constexpr unsigned kDirectSpdForceQueueClasses = 21U;

/** Convert a bounded s/p/d task histogram to disjoint exclusive queue offsets.
 * The producer admits each task once before this scan; grouping only changes
 * execution order, never shell screening, AO tiling or primitive ownership.
 * Counts must sum to at most the producer's fixed queue capacity.
 */
GENERATIVEQC_FORCE_QUEUE_HD constexpr std::uint32_t direct_force_class_queue_offsets(
    std::uint32_t (&counts)[kDirectSpdForceQueueClasses]) noexcept {
  std::uint32_t total = 0U;
  for (auto& count : counts) {
    const auto size = count;
    count = total;
    total += size;
  }
  return total;
}

}  // namespace generativeqc::scf::cuda_execution

#undef GENERATIVEQC_FORCE_QUEUE_HD
