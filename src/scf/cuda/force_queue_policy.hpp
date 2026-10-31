#ifndef GENERATIVEQC_SCF_CUDA_FORCE_QUEUE_POLICY_HPP
#define GENERATIVEQC_SCF_CUDA_FORCE_QUEUE_POLICY_HPP

#include <bit>
#include <cstdint>

namespace generativeqc::scf::cuda_policy {

/** Prefer the existing angular queue when the complete topology cannot supply
 * one persistent worker round per generated force class on average. This is a
 * conservative portable scheduling heuristic, not a scientific screening rule.
 * Callers retain generated consumers for bounded streaming and tuned profiles.
 */
inline constexpr bool portable_force_prefers_angular_queue(
    std::uint64_t shell_quartets, std::uint64_t generated_class_mask,
    unsigned persistent_worker_blocks) noexcept {
  const auto classes = static_cast<std::uint64_t>(std::popcount(generated_class_mask));
  // A 64-bit mask and 32-bit worker count cannot overflow this product. A
  // missing topology/worker description leaves the generated route unchanged.
  return shell_quartets != 0U && classes != 0U && persistent_worker_blocks != 0U &&
         shell_quartets <= classes * persistent_worker_blocks;
}

}  // namespace generativeqc::scf::cuda_policy

#endif
