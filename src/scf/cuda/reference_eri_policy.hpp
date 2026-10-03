#pragma once

#include <cstddef>

namespace generativeqc::scf::cuda_execution {

/** Admit an optional s/p reference cache after all mandatory workspaces.
 * The 256 MiB ceiling bounds quartic storage independently of the caller's
 * budget. Higher angular momentum retains the qualified direct evaluator;
 * force consumers keep their separate generated derivative ownership.
 * Zero means that execution must use the already admitted direct fallback.
 */
constexpr std::size_t reference_eri_cache_bytes(std::size_t elements, unsigned max_angular,
                                                bool compute_forces, std::size_t required,
                                                std::size_t budget) noexcept {
  constexpr std::size_t limit = 256ULL << 20;
  if (compute_forces || max_angular > 1 || elements > limit / sizeof(double) || required > budget)
    return 0;
  const auto bytes = elements * sizeof(double);
  return bytes <= budget - required ? bytes : 0;
}

}  // namespace generativeqc::scf::cuda_execution
