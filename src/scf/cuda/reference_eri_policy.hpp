#pragma once

#include <cstddef>

namespace generativeqc::scf::cuda_execution {

inline constexpr std::size_t kReferenceEriCacheLimit = 256ULL << 20;

/** Use bounded shell-quartet work when the reference cannot use an ERI cache.
 * Keep the established s/p cache and its matrix-direct low-budget fallback.
 * d/f and larger s/p references instead share the ordinary exact quartet owner,
 * including one public-to-Cartesian density transform per Fock build. Angular
 * domains outside that owner's coverage retain the matrix-direct evaluator.
 * This topology decision is shared by host packing and device admission.
 */
constexpr bool reference_quartet_direct(std::size_t nbf, unsigned max_angular) noexcept {
  if (!nbf || max_angular > 3) return false;
  if (max_angular > 1) return true;
  std::size_t elements = 1;
  for (unsigned axis = 0; axis < 4; ++axis) {
    if (nbf > kReferenceEriCacheLimit / sizeof(double) / elements) return true;
    elements *= nbf;
  }
  return false;
}

/** Admit an optional s/p reference cache after all mandatory workspaces.
 * The 256 MiB ceiling bounds quartic storage independently of the caller's
 * budget. Higher angular momentum retains the qualified direct evaluator;
 * force consumers keep their separate generated derivative ownership.
 * Zero means that execution must use the already admitted direct fallback.
 */
constexpr std::size_t reference_eri_cache_bytes(std::size_t elements, unsigned max_angular,
                                                bool compute_forces, std::size_t required,
                                                std::size_t budget) noexcept {
  constexpr std::size_t limit = kReferenceEriCacheLimit;
  if (compute_forces || max_angular > 1 || elements > limit / sizeof(double) || required > budget)
    return 0;
  const auto bytes = elements * sizeof(double);
  return bytes <= budget - required ? bytes : 0;
}

}  // namespace generativeqc::scf::cuda_execution
