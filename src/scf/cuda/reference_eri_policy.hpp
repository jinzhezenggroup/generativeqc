#pragma once

#include <cstddef>
#include <string_view>

namespace generativeqc::scf::cuda_execution {

inline constexpr std::size_t kReferenceEriCacheLimit = 256ULL << 20;

enum class ReferenceResidentValuesSelection { automatic, disabled, forced, invalid };

/** Unset selects bounded automatic routing; explicit 0/1 retain A/B controls. */
constexpr ReferenceResidentValuesSelection reference_resident_values_selection(
    const char* value) noexcept {
  if (!value || std::string_view(value) == "auto")
    return ReferenceResidentValuesSelection::automatic;
  if (std::string_view(value) == "0") return ReferenceResidentValuesSelection::disabled;
  if (std::string_view(value) == "1") return ReferenceResidentValuesSelection::forced;
  return ReferenceResidentValuesSelection::invalid;
}

/** Conservative cold/fallback-domain heuristic, separate from memory legality.
 * 128 Cartesian AOs imply at least 256 MiB of canonical values. Smaller and
 * complete generated schedules retain their low-setup path. The iteration
 * limit excludes short requests; it is not a prediction of actual convergence.
 * A known warm bucket is never charged a fresh source by automatic routing.
 */
constexpr const char* reference_resident_values_auto_refusal(
    std::size_t nbf, std::size_t direct_nbf, unsigned maximum_iterations, unsigned maximum_angular,
    bool cold_reference, bool generic_fock) noexcept {
  if (!cold_reference) return "auto: reused reference bucket";
  if (maximum_angular != 3 || !generic_fock) return "auto: retain generated/lower-angular schedule";
  if (nbf < 64 || direct_nbf < 128) return "auto: source below cold reuse threshold";
  if (maximum_iterations < 8) return "auto: short iteration limit";
  return nullptr;
}

/** Phase-local canonical values amortize setup only for repeated exact actions.
 * Small references retain their established schedule. Metadata, compensation,
 * scratch and host preparation must fit before optional quartic storage; a
 * rejected allowance leaves the already admitted exact Fock route untouched.
 */
constexpr std::size_t reference_resident_value_allowance(std::size_t nbf,
                                                         unsigned maximum_iterations,
                                                         std::size_t required, std::size_t overhead,
                                                         std::size_t budget,
                                                         std::size_t ceiling) noexcept {
  if (nbf < 64 || maximum_iterations < 4 || required > budget || overhead > budget - required)
    return 0;
  const auto remaining = budget - required - overhead;
  return remaining < ceiling ? remaining : ceiling;
}

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
