#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace generativeqc::scf::cuda_execution {

enum class DirectJkSourceSelection { automatic, disabled, forced, invalid };

/** RHF keeps its qualified domain; KS also admits bounded screened SPD sources. */
enum class DirectJkSourceAutoPolicy { generic_f, screened_cold };

/** Capture an experimental selection once per phase, never in kernel replay.
 * Callers supply method-specific controls and the incumbent value ceiling.
 * Unset controls preserve complete residency; invalid syntax never truncates.
 */
struct DirectJkResidentBucketPolicy {
  std::uint32_t mask{(1U << 28) - 1};
  std::size_t ceiling{8ULL << 30};
  bool valid{true};
};

inline DirectJkResidentBucketPolicy direct_jk_resident_bucket_policy(
    const char* mask, const char* ceiling, std::size_t default_ceiling = 8ULL << 30) noexcept {
  DirectJkResidentBucketPolicy policy;
  policy.ceiling = default_ceiling;
  const auto parse = [](const char* text, auto& output) {
    if (!text) return true;
    std::string_view value(text);
    int base = 10;
    if (value.starts_with("0x")) {
      value.remove_prefix(2);
      base = 16;
    }
    if (value.empty()) return false;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), output, base);
    return result.ec == std::errc{} && result.ptr == value.data() + value.size();
  };
  policy.valid =
      parse(mask, policy.mask) && policy.mask < (1U << 28) && parse(ceiling, policy.ceiling);
  return policy;
}

/** Unset is automatic; forced never means unbounded or scientifically unchecked. */
constexpr DirectJkSourceSelection direct_jk_source_selection(const char* value) noexcept {
  if (!value || std::string_view(value) == "auto") return DirectJkSourceSelection::automatic;
  if (std::string_view(value) == "0") return DirectJkSourceSelection::disabled;
  if (std::string_view(value) == "1") return DirectJkSourceSelection::forced;
  return DirectJkSourceSelection::invalid;
}

/** Cold resource candidates, not a prediction of convergence or profitability.
 * The screened policy adds through-d sources only with a positive immutable
 * geometry mask. The canonical census and complete admission remain decisive.
 */
constexpr const char* direct_jk_source_auto_refusal(
    std::size_t nbf, std::size_t direct_nbf, unsigned maximum_iterations, unsigned maximum_angular,
    bool cold_phase, bool generic_fock,
    DirectJkSourceAutoPolicy policy = DirectJkSourceAutoPolicy::generic_f,
    double screening = 0.0) noexcept {
  if (!cold_phase) return "auto: reused reference bucket";
  if (policy == DirectJkSourceAutoPolicy::screened_cold && maximum_angular <= 2) {
    if (!(screening > 0.0)) return "auto: screened SPD source requires a positive mask";
  } else if (maximum_angular != 3 || !generic_fock) {
    return "auto: retain generated/lower-angular schedule";
  }
  if (nbf < 64 || direct_nbf < 128) return "auto: source below cold reuse threshold";
  if (maximum_iterations < 8) return "auto: short iteration limit";
  return nullptr;
}

/** Expand screened SPD candidates without widening the incumbent f-shell cap. */
constexpr std::size_t direct_jk_source_value_ceiling(DirectJkSourceSelection selection,
                                                     DirectJkSourceAutoPolicy policy,
                                                     unsigned maximum_angular) noexcept {
  if (selection == DirectJkSourceSelection::forced ||
      (policy == DirectJkSourceAutoPolicy::screened_cold && maximum_angular <= 2))
    return 16ULL << 30;
  return 8ULL << 30;
}

/** Admit quartic values only after the complete consumer and source overhead. */
constexpr std::size_t direct_jk_source_allowance(std::size_t nbf, unsigned maximum_iterations,
                                                 std::size_t required, std::size_t overhead,
                                                 std::size_t budget, std::size_t ceiling) noexcept {
  if (nbf < 64 || maximum_iterations < 4 || required > budget || overhead > budget - required)
    return 0;
  const auto remaining = budget - required - overhead;
  return remaining < ceiling ? remaining : ceiling;
}

}  // namespace generativeqc::scf::cuda_execution
