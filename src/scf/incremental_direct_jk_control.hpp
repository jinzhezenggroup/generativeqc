#pragma once

#include <string_view>

namespace generativeqc::scf {
enum class IncrementalDirectJkControl { Disabled, Enabled, Invalid };

// Shared vocabulary only: callers retain their error type and admission policy.
// In particular, validate both selectors before combining their enabled states.
constexpr IncrementalDirectJkControl classify_incremental_direct_jk_control(const char* value) {
  if (value == nullptr) return IncrementalDirectJkControl::Disabled;
  const std::string_view setting(value);
  if (setting == "0" || setting == "off") return IncrementalDirectJkControl::Disabled;
  if (setting == "1" || setting == "on") return IncrementalDirectJkControl::Enabled;
  return IncrementalDirectJkControl::Invalid;
}
}  // namespace generativeqc::scf
