#pragma once

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace generativeqc::core {

/** Policy refusal, not allocation failure or a failed scientific calculation.
 * Deliberately not a runtime_error/length_error: method adapters must preserve
 * this distinction until the public exception mapper sees it. */
class WorkLimitError final : public std::exception {
 public:
  explicit WorkLimitError(std::string message) : message_(std::move(message)) {}
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

/** Explicit finite process policy shared by unrestricted primal, DF and response.
 * Zero, signs, whitespace and overflow are rejected; no value disables safety
 * admission. Read per admission so changing policy never changes artifact identity. */
inline std::size_t correlation_work_limit() {
  const char* configured = std::getenv("GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK");
  if (!configured) return 100000000000ULL;
  const std::string_view text(configured);
  std::size_t limit{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), limit);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !limit)
    throw std::invalid_argument(
        "GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK must be a positive size_t decimal integer");
  return limit;
}

/** Checked callers supply semantic scalar work, not measured FLOPs or runtime.
 * Diagnostics retain the dimensions and simultaneous numeric-storage estimate
 * even when a work override is needed independently of the memory budget. */
inline void admit_correlation_work(std::string_view phase, std::size_t estimated,
                                   std::size_t numeric_bytes, std::size_t nbf, std::size_t alpha,
                                   std::size_t beta, unsigned iterations = 0) {
  const auto allowed = correlation_work_limit();
  if (estimated <= allowed) return;
  throw WorkLimitError("correlation work-policy rejection: phase=" + std::string(phase) +
                       " nbf=" + std::to_string(nbf) + " nocc_alpha=" + std::to_string(alpha) +
                       " nocc_beta=" + std::to_string(beta) + " nvir_alpha=" +
                       std::to_string(nbf - alpha) + " nvir_beta=" + std::to_string(nbf - beta) +
                       " iteration_ceiling=" + std::to_string(iterations) + " estimated_work=" +
                       std::to_string(estimated) + " allowed_work=" + std::to_string(allowed) +
                       " numeric_storage_bytes=" + std::to_string(numeric_bytes));
}
}  // namespace generativeqc::core
