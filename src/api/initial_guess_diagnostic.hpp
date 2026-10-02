#pragma once

#include <optional>

#include "api/error.hpp"
#include "scf/initial_guess/preliminary_types.hpp"

namespace generativeqc::api {
inline generativeqc_status copy_initial_guess_diagnostic(
    const std::optional<scf::initial_guess::PreliminaryDiagnostic>& source,
    generativeqc_initial_guess_diagnostic* out) {
  if (out && !valid_descriptor(out)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if (!source) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  if (out) {
    const auto& value = *source;
    *out = {sizeof(*out),
            GENERATIVEQC_ABI_VERSION,
            value.requested_kind,
            static_cast<std::uint32_t>(value.outcome),
            value.preliminary_iterations,
            value.preliminary_fock_builds,
            value.target_attempts,
            value.discarded_target_iterations,
            value.discarded_target_fock_builds,
            value.preparation_numeric_capacity,
            value.preparation_seconds,
            value.work_counters_complete ? 1U : 0U};
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace generativeqc::api
