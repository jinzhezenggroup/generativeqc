#include "core/correlation_convergence.hpp"
#include "generativeqc/generativeqc.h"

extern "C" generativeqc_status generativeqc_correlation_convergence_query_v1(
    const generativeqc_correlation_budget_v1* descriptor,
    generativeqc_correlation_limits_v1* output) {
  if (!descriptor || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION ||
      output->struct_size < sizeof(*output) || output->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  try {
    const auto policy = generativeqc::core::CorrelationConvergence::from_budget(
        {descriptor->energy_abs, descriptor->force_max_abs, descriptor->force_rms});
    // Finish reading input before publishing, including for overlapping buffers.
    *output = {sizeof(*output),           GENERATIVEQC_ABI_VERSION, policy.scale,
               policy.scf_energy(),       policy.scf_density(),     policy.cc_energy(),
               policy.cc_residual(),      policy.lambda_residual(), policy.response_solve(),
               policy.orbital_residual(), policy.stationarity()};
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (...) {
    // No C++ exception may escape the public C boundary, including allocation
    // failure while the shared validator constructs an invalid-input diagnostic.
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}
