#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#include "scf/types.hpp"

namespace generativeqc::scf {

/** Preserve the last evaluated physical iterate without publishing a reference. */
inline std::string reference_failure_detail(const ScfResult& result, generativeqc_status status,
                                            std::string_view message) {
  const char* reason = "execution_failure";
  if (status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE)
    reason = "numerical_failure";
  else if (status == GENERATIVEQC_STATUS_CUDA_ERROR)
    reason = "cuda_error";
  else if (status == GENERATIVEQC_STATUS_NOT_CONVERGED)
    reason = result.converged ? "physical_reference_unavailable" : "scf_not_converged";
  std::ostringstream detail;
  detail << std::setprecision(17) << message << "; scf_status=" << static_cast<int>(status)
         << "; stopping_reason=" << reason << "; iterations=" << result.iterations
         << "; energy=" << result.energy << "; energy_change=" << result.energy_change
         << "; density_rms=" << result.density_rms
         << "; physical_residual_rms=" << result.physical_residual_rms;
  return detail.str();
}

}  // namespace generativeqc::scf
