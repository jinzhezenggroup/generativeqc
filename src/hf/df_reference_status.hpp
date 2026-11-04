#pragma once

#include <sstream>
#include <vector>

#include "generativeqc/generativeqc.hpp"
#include "scf/cuda_batch.hpp"

namespace generativeqc::hf {
/** Fail closed without conflating provider refusal, convergence and export.
 * Kept independent of CUDA execution so failure/status publication is testable
 * even on hosts without a device or a CUDA-enabled library. */
inline void require_df_reference_result(const std::vector<scf::RhfBucketItem>& results) {
  if (results.size() != 1)
    throw Error(GENERATIVEQC_STATUS_INTERNAL_ERROR,
                "native DF-HF reference: expected one backend result, got " +
                    std::to_string(results.size()));
  const auto& item = results.front();
  if (item.status == GENERATIVEQC_STATUS_SUCCESS && item.scf.converged && item.scf.reference)
    return;
  auto status = item.status;
  if (status == GENERATIVEQC_STATUS_SUCCESS)
    status = item.scf.converged ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
                                : GENERATIVEQC_STATUS_NOT_CONVERGED;
  std::ostringstream detail;
  detail << "native DF-HF reference: backend_status=" << item.status
         << " converged=" << item.scf.converged
         << " physical_reference_exported=" << bool(item.scf.reference)
         << " iterations=" << item.scf.iterations << " energy_change=" << item.scf.energy_change
         << " density_rms=" << item.scf.density_rms
         << " physical_residual_rms=" << item.scf.physical_residual_rms;
  if (!item.detail.empty()) detail << " detail=" << item.detail;
  throw Error(status, detail.str());
}
}  // namespace generativeqc::hf
