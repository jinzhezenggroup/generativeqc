#include <algorithm>

#include "molecule/basis.hpp"
#include "scf/cuda/df_md_layout.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda_density_fitting_md.hpp"

namespace generativeqc::scf {

std::size_t cuda_df_md_coulomb_reservation(const core::System& orbital,
                                           const core::System& auxiliary,
                                           std::size_t value_budget) {
  for (const auto& shell : orbital.shells)
    if (shell.angular_momentum > 3) return 0;
  for (const auto& shell : auxiliary.shells)
    if (shell.angular_momentum > 4) return 0;
  try {
    const auto pairs =
        cuda_execution::df_md_layout_device_bytes(orbital, false, value_budget, value_budget);
    const auto aux = cuda_execution::df_md_layout_device_bytes(auxiliary, true,
                                                               value_budget - pairs, value_budget);
    return pairs + aux;  // The subtracted cap proves representability.
  } catch (const std::length_error&) {
    return 0;
  }
}

generativeqc_status try_install_cuda_df_md_coulomb(
    CudaDensityFittingJkPlan* plan, const core::System& orbital, const core::System& auxiliary,
    std::size_t device_budget, std::size_t host_budget,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  detail.clear();
  if (!plan || plan->md_coulomb || plan->persistent_scf_state || diagnostics.size() != 1 ||
      molecule::ao_count(orbital) != plan->nbf || molecule::ao_count(auxiliary) != plan->naux) {
    detail = "DF Hermite installation requires a fresh compatible single-system plan";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!plan->streamed || !plan->integral_source || plan->batch_size != 1 ||
      plan->metric_full_rank.size() != 1 || !plan->metric_full_rank[0] || !plan->metric_project ||
      !plan->metric_rotate || plan->row_tile * plan->nbf * plan->auxiliary_tile < plan->naux) {
    detail = "DF Hermite fallback: requires a streamed full-rank vector-metric plan";
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  for (const auto& shell : orbital.shells)
    if (shell.angular_momentum > 3) {
      detail = "DF Hermite fallback: orbital angular momentum exceeds f";
      return GENERATIVEQC_STATUS_SUCCESS;
    }
  for (const auto& shell : auxiliary.shells)
    if (shell.angular_momentum > 4) {
      detail = "DF Hermite fallback: auxiliary angular momentum exceeds g";
      return GENERATIVEQC_STATUS_SUCCESS;
    }
  auto& record = diagnostics.front();
  // Reserve against both setup peak and lazy SCF reservations, even though
  // setup and optional-owner preparation are sequential. Never borrow force
  // capacity or enlarge the user's enclosing allowance to admit this path.
  const auto used_device = std::max(record.peak_device_bytes, record.device_resident_bytes);
  if (device_budget <= used_device || host_budget <= record.host_resident_bytes) {
    detail = "DF Hermite fallback: no remaining owner capacity";
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  auto error = cudaSetDevice(plan->device_id);
  if (error != cudaSuccess)
    return cuda_df::cuda_failure(error, "select DF Hermite owner device", detail);
  cuda_execution::CudaDfMdResources resources;
  auto status = cuda_execution::create_cuda_df_md_owner(
      plan->device_id, plan->stream, orbital, auxiliary, device_budget - used_device,
      host_budget - record.host_resident_bytes, &plan->md_coulomb, resources, detail);
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) {
    detail = "DF Hermite fallback: " + detail;
    return GENERATIVEQC_STATUS_SUCCESS;
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  // Subtracted limits above prove that each sum is representable. Host layout
  // temporaries die at installation; the previous plan setup peak is retained.
  record.peak_device_bytes = used_device + resources.device_bytes;
  record.device_resident_bytes += resources.device_bytes;
  record.peak_host_bytes =
      std::max(record.peak_host_bytes, record.host_resident_bytes + resources.host_peak_bytes);
  record.host_resident_bytes += resources.host_resident_bytes;
  record.md_coulomb = true;
  record.md_coulomb_device_bytes = resources.device_bytes;
  plan->md_coulomb_resources = resources;
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::scf
