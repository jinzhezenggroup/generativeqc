#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/host_cache_budget.hpp"
#include "scf/cuda/df_host_panels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_plan_setup.hpp"
#include "scf/df_exchange_policy.hpp"

namespace generativeqc::scf {
using namespace cuda_df;

// Public plan adapters retain their opaque ABI and source-transfer contract.
generativeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, std::size_t automatic_rhf_rank) {
  return create_cuda_density_fitting_jk_plan_tiled_impl(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, nullptr, false, {}, automatic_rhf_rank);
}

generativeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, false);
}

generativeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool retain_three_center) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, retain_three_center, {});
}

generativeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool retain_three_center, DfValueStorageOptions storage,
    std::size_t automatic_rhf_rank) {
  if (plan != nullptr) *plan = nullptr;
  if (source == nullptr || *source == nullptr) {
    detail = "source-backed CUDA DF plan requires a source";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  // The implementation owns the handle for the duration of this call,
  // including validation/allocation failures.  It destroys the handle on
  // every failure path; clear the caller slot unconditionally below.
  generativeqc_status status = create_cuda_density_fitting_jk_plan_tiled_impl(
      device_id, batch_size, nbf, naux, metrics, {}, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, *source, retain_three_center, storage,
      automatic_rhf_rank);
  if (status == GENERATIVEQC_STATUS_SUCCESS) {
    *source = nullptr;  // ownership transfers to the prepared plan
  } else {
    // The implementation has already destroyed the transferred source on
    // all failure paths.  Keep the caller handle null to make cleanup safe.
    *source = nullptr;
  }
  return status;
}

generativeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail,
    std::size_t automatic_rhf_rank) {
  // The compatibility API is the resident/default path: keep the complete
  // auxiliary dimension so large-AO warm replays can capture the SCF Graph.
  // Budgeted callers use the tiled entry point directly and may select host
  // streaming when the full transformed tensor cannot fit.
  const std::size_t resident_auxiliary_tile = auxiliary_tile == 0U ? naux : auxiliary_tile;
  return create_cuda_density_fitting_jk_plan_tiled(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold,
      resident_auxiliary_tile, 0, plan, diagnostics, detail, automatic_rhf_rank);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan* plan) noexcept {
  if (plan == nullptr) return;
  release(*plan);
  delete plan;
}

CudaDensityFittingSourceCounters cuda_density_fitting_jk_plan_source_counters(
    const CudaDensityFittingJkPlan* plan) noexcept {
  return plan == nullptr ? CudaDensityFittingSourceCounters{}
                         : cuda_density_fitting_integral_source_counters(plan->integral_source);
}

std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan == nullptr ? 0U : plan->batch_size;
}

void set_cuda_density_fitting_scf_value_budget(CudaDensityFittingJkPlan* plan,
                                               std::size_t budget) noexcept {
  if (plan) plan->scf_value_budget_bytes = budget;
}

std::size_t cuda_density_fitting_scf_value_budget(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->scf_value_budget_bytes : 0;
}

void set_cuda_density_fitting_scf_diis_history(CudaDensityFittingJkPlan* plan,
                                               unsigned history) noexcept {
  if (plan) plan->scf_diis_history = history;
}

unsigned cuda_density_fitting_scf_diis_history(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->scf_diis_history : 0;
}

bool cuda_density_fitting_scf_policy_matches(const CudaDensityFittingJkPlan* plan) noexcept {
  const char* resident = std::getenv("GENERATIVEQC_DF_RESIDENT_EXCHANGE");
  const char* diis = std::getenv("GENERATIVEQC_DF_DIIS_DOTS");
  return plan != nullptr &&
         (!resident || std::strcmp(resident, "auto") == 0 || std::strcmp(resident, "full") == 0 ||
          std::strcmp(resident, "flat") == 0 || std::strcmp(resident, "legacy") == 0) &&
         (!diis || std::strcmp(diis, "auto") == 0 || std::strcmp(diis, "serial") == 0) &&
         cuda_density_fitting_host_policy_matches(plan) &&
         plan->resident_exchange_enabled == df_resident_exchange_requested() &&
         plan->triangular_exchange == df_triangular_exchange_requested() &&
         plan->flat_dense_exchange == df_flat_dense_exchange_requested() &&
         plan->cooperative_diis == df_cooperative_diis_requested() &&
         plan->occupied_scf_reserved == df_occupied_exchange_requested(plan->nbf, plan->naux,
                                                                       plan->batch_size,
                                                                       plan->automatic_rhf_rank);
}

DfPairStorage cuda_density_fitting_pair_storage(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->value_storage.pairs : DfPairStorage::Dense;
}

bool cuda_density_fitting_jk_plan_matches(const CudaDensityFittingJkPlan* plan, std::size_t item,
                                          std::size_t nbf, std::size_t naux,
                                          double relative_threshold) noexcept {
  return plan && item < plan->batch_size && plan->nbf == nbf && plan->naux == naux &&
         plan->metric_relative_threshold == relative_threshold;
}

bool cuda_density_fitting_host_policy_matches(const CudaDensityFittingJkPlan* plan) noexcept {
  std::size_t limit;
  bool automatic;
  return plan && host_panel_policy(limit, automatic) && limit == plan->host_panel_policy_limit &&
         automatic == plan->host_panel_automatic;
}

generativeqc_status prepare_cuda_density_fitting_host_panels(
    CudaDensityFittingJkPlan* plan, std::size_t host_budget,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  std::size_t requested;
  bool automatic;
  if (!plan || !host_panel_policy(requested, automatic)) {
    detail = "invalid generated DF host cache byte ceiling";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  // This API is setup-only. A populated source can never be rebound or resized
  // while queued consumers borrow its pageable backing.
  if (!plan->host_raw_three_center.empty()) {
    if (requested == plan->host_panel_policy_limit && automatic == plan->host_panel_automatic &&
        diagnostics.size() == 1 && diagnostics.front().peak_host_bytes <= host_budget)
      return GENERATIVEQC_STATUS_SUCCESS;
    detail = "cannot change an admitted DF host source policy";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  plan->host_panel_policy_limit = requested;
  plan->host_panel_automatic = automatic;
  if (!plan->streamed || !plan->integral_source || plan->batch_size != 1 ||
      plan->persistent_scf_state || plan->md_coulomb || diagnostics.size() != 1 || !host_budget)
    return GENERATIVEQC_STATUS_SUCCESS;
  const auto available = automatic ? runtime::automatic_host_cache_bytes() : requested;
  const auto existing = diagnostics.front().peak_host_bytes;
  std::size_t elements, bytes;
  if (existing >= host_budget || !checked_multiply(plan->matrix_elements, plan->naux, elements) ||
      !checked_bytes(elements, bytes) || bytes > host_budget - existing || bytes > available)
    return GENERATIVEQC_STATUS_SUCCESS;
  std::vector<double> values;
  try {
    values.resize(elements);
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_SUCCESS;  // Optional RAM never displaces the source.
  }
  const auto capacity = vector_capacity_bytes(values);
  if (capacity > host_budget - existing || capacity > available) return GENERATIVEQC_STATUS_SUCCESS;
  struct Drain {
    cudaStream_t stream;
    ~Drain() { (void)cudaStreamSynchronize(stream); }
  } drain{plan->stream};
  const auto tile_capacity = plan->row_tile * plan->nbf * plan->auxiliary_tile;
  const auto pair_tile = std::min(plan->matrix_elements, plan->row_tile * plan->nbf);
  for (std::size_t first = 0; first < plan->matrix_elements; first += pair_tile) {
    const auto pairs = std::min(pair_tile, plan->matrix_elements - first);
    const auto auxiliary_tile = std::min(plan->naux, tile_capacity / pairs);
    for (std::size_t begin = 0; begin < plan->naux; begin += auxiliary_tile) {
      const auto count = std::min(auxiliary_tile, plan->naux - begin);
      const auto status = generate_cuda_density_fitting_raw_tile(
          plan->integral_source, 0, first, pairs, begin, count, -1, plan->stream,
          plan->exchange_intermediate, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      auto error =
          cudaMemcpy2DAsync(values.data() + first * plan->naux + begin, plan->naux * sizeof(double),
                            plan->exchange_intermediate, count * sizeof(double),
                            count * sizeof(double), pairs, cudaMemcpyDeviceToHost, plan->stream);
      if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
      if (error != cudaSuccess)
        return cuda_failure(error, "prepare immutable DF host panels", detail);
    }
  }
  if (!finite_values(values)) {
    detail = "nonfinite generated DF host source";
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
  plan->host_raw_three_center = std::move(values);
  diagnostics.front().host_resident_bytes += capacity;
  diagnostics.front().peak_host_bytes += capacity;
  diagnostics.front().host_raw_factor_bytes = capacity;
  runtime::cuda_trace::trace_counter("host_raw_source_setup_bytes", bytes);
  return GENERATIVEQC_STATUS_SUCCESS;
}

}  // namespace generativeqc::scf
