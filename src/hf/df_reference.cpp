#include "hf/df_reference.hpp"

#include <algorithm>
#include <stdexcept>

#include "posthf/capacity.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/df_response_weights.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::hf {
DFReference::~DFReference() {
#if GENERATIVEQC_HAS_CUDA
  scf::destroy_cuda_density_fitting_jk_plan(plan_);
#endif
}

scf::ScfResult DFReference::solve(const core::System& orbital, const core::System& auxiliary,
                                  scf::ScfOptions options, int device,
                                  const std::vector<double>* initial_density) {
#if GENERATIVEQC_HAS_CUDA
  if (plan_ || identity_) throw std::logic_error("DF reference owners are single-frame");
  const auto budget = options.reference_memory_budget_bytes;
  // Reserve both sides of the existing DF planner's separate host/device caps.
  // Keeping this bounded lease live through correlation is conservative, not a
  // claim that all reserved bytes are allocated. Never consume the entire CC cap.
  const auto value_budget = std::min<std::size_t>(budget / 4, 4ULL << 30);
  if (!value_budget) throw std::length_error("DF reference has no numeric allowance");
  orbital_ = orbital;
  auxiliary_ = auxiliary;
  device_ = device;
  options.compute_forces = false;
  options.export_physical_reference = true;
  options.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
  options.density_fitting_host_retry = false;
  options.density_fitting_memory_budget_bytes = value_budget;
  std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
  auto results = scf::run_rhf_density_fitting_cuda_bucket_cached(
      &plan_, {*orbital_}, auxiliary_, options, {initial_density}, device, &diagnostics, &data_);
  if (results.size() == 1 && results.front().status == GENERATIVEQC_STATUS_OUT_OF_MEMORY)
    throw std::bad_alloc();
  if (results.size() != 1 || results.front().status != GENERATIVEQC_STATUS_SUCCESS ||
      !results.front().scf.converged || !results.front().scf.reference)
    throw std::runtime_error("native DF-HF reference did not converge");
  if (diagnostics.size() != 1)
    throw std::logic_error("DF reference has no unique forward metric receipt");
  metric_diagnostic_ = diagnostics.front();
  auto result = std::move(results.front().scf);
  identity_.emplace(orbital, *result.reference);
  reference_ = result.reference;
  storage_bytes_ = posthf::checked_add(
      posthf::checked_mul(2, value_budget),
      posthf::checked_add(identity_->storage_bytes(),
                          posthf::checked_add(posthf::source_capacity(orbital),
                                              posthf::source_capacity(auxiliary))));
  if (storage_bytes_ > budget) throw std::length_error("DF reference exceeds numeric allowance");
  return result;
#else
  throw std::runtime_error("DF reference requires a CUDA-enabled build");
#endif
}

bool DFReference::matches(const core::System& system, const PhysicalReference& reference,
                          int device) const {
  return plan_ && identity_ && reference_.get() == &reference && reference.density_fitted &&
         device == device_ && identity_->matches(system, reference);
}

std::vector<double> DFReference::quadratic_gradient(std::span<const double> density,
                                                    std::size_t maximum_bytes) {
#if GENERATIVEQC_HAS_CUDA
  if (!identity_ || data_.size() != 1 || !data_[0] || !maximum_bytes)
    throw std::invalid_argument("unprepared DF reference derivative");
  const auto& data = *data_[0];
  const scf::DensityFittingDensityResponse term{density, 1.0, 0.25};
  std::vector<double> gradient;
  std::string detail;
  scf::DfGradientResources resources;
  const auto code = scf::execute_cuda_density_fitting_generated_force_response(
      plan_, 0, *orbital_, *auxiliary_, data.raw.three_center, data.raw.metric, {&term, 1}, 0,
      maximum_bytes / 2, 0, gradient, detail, &resources);
  if (code == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (code != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
  if (posthf::checked_add(resources.host_bytes, resources.device_bytes) > maximum_bytes)
    throw std::length_error("DF reference derivative exceeded admitted capacity");
  return gradient;
#else
  throw std::runtime_error("DF reference derivative requires CUDA");
#endif
}
}  // namespace generativeqc::hf
