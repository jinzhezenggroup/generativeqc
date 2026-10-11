#include "scf/cuda/direct_jk_source_lease.hpp"

#include <algorithm>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/df_progress_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/direct_jk_source_policy.hpp"
#include "scf/cuda/runtime_support.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
bool optional_refusal(generativeqc_status status) {
  return status == GENERATIVEQC_STATUS_OUT_OF_MEMORY ||
         status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}
}  // namespace

DirectJkSourceLease::~DirectJkSourceLease() {
  // Consumers have retired captured borrowers; drain outstanding work before
  // releasing the immutable source and compensation scratch.
  if (stream_) (void)cudaStreamSynchronize(stream_);
  release_values();
}

void DirectJkSourceLease::release_values() noexcept {
  direct_.reset();
  if (scratch_) (void)runtime::resource_cuda_free(scratch_);
  if (error_) (void)runtime::resource_cuda_free(error_);
  if (census_) (void)runtime::resource_cuda_free(census_);
  scratch_ = nullptr;
  error_ = nullptr;
  census_ = nullptr;
  last_error_ = nullptr;
}

generativeqc_status DirectJkSourceLease::prepare(
    int device, const std::vector<core::System>& systems, unsigned maximum_iterations,
    std::size_t direct_nbf, std::size_t required, std::size_t budget, const char* selection_value,
    const char* trace_name, bool cold_phase, bool generic_fock, double screening,
    std::size_t value_ceiling, bool own_outputs, DirectJkSourceAutoPolicy auto_policy,
    std::uint32_t bucket_mask, unsigned spins) {
  using namespace posthf;
  if (prepared_) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  prepared_ = true;
  if (spins != 1 && spins != 2) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  runtime::df_progress::Scope trace(trace_name, "cuda_completed");
  const auto finish = [&](generativeqc_status status, bool admitted = false) {
    const auto diagnostic = cuda_direct_jk_plan_diagnostic(direct_.get());
    completed_values_ = diagnostic.resident_values_completed;
    runtime::df_progress::Scope::label("selection", selection_);
    runtime::df_progress::Scope::label("admission", reason_.c_str());
    runtime::df_progress::Scope::number("source_values_submitted",
                                        diagnostic.resident_values_submitted);
    runtime::df_progress::Scope::number("source_values_completed",
                                        diagnostic.resident_values_completed);
    runtime::df_progress::Scope::number("source_values_built", value_count_);
    runtime::df_progress::Scope::number("source_value_bytes", value_bytes_);
    runtime::df_progress::Scope::number("additional_numeric_capacity_bytes", capacity_bytes_);
    // A host publication allocation can refuse after every ERI has completed.
    // Observe that discarded work before retiring the unpublished provider.
    if (status != GENERATIVEQC_STATUS_SUCCESS || !admitted) release_values();
    trace.finish(status == GENERATIVEQC_STATUS_SUCCESS ? (admitted ? "accepted" : "refused")
                                                       : "failed");
    return status;
  };
  const auto refuse_optional = [&] {
    // Provider preparation may fail in metadata/scratch cudaMalloc before its
    // optional-value allocator clears last-error. Retire only that OOM, and
    // fence the borrowed stream so a concurrent execution fault cannot become
    // an apparently successful exact fallback.
    const auto pending = cudaGetLastError();
    const auto drain = cudaStreamSynchronize(stream_);
    if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation)
      return finish(cuda_status(pending));
    if (drain != cudaSuccess) return finish(cuda_status(drain));
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  };
  const auto selection = direct_jk_source_selection(selection_value);
  if (selection == DirectJkSourceSelection::disabled) {
    selection_ = "disabled";
    reason_ = "disabled";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  if (selection == DirectJkSourceSelection::invalid) {
    selection_ = "invalid";
    reason_ = "invalid resident value selection (expected auto, 0 or 1)";
    return finish(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  }
  selection_ = selection == DirectJkSourceSelection::forced ? "forced" : "auto";
  if (bucket_mask >= (1U << 28)) {
    reason_ = "invalid resident bucket mask or value ceiling";
    return finish(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  }
  if (!bucket_mask || !value_ceiling) {
    reason_ = "no resident bucket selected";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  if (systems.size() != 1) return finish(GENERATIVEQC_STATUS_SUCCESS);
  const auto& system = systems.front();
  const auto nbf = molecule::ao_count(system);
  if (selection == DirectJkSourceSelection::automatic) {
    unsigned maximum_angular = 0;
    for (const auto& shell : system.shells)
      maximum_angular = std::max(maximum_angular, shell.angular_momentum);
    if (const auto* refusal =
            direct_jk_source_auto_refusal(nbf, direct_nbf, maximum_iterations, maximum_angular,
                                          cold_phase, generic_fock, auto_policy, screening)) {
      reason_ = refusal;
      return finish(GENERATIVEQC_STATUS_SUCCESS);
    }
  }
  std::size_t primitives = 0;
  for (const auto& shell : system.shells)
    primitives = checked_add(primitives, shell.primitives.size());
  auto direct_bound = cuda_direct_coulomb_device_bytes(1, nbf, system.atoms.size(),
                                                       system.shells.size(), primitives, 0);
  if (screening > 0.0) {
    std::size_t layout_bytes = 0;
    const auto layout_status = cuda_direct_jk_resident_layout_device_bytes(
        systems.size(), direct_nbf, layout_bytes, reason_);
    if (optional_refusal(layout_status)) return refuse_optional();
    if (layout_status != GENERATIVEQC_STATUS_SUCCESS) return finish(layout_status);
    // Ordinary Direct admission may refuse optional rows and remain exact.
    // A compact source needs those rows, including transient sort/scan space.
    direct_bound = checked_add(direct_bound, layout_bytes);
  }
  matrix_elements_ = checked_mul(nbf, nbf);
  output_elements_ = own_outputs ? checked_mul(1U + spins, matrix_elements_) : 0;
  correction_elements_ = checked_mul(2U * spins, checked_mul(direct_nbf, direct_nbf));
  const auto scratch_bytes =
      checked_mul(checked_add(output_elements_, correction_elements_), sizeof(double));
  // Match the common response owner's conservative host-preparation inventory;
  // explicit scratch includes the diagnostic integer and last-action census.
  const auto bucket_layout_bytes =
      bucket_mask == (1U << 28) - 1 ? 0 : checked_mul(systems.size(), 28 * sizeof(std::size_t));
  const auto overhead = checked_add(
      bucket_layout_bytes,
      checked_add(
          checked_add(checked_mul(5, direct_bound), checked_mul(6, source_capacity(system))),
          checked_add(scratch_bytes, sizeof(int) + 2 * sizeof(std::uint64_t))));
  const auto numeric_allowance = direct_jk_source_allowance(nbf, maximum_iterations, required,
                                                            overhead, budget, value_ceiling);
  auto allowance = numeric_allowance;
  if (!allowance) {
    reason_ = "reuse or numeric capacity refusal";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  std::size_t available = 0, total = 0;
  auto error = cudaMemGetInfo(&available, &total);
  if (error != cudaSuccess) return finish(cuda_status(error));
  const auto device_overhead =
      checked_add(direct_bound, scratch_bytes + sizeof(int) + 2 * sizeof(std::uint64_t));
  // Leave room for allocator/context rounding; neither free memory nor this
  // reserve replaces the complete numeric admission above.
  const auto reserve = checked_add(device_overhead, 256ULL << 20);
  allowance = std::min(allowance, available > reserve ? available - reserve : 0);
  if (!allowance) {
    reason_ = "free device capacity refusal";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  CudaDirectJkPlan* raw = nullptr;
  CudaDirectJkDiagnostic diagnostic;
  capacity_bytes_ = overhead;
  auto status = create_cuda_direct_jk_resident_plan_on_stream(
      device, systems, screening, direct_bound, stream_, &raw, diagnostic, reason_);
  direct_.reset(raw);
  if (optional_refusal(status)) return refuse_optional();
  if (status != GENERATIVEQC_STATUS_SUCCESS) return finish(status);
  status = prepare_cuda_direct_jk_resident_layout(direct_.get(), reason_);
  if (optional_refusal(status)) return refuse_optional();
  if (status != GENERATIVEQC_STATUS_SUCCESS) return finish(status);
  diagnostic = cuda_direct_jk_plan_diagnostic(direct_.get());
  if (diagnostic.device_bytes > direct_bound ||
      diagnostic.host_preparation_bytes + diagnostic.host_bytes >
          checked_add(checked_mul(4, direct_bound), checked_mul(6, source_capacity(system)))) {
    reason_ = "Direct provider exceeded admitted inventory";
    return finish(GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  }
  // The conservative preparation bound is charged to the complete numeric
  // budget. Once that storage exists, live free memory already excludes it;
  // reserve only the remaining scratch/diagnostics and the same runtime margin.
  error = cudaMemGetInfo(&available, &total);
  if (error != cudaSuccess) return finish(cuda_status(error));
  const auto remaining_reserve =
      checked_add(scratch_bytes + sizeof(int) + 2 * sizeof(std::uint64_t), 256ULL << 20);
  allowance = std::min(numeric_allowance,
                       available > remaining_reserve ? available - remaining_reserve : 0);
  source_value_count_ = cuda_direct_jk_resident_value_bytes(direct_.get()) / sizeof(double);
  source_screening_ = screening;
  const auto required_values =
      bucket_mask == (1U << 28) - 1
          ? source_value_count_ * sizeof(double)
          : cuda_direct_jk_selected_resident_value_bytes(direct_.get(), bucket_mask);
  const auto compensation_elements = cuda_direct_jk_compensation_elements(direct_.get(), spins);
  runtime::df_progress::Scope::number("required_source_value_bytes", required_values);
  runtime::df_progress::Scope::number("source_value_allowance_bytes", allowance);
  runtime::df_progress::Scope::number("free_device_bytes_after_preparation", available);
  runtime::df_progress::Scope::number("source_preparation_device_bound_bytes", direct_bound);
  runtime::df_progress::Scope::number("canonical_compensation_elements", compensation_elements);
  runtime::df_progress::Scope::number("required_compensation_elements", correction_elements_);
  if (!required_values || required_values > allowance ||
      compensation_elements != correction_elements_) {
    reason_ = "canonical domain or value capacity refusal";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  // Admission is a transient peak, not retained storage. Keep the reservation
  // even when a later publication refuses and the exact fallback succeeds.
  capacity_bytes_ = checked_add(overhead, required_values);
  error = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&scratch_), scratch_bytes);
  if (error == cudaSuccess)
    error = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&error_), sizeof(int));
  if (error == cudaSuccess)
    error = runtime::resource_cuda_malloc(reinterpret_cast<void**>(&census_),
                                          2 * sizeof(std::uint64_t));
  if (error != cudaSuccess) {
    reason_ = "optional scratch allocation refusal";
    if (error == cudaErrorMemoryAllocation) {
      const auto pending = cudaGetLastError();
      return finish(pending == cudaSuccess || pending == cudaErrorMemoryAllocation
                        ? GENERATIVEQC_STATUS_SUCCESS
                        : cuda_status(pending));
    }
    return finish(cuda_status(error));
  }
  runtime::df_progress::Scope::number("source_values_requested", required_values / sizeof(double));
  runtime::df_progress::Scope::number("admitted_optional_capacity_bytes", capacity_bytes_);
  status = bucket_mask == (1U << 28) - 1
               ? prepare_cuda_direct_jk_resident_values(direct_.get(), allowance, reason_)
               : prepare_cuda_direct_jk_selected_resident_values(direct_.get(), allowance,
                                                                 bucket_mask, reason_);
  if (optional_refusal(status)) {
    return refuse_optional();
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS) return finish(status);
  diagnostic = cuda_direct_jk_plan_diagnostic(direct_.get());
  value_bytes_ = diagnostic.resident_value_bytes;
  value_count_ = diagnostic.resident_value_count;
  capacity_bytes_ = checked_add(overhead, value_bytes_);
  reason_ = value_count_ == source_value_count_
                ? "admitted phase-local canonical FP64 values"
                : "admitted partial canonical FP64 values plus exact Direct buckets";
  return finish(GENERATIVEQC_STATUS_SUCCESS, true);
}

generativeqc_status DirectJkSourceLease::enqueue(FockBuildSpec spec, const double* density,
                                                 double* coulomb, double* exchange,
                                                 int* numerical_error, double screening,
                                                 std::string& detail, const double* beta,
                                                 double* beta_exchange) {
  if (!active()) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* correction = scratch_ + output_elements_;
  last_error_ = numerical_error ? numerical_error : error_;
  const auto status = enqueue_cuda_direct_jk_compensated_device(
      direct_.get(), spec, density, matrix_elements_, coulomb, exchange, correction,
      correction_elements_, last_error_, screening, census_, detail, true, beta, beta_exchange);
  if (status == GENERATIVEQC_STATUS_SUCCESS) last_screening_ = screening;
  return status;
}

generativeqc_status DirectJkSourceLease::audit() {
  if (!active() || !last_error_) return GENERATIVEQC_STATUS_SUCCESS;
  int failure = 0;
  std::uint64_t census[2]{};
  auto error = cudaMemcpyAsync(&failure, last_error_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(census, census_, sizeof(census), cudaMemcpyDeviceToHost, stream_);
  // Host destinations must survive completion even on a failed submission.
  const auto drain = cudaStreamSynchronize(stream_);
  if (error != cudaSuccess) return cuda_status(error);
  if (drain != cudaSuccess) return cuda_status(drain);
  const auto uncached = source_value_count_ - value_count_;
  // Compact rows freeze the preparation predicate, not a stricter consumer
  // predicate. Every row is still visited; only Direct evaluations can shrink.
  const bool valid_evaluations = last_screening_ == source_screening_
                                     ? census[1] == uncached
                                     : last_screening_ > source_screening_ && census[1] <= uncached;
  if (failure || census[0] != source_value_count_ || !valid_evaluations)
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  return GENERATIVEQC_STATUS_SUCCESS;
}

void DirectJkSourceLease::observe_completed(std::size_t physical_focks) const noexcept {
  runtime::df_progress::Scope::label("phase_resident_values_selection", selection_);
  runtime::df_progress::Scope::label("resident_value_admission", reason_.c_str());
  runtime::df_progress::Scope::number("phase_resident_value_bytes", value_bytes_);
  runtime::df_progress::Scope::number("phase_resident_values_built", value_count_);
  runtime::df_progress::Scope::number("phase_canonical_source_values", source_value_count_);
  runtime::df_progress::Scope::number("phase_direct_values_per_action",
                                      active() ? source_value_count_ - value_count_ : 0);
  runtime::df_progress::Scope::number("phase_resident_fock_actions", active() ? physical_focks : 0);
  runtime::df_progress::Scope::number("phase_additional_numeric_capacity_bytes", capacity_bytes_);
}

}  // namespace generativeqc::scf::cuda_execution
