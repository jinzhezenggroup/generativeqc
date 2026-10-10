#include "scf/cuda/rhf_resident_values.hpp"

#include <algorithm>
#include <cstdlib>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/df_progress_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/reference_eri_policy.hpp"
#include "scf/cuda/runtime_support.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
bool optional_refusal(generativeqc_status status) {
  return status == GENERATIVEQC_STATUS_OUT_OF_MEMORY ||
         status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}
}  // namespace

RhfResidentValues::~RhfResidentValues() {
  // Failure exits may still have submitted graph work. Drain before retiring
  // graphs, then release their borrowed source and scratch in that order.
  if (stream_) (void)cudaStreamSynchronize(stream_);
  graphs.reset();
  release_values();
}

void RhfResidentValues::release_values() noexcept {
  direct_.reset();
  if (scratch_) (void)runtime::resource_cuda_free(scratch_);
  if (error_) (void)runtime::resource_cuda_free(error_);
  if (census_) (void)runtime::resource_cuda_free(census_);
  scratch_ = nullptr;
  error_ = nullptr;
  census_ = nullptr;
}

generativeqc_status RhfResidentValues::prepare(int device, const std::vector<core::System>& systems,
                                               unsigned maximum_iterations, std::size_t direct_nbf,
                                               std::size_t required, std::size_t budget,
                                               bool cold_reference, bool generic_fock) {
  using namespace posthf;
  if (prepared_) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  prepared_ = true;
  runtime::df_progress::Scope trace("cuda_rhf_resident_values_prepare", "cuda_completed");
  const auto finish = [&](generativeqc_status status, bool admitted = false) {
    const auto diagnostic = cuda_direct_jk_plan_diagnostic(direct_.get());
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
  const auto selection =
      reference_resident_values_selection(std::getenv("GENERATIVEQC_RHF_RESIDENT_VALUES"));
  if (selection == ReferenceResidentValuesSelection::disabled) {
    selection_ = "disabled";
    reason_ = "disabled";
    return finish(GENERATIVEQC_STATUS_SUCCESS);
  }
  if (selection == ReferenceResidentValuesSelection::invalid) {
    selection_ = "invalid";
    reason_ = "invalid GENERATIVEQC_RHF_RESIDENT_VALUES (expected auto, 0 or 1)";
    return finish(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  }
  selection_ = selection == ReferenceResidentValuesSelection::forced ? "forced" : "auto";
  if (systems.size() != 1) return finish(GENERATIVEQC_STATUS_SUCCESS);
  const auto& system = systems.front();
  const auto nbf = molecule::ao_count(system);
  if (selection == ReferenceResidentValuesSelection::automatic) {
    unsigned maximum_angular = 0;
    for (const auto& shell : system.shells)
      maximum_angular = std::max(maximum_angular, shell.angular_momentum);
    if (const auto* refusal = reference_resident_values_auto_refusal(
            nbf, direct_nbf, maximum_iterations, maximum_angular, cold_reference, generic_fock)) {
      reason_ = refusal;
      return finish(GENERATIVEQC_STATUS_SUCCESS);
    }
  }
  std::size_t primitives = 0;
  for (const auto& shell : system.shells)
    primitives = checked_add(primitives, shell.primitives.size());
  const auto direct_bound = cuda_direct_coulomb_device_bytes(1, nbf, system.atoms.size(),
                                                             system.shells.size(), primitives, 0);
  matrix_elements_ = checked_mul(nbf, nbf);
  correction_elements_ = checked_mul(2, checked_mul(direct_nbf, direct_nbf));
  const auto scratch_bytes = checked_mul(
      checked_add(checked_mul(2, matrix_elements_), correction_elements_), sizeof(double));
  // Match the common response owner's conservative host-preparation inventory;
  // explicit scratch includes the diagnostic integer and last-action census.
  const auto overhead = checked_add(
      checked_add(checked_mul(5, direct_bound), checked_mul(6, source_capacity(system))),
      checked_add(scratch_bytes, sizeof(int) + 2 * sizeof(std::uint64_t)));
  auto allowance = reference_resident_value_allowance(nbf, maximum_iterations, required, overhead,
                                                      budget, 8ULL << 30);
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
  auto status = create_cuda_direct_jk_plan_on_stream(device, systems, 0, 0.0, direct_bound, stream_,
                                                     &raw, diagnostic, reason_);
  direct_.reset(raw);
  if (optional_refusal(status)) return refuse_optional();
  if (status != GENERATIVEQC_STATUS_SUCCESS) return finish(status);
  if (diagnostic.device_bytes > direct_bound ||
      diagnostic.host_preparation_bytes + diagnostic.host_bytes >
          checked_add(checked_mul(4, direct_bound), checked_mul(6, source_capacity(system)))) {
    reason_ = "Direct provider exceeded admitted inventory";
    return finish(GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  }
  const auto required_values = cuda_direct_jk_resident_value_bytes(direct_.get());
  if (!required_values || required_values > allowance ||
      cuda_direct_jk_compensation_elements(direct_.get()) != correction_elements_) {
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
  status = prepare_cuda_direct_jk_resident_values(direct_.get(), allowance, reason_);
  if (optional_refusal(status)) {
    return refuse_optional();
  }
  if (status != GENERATIVEQC_STATUS_SUCCESS) return finish(status);
  diagnostic = cuda_direct_jk_plan_diagnostic(direct_.get());
  value_bytes_ = diagnostic.resident_value_bytes;
  value_count_ = diagnostic.resident_value_count;
  capacity_bytes_ = checked_add(overhead, value_bytes_);
  reason_ = "admitted phase-local canonical FP64 values";
  return finish(GENERATIVEQC_STATUS_SUCCESS, true);
}

generativeqc_status RhfResidentValues::enqueue(const double* density, const double* hcore,
                                               double* fock) {
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.derivative_order = 0;
  auto* coulomb = scratch_;
  auto* exchange = coulomb + matrix_elements_;
  auto* correction = exchange + matrix_elements_;
  std::string detail;
  auto status = enqueue_cuda_direct_jk_compensated_device(
      direct_.get(), spec, density, matrix_elements_, coulomb, exchange, correction,
      correction_elements_, error_, 0.0, census_, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
  cuda_df::launch_assemble_rhf_fock_kernel((matrix_elements_ + 255) / 256, 256, 0, stream_,
                                           matrix_elements_, hcore, coulomb, exchange, fock);
  return cuda_status(cudaPeekAtLastError());
}

generativeqc_status RhfResidentValues::audit() {
  if (!active()) return GENERATIVEQC_STATUS_SUCCESS;
  int failure = 0;
  std::uint64_t census[2]{};
  auto error = cudaMemcpyAsync(&failure, error_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(census, census_, sizeof(census), cudaMemcpyDeviceToHost, stream_);
  // Host destinations must survive completion even on a failed submission.
  const auto drain = cudaStreamSynchronize(stream_);
  if (error != cudaSuccess) return cuda_status(error);
  if (drain != cudaSuccess) return cuda_status(drain);
  if (failure || census[0] != value_count_ || census[1] != 0)
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  return GENERATIVEQC_STATUS_SUCCESS;
}

void RhfResidentValues::observe_completed(std::size_t physical_focks) const noexcept {
  runtime::df_progress::Scope::label("phase_resident_values_selection", selection_);
  runtime::df_progress::Scope::label("resident_value_admission", reason_.c_str());
  runtime::df_progress::Scope::number("phase_resident_value_bytes", value_bytes_);
  runtime::df_progress::Scope::number("phase_resident_values_built", value_count_);
  runtime::df_progress::Scope::number("phase_resident_fock_actions", active() ? physical_focks : 0);
  runtime::df_progress::Scope::number("phase_additional_numeric_capacity_bytes", capacity_bytes_);
}

}  // namespace generativeqc::scf::cuda_execution
