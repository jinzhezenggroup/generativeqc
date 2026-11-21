#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "dft/split_hybrid_cuda_domain.hpp"
#include "generativeqc/ks_hessian.h"
#include "r2scan_cuda_domain.hpp"

#if GENERATIVEQC_HAS_NATIVE_HESSIAN
#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "methods/dft_method.hpp"
#include "runtime/cuda_resources.cuh"
#endif

namespace {
/** Compare byte intervals without adding a byte count to a pointer address. */
bool overlaps(const void* first, std::size_t first_size, const void* second,
              std::size_t second_size) noexcept {
  const auto first_start = reinterpret_cast<std::uintptr_t>(first);
  const auto second_start = reinterpret_cast<std::uintptr_t>(second);
  return first_size && second_size &&
         (first_start <= second_start ? second_start - first_start < first_size
                                      : first_start - second_start < second_size);
}
}  // namespace

extern "C" generativeqc_status generativeqc_ks_hessian_default_options_v2(
    generativeqc_ks_hessian_options_v2* options) {
  if (!options) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *options = {sizeof(*options),
              GENERATIVEQC_ABI_VERSION,
              2ULL << 30,
              64ULL << 20,
              128ULL << 20,
              64ULL << 20,
              64ULL << 20,
              16'777'216,
              30,
              200,
              2,
              1e-10,
              1e-12,
              0};
  return GENERATIVEQC_STATUS_SUCCESS;
}

extern "C" std::uint32_t generativeqc_ks_hessian_cuda_version_v2() {
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  return 2;
#else
  return 0;
#endif
}

extern "C" std::uint32_t generativeqc_ks_hessian_cuda_resource_version_v1() {
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  return 1;
#else
  return 0;
#endif
}

extern "C" std::uint32_t generativeqc_ks_hessian_cuda_r2scan_domain_v1() {
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  if (generativeqc::dft::kCudaR2scanVariational)
    return generativeqc::dft::kCudaR2scanVariationalDomain;
#endif
  return 0;
}

extern "C" std::uint32_t generativeqc_ks_hessian_cuda_m062x_domain_v1() {
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  if (generativeqc::dft::kCudaM062XVariational)
    return generativeqc::dft::kCudaM062XVariationalDomain;
#endif
  return 0;
}

extern "C" std::uint32_t generativeqc_ks_hessian_cuda_mn15_domain_v1() {
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  if (generativeqc::dft::kCudaMN15Variational) return generativeqc::dft::kCudaMN15VariationalDomain;
#endif
  return 0;
}

extern "C" generativeqc_status generativeqc_ks_hessian_cuda_v2(
    generativeqc_batch* batch, std::size_t index, const generativeqc_ks_hessian_options_v2* options,
    const double* direction, std::size_t direction_count, double* output, std::size_t output_count,
    generativeqc_ks_hessian_receipt_v2* receipt) {
  if (!batch || !batch->context || !batch->plan || !output || !options || !receipt)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  if (!generativeqc::api::valid_descriptor(options) ||
      !generativeqc::api::valid_descriptor(receipt))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  batch->context->last_detail.clear();
  if (index >= batch->atom_counts.size() || !batch->atom_counts[index] ||
      batch->atom_counts[index] > std::numeric_limits<std::size_t>::max() / 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const std::size_t coordinates = 3 * std::size_t(batch->atom_counts[index]);
  const bool matrix = !direction;
  if ((matrix &&
       (direction_count || coordinates > std::numeric_limits<std::size_t>::max() / coordinates)) ||
      (!matrix && (direction_count != coordinates || options->response_fp32_proposal)) ||
      options->response_fp32_proposal > 1 ||
      output_count != (matrix ? coordinates * coordinates : coordinates) ||
      output_count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(double)) ||
      2 * output_count * sizeof(double) > options->output_bytes || !options->device_bytes ||
      !options->integral_bytes || !options->grid_bytes || !options->response_bytes ||
      !options->max_nonlocal_pairs || !options->restart || !options->max_iterations ||
      !options->reorthogonalizations || !std::isfinite(options->relative_tolerance) ||
      options->relative_tolerance < 0 || options->relative_tolerance >= 1 ||
      !std::isfinite(options->absolute_tolerance) || options->absolute_tolerance < 0)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  const auto output_bytes = output_count * sizeof(double);
  const auto direction_bytes = direction_count * sizeof(double);
  if (overlaps(output, output_bytes, direction, direction_bytes) ||
      overlaps(output, output_bytes, receipt, sizeof(*receipt)) ||
      overlaps(output, output_bytes, options, sizeof(*options)) ||
      overlaps(receipt, sizeof(*receipt), direction, direction_bytes) ||
      overlaps(receipt, sizeof(*receipt), options, sizeof(*options)) ||
      overlaps(direction, direction_bytes, options, sizeof(*options))) {
    batch->context->last_detail = "native Hessian host input and output regions must be disjoint";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (direction && !std::all_of(direction, direction + direction_count,
                                [](double value) { return std::isfinite(value); }))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
#if GENERATIVEQC_HAS_NATIVE_HESSIAN
  try {
    const auto started = std::chrono::steady_clock::now();
    generativeqc::runtime::DeviceResourceScope resource_scope(batch->resource_ledger);
    const auto external_bytes = (output_count + direction_count) * sizeof(double);
    if (external_bytes >= options->device_bytes) {
      batch->context->last_detail = "native Hessian result exceeds the device capacity";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    generativeqc::dft::CudaKsFinalStateToken token;
    std::string detail;
    auto status =
        generativeqc::methods::detail::dft_final_state_token(*batch->plan, index, token, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      batch->context->last_detail = detail;
      return status;
    }
    if (token.identity.determinant.model.backend != generativeqc::scf::FockBackend::Cuda) {
      batch->context->last_detail = "native CUDA Hessian requires an accepted CUDA KS state";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (!token.strict_physical_fock) {
      batch->context->last_detail = "native Hessian requires a fresh strict-FP64 physical SCF Fock";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (token.relaxed_stationarity) {
      batch->context->last_detail =
          "native Hessian cannot use an SCF state with relaxed application stationarity";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    generativeqc::runtime::CudaDeviceScope device(token.identity.model.device);
    generativeqc::dft::CudaKsGridReference reference(*batch->plan, index, token);
    status = reference.prepare(detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      batch->context->last_detail = detail;
      return status;
    }
    // The canonical validation may allocate through the retained SCF owner.
    // Response buffers may use neither its live bytes nor unspent promises.
    auto device_bytes = options->device_bytes;
    if (batch->resource_ledger) {
      const std::lock_guard guard(generativeqc::runtime::device_resource_mutex);
      device_bytes = std::min(
          device_bytes, generativeqc::runtime::device_resource_unpromised(*batch->resource_ledger));
    }
    if (external_bytes >= device_bytes) {
      batch->context->last_detail = "global device ledger has no capacity for Hessian response";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    const auto stream = reference.reference().stream;
    generativeqc::runtime::OwnedCudaBuffer<double> candidate(token.identity.model.device,
                                                             output_count, stream);
    generativeqc::runtime::OwnedCudaBuffer<double> motion;
    if (!matrix) {
      motion.allocate(token.identity.model.device, direction_count, stream);
      generativeqc::runtime::cuda_resource_check(cudaMemcpyAsync(motion.get(), direction,
                                                                 direction_count * sizeof(double),
                                                                 cudaMemcpyHostToDevice, stream));
    }
    generativeqc::dft::CudaKsWb97mvHessianOptions policy;
    policy.max_device_bytes = device_bytes - external_bytes;
    policy.grid.max_workspace_bytes = policy.geometry.max_workspace_bytes = options->grid_bytes;
    policy.induced.grid.max_workspace_bytes = options->grid_bytes;
    policy.grid.max_pair_evaluations = policy.induced.grid.max_pair_evaluations =
        options->max_nonlocal_pairs;
    policy.one_electron.max_workspace_bytes = policy.exact.max_workspace_bytes =
        options->integral_bytes;
    policy.raw_df.max_workspace_bytes = policy.fitted.max_workspace_bytes =
        policy.ecp.max_workspace_bytes = options->integral_bytes;
    policy.cpks.max_workspace_bytes = policy.cpks.gmres.max_workspace_bytes =
        options->response_bytes;
    policy.cpks.gmres.restart = options->restart;
    policy.cpks.gmres.max_iterations = options->max_iterations;
    policy.cpks.gmres.reorthogonalizations = options->reorthogonalizations;
    policy.cpks.gmres.relative_tolerance = options->relative_tolerance;
    policy.cpks.gmres.absolute_tolerance = options->absolute_tolerance;
    if (options->response_fp32_proposal)
      policy.cpks.resident_precision =
          generativeqc::response::CudaCpksResidentPrecision::fp32_proposal;
    generativeqc::dft::CudaKsWb97mvHessian endpoint;
    using Mode = generativeqc::dft::CudaKsWb97mvHessianMode;
    status = endpoint.prepare(*batch->plan, index, token,
                              matrix ? Mode::dense_and_action : Mode::action_only, policy, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      batch->context->last_detail = detail;
      return status;
    }
    generativeqc::dft::CudaKsWb97mvHessianResult result;
    const auto error = matrix ? endpoint.hessian(candidate.get(), stream, result)
                              : endpoint.apply(motion.get(), candidate.get(), stream, result);
    if (error == cudaErrorInvalidValue && result.failure_check)
      throw std::runtime_error(result.failure_check);
    generativeqc::runtime::cuda_resource_check(error);
    if (!result.converged()) {
      batch->context->last_detail =
          "native stationary response did not converge; no Hessian result published";
      return GENERATIVEQC_STATUS_NOT_CONVERGED;
    }
    std::vector<double> published(output_count);
    // Drain even a failed copy submission before releasing its host target.
    const auto transfer_status =
        cudaMemcpyAsync(published.data(), candidate.get(), output_count * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
    const auto completion_status = cudaStreamSynchronize(stream);
    generativeqc::runtime::cuda_resource_check(transfer_status);
    generativeqc::runtime::cuda_resource_check(completion_status);
    const auto work = endpoint.work();
    generativeqc_ks_hessian_receipt_v2 completed{};
    completed.struct_size = sizeof(completed);
    completed.abi_version = GENERATIVEQC_ABI_VERSION;
    completed.device_bytes = work.device_bytes + external_bytes;
    completed.completed_directions = work.completed_directions;
    completed.induced_fock_actions = work.induced.actions;
    completed.exact_primitive_components = work.exact.component_evaluations;
    completed.one_electron_primitive_components = work.one_electron.integrals.component_evaluations;
    completed.df_value_passes = work.raw_df.value_passes + work.raw_df_range.value_passes;
    completed.df_contraction_passes =
        work.raw_df.contraction_passes + work.raw_df_range.contraction_passes;
    completed.grid_right_directions = work.grid.nuclear_directions;
    completed.grid_mixed_contractions = work.grid.nuclear_contractions;
    completed.vv10_reference_pairs = work.induced.grid.reference_pairs + work.grid.reference_pairs +
                                     work.induced.separate_vv10.reference_pairs +
                                     work.separate_vv10.reference_pairs;
    completed.response_iterations = result.response.gmres.iterations;
    completed.response_workspace_bytes = result.response.gmres.workspace_bytes;
    completed.ecp_workspace_bytes = work.ecp.workspace_bytes;
    completed.df_workspace_bytes = work.raw_df.workspace_bytes + work.raw_df_range.workspace_bytes;
    completed.response_rhs_norm = result.response.gmres.rhs_norm;
    completed.response_residual_norm = result.response.gmres.residual_norm;
    completed.response_tolerance = result.response.gmres.tolerance;
    completed.cpks_operator_bytes = work.cpks_operator_bytes;
    completed.cpks_physical_checks = work.cpks_physical_checks;
    completed.cpks_cached_actions = work.cpks_cached_actions;
    completed.cpks_proposal_iterations = work.cpks_proposal_iterations;
    completed.cpks_proposal_seeds = work.cpks_proposal_seeds;
    completed.cpks_strict_iterations = work.cpks_strict_iterations;
    completed.execution_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::copy(published.begin(), published.end(), output);
    *receipt = completed;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&batch->context->last_detail);
  }
#else
  batch->context->last_detail =
      "CUDA KS Hessian requires a linked build-time AOT package; Python/JIT fallback is "
      "disabled";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
