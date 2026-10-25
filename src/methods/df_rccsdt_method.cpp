#include "methods/df_rccsdt_method.hpp"

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "methods/correlation_convergence.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "methods/df_ccsdt_force.hpp"
#endif
#include "methods/rccsd_method.hpp"

namespace generativeqc::methods::detail {
namespace {

constexpr double kQualifiedMetricThreshold = 1e-10;

void validate_descriptor(const generativeqc_method_descriptor& descriptor,
                         const runtime::ExecutionContext& execution) {
  if (!execution.cuda_requested())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) currently requires CUDA");
  if (descriptor.density_fitting_auxiliary_basis == nullptr)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) requires an explicit auxiliary basis");
  const bool correlation_mp2 =
      descriptor.method == GENERATIVEQC_METHOD_MP2 &&
      descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA;
  if (!correlation_mp2 && descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CUDA &&
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_AUTO &&
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_FULL_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DF-RCCSD(T) requires density_fitting='cuda' or 'auto'");
  const double threshold = descriptor.density_fitting_relative_threshold == 0.0
                               ? kQualifiedMetricThreshold
                               : descriptor.density_fitting_relative_threshold;
  if (!std::isfinite(threshold) || threshold <= 0.0 || threshold >= 1.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) metric threshold must lie in (0,1)");
  if (threshold != kQualifiedMetricThreshold)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "public DF-RCCSD(T) currently qualifies density_fitting_relative_threshold=1e-10");
  if (descriptor.density_fitting_memory_budget_bytes != 0)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "DF-RCCSD(T) uses correlation_memory_budget_bytes for complete endpoint admission; "
        "a separate density-fitting memory budget is not yet qualified");
  if (descriptor.screening_tolerance != 0.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) requires screening_tolerance=0");
  if (descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "DF-RCCSD(T) requires FP64 precision");
  if (descriptor.ccsd_frozen_core != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DF-RCCSD(T) frozen-core references are not implemented");
}

class DfRccsdtPrepared final : public PreparedCalculation {
 public:
  DfRccsdtPrepared(Capabilities capabilities, runtime::ExecutionContext execution,
                   core::System system, core::System auxiliary,
                   const generativeqc_method_descriptor& descriptor)
      : PreparedCalculation(execution.backend(), execution.device_id()),
        capabilities_(capabilities),
        execution_(std::move(execution)),
        system_(std::move(system)),
        auxiliary_(std::move(auxiliary)),
        descriptor_(descriptor) {
    convergence_ = correlation_convergence(descriptor);
    preconvergence_ = correlation_preconvergence(descriptor);
    descriptor_.struct_size = sizeof(descriptor_);
    full_df_reference_ = descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_FULL_CUDA;
    // Preserve the legacy descriptor contract of the CC solver. The explicit
    // demand carries the RI-JK model through reference, companions and response.
    descriptor_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.density_fitting_memory_budget_bytes = 0;
    descriptor_.ks_options = nullptr;
    descriptor_.initial_guess = nullptr;
    if (descriptor_.method == GENERATIVEQC_METHOD_MP2) capabilities_.supports_batch = false;
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }

  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    return execution_.resources();
  }
  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_correlation_;
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_performance_;
  }
  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
  }

  Result execute_impl(bool compute_forces) override { return execute_group(compute_forces, 0); }
  Result execute_components_impl(bool compute_forces, unsigned companions) override {
    const auto allowed =
        descriptor_.method == GENERATIVEQC_METHOD_MP2 ? kHfComponent : kHfComponent | kMp2Component;
    if (companions & ~allowed)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "unsupported DF correlation companions");
    return execute_group(compute_forces, companions);
  }
  Result execute_group(bool compute_forces, unsigned companions) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
#if GENERATIVEQC_HAS_CUDA
    auto native =
        companions || descriptor_.method == GENERATIVEQC_METHOD_MP2 || full_df_reference_ ||
                convergence_.relaxed() || preconvergence_ != core::DfPreconvergence::Legacy
            ? run_df_correlation_group_native(
                  execution_, system_, auxiliary_, descriptor_, compute_forces,
                  {descriptor_.method == GENERATIVEQC_METHOD_MP2, companions, full_df_reference_},
                  convergence_, preconvergence_)
            : run_df_ccsdt_native(execution_, system_, auxiliary_, descriptor_, compute_forces);
    native.primal.triples_seconds = native.triples_seconds;
    // Publish detached response work only after the entire endpoint succeeds.
    // Action counters cover the shared transpose operator, not the separate
    // independent audit. Phase times include their nested preparation/replay.
    if (compute_forces) {
      auto& performance = native.primal;
      const auto& lambda = native.lambda;
      performance.df_response_available = true;
      performance.lambda_seconds = native.lambda_seconds;
      performance.source_response_seconds = native.source_response_seconds;
      performance.orbital_seconds = native.orbital_seconds;
      performance.lambda_fp32_executed = lambda.df_fp32_gemm_calls > 0;
      performance.lambda_fp32_refused = lambda.df_mixed_precision_refused;
      performance.lambda_fp32_iterations = lambda.df_mixed_iterations;
      performance.lambda_fp64_iterations = lambda.df_strict_iterations;
      performance.lambda_fp32_actions = lambda.df_mixed_actions;
      performance.lambda_fp64_actions = lambda.df_strict_actions;
      performance.lambda_fp32_gemm_calls = lambda.df_fp32_gemm_calls;
      performance.lambda_fp32_cast_elements = lambda.df_precision_cast_elements;
      performance.lambda_auxiliary_batch_size = lambda.df_auxiliary_batch_size;
      performance.lambda_numeric_capacity_bytes = lambda.numeric_capacity_bytes;
      performance.lambda_owned_device_bytes = lambda.owned_device_bytes;
    }
    last_correlation_ = native.correlation;
    last_performance_ = native.primal;
    return std::move(native.method_result);
#else
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) requires a CUDA-enabled build");
#endif
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::System system_;
  core::System auxiliary_;
  bool full_df_reference_{};
  generativeqc_method_descriptor descriptor_{};
  core::CorrelationConvergence convergence_;
  core::DfPreconvergence preconvergence_{core::DfPreconvergence::Legacy};
  std::optional<generativeqc_correlation_diagnostic> last_correlation_;
  std::optional<CcPerformanceDiagnostic> last_performance_;
  mutable std::mutex mutex_;
};

}  // namespace

generativeqc_status validate_df_rccsdt_system(generativeqc_method method,
                                              const core::System& system, std::string& detail) {
  return validate_rccsd_system(method, system, detail);
}

std::unique_ptr<PreparedCalculation> prepare_df_rccsdt_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* auxiliary_basis) {
  runtime::ExecutionContext execution(context);
  validate_descriptor(descriptor, execution);
  if (auxiliary_basis == nullptr)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "auxiliary basis must be resolved at the API boundary");
  core::System auxiliary = *auxiliary_basis;
  if (auxiliary.atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) auxiliary geometry differs from the orbital system");
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    if (auxiliary.atoms[atom].atomic_number != system.atoms[atom].atomic_number ||
        auxiliary.atoms[atom].position != system.atoms[atom].position)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DF-RCCSD(T) auxiliary geometry differs from the orbital system");
  return std::make_unique<DfRccsdtPrepared>(capabilities, std::move(execution), system,
                                            std::move(auxiliary), descriptor);
}

}  // namespace generativeqc::methods::detail
