#include "methods/df_rccsdt_method.hpp"

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "api/handles.hpp"
#include "methods/df_ccsdt_force.hpp"
#include "methods/rccsd_method.hpp"

namespace generativeqc::methods::detail {
namespace {

constexpr double kQualifiedMetricThreshold = 1e-10;

void validate_descriptor(const generativeqc_method_descriptor& descriptor,
                         const runtime::ExecutionContext& execution) {
  if (!execution.cuda_requested())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) energy currently requires CUDA");
  if (descriptor.density_fitting_auxiliary_basis == nullptr)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) requires an explicit auxiliary basis");
  if (descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CUDA &&
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_AUTO)
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
      : capabilities_(capabilities),
        execution_(std::move(execution)),
        system_(std::move(system)),
        auxiliary_(std::move(auxiliary)),
        descriptor_(descriptor) {
    // Keep the RHF reference conventional. The auxiliary is passed separately
    // as the correlation-only DF Hamiltonian source.
    descriptor_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.density_fitting_memory_budget_bytes = 0;
    descriptor_.ks_options = nullptr;
    descriptor_.initial_guess = nullptr;
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

  Result execute(bool compute_forces) override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
    if (compute_forces)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "public DF-RCCSD(T) forces remain unqualified; request energy only (#158)");
#if GENERATIVEQC_HAS_CUDA
    auto native = run_df_ccsdt_native(execution_, system_, auxiliary_, descriptor_, false);
    native.primal.triples_seconds = native.triples_seconds;
    last_correlation_ = native.correlation;
    last_performance_ = native.primal;
    return std::move(native.method_result);
#else
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) energy requires a CUDA-enabled build");
#endif
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::System system_;
  core::System auxiliary_;
  generativeqc_method_descriptor descriptor_{};
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
    const generativeqc_method_descriptor& descriptor) {
  runtime::ExecutionContext execution(context);
  validate_descriptor(descriptor, execution);
  core::System auxiliary = descriptor.density_fitting_auxiliary_basis->data;
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
