#pragma once
#include <memory>
#include <optional>

#include "cc/df_source.hpp"
#include "cc/solver.hpp"
#include "hf/reference.hpp"
#include "methods/method.hpp"
#include "scf/types.hpp"

namespace generativeqc::integrals {
class ElectronInteractionSource;
}
namespace generativeqc::hf {
class DFReference;
}
namespace generativeqc::scf {
class PreparedFockPlan;
struct CudaRhfBucketPlan;
}  // namespace generativeqc::scf
namespace generativeqc::methods::detail {
struct RccsdNativeState {
  std::shared_ptr<const hf::PhysicalReference> reference;
  /** Present only for the explicitly requested full RI-JK reference model. */
  std::shared_ptr<hf::DFReference> df_reference;
  // Sole source owner from this exact RHF: resident storage when eligible,
  // otherwise compact immutable metadata. Shared retirement covers all CC stages.
  std::shared_ptr<const integrals::ElectronInteractionSource> reference_interaction_source;
  cc::Problem problem;
  cc::SolverResult solved;
  // Opt-in immutable DF source/frame owner, retained across CC response.
  cc::DFSourceResult df_source;
  // Correlation-only MP2 bypasses CC iteration and retains only its demanded VJP.
  cc::DFMP2Result df_mp2;
  bool df_mp2_mode{};
  std::vector<double> eps_o, eps_v;
  generativeqc_correlation_diagnostic diagnostic{};
  CcPerformanceDiagnostic performance{};
  Result result;
  std::size_t budget{};
  std::size_t external_reservation_bytes{};
  // Full numeric reservation is separate from directly owned device telemetry.
  std::size_t reference_execution_plan_bytes{};
  std::size_t reference_execution_plan_device_bytes{};
  double reference_energy_change{};
  double reference_density_rms{};
  int reference_iterations{};
  /** Complete reference operator census, distinct from SCF iteration count. */
  scf::PrecisionProvenance reference_work;
};

// The optional auxiliary selects the internal correlation-only DF Hamiltonian.
// The descriptor remains conventional; df_reference explicitly selects RI-JK
// and retains its immutable owner. Otherwise the reference is exact FP64 RHF.
// Response retention
// keeps the original source/frame alive and charges it beside the CC solve;
// energy-only calls release it normally. Public method registration is separate.
// An absent mixed request selects the qualified native DF auto policy; explicit
// true/false retain diagnostic warmup and exact-FP64 ablations at every size.
RccsdNativeState run_rccsd_native_state(
    runtime::ExecutionContext&, const core::System&, const generativeqc_method_descriptor&,
    std::unique_ptr<scf::PreparedFockPlan>* prepared_exact_cache = nullptr,
    const std::vector<double>* initial_density = nullptr, bool* warm_start_fallback = nullptr,
    std::size_t external_reservation_bytes = 0, const core::System* correlation_auxiliary = nullptr,
    bool retain_df_response = false, bool df_matrix_gemm = true,
    scf::CudaRhfBucketPlan** cuda_reference_plan = nullptr,
    std::size_t df_auxiliary_batch_limit = 8, bool derived_denominators = true,
    bool packed_diis = false, std::optional<bool> df_mixed_precision = std::nullopt,
    double df_mixed_precision_switch_residual =
        cc::SolverOptions{}.df_mixed_precision_switch_residual,
    bool df_mp2_mode = false, bool df_reference = false);
generativeqc_status validate_rccsd_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System* auxiliary_basis = nullptr);
std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities&, core::ContextState&,
                                                   std::vector<core::System>,
                                                   const generativeqc_method_descriptor&,
                                                   generativeqc_batch_flags,
                                                   const core::System* auxiliary_basis = nullptr);
}  // namespace generativeqc::methods::detail
