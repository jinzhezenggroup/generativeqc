#pragma once
#include <memory>

#include "cc/solver.hpp"
#include "hf/reference.hpp"
#include "methods/method.hpp"

namespace generativeqc::scf {
class PreparedFockPlan;
}
namespace generativeqc::methods::detail {
struct RccsdNativeState {
  std::shared_ptr<const hf::PhysicalReference> reference;
  cc::Problem problem;
  cc::SolverResult solved;
  std::vector<double> eps_o, eps_v;
  generativeqc_correlation_diagnostic diagnostic{};
  CcPerformanceDiagnostic performance{};
  Result result;
  std::size_t budget{};
};

RccsdNativeState run_rccsd_native_state(
    runtime::ExecutionContext&, const core::System&, const generativeqc_method_descriptor&,
    std::unique_ptr<scf::PreparedFockPlan>* prepared_exact_cache = nullptr);
generativeqc_status validate_rccsd_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities&, core::ContextState&,
                                                   std::vector<core::System>,
                                                   const generativeqc_method_descriptor&,
                                                   generativeqc_batch_flags);
}  // namespace generativeqc::methods::detail
