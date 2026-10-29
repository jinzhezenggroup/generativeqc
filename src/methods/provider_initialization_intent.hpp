#pragma once

#include "methods/method.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/provider_initialization_intent.hpp"

namespace generativeqc::methods {

/** Validate native-owned metadata before choosing the existing scalar factory.
 * This is neither a resource voucher nor a context/provider qualification query. */
inline void require_provider_initialization_intent_scope(
    const core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor) {
  if (context.requested_backend != core::kCudaBackend ||
      context.executed_backend != core::kCudaBackend || context.device_id < 0 ||
      system.multiplicity != 1 ||
      (descriptor.method != GENERATIVEQC_METHOD_MP2 &&
       descriptor.method != GENERATIVEQC_METHOD_RCCSD) ||
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
      descriptor.density_fitting_auxiliary_basis || descriptor.ks_options)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "provider intent requires scalar restricted conventional CUDA MP2/RCCSD");
  if (runtime::active_device_resource_ledger)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "provider intent has unknown active ledger accounting basis");
}

/** The private registry proposal preserves the existing validators before
 * selecting an in-place factory. The MP2/RCCSD detail factories and native
 * first-reference wiring are implemented in their incumbent translation units;
 * each retains an inert destination before initialization. Registry/public ABI
 * integration remains unregistered and optional capabilities remain zero. */
void prepare_calculation_with_provider_initialization_intent_into(
    core::ContextState&, const core::System&, const generativeqc_method_descriptor&,
    const core::System*, scf::ProviderInitializationIntent,
    std::unique_ptr<PreparedCalculation>& destination);

namespace detail {
void prepare_mp2_calculation_with_provider_initialization_intent_into(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System*, scf::ProviderInitializationIntent,
    std::unique_ptr<PreparedCalculation>& destination);
void prepare_rccsd_calculation_with_provider_initialization_intent_into(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System*, scf::ProviderInitializationIntent,
    std::unique_ptr<PreparedCalculation>& destination);
}  // namespace detail

}  // namespace generativeqc::methods
