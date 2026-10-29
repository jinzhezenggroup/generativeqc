#include <new>
#include <stdexcept>

#include "api/provider_initialization_intent.hpp"
#include "generativeqc/provider_initialization.h"

extern "C" {

uint32_t generativeqc_provider_initialization_intent_version(void) { return 1; }

uint32_t generativeqc_provider_initialization_capabilities_v1(void) {
  return generativeqc::api::provider_initialization_intent_capabilities();
}

generativeqc_status generativeqc_calculation_prepare_with_provider_initialization_v1(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* method,
    const generativeqc_provider_initialization_intent_v1* intent,
    generativeqc_calculation** calculation) {
  if (!intent) return generativeqc_calculation_prepare(context, system, method, calculation);
  if (!context || !system || !method || !calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (intent->struct_size < sizeof(*intent) || intent->abi_version != 1) {
    *calculation = nullptr;
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  }
  const auto snapshot = *intent;
  *calculation = nullptr;
  if (snapshot.reserved != 0) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (snapshot.mode == GENERATIVEQC_PROVIDER_INITIALIZATION_LEGACY_DEFAULT) {
    if (snapshot.allowance_bytes != 0) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    return generativeqc_calculation_prepare(context, system, method, calculation);
  }
  if (snapshot.mode != GENERATIVEQC_PROVIDER_INITIALIZATION_PENDING)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (method->struct_size < sizeof(*method) || method->abi_version != GENERATIVEQC_ABI_VERSION)
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  if ((method->method != GENERATIVEQC_METHOD_MP2 && method->method != GENERATIVEQC_METHOD_RCCSD) ||
      method->density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
      method->density_fitting_auxiliary_basis || method->ks_options)
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  try {
    const generativeqc::scf::ProviderInitializationIntent pending(snapshot.allowance_bytes);
    return generativeqc::api::prepare_calculation_with_provider_initialization_intent(
        context, system, method, pending, calculation);
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

}  // extern "C"
