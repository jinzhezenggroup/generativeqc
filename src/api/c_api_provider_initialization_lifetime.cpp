#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/provider_initialization_lifetime.hpp"
#include "generativeqc/provider_initialization.h"
#include "methods/provider_initialization_lifetime.hpp"

namespace generativeqc::api {
generativeqc_status close_provider_initialization_current(
    generativeqc_calculation* calculation) noexcept {
  if (!calculation) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!calculation->pending_provider_initialization || !calculation->plan)
    return GENERATIVEQC_STATUS_SUCCESS;
  try {
    auto* owner = dynamic_cast<methods::ProviderInitializationCloseable*>(calculation->plan.get());
    if (!owner) {
      calculation->context->last_detail = "native calculation has no pending owner close interface";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    return owner->close_provider_initialization_current(calculation->context->last_detail);
  } catch (...) {
    return map_exception(&calculation->context->last_detail);
  }
}

generativeqc_status close_provider_initialization_context(generativeqc_context* context) noexcept {
  if (!context) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    auto* calculation = context->provider_initialization_calculation;
    if (!calculation) return GENERATIVEQC_STATUS_SUCCESS;
    const auto status = close_provider_initialization_current(calculation);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    // An unpublished failed candidate has no caller handle. Its owning context
    // keeps it reachable until this status-bearing close confirms full closure.
    if (context->provider_initialization_failed_candidate) {
      context->provider_initialization_calculation = nullptr;
      context->provider_initialization_failed_candidate = false;
      delete calculation;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception(&context->last_detail);
  }
}
}  // namespace generativeqc::api

extern "C" generativeqc_status generativeqc_context_close_provider_initialization_v1(
    generativeqc_context* context) {
  return generativeqc::api::close_provider_initialization_context(context);
}
