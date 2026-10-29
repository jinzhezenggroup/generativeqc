#pragma once

#include <cstdint>

#include "generativeqc/generativeqc.h"
#include "scf/provider_initialization_intent.hpp"

namespace generativeqc::api {

/** Implemented by the existing preparation owner, not a second publisher.
 * The owner validates actual context/method/system/accounting constraints under
 * its lock, creates one private candidate and publishes only after admission. */
generativeqc_status prepare_calculation_with_provider_initialization_intent(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* method, scf::ProviderInitializationIntent intent,
    generativeqc_calculation** calculation) noexcept;

/** Compiled support only. The existing native preparation owner defines it. */
std::uint32_t provider_initialization_intent_capabilities() noexcept;

}  // namespace generativeqc::api
