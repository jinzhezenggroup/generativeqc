#pragma once

#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Common conventional UHF admission for native UCCSD and UCCSD(T). */
generativeqc_status validate_uccsd_system(generativeqc_method method, const core::System& system,
                                          std::string& detail);

/** Prepare the native energy-only unrestricted CC owner. Unsupported backends,
 * derivative properties and optional Hamiltonians never select prototypes. */
std::unique_ptr<PreparedCalculation> prepare_uccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor);

}  // namespace generativeqc::methods::detail
