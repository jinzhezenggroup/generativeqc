#pragma once

#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Common conventional UHF admission for native UCCSD and UCCSD(T). */
generativeqc_status validate_uccsd_system(generativeqc_method method, const core::System& system,
                                          std::string& detail);

/** Prepare the resource-bounded native CPU/CUDA unrestricted energy/force owner.
 * Optional fitted operators preserve their explicitly selected UHF reference;
 * unsupported requests never select prototypes or reference-oracle work. */
std::unique_ptr<PreparedCalculation> prepare_uccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor,
    const core::System* auxiliary_basis = nullptr);

}  // namespace generativeqc::methods::detail
