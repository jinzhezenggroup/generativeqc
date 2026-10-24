#pragma once

#include "core/correlation_convergence.hpp"
#include "methods/application_precision.hpp"

namespace generativeqc::methods::detail {

/** Decode while the extended descriptor is alive. Prepared owners retain only
 * this value, never a pointer beyond a copied legacy descriptor prefix. */
inline core::CorrelationConvergence correlation_convergence(
    const generativeqc_method_descriptor& descriptor) {
  return core::CorrelationConvergence::from_budget(application_precision_budget(descriptor));
}

}  // namespace generativeqc::methods::detail
