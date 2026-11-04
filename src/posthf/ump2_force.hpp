#pragma once
#include "hf/reference.hpp"
#include "integrals/correlation_backend.hpp"
#include "posthf/mp2_force.hpp"
#include "response/unrestricted.hpp"
namespace generativeqc::posthf {
class DensityFittedInteractionSource;
}
namespace generativeqc::mp2 {
/** Complete real conventional UMP2 analytic force. Borrowed owners stay live;
 * any response/denominator/resource failure publishes no force. CUDA uses
 * device integral transforms/derivatives with the existing host scalar
 * cotangent assembly, orbital projections and Krylov controller explicitly. */
ConventionalForceResult unrestricted_force_cpu(
    const hf::UnrestrictedPhysicalReference& reference,
    const response::SpinResponseProblem* response_problem,
    const integrals::ElectronInteractionSource& source, std::size_t budget,
    double denominator_threshold, bool cuda = false, int device = 0,
    const posthf::DensityFittedInteractionSource* fitted = nullptr,
    const integrals::ElectronInteractionSource* independent_reference = nullptr,
    const integrals::CorrelationIntegralBackend& backend = {});
}  // namespace generativeqc::mp2
