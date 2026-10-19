#pragma once
#include "hf/reference.hpp"
#include "posthf/mp2_force.hpp"
#include "response/unrestricted.hpp"
namespace generativeqc::mp2 {
/** Complete real conventional UMP2 analytic force. Borrowed owners stay live;
 * any response/denominator/resource failure publishes no force. */
ConventionalForceResult unrestricted_force_cpu(
    const hf::UnrestrictedPhysicalReference& reference,
    const response::SpinResponseProblem* response_problem,
    const integrals::ElectronInteractionSource& source, std::size_t budget,
    double denominator_threshold);
}  // namespace generativeqc::mp2
