#pragma once

#include <cstddef>
#include <vector>

#include "cc/uccsd_solver.hpp"
#include "integrals/correlation_backend.hpp"
#include "response/native_gmres.hpp"

namespace generativeqc::response {
class LinearResponseProblem;
}
namespace generativeqc::posthf {
class DensityFittedInteractionSource;
}

namespace generativeqc::cc {

/** Complete, checked admission for native unrestricted analytic response.
 * Dense raw-MO weights are resource-bounded rather than AO-count-bounded.
 * This is a conservative numeric payload peak, not an allocation measurement. */
std::size_t unrestricted_force_capacity(std::size_t nbf, std::size_t alpha, std::size_t beta,
                                        const SolverOptions& options, bool triples,
                                        std::size_t reference_capacity,
                                        bool separate_reference = false);

struct UnrestrictedForce {
  std::vector<double> forces;
  response::GmresResult lambda, orbital;
  std::size_t numeric_capacity_bytes{};
  double stationarity_residual{};
};

/** Analytic derivative of a freshly qualified physical UCCSD/(T) endpoint.
 * The borrowed orbital problem must come from the identical UHF Fock owner.
 * No iterative/shifted denominator, finite difference or reference oracle is
 * differentiated. Any failed stationarity check prevents force publication. */
UnrestrictedForce unrestricted_force(
    const hf::UnrestrictedPhysicalReference& reference,
    const integrals::ElectronInteractionSource& source, const UnrestrictedSolve& parent,
    const SolverOptions& options, bool triples, std::size_t reference_capacity,
    const response::LinearResponseProblem* orbital_problem, bool cuda = false, int device = 0,
    const posthf::DensityFittedInteractionSource* fitted = nullptr,
    const integrals::ElectronInteractionSource* independent_reference = nullptr,
    const integrals::CorrelationIntegralBackend& backend = {});

}  // namespace generativeqc::cc
