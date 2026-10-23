#pragma once

#include "scf/initial_guess/density.hpp"
#include "scf/initial_guess/preliminary_types.hpp"

namespace generativeqc::scf::initial_guess {

struct HuckelDensityResult {
  Matrix density;
  std::size_t atomic_solves{}, occupied_atomic_orbitals{};
  /** Exact molecular C[AO, occupied] used to form D = 2 C C^T. Keeping this
   * existing payload avoids refactorizing D for a native fitted consumer. */
  Matrix coefficients;
};

/** Neutral H-Ar configurations used by spherical average atomic RHF.
 * Return the electron counts in s and p channels, not molecular occupations. */
std::pair<unsigned, unsigned> huckel_atomic_configuration(int atomic_number);

/** Largest atomic s/p integral problem after removing unoccupied channels. */
std::size_t huckel_maximum_atomic_aos(const core::System& system);

/** Native Modified Hückel (updated Wolfsberg-Helmholtz rule).
 * CPU atomic RHF and the occupied-atomic-subspace eigenproblem are explicit
 * production preparation, including for CUDA targets. No PySCF is called.
 * Identical normalized element/basis problems are reused only within this
 * invocation; no geometry-dependent density or unbounded global cache survives.
 * Atomic work is added to the preliminary diagnostic. The caller still applies
 * strict same-basis target admission, optionally using its CUDA eigen owner.
 */
HuckelDensityResult huckel_density(const core::System& system,
                                   const integrals::IntegralData& target,
                                   const PreliminaryOptions& policy,
                                   PreliminaryDiagnostic& diagnostic);

}  // namespace generativeqc::scf::initial_guess
