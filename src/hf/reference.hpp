#pragma once

#include <cstddef>
#include <vector>

#include "core/electronic_reference.hpp"

namespace generativeqc::hf {

/** Owned physical canonical RHF state used by bounded post-HF consumers.
 *
 * This is method state, not SCF solver state. Keeping the canonical RHF
 * reference under the HF owner prevents post-HF/CC consumers from depending on
 * the generic SCF implementation solely to describe an RHF reference.
 */
struct PhysicalReference {
  std::size_t nbf{};
  std::size_t nocc{};
  std::vector<double> overlap, hcore, fock, coefficients, orbital_energies, density;
  /** Optional native force diagnostic: the actual occupation-weighted W used
   * by Pulay response, detached before force assembly. Energy-only/post-HF
   * exports leave this empty and incur no additional matrix reservation. */
  std::vector<double> weighted_density;
  double energy{};
  double commutator_residual{};
  double canonical_density_drift{};
  double eigen_residual{};
  std::size_t numeric_capacity_bytes{};

  /** Borrow this owned RHF state through the method-neutral core contract. */
  [[nodiscard]] core::ElectronicReferenceView electronic_reference() const noexcept {
    core::ElectronicReferenceView view;
    view.basis_functions = nbf;
    view.spin_channels = 1;
    view.overlap = overlap;
    view.hcore = hcore;
    view.energy = energy;
    view.channels[0] = {nocc, coefficients, orbital_energies, density, fock, weighted_density};
    return view;
  }
};

}  // namespace generativeqc::hf
