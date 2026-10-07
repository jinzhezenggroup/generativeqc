#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "core/electronic_reference.hpp"
#include "molecule/basis_geometry_identity.hpp"

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

/** Owned, validated physical UHF determinant for unrestricted post-HF energy.
 * Each channel retains the unshifted physical Fock and its own canonical frame.
 * The view borrows this object and must not outlive it. */
struct UnrestrictedPhysicalReference {
  std::size_t nbf{};
  std::array<std::size_t, 2> nocc{};
  std::vector<double> overlap, hcore;
  std::array<std::vector<double>, 2> fock, coefficients, orbital_energies, density;
  double energy{};
  double commutator_residual{}, canonical_density_drift{}, canonical_error{};
  std::optional<molecule::BasisGeometryIdentity> source_identity;
  int source_charge{}, source_electrons{};
  unsigned source_multiplicity{};

  [[nodiscard]] core::ElectronicReferenceView electronic_reference() const noexcept {
    core::ElectronicReferenceView view;
    view.basis_functions = nbf;
    view.spin_channels = 2;
    view.overlap = overlap;
    view.hcore = hcore;
    view.energy = energy;
    for (std::size_t spin = 0; spin < 2; ++spin)
      view.channels[spin] = {nocc[spin],    coefficients[spin], orbital_energies[spin],
                             density[spin], fock[spin],         {}};
    return view;
  }
};

}  // namespace generativeqc::hf
