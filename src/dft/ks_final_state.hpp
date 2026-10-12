#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/electronic_reference.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/scf_diagnostic.hpp"
#include "scf/solver/final_state.hpp"

namespace generativeqc::dft {

/** Immutable semilocal model and owner identity not represented by the common
 * mean-field determinant identity. The owner and solve epoch prevent a
 * numerically identical replacement plan from authorizing an old state. */
struct KsModelIdentity {
  std::uint32_t version{1};
  std::uint32_t scf_domain_version{1};
  GridSpec grid;
  std::size_t tile_points{};
  /** Stable generated semilocal/registered functional code; immutable provenance. */
  std::uint32_t functional{};
  unsigned spins{};
  // CPU is exactly -1; CUDA is a nonnegative visible device ordinal. The
  // determinant's resolved Fock backend is authoritative, never inferred here.
  int device{};
  std::uint64_t owner{};
  double semilocal_exchange_scale{1.0};
  double semilocal_correlation_scale{1.0};
  // The complete physical model, not only its primary Coulomb/exchange owner.
  std::optional<scf::ResolvedFockBuild> range_correction{};
  std::optional<nlc::Vv10Parameters> nonlocal_correlation{};
  nlc::Vv10DensityDomain nonlocal_density_domain{nlc::Vv10DensityDomain::StrictPositive};
  bool operator==(const KsModelIdentity&) const = default;
};

struct KsFinalStateIdentity {
  scf::solver::FinalStateIdentity determinant;
  KsModelIdentity model;
  bool operator==(const KsFinalStateIdentity&) const = default;
};

/** Physical KS evaluation at exactly the tagged density. Components use the
 * DFT energy convention and are not replaced by the quadratic HF formula. */
struct KsPhysicalState {
  KsFinalStateIdentity identity;
  bool physical{};
  std::vector<scf::reference::Matrix> density, fock;
  EnergyComponents components;
  double reported_energy{};
  double physical_residual{};
};

/** Canonical frame produced from the physical non-DIIS Fock. */
struct KsFinalStateCandidate {
  KsFinalStateIdentity identity;
  std::uint64_t fock_density_generation{};
  bool physical_origin{};
  std::vector<scf::reference::EigenResult> spins;
};

struct KsFinalStateDiagnostic {
  scf::solver::FinalStateDiagnostic determinant;
  double component_energy{};
  double reported_energy_error{};
  double physical_residual{};
};

/** Scalar evidence bound by a native owner to its exact physical final frame.
 * The owner must authenticate the token, generations, device and solver result
 * before supplying products; copied identity labels alone prove no lifetime. */
struct KsFinalStateEvidence {
  KsFinalStateIdentity identity;
  std::uint64_t fock_density_generation{};
  bool physical_origin{};
  EnergyComponents components;
  double reported_energy{}, physical_residual{};
};

/** Shared KS component-energy and determinant gates for resident consumers.
 * Requires canonical products and the current physical F[D] generation; never
 * substitutes the quadratic HF energy for the KS component energy. Failed
 * evidence clears output and does not authorize stationary weights. */
bool validate_ks_final_state_evidence(const KsFinalStateIdentity& current, std::size_t dimension,
                                      const KsFinalStateEvidence& evidence,
                                      const scf::solver::FinalStateLimits& limits,
                                      const scf::solver::FinalStateDiagnostic& products,
                                      KsFinalStateDiagnostic& output, std::string& detail);

/** Detached output authorized for a later stationary-gradient consumer. W is
 * absent unless explicitly requested after every identity and numerical gate
 * has passed. */
struct VerifiedKsFinalState {
  KsFinalStateIdentity identity;
  std::vector<scf::reference::Matrix> density, fock, weighted_density;
  std::vector<scf::reference::EigenResult> orbitals;
  EnergyComponents components;
  KsFinalStateDiagnostic diagnostic;
};

/** Borrow a validated RKS/UKS snapshot through the common core reference
 * contract. The caller retains overlap/hcore and state storage ownership. */
core::ElectronicReferenceView electronic_reference(const VerifiedKsFinalState& state,
                                                   const scf::reference::Matrix& overlap,
                                                   const scf::reference::Matrix& hcore);

/** Preserve shared physical/canonical gates and KS component-energy ownership.
 * An optional backend supplies only authenticated validation products; the
 * shared policy still accepts/rejects, and orbital-energy W is unchanged. */
bool validate_ks_final_state(const KsFinalStateIdentity& current,
                             const scf::reference::Matrix& overlap,
                             const scf::reference::Matrix& hcore, const KsPhysicalState& physical,
                             const KsFinalStateCandidate& candidate,
                             const scf::solver::FinalStateLimits& limits,
                             bool compute_weighted_density, VerifiedKsFinalState& output,
                             std::string& detail,
                             const scf::solver::FinalStateOperations* operations = nullptr);

}  // namespace generativeqc::dft
