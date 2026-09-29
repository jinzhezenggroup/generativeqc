#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "response/native_gmres.hpp"

namespace generativeqc::posthf {
class MOBlockProvider;
class DensityFittedBlockProvider;
}  // namespace generativeqc::posthf
namespace generativeqc::hf {
struct PhysicalReference;
}

namespace generativeqc::mp2 {

struct EnergyAdjoint {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> integrals_iajb;
  std::vector<double> orbital_energies;
  const char* equation_hash{};
};

/** Structured relaxed two-electron MP2 cotangent.
 *
 * The complete N^4 MO tensor is intentionally not owned here.  The canonical
 * MP2 correlation adjoint retains its natural ijab block, while every
 * reference/orbital-energy/Z-vector contribution is represented by the
 * Fock-like rank-2 multiplier f[p,q]:
 *
 *   W[p,q,r,s] = 2 f[p,q] delta_occ(r,s)
 *              - f[p,s] delta_occ(q,r)
 *              + G[p,q,r,s]_(iajb only).
 *
 * Versioning keeps the producer/consumer contract explicit as additional
 * factorized terms are introduced.
 */
struct FactorizedTwoElectronWeights {
  static constexpr unsigned current_version = 1;
  unsigned version{current_version};
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> fock;
  std::vector<double> correlation_iajb;
};

struct OrbitalRhs {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> energy_gradient;
  std::vector<double> response_rhs;
  std::vector<double> one_electron;
  // Dense oracle-only representation. Streamed production leaves this empty
  // and retains only the O(N^2) Fock-like multiplier below.
  std::vector<double> two_electron;
  std::vector<double> fock_weights;
};

struct LagrangianWeights {
  std::size_t orbitals{};
  std::size_t occupied{};
  std::vector<double> one_electron;
  // Dense oracle/legacy consumer representation. Production MP2 force paths
  // leave this empty and use two_electron_factors.
  std::vector<double> two_electron;
  FactorizedTwoElectronWeights two_electron_factors;
  std::vector<double> overlap;
  double stationarity_residual{};
};

struct DensityFittedLagrangianWeights {
  std::size_t orbitals{};
  std::size_t auxiliary{};
  std::vector<double> overlap;
  std::vector<double> one_electron;
  std::vector<double> three_center;
  std::vector<double> metric;
  std::size_t workspace_bytes{};
  std::size_t planned_peak_bytes{};
};

struct GradientResourcePlan {
  std::size_t provider_bytes{};
  std::size_t adjoint_bytes{};
  std::size_t response_bytes{};
  std::size_t relaxed_weight_bytes{};
  std::size_t shell_cotangent_bytes{};
  std::size_t derivative_staging_bytes{};
  std::size_t derivative_backend_staging_bytes{};
  std::size_t candidate_output_bytes{};
  std::size_t peak_bytes{};
};

struct DensityFittedGradientResourcePlan {
  std::size_t provider_bytes{};
  std::size_t adjoint_bytes{};
  std::size_t response_bytes{};
  std::size_t relaxed_weight_bytes{};
  std::size_t reverse_result_bytes{};
  std::size_t reverse_workspace_bytes{};
  std::size_t derivative_staging_bytes{};
  std::size_t candidate_output_bytes{};
  std::size_t peak_bytes{};
};

EnergyAdjoint canonical_energy_adjoint(std::span<const double> integrals_iajb,
                                       std::span<const double> orbital_energies,
                                       std::size_t occupied, double denominator_threshold);
OrbitalRhs canonical_orbital_rhs(std::span<const double> hcore_mo, std::span<const double> eri_mo,
                                 const EnergyAdjoint& adjoint, double same_space_threshold);
OrbitalRhs canonical_orbital_rhs_streamed(const hf::PhysicalReference& reference,
                                          std::span<const double> hcore_mo,
                                          const posthf::MOBlockProvider& provider,
                                          const EnergyAdjoint& adjoint, double same_space_threshold,
                                          bool cuda = false, int device_id = 0);
LagrangianWeights canonical_lagrangian_weights(std::span<const double> hcore_mo,
                                               std::span<const double> eri_mo,
                                               const EnergyAdjoint& adjoint,
                                               std::span<const double> response,
                                               double same_space_threshold);
LagrangianWeights canonical_lagrangian_weights_streamed(const hf::PhysicalReference& reference,
                                                        std::span<const double> hcore_mo,
                                                        const posthf::MOBlockProvider& provider,
                                                        EnergyAdjoint adjoint,
                                                        std::span<const double> response,
                                                        double same_space_threshold,
                                                        bool cuda = false, int device_id = 0);

bool valid_factorized_two_electron_weights(const FactorizedTwoElectronWeights& weights);
double factorized_two_electron_weight(const FactorizedTwoElectronWeights& weights,
                                      std::size_t p, std::size_t q, std::size_t r,
                                      std::size_t s);
double two_electron_weight(const LagrangianWeights& weights, std::size_t p, std::size_t q,
                           std::size_t r, std::size_t s);
/** Reverse relaxed MO-basis MP2 weights through the RI factorization.
 *
 * This is the C2 producer boundary: it returns AO one-/overlap weights and
 * raw three-center/metric cotangents.  The metric pullback is delegated to the
 * shared fixed-rank symmetric inverse-square-root rule (#466).
 */
DensityFittedLagrangianWeights density_fitted_lagrangian_weights(
    const hf::PhysicalReference& reference, const posthf::DensityFittedBlockProvider& provider,
    const LagrangianWeights& weights, std::size_t maximum_bytes);

GradientResourcePlan conventional_gradient_plan(
    std::size_t orbitals, std::size_t occupied, std::size_t provider_bytes,
    const response::GmresPlan& response_plan, std::size_t maximum_shell_ao_count,
    std::size_t coordinate_count, std::size_t candidate_output_bytes, std::size_t budget_bytes,
    std::size_t derivative_backend_staging_bytes = 0);

DensityFittedGradientResourcePlan density_fitted_gradient_plan(
    std::size_t orbitals, std::size_t occupied, std::size_t auxiliaries, std::size_t provider_bytes,
    const response::GmresPlan& response_plan, std::size_t cartesian_orbitals,
    std::size_t cartesian_auxiliaries, std::size_t coordinate_count,
    std::size_t candidate_output_bytes, std::size_t budget_bytes);

}  // namespace generativeqc::mp2
