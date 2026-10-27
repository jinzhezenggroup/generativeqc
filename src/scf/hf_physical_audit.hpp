#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/density_fitting.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/hf_host_resources.hpp"
#include "scf/reference/mean_field.hpp"

namespace generativeqc::scf::physical_audit {

/** An unsupported metadata domain is different from a failed admitted audit. */
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/** Exact HF model controls, with a conservative total numeric host allowance.
 * DIIS history is used only to preserve the incumbent full-HF admission bound;
 * this operation never iterates, extrapolates a Fock, or modifies the density.
 */
struct Controls {
  bool unrestricted{}, fitted{};
  std::uint32_t diis_history{8};
  std::uint64_t maximum_host_bytes{};
  double screening_tolerance{1e-12}, metric_relative_threshold{1e-10};
};

struct Capacity {
  std::uint64_t ao_count{}, spin_count{}, density_count{}, overlap_count{};
  std::uint64_t provider_bytes{}, workspace_bytes{}, required_host_bytes{};
};

/** Detached physical observations; no prepared provider survives publication. */
struct Result {
  std::vector<double> overlap;
  double energy{}, energy_difference{}, maximum_commutator{}, rms_commutator{};
  std::uint64_t fock_builds{1}, auxiliary_rank{};
};

/** Query the existing HF fleet envelope before any integral/provider work.
 * The additional workspace preserves the prior audit's allowance for density,
 * Fock, overlap, products and immutable/hash publication copies. It bounds
 * explicit numeric/value metadata, not allocator overhead or process RSS.
 */
inline Capacity capacity(const core::System& orbital, const core::System* auxiliary,
                         const Controls& controls) {
  using hf_host_resource_detail::product;
  using hf_host_resource_detail::sum;
  if (!controls.maximum_host_bytes || !std::isfinite(controls.screening_tolerance) ||
      controls.screening_tolerance < 0 || !std::isfinite(controls.metric_relative_threshold) ||
      controls.metric_relative_threshold <= 0 || controls.metric_relative_threshold >= 1)
    throw std::invalid_argument("invalid physical HF audit controls");
  const auto* fitting = controls.fitted ? (auxiliary ? auxiliary : &orbital) : nullptr;
  if (!orbital.ecp_terms.empty() || (fitting && !fitting->ecp_terms.empty()))
    throw Unsupported("native physical HF audit requires all-electron CPU inputs");
  auto angular = [](const core::System& system) {
    unsigned result = 0;
    for (const auto& shell : system.shells) result = std::max(result, shell.angular_momentum);
    return result;
  };
  auto primitives = [](const core::System& system) {
    std::uint64_t count = 0;
    for (const auto& shell : system.shells)
      count = hf_host_resource_detail::sum({count, shell.primitives.size()});
    return count;
  };
  const auto n = molecule::ao_count(orbital);
  const std::uint64_t spins = controls.unrestricted ? 2 : 1;
  const auto l = angular(orbital), auxiliary_l = fitting ? angular(*fitting) : 0;
  if (l > 4 || auxiliary_l > 4)
    throw Unsupported("native physical HF audit requires through-g CPU topology");
  CpuHfHostFleetResources fleet;
  fleet.append({{orbital.atoms.size(), n, molecule::cartesian_ao_count(orbital), spins, l,
                 controls.diis_history, fitting ? molecule::ao_count(*fitting) : 0,
                 fitting ? molecule::cartesian_ao_count(*fitting) : 0, auxiliary_l, 0},
                orbital.shells.size(),
                primitives(orbital),
                fitting ? fitting->shells.size() : 0,
                fitting ? primitives(*fitting) : 0});
  const auto matrix = product({n, n});
  const auto workspace = product({8, matrix, sum({product({12, spins}), 8})});
  const auto provider = fleet.inventory()[2];
  return {n,        spins,     product({matrix, spins}),  matrix,
          provider, workspace, sum({provider, workspace})};
}

/** Rebuild one physical F[D] and use the existing HF energy/commutator owners.
 * Admission precedes input copies and provider construction. All intermediates
 * are local; a failure exposes no partial result and cannot trigger another SCF.
 */
inline Result evaluate(const core::System& orbital, const core::System* auxiliary,
                       const Controls& controls, std::span<const double> density,
                       double target_energy) {
  const auto required = capacity(orbital, auxiliary, controls);
  if (density.size() != required.density_count || !std::isfinite(target_energy) ||
      !std::all_of(density.begin(), density.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("invalid physical HF audit density or target energy");
  if (required.required_host_bytes > controls.maximum_host_bytes) throw std::bad_alloc();
  auto spec = make_hf_fock_spec(
      controls.unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
      controls.fitted ? FockApproximation::DensityFitted : FockApproximation::Exact);
  spec.derivative_order = 0;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cpu, controls.screening_tolerance,
                                           controls.metric_relative_threshold);
  PreparedFockPlan plan(orbital, auxiliary, strategy);
  const auto n = required.ao_count, matrix = required.overlap_count;
  const auto& ints = plan.one_electron();
  const std::vector<double> alpha(density.begin(), density.begin() + matrix);
  const std::vector<double> beta =
      controls.unrestricted ? std::vector<double>(density.begin() + matrix, density.end())
                            : std::vector<double>{};
  const auto jk = plan.build(alpha, beta);
  const auto fock = assemble_fock(strategy, ints.hcore, jk);
  // Share the established physical energy expressions with the ordinary HF
  // owner. The caller's density is never normalized or silently symmetrized.
  const auto electronic =
      controls.unrestricted
          ? reference::uhf_electronic_energy(alpha, beta, ints.hcore, fock.alpha, fock.beta)
          : reference::electronic_energy(alpha, ints.hcore, fock.alpha);
  const auto residual_alpha = reference::commutator_residual(fock.alpha, alpha, ints.overlap, n);
  auto residual = residual_alpha;
  if (controls.unrestricted) {
    const auto residual_beta = reference::commutator_residual(fock.beta, beta, ints.overlap, n);
    residual = reference::concatenate(residual_alpha, residual_beta);
  }
  Result result;
  result.energy = electronic + ints.nuclear_repulsion;
  result.energy_difference = std::abs(result.energy - target_energy);
  result.maximum_commutator = reference::residual_max_abs(residual);
  result.rms_commutator = reference::residual_rms(residual);
  if (!std::isfinite(result.energy) || !std::isfinite(result.energy_difference) ||
      !std::isfinite(result.maximum_commutator) || !std::isfinite(result.rms_commutator))
    throw std::runtime_error("nonfinite physical HF audit result");
  result.overlap = ints.overlap;
  // The CPU fitted source owns the metric policy. Its diagnostics are reported
  // separately from physical-model identity and never select a new approximation.
  if (const auto* fitted = plan.cpu_fitted_data())
    result.auxiliary_rank = fitted->three_center.effective_rank;
  return result;
}

}  // namespace generativeqc::scf::physical_audit
