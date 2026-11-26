#pragma once

#include <algorithm>

#include "methods/method.hpp"
#include "methods/scf_common_options.hpp"
#include "molecule/basis.hpp"
#include "scf/hf_host_resources.hpp"

namespace generativeqc::api {

/** Size the ordinary CPU HF owner from its validated system, not a caller quote.
 * CPU HF reconstructs its Fock/integral plan inside execute; the serialized
 * workspace includes that full reconstruction and does not persist across calls.
 */
inline std::array<std::uint64_t, 3> cpu_direct_hf_host_inventory(
    core::Backend backend, const core::System& system,
    const generativeqc_method_descriptor& method) {
  if (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
      (method.method != GENERATIVEQC_METHOD_RHF && method.method != GENERATIVEQC_METHOD_UHF) ||
      methods::detail::checked_density_fitting_mode(method) != GENERATIVEQC_DENSITY_FITTING_NONE ||
      methods::detail::checked_precision_mode(method) != GENERATIVEQC_PRECISION_FP64 ||
      method.density_fitting_auxiliary_basis || method.ks_options ||
      (method.initial_guess && method.initial_guess->kind))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "host lease requires ordinary Direct FP64 CPU RHF or UHF");
  if (!system.ecp_terms.empty() ||
      std::ranges::any_of(system.atoms, [](const auto& atom) { return atom.ecp_core != 0; }))
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "Direct CPU HF host lease requires an all-electron system");

  std::uint64_t angular = 0;
  std::uint64_t primitives = 0;
  for (const auto& shell : system.shells) {
    angular = std::max<std::uint64_t>(angular, shell.angular_momentum);
    primitives = scf::hf_host_resource_detail::sum({primitives, shell.primitives.size()});
  }
  if (angular > 4)
    throw methods::MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                               "Direct CPU HF host lease requires a through-g orbital basis");

  const auto options = methods::detail::common_scf_options(method);
  scf::CpuHfHostFleetResources fleet;
  fleet.append(
      {{system.atoms.size(), molecule::ao_count(system), molecule::cartesian_ao_count(system),
        method.method == GENERATIVEQC_METHOD_UHF ? 2u : 1u, angular, options.diis_history, 0, 0, 0,
        0},
       system.shells.size(),
       primitives,
       0,
       0});
  return fleet.inventory();
}

}  // namespace generativeqc::api
