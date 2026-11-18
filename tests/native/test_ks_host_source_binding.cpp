#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dft/ao_grid.hpp"
#include "methods/dft_method.hpp"
#include "methods/named_ks_catalog.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"

namespace {
using namespace generativeqc;
using methods::detail::KsHostDerivativeSources;

void require(bool condition, const std::string& detail) {
  if (!condition) throw std::runtime_error(detail);
}

bool empty(const KsHostDerivativeSources& sources) {
  return !sources.system && !sources.fock && !sources.basis && !sources.grid && !sources.dispersion;
}

core::System molecule(unsigned spins) {
  core::System system;
  system.multiplicity = spins;
  system.basis_representation = core::kSphericalBasis;
  for (unsigned atom = 0; atom < spins + 1; ++atom) {
    system.atoms.push_back({1, {0.17 * atom * atom, -0.08 * atom, 1.6 * atom}});
    for (const auto exponent : {1.2, 0.35}) system.shells.push_back({atom, 0, {{exponent, 1}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  return system;
}

void binding_case(unsigned spins, bool dispersion) {
  const auto name =
      std::string(dispersion ? "pbe0-d3(bj)-" : "b3lyp-") + (spins == 1 ? "rks" : "uks");
  const auto* named = methods::named_ks::find_method(name);
  require(named, "test requires the canonical native method catalog");
  generativeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = GENERATIVEQC_ABI_VERSION;
  options.grid_version = 1;
  options.radial_points = 12;
  options.angular_polar = 4;
  options.angular_azimuth = 8;
  options.partition_iterations = 3;
  options.coincident_tolerance = 1e-12;
  options.tile_points = 64;
  methods::named_ks::electronic_options(*named, options);
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method = named->carrier;
  descriptor.max_iterations = 200;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = 1e-12;
  descriptor.density_tolerance = 1e-10;
  descriptor.screening_tolerance = 1e-14;
  descriptor.ks_options = &options;
  core::ContextState context;
  context.device_id = -1;
  const auto* capabilities = methods::find_capabilities(descriptor.method);
  require(capabilities, "native DFT carrier is unavailable");
  auto systems = std::vector<core::System>{molecule(spins), molecule(spins)};
  auto batch =
      dispersion
          ? methods::detail::prepare_composed_dft_batch(
                *capabilities, context, std::move(systems), descriptor,
                GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, nullptr, named->dispersion, 16U << 20)
          : methods::detail::prepare_dft_batch(*capabilities, context, std::move(systems),
                                               descriptor, GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  std::string detail;
  dft::CudaKsFinalStateToken token;
  KsHostDerivativeSources sources;
  require(methods::detail::borrow_dft_host_derivative_sources(*batch, 0, token, sources, detail) !=
                  GENERATIVEQC_STATUS_SUCCESS &&
              empty(sources),
          "unexecuted owner published CPU derivative sources");
  const auto result = batch->execute({}, false);
  require(result.size() == 2, "native batch dropped an input");
  for (const auto& item : result)
    require(item.status == GENERATIVEQC_STATUS_SUCCESS && item.calculation.convergence.converged,
            "native source-binding fixture failed actual SCF");
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(methods::detail::borrow_dft_host_derivative_sources(*batch, 0, token, sources, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(sources.system && sources.fock && sources.basis && sources.grid && sources.dispersion,
          "native source binding is incomplete");
  const core::System* integral_system{};
  require(methods::detail::borrow_dft_integral_system(*batch, 0, token, integral_system, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              integral_system == sources.system,
          "native Hessian and integral source bindings have different owners");
  dft::VerifiedKsFinalState frame;
  require(methods::detail::read_dft_final_state(*batch, 0, token, true, frame, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  const auto reference = dft::electronic_reference(frame, sources.fock->one_electron().overlap,
                                                   sources.fock->one_electron().hcore);
  require(reference.basis_functions == sources.basis->nao && reference.spin_channels == spins &&
              sources.grid->point_count() > 0 &&
              sources.grid->owners().size() == sources.grid->point_count(),
          "borrowed basis/grid/Fock do not describe the verified native frame");
  require(sources.dispersion->kind == named->dispersion.kind &&
              sources.dispersion->s8 == named->dispersion.s8 &&
              sources.dispersion->s9 == named->dispersion.s9 &&
              sources.dispersion->a2 == named->dispersion.a2,
          "native correction identity was lost or inferred from the electronic token");
  const auto stable = sources;
  auto forged = token;
  ++forged.identity.model.owner;
  for (const auto index : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
    sources = stable;
    require(methods::detail::borrow_dft_host_derivative_sources(
                *batch, index, index == 0 ? forged : token, sources, detail) !=
                    GENERATIVEQC_STATUS_SUCCESS &&
                empty(sources),
            "wrong-owner/index source refusal retained a partial binding");
  }
  require(methods::detail::borrow_dft_host_derivative_sources(*batch, 0, token, sources, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              sources.system == stable.system && sources.fock == stable.fock &&
              sources.basis == stable.basis && sources.grid == stable.grid,
          "source refusal corrupted recovery or reconstructed borrowed owners");
  const auto replay = batch->execute({}, false);
  require(replay[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              replay[0].calculation.convergence.converged,
          "native warm replay failed");
  require(methods::detail::borrow_dft_host_derivative_sources(*batch, 0, token, sources, detail) !=
                  GENERATIVEQC_STATUS_SUCCESS &&
              empty(sources),
          "replay did not revoke the old source binding token");
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::borrow_dft_host_derivative_sources(
                  *batch, 0, token, sources, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "current native epoch failed source binding after replay");
  const auto previous = token;
  std::vector<double> positions;
  for (const auto& atom : sources.system->atoms)
    positions.insert(positions.end(), atom.position.begin(), atom.position.end());
  positions[3] += 0.07;
  methods::Coordinates moved(2);
  moved[0] = positions;
  const auto displaced = batch->execute(moved, false);
  require(displaced[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              displaced[0].calculation.convergence.converged,
          "native geometry update failed actual SCF");
  require(methods::detail::borrow_dft_host_derivative_sources(
              *batch, 0, previous, sources, detail) != GENERATIVEQC_STATUS_SUCCESS &&
              empty(sources),
          "displaced geometry accepted an old source token");
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              methods::detail::borrow_dft_host_derivative_sources(
                  *batch, 0, token, sources, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              sources.system->atoms[1].position[0] == positions[3] &&
              sources.basis->packed[3] == positions[3] &&
              sources.basis->nao == sources.fock->one_electron().nbf,
          "geometry rebuild did not rebind the actual matching native sources");
  batch->invalidate_result();
  require(methods::detail::borrow_dft_host_derivative_sources(*batch, 0, token, sources, detail) !=
                  GENERATIVEQC_STATUS_SUCCESS &&
              empty(sources),
          "explicit final-state revocation retained derivative eligibility");
}
}  // namespace

int main() {
  try {
    for (const auto spins : {1U, 2U})
      for (const auto dispersion : {false, true}) binding_case(spins, dispersion);
    std::cout << "native CPU source ownership, correction identity and token revocation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
