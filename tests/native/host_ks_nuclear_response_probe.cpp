#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "dft/host_ks_nuclear_hessian.hpp"
#include "dft/host_ks_nuclear_response.hpp"
#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

core::System system_for(const double* centers, unsigned spins, unsigned residual) {
  core::System system;
  const auto atoms = spins == 1 ? 2U : 3U;
  for (unsigned atom = 0; atom < atoms; ++atom) {
    system.atoms.push_back({atom == 0 && residual ? 11 : 1,
                            {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]},
                            atom == 0 && residual ? 10 : 0});
    for (double exponent : {1.2, .35}) system.shells.push_back({atom, 0, {{exponent, 1}}});
  }
  if (residual)
    for (int channel = -1; channel <= (residual == 2 ? 3 : 2); ++channel)
      for (unsigned power = 0; power <= 4; ++power) {
        if (residual == 1 && power != 2) continue;
        system.ecp_terms.push_back({0, channel, power, .67, -.012});
        system.ecp_terms.push_back({0, channel, power, 1.23, .0081});
      }
  system.multiplicity = spins;
  system.basis_representation = core::kSphericalBasis;
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
  return system;
}

/** A genuine native stationary frame; no oracle density/Fock seeds this owner. */
struct Handle {
  core::System system;
  dft::AoBasis basis;
  dft::MolecularGrid grid;
  scf::PreparedFockPlan fock;
  dft::VerifiedKsFinalState state;
  dft::KsFinalStateIdentity identity;
  std::unique_ptr<dft::HostGridGeometry> geometry;
  std::unique_ptr<dft::HostKsNuclearResponse> source;
  std::vector<double> atomic_weights;
  std::unique_ptr<dft::HostGridColumnGeometry> column_geometry;
  std::unique_ptr<dft::HostKsNuclearHessian> hessian;
  std::string method;
  std::size_t validations{}, fail_at{};
  bool stale{};
  int hessian_failure{};
  double* active_right{};

  Handle(const double* centers, unsigned spins, unsigned residual, std::size_t tile)
      : system(system_for(centers, spins, residual)),
        basis(system),
        grid(system),
        fock(system, nullptr,
             scf::resolve_fock_build(
                 scf::make_global_hybrid_fock_spec(
                     spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted, .2),
                 scf::FockBackend::Cpu, 0, 1e-10)) {
    const dft::host_hessian_aot::MethodBinding* binding{};
    const dft::host_hessian_aot::Program* motion{};
    for (const auto& entry : dft::host_hessian_aot::method_bindings())
      if (entry.spins == spins) binding = &entry;
    for (const auto& entry : dft::host_hessian_aot::programs())
      if (entry.name == "ao_first_motion") motion = &entry;
    if (!binding || !motion) throw std::invalid_argument("missing test method/motion closure");
    method = binding->identity;
    scf::ScfOptions options;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-11;
    options.screening_tolerance = 0;
    options.max_iterations = 200;
    options.compute_forces = false;
    options.retain_ks_state = true;
    options.xc_tile_points = tile;
    options.resolved_fock_build = fock.strategy();
    const auto scf = spins == 1 ? scf::run_b3lyp_rks(fock, basis, grid, options, .2)
                                : scf::run_b3lyp_uks(fock, basis, grid, options, .2);
    if (!scf.converged) throw std::runtime_error("nuclear test native SCF did not converge");
    const auto owner = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this));
    identity.determinant.factor = {owner, owner, 1, 1};
    identity.determinant.solve_epoch = 1;
    identity.determinant.model = fock.strategy();
    const auto electrons = static_cast<std::size_t>(system.electron_count);
    identity.determinant.occupied =
        spins == 1 ? std::vector<std::size_t>{electrons / 2}
                   : std::vector<std::size_t>{(electrons + 1) / 2, (electrons - 1) / 2};
    identity.model.grid = grid.spec();
    identity.model.tile_points = tile;
    identity.model.functional = binding->functional;
    identity.model.scf_domain_version = binding->scf_domain;
    identity.model.spins = spins;
    identity.model.device = -1;
    identity.model.owner = owner;
    dft::KsPhysicalState physical;
    physical.identity = identity;
    physical.physical = true;
    physical.components = scf.dft_diagnostic.components;
    physical.reported_energy = scf.energy;
    physical.physical_residual = scf.dft_diagnostic.physical_residual;
    dft::KsFinalStateCandidate candidate;
    candidate.identity = identity;
    candidate.physical_origin = true;
    candidate.fock_density_generation = identity.determinant.factor.density_generation;
    const auto matrix = basis.nao * basis.nao;
    for (unsigned spin = 0; spin < spins; ++spin) {
      physical.density.emplace_back(scf.density.begin() + spin * matrix,
                                    scf.density.begin() + (spin + 1) * matrix);
      physical.fock.emplace_back(scf.ks_physical_fock.begin() + spin * matrix,
                                 scf.ks_physical_fock.begin() + (spin + 1) * matrix);
      candidate.spins.push_back(scf::reference::generalized_eigen(
          physical.fock.back(), fock.overlap_orthogonalizer(), basis.nao));
    }
    scf::solver::FinalStateLimits final_limits;
    final_limits.density_tolerance = 1e-10;
    final_limits.energy_tolerance = 1e-11;
    final_limits.stationarity_tolerance = 1e-10;
    final_limits.require_canonicality = true;
    std::string detail;
    if (!dft::validate_ks_final_state(identity, fock.one_electron().overlap,
                                      fock.one_electron().hcore, physical, candidate, final_limits,
                                      true, state, detail))
      throw std::runtime_error("nuclear test stationary frame failed: " + detail);
    geometry = std::make_unique<dft::HostGridGeometry>(
        basis, grid, tile, dft::HostGridGeometry::plan(basis, grid, tile), motion->identity,
        [this] { validate(); });
    bind(0);
  }

  void validate() {
    ++validations;
    if (stale || (fail_at && validations >= fail_at))
      throw std::runtime_error("stale nuclear test owner");
    if (hessian_failure == 1 && hessian && hessian->work().xc.columns_completed) {
      hessian_failure = 0;
      active_right[0] += .125;
    }
    if (hessian_failure == 2 && hessian) {
      std::vector<double> direction(3 * system.atoms.size()), output(2 + 2 * direction.size());
      hessian->apply(direction, output, {});
    }
  }

  void bind(unsigned refusal, std::size_t panel_bytes = 0, std::size_t response_bytes = 0) {
    dft::HostKsNuclearLimits limits;
    limits.max_numeric_bytes = refusal == 1 ? 1 : 64U << 20;
    if (response_bytes) limits.max_numeric_bytes = response_bytes;
    limits.coordinate_source_bytes = panel_bytes;
    limits.perturbation_map_terms = refusal == 6 ? 1 : 1ULL << 34;
    limits.assembly_map_terms = refusal == 5 ? 1 : 1ULL << 34;
    limits.integrals.limits = dft::HostKsNuclearIntegrals::plan(basis, identity.model.spins);
    limits.xc = dft::HostGgaNuclearXc::plan(*geometry, identity.model.spins);
    limits.ecp = integrals::ecp_direction_plan(system).limits;
    if (refusal == 2) --limits.integrals.limits.eri_primitives;
    if (refusal == 3) --limits.xc.map_terms;
    if (refusal == 4) --limits.ecp.workspace_bytes;
    auto replacement = std::make_unique<dft::HostKsNuclearResponse>(
        state, system, fock, *geometry, method, limits, [this](const auto& current) {
          validate();
          if (current != identity) throw std::runtime_error("wrong nuclear test identity");
        });
    source = std::move(replacement);
  }

  void bind_hessian(bool bj, unsigned refusal, std::size_t batch_bytes = 0) {
    if (!column_geometry) {
      atomic_weights = grid.atomic_weights();
      column_geometry = std::make_unique<dft::HostGridColumnGeometry>(
          *geometry, atomic_weights, dft::HostGridColumnGeometry::plan(*geometry),
          [this] { validate(); });
    }
    dft::HostKsHessianLimits limits;
    limits.max_numeric_bytes = refusal == 1 ? 1 : 128U << 20;
    limits.max_assembly_numeric_bytes = dft::HostKsNuclearHessian::assembly_bytes(
        system.atoms.size(), basis.nao, identity.model.spins);
    limits.max_assembly_numeric_bytes += batch_bytes;
    limits.max_nuclear_pairs = system.atoms.size() * (system.atoms.size() - 1) / 2;
    limits.response.max_numeric_bytes = 64U << 20;
    limits.response.perturbation_map_terms = 1ULL << 34;
    limits.response.assembly_map_terms = 1ULL << 34;
    limits.response.integrals.limits =
        dft::HostKsNuclearIntegrals::plan(basis, identity.model.spins);
    limits.response.xc = dft::HostGgaNuclearXc::plan(*geometry, identity.model.spins);
    limits.response.ecp = integrals::ecp_direction_plan(system).limits;
    limits.integrals.limits = dft::HostKsNuclearIntegralCurvature::plan(
        basis, identity.model.spins, dft::HostNuclearIntegralCurvatureSchedule::Column);
    limits.xc = dft::HostGgaNuclearColumn::plan(*column_geometry, identity.model.spins);
    limits.ecp = integrals::ecp_curvature_plan(system).limits;
    const auto d3_plan = dft::dispersion::HostD3Column::plan(system.atoms.size());
    limits.dispersion = {d3_plan.numeric_bytes, d3_plan.pair_visits, d3_plan.reference_terms};
    limits.energy_absolute_tolerance = 2e-9;
    limits.energy_relative_tolerance = 2e-9;
    if (refusal == 2) --limits.max_assembly_numeric_bytes;
    if (refusal == 3) --limits.integrals.limits.eri_primitives;
    if (refusal == 4) --limits.xc.map_terms;
    if (refusal == 5) --limits.ecp.workspace_bytes;
    if (refusal == 6) --limits.max_nuclear_pairs;
    if (refusal == 7) --limits.dispersion.max_pair_visits;
    std::optional<dft::dispersion::D3Parameters> parameters;
    if (bj) parameters = dft::dispersion::D3Parameters{1, 1.9889, .3981, 4.4211, 0};
    auto next = std::make_unique<dft::HostKsNuclearHessian>(
        state, system, fock, *column_geometry, method, parameters, limits,
        [this](const auto& current) {
          validate();
          if (current != identity) throw std::runtime_error("wrong Hessian test identity");
        });
    hessian = std::move(next);
  }
};
}  // namespace

extern "C" const char* host_ks_nuclear_error() { return error.c_str(); }
extern "C" int host_ks_nuclear_point(const double* rho, const double* gradient, double* output) {
  try {
    double cartesian[2][3];
    for (unsigned spin = 0; spin < 2; ++spin) std::copy_n(gradient + 3 * spin, 3, cartesian[spin]);
    const auto value = dft::evaluate_b3lyp_point(rho, cartesian);
    output[0] = value.energy;
    std::copy_n(value.rho, 2, output + 1);
    for (unsigned spin = 0; spin < 2; ++spin)
      std::copy_n(value.gradient[spin], 3, output + 3 + 3 * spin);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void* host_ks_nuclear_create(const double* centers, unsigned spins, unsigned residual,
                                        std::size_t tile) {
  try {
    return new Handle(centers, spins, residual, tile);
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_ks_nuclear_destroy(Handle* handle) { delete handle; }
extern "C" void host_ks_nuclear_sizes(Handle* handle, std::size_t* output) {
  const auto plan = handle->source->resources({});
  const std::size_t values[]{
      handle->basis.nao,          handle->system.atoms.size(), handle->identity.model.spins,
      handle->grid.point_count(), plan.fixed_bound_bytes,      plan.gmres_workspace_bytes,
      plan.assembly_map_terms};
  std::copy(std::begin(values), std::end(values), output);
}
extern "C" void host_ks_nuclear_frame(Handle* handle, double* density, double* weights,
                                      double* points, double* grid_weights) {
  const auto frame = dft::electronic_reference(handle->state, handle->fock.one_electron().overlap,
                                               handle->fock.one_electron().hcore);
  for (std::size_t spin = 0; spin < frame.spin_channels; ++spin) {
    const auto& channel = frame.channels[spin];
    std::copy(channel.density.begin(), channel.density.end(),
              density + spin * channel.density.size());
    std::copy(channel.weighted_density.begin(), channel.weighted_density.end(),
              weights + spin * channel.weighted_density.size());
  }
  std::copy(handle->grid.points().begin(), handle->grid.points().end(), points);
  std::copy(handle->grid.weights().begin(), handle->grid.weights().end(), grid_weights);
}
extern "C" void host_ks_nuclear_failure(Handle* handle, std::size_t fail_at, int stale) {
  handle->fail_at = fail_at;
  handle->stale = stale;
}
extern "C" int host_ks_nuclear_bind(Handle* handle, unsigned refusal) {
  try {
    handle->bind(refusal);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_ks_nuclear_panel(Handle* handle, std::size_t panel_bytes,
                                     std::size_t response_bytes, unsigned prepare) {
  try {
    if (!prepare) {
      handle->bind(0, panel_bytes, response_bytes);
      return handle->source->admit_coordinate_sources({}) ? 1 : 0;
    }
    handle->source->prepare_coordinate_sources();
    return 1;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_ks_nuclear_source(Handle* handle, const double* direction,
                                      std::size_t direction_size, double* overlap, double* fock) {
  try {
    handle->validations = 0;
    const auto matrix = handle->basis.nao * handle->basis.nao;
    handle->source->source({direction, direction_size}, {overlap, matrix},
                           {fock, matrix * handle->identity.model.spins});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_ks_nuclear_cache(Handle* handle, std::size_t response_bytes,
                                     std::size_t* output) {
  try {
    handle->bind(0, 0, response_bytes);
    const auto added = handle->source->admit_response_cache({});
    const auto plan = handle->source->resources({});
    output[0] = added;
    output[1] = plan.fixed_bound_bytes;
    output[2] = plan.gmres_workspace_bytes;
    output[3] = plan.response_ao_cache_bytes;
    output[4] = plan.response_ao_prefix_points;
    return added ? 1 : 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_ks_nuclear_cache_work(Handle* handle, std::size_t* output) {
  const auto work = handle->source->work();
  output[0] = work.response_xc_point_visits;
  output[1] = work.response_xc_evaluated_points;
  output[2] = work.response_xc_cached_points;
  output[3] = work.induced_fock_actions;
  output[4] = work.response_ao_tile_reads;
  output[5] = work.response_ao_evaluated_points;
  output[6] = work.response_ao_cached_points;
}
extern "C" int host_ks_nuclear_solve(Handle* handle, const double* direction,
                                     std::size_t direction_size, double* density, double* weights,
                                     std::size_t iterations, std::size_t workspace,
                                     double* residual) {
  try {
    handle->validations = 0;
    response::GmresOptions options;
    options.relative_tolerance = 1e-11;
    options.absolute_tolerance = 1e-13;
    options.max_iterations = iterations;
    options.restart = 16;
    options.max_workspace_bytes = workspace;
    const auto matrices = handle->basis.nao * handle->basis.nao * handle->identity.model.spins;
    const auto result = handle->source->solve({direction, direction_size}, {density, matrices},
                                              {weights, matrices}, options);
    residual[0] = result.physical_residual_norm;
    residual[1] = result.relative_physical_residual;
    residual[2] = handle->validations;
    return result.converged() ? 0 : 1;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" std::size_t host_ks_nuclear_validations(Handle* handle) { return handle->validations; }

extern "C" int host_ks_nuclear_batch_plan(Handle* handle, std::size_t count, std::size_t* output) {
  try {
    const auto plan = handle->source->resources({}, count);
    output[0] = plan.fixed_bound_bytes;
    output[1] = plan.gmres_workspace_bytes;
    output[2] = plan.admitted;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_nuclear_batch_solve(Handle* handle, const double* directions,
                                           std::size_t count, double* density, double* weights,
                                           std::size_t iterations, double* residuals,
                                           std::size_t* output_work) {
  try {
    if (count > 4) throw std::invalid_argument("probe batch is too wide");
    handle->validations = 0;
    response::GmresOptions options;
    options.relative_tolerance = 0;
    options.absolute_tolerance = 1e-12;
    options.max_iterations = iterations;
    const auto dimension = 3 * handle->system.atoms.size();
    const auto matrices = handle->basis.nao * handle->basis.nao * handle->identity.model.spins;
    std::array<dft::HostKsNuclearRequest, 4> requests;
    for (std::size_t index = 0; index < count; ++index)
      requests[index] = {{directions + index * dimension, dimension},
                         {density + index * matrices, matrices},
                         {weights + index * matrices, matrices}};
    const auto before = handle->source->work();
    const auto result = handle->source->solve_batch({requests.data(), count}, options);
    const auto after = handle->source->work();
    bool converged = true;
    for (std::size_t index = 0; index < count; ++index) {
      residuals[index] = result.responses[index].physical_residual_norm;
      output_work[index] = static_cast<std::size_t>(result.responses[index].status);
      converged &= result.responses[index].converged();
    }
    output_work[count] = after.source_attempts - before.source_attempts;
    output_work[count + 1] = after.solves_completed - before.solves_completed;
    output_work[count + 2] = after.induced_fock_actions - before.induced_fock_actions;
    output_work[count + 3] = after.response_ao_tile_reads - before.response_ao_tile_reads;
    return converged ? 0 : 1;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" double* host_ks_nuclear_borrowed(Handle* handle, unsigned owner) {
  if (owner == 1) return handle->basis.packed.data();
  if (owner == 2) return const_cast<double*>(handle->grid.points().data());
  if (owner == 3) return handle->state.weighted_density[0].data();
  if (owner == 4) return handle->state.orbitals[0].vectors.data();
  if (owner == 5) return handle->state.orbitals[0].values.data();
  if (owner == 6) return &handle->state.components.one_electron;
  const auto frame = dft::electronic_reference(handle->state, handle->fock.one_electron().overlap,
                                               handle->fock.one_electron().hcore);
  return const_cast<double*>(frame.channels[0].density.data());
}

extern "C" int host_ks_hessian_bind(Handle* handle, int bj, unsigned refusal) {
  try {
    handle->bind_hessian(bj, refusal);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_hessian_batch_bind(Handle* handle, int bj, std::size_t extra,
                                          std::size_t maximum) {
  try {
    handle->bind_hessian(bj, 0, extra);
    const auto width = handle->hessian->admit_response_batch({}, maximum);
    handle->hessian->admit_response_cache({});
    return static_cast<int>(width);
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_hessian_batch_apply(Handle* handle, std::size_t first, std::size_t count,
                                           double* output, std::size_t iterations,
                                           double* residuals, int failure) {
  try {
    handle->validations = 0;
    handle->hessian_failure = failure;
    response::GmresOptions options;
    options.relative_tolerance = 0;
    options.absolute_tolerance = 1e-12;
    options.max_iterations = iterations;
    const auto results = handle->hessian->apply_coordinates(first, count, options);
    bool converged = true;
    for (std::size_t index = 0; index < count; ++index) {
      residuals[index] = results[index].response.physical_residual_norm;
      converged &= results[index].converged();
    }
    if (!converged) return 1;
    const auto fields = 2 + 6 * handle->system.atoms.size();
    for (std::size_t index = 0; index < count; ++index) {
      const auto column = handle->hessian->coordinate_column(index);
      std::copy(column.begin(), column.end(), output + index * fields);
    }
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_ks_hessian_apply(Handle* handle, const double* right, std::size_t right_size,
                                     double* output, std::size_t output_size,
                                     std::size_t iterations, std::size_t workspace,
                                     double* diagnostics, int failure) {
  try {
    handle->validations = 0;
    handle->active_right = const_cast<double*>(right);
    handle->hessian_failure = failure;
    response::GmresOptions options;
    options.relative_tolerance = 1e-11;
    options.absolute_tolerance = 1e-13;
    options.max_iterations = iterations;
    options.restart = 16;
    options.max_workspace_bytes = workspace;
    const auto result = handle->hessian->apply({right, right_size}, {output, output_size}, options);
    diagnostics[0] = result.response.physical_residual_norm;
    diagnostics[1] = result.response.relative_physical_residual;
    diagnostics[2] = result.reconstructed_energy;
    diagnostics[3] = result.verified_energy;
    diagnostics[4] = result.energy_error;
    return result.converged() ? 0 : 1;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_ks_hessian_resources(Handle* handle, std::size_t* output) {
  const auto plan = handle->hessian->resources({});
  const auto work = handle->hessian->work();
  const std::size_t values[]{plan.assembly_bytes,
                             plan.reserved_numeric_bytes,
                             plan.fixed_bound_bytes,
                             plan.gmres_bytes,
                             work.columns_started,
                             work.columns_completed,
                             work.nuclear_pairs,
                             work.response.solve_attempts,
                             work.response.solves_completed,
                             work.integrals.eri_primitives,
                             work.xc.point_visits,
                             work.dispersion.energy_pairs};
  std::copy(std::begin(values), std::end(values), output);
}
extern "C" int host_ks_hessian_oracle_grid(const double* centers, unsigned spins, unsigned residual,
                                           double* points, double* weights, std::size_t count) {
  try {
    const auto system = system_for(centers, spins, residual);
    const dft::MolecularGrid grid(system);
    if (count != grid.point_count())
      throw std::invalid_argument("oracle primal grid shape mismatch");
    std::copy(grid.points().begin(), grid.points().end(), points);
    std::copy(grid.weights().begin(), grid.weights().end(), weights);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

#ifdef GENERATIVEQC_NUCLEAR_RESPONSE_STANDALONE
int main() {
  try {
    const double centers[]{0, 0, 0, .15, -.08, 1.6, 1.8, .25, .3};
    const double motion[]{.12, -.09, .07, -.03, .06, -.1, .04, -.02, .08};
    for (unsigned spins : {1U, 2U})
      for (unsigned residual : {0U, 1U, 2U}) {
        Handle owner(centers, spins, residual, 256);
        const auto count = spins * owner.basis.nao * owner.basis.nao;
        std::vector<double> density(count), weights(count);
        response::GmresOptions options;
        options.relative_tolerance = 1e-11;
        options.absolute_tolerance = 1e-13;
        options.max_workspace_bytes = 16U << 20;
        const auto result =
            owner.source->solve({motion, 3 * owner.system.atoms.size()}, density, weights, options);
        if (!result.converged() || result.physical_residual_norm > 2e-11)
          throw std::runtime_error("standalone nuclear response did not converge");
        for (const auto& panel : {density, weights})
          if (!std::all_of(panel.begin(), panel.end(),
                           [](double value) { return std::isfinite(value); }))
            throw std::runtime_error("standalone nuclear response is nonfinite");
      }
    std::cout << "native composed nuclear RKS/UKS, empty/analytic/grid ECP, no Python owner\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
#endif

#ifdef GENERATIVEQC_NUCLEAR_HESSIAN_STANDALONE
int main() {
  try {
    const double centers[]{0, 0, 0, .15, -.08, 1.6, 1.8, .25, .3};
    const double motion[]{.12, -.09, .07, -.03, .06, -.1, .04, -.02, .08};
    for (unsigned spins : {1U, 2U}) {
      Handle owner(centers, spins, 1, 256);
      owner.bind_hessian(true, 0);
      const auto dimension = 3 * owner.system.atoms.size();
      std::vector<double> output(2 + 2 * dimension);
      response::GmresOptions options;
      options.relative_tolerance = 1e-11;
      options.absolute_tolerance = 1e-13;
      options.max_workspace_bytes = 16U << 20;
      const auto result = owner.hessian->apply({motion, dimension}, output, options);
      if (!result.converged() || result.energy_error > 2e-9 ||
          !std::all_of(output.begin(), output.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("native stationary Hessian column failed qualification");
    }
    std::cout << "native stationary RKS/UKS columns: coupled D/full W, residual ECP, ionic nuclei, "
                 "D3(BJ)\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
#endif
