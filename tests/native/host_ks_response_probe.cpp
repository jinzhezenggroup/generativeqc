#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/host_ks_perturbation.hpp"
#include "dft/host_ks_response.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

core::System hydrogen_system(unsigned spins) {
  core::System system;
  system.atoms = {{1, {0, 0, 0}}, {1, {.15, -.08, 1.6}}};
  if (spins == 2) system.atoms.push_back({1, {1.8, .25, .3}});
  system.multiplicity = spins;
  system.basis_representation = core::kSphericalBasis;
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    for (double exponent : {1.2, .35})
      system.shells.push_back({static_cast<std::uint32_t>(atom), 0, {{exponent, 1}}});
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return system;
}

/** Test-only owner: ordinary cold native SCF, then canonical validation of its
 * retained physical F[D]. No oracle frame or density enters native execution.
 */
struct Handle {
  core::System system;
  dft::AoBasis basis;
  dft::MolecularGrid grid;
  scf::PreparedFockPlan fock;
  scf::ScfResult scf;
  dft::VerifiedKsFinalState state;
  dft::KsFinalStateIdentity current;
  std::unique_ptr<dft::HostKsResponse> response;
  std::unique_ptr<dft::HostKsPerturbation> perturbation;
  std::vector<double> retired_coefficients;
  int failure{};
  bool alive{true};

  Handle(unsigned spins, std::size_t tile)
      : system(hydrogen_system(spins)),
        basis(system),
        grid(system),
        fock(system, nullptr,
             scf::resolve_fock_build(
                 scf::make_global_hybrid_fock_spec(
                     spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted, .2),
                 scf::FockBackend::Cpu, 0, 1e-10)) {
    scf::ScfOptions options;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-11;
    options.screening_tolerance = 0;
    options.max_iterations = 150;
    options.compute_forces = false;
    options.retain_ks_state = true;
    options.xc_tile_points = tile;
    options.resolved_fock_build = fock.strategy();
    scf = spins == 1 ? scf::run_b3lyp_rks(fock, basis, grid, options, .2)
                     : scf::run_b3lyp_uks(fock, basis, grid, options, .2);
    if (!scf.converged) throw std::runtime_error("native qualification SCF did not converge");
    const dft::host_hessian_aot::MethodBinding* binding = nullptr;
    for (const auto& candidate : dft::host_hessian_aot::method_bindings())
      if (candidate.spins == spins) binding = &candidate;
    if (!binding) throw std::runtime_error("qualification method binding is absent");
    const auto owner = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(this));
    current.determinant.factor = {owner, owner, 1, 1};
    current.determinant.solve_epoch = 1;
    current.determinant.model = fock.strategy();
    const auto electrons = static_cast<std::size_t>(system.electron_count);
    current.determinant.occupied =
        spins == 1 ? std::vector<std::size_t>{electrons / 2}
                   : std::vector<std::size_t>{(electrons + 1) / 2, (electrons - 1) / 2};
    current.model.grid = grid.spec();
    current.model.tile_points = tile;
    current.model.functional = binding->functional;
    current.model.scf_domain_version = binding->scf_domain;
    current.model.spins = spins;
    current.model.device = -1;
    current.model.owner = owner;
    dft::KsPhysicalState physical;
    physical.identity = current;
    physical.physical = true;
    physical.components = scf.dft_diagnostic.components;
    physical.reported_energy = scf.energy;
    physical.physical_residual = scf.dft_diagnostic.physical_residual;
    dft::KsFinalStateCandidate candidate;
    candidate.identity = current;
    candidate.physical_origin = true;
    candidate.fock_density_generation = current.determinant.factor.density_generation;
    const auto matrix = basis.nao * basis.nao;
    const auto orthogonalizer = fock.overlap_orthogonalizer();
    for (unsigned spin = 0; spin < spins; ++spin) {
      physical.density.emplace_back(scf.density.begin() + spin * matrix,
                                    scf.density.begin() + (spin + 1) * matrix);
      physical.fock.emplace_back(scf.ks_physical_fock.begin() + spin * matrix,
                                 scf.ks_physical_fock.begin() + (spin + 1) * matrix);
      candidate.spins.push_back(
          scf::reference::generalized_eigen(physical.fock.back(), orthogonalizer, basis.nao));
    }
    scf::solver::FinalStateLimits limits;
    limits.density_tolerance = 1e-10;
    limits.energy_tolerance = 1e-11;
    limits.stationarity_tolerance = 1e-10;
    limits.require_canonicality = true;
    std::string detail;
    if (!dft::validate_ks_final_state(current, fock.one_electron().overlap,
                                      fock.one_electron().hcore, physical, candidate, limits, true,
                                      state, detail))
      throw std::runtime_error("native stationary qualification failed: " + detail);
  }

  void bind(std::size_t budget, std::string_view method, int mutation) {
    auto shape = dft::HostGgaResponseShape{basis.nao, grid.point_count(), current.model.spins,
                                           current.model.tile_points};
    if (mutation == 1) ++state.identity.model.functional;
    if (mutation == 2) ++state.identity.model.scf_domain_version;
    if (mutation == 3) state.identity.model.semilocal_exchange_scale = .9;
    if (mutation == 4) state.identity.model.device = 0;
    if (mutation == 5) state.stationarity_tolerance.reset();
    if (mutation == 6) --shape.points;
    if (mutation == 7) ++shape.tile_points;
    if (mutation == 8) {
      state.orbitals[0].values.back() = state.orbitals[0].values.front();
    }
    response = std::make_unique<dft::HostKsResponse>(
        state, system, fock, shape, method,
        [this](std::size_t begin, std::size_t count, std::span<double> ao,
               std::span<double> weights) {
          if (failure == 1) throw std::runtime_error("injected native grid failure");
          if (failure == 4) {
            std::vector<double> vector(response->dimension());
            response->apply(vector, vector);
          }
          if (failure == 8) {
            std::vector<double> overlap(basis.nao * basis.nao);
            std::vector<double> matrices(current.model.spins * overlap.size());
            perturbation->solve(overlap, matrices, matrices, matrices, {});
          }
          basis.evaluate(grid.points().data() + 3 * begin, count, 1, 0, basis.nao, ao.data(),
                         ao.size());
          std::copy_n(grid.weights().begin() + begin, count, weights.begin());
          if (failure == 2) alive = false;
          if (failure == 3 && begin + count == grid.point_count())
            ao.back() = std::numeric_limits<double>::quiet_NaN();
        },
        [this](const dft::KsFinalStateIdentity& expected) {
          if (!alive || current != expected)
            throw std::runtime_error("native stationary owner or solve epoch expired");
        },
        budget);
  }
};
}  // namespace

extern "C" const char* host_ks_error() { return error.c_str(); }
extern "C" void* host_ks_create(unsigned spins, std::size_t tile, std::size_t budget,
                                const char* method, int mutation) {
  try {
    auto handle = std::make_unique<Handle>(spins, tile);
    handle->bind(budget, method, mutation);
    return handle.release();
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_ks_destroy(Handle* handle) { delete handle; }
extern "C" int host_ks_rebind(Handle* handle, std::size_t budget, const char* method) {
  try {
    handle->perturbation.reset();
    handle->response.reset();
    handle->bind(budget, method, 0);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

/** Export the exact native frame and grid so the independent test oracle uses
 * identical quadrature, normalization and orbital gauge, never a new SCF seed.
 */
extern "C" void host_ks_sizes(Handle* handle, std::size_t* sizes) {
  sizes[0] = handle->basis.nao;
  sizes[1] = handle->grid.point_count();
  sizes[2] = handle->current.model.spins;
  sizes[3] = handle->response->dimension();
  sizes[4] = handle->current.determinant.occupied[0];
  sizes[5] = sizes[2] == 2 ? handle->current.determinant.occupied[1] : 0;
}
extern "C" void host_ks_frame(Handle* handle, double* density, double* coefficients,
                              double* energies, double* overlap, double* hcore, double* points,
                              double* weights, double* summary) {
  const auto matrix = handle->basis.nao * handle->basis.nao;
  for (std::size_t spin = 0; spin < handle->current.model.spins; ++spin) {
    std::copy(handle->state.density[spin].begin(), handle->state.density[spin].end(),
              density + spin * matrix);
    std::copy(handle->state.orbitals[spin].vectors.begin(),
              handle->state.orbitals[spin].vectors.end(), coefficients + spin * matrix);
    std::copy(handle->state.orbitals[spin].values.begin(),
              handle->state.orbitals[spin].values.end(), energies + spin * handle->basis.nao);
  }
  const auto& integrals = handle->fock.one_electron();
  std::copy(integrals.overlap.begin(), integrals.overlap.end(), overlap);
  std::copy(integrals.hcore.begin(), integrals.hcore.end(), hcore);
  std::copy(handle->grid.points().begin(), handle->grid.points().end(), points);
  std::copy(handle->grid.weights().begin(), handle->grid.weights().end(), weights);
  summary[0] = handle->scf.energy;
  summary[1] = handle->scf.dft_diagnostic.physical_residual;
  summary[2] = handle->scf.iterations;
  summary[3] = handle->scf.fock_builds;
}
extern "C" void host_ks_failure(Handle* handle, int failure) {
  if (!handle->retired_coefficients.empty()) {
    handle->state.orbitals[0].vectors.swap(handle->retired_coefficients);
    handle->retired_coefficients.clear();
  }
  if (failure == 7) {
    auto replacement = handle->state.orbitals[0].vectors;
    replacement.swap(handle->state.orbitals[0].vectors);
    // Retain the old allocation until recovery so rejection tests the changed
    // frame identity, not undefined access through a dangling borrowed span.
    handle->retired_coefficients = std::move(replacement);
  }
  handle->failure = failure;
  handle->alive = failure != 5;
  handle->current.determinant.solve_epoch = failure == 6 ? 2 : 1;
}
extern "C" int host_ks_apply(Handle* handle, const double* input, std::size_t count, double* output,
                             std::size_t output_count) {
  try {
    handle->response->apply({input, count}, {output, output_count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_ks_alias_frame(Handle* handle, const double* input) {
  return host_ks_apply(handle, input, handle->response->dimension(),
                       handle->state.orbitals[0].vectors.data(), handle->response->dimension());
}
extern "C" int host_ks_solve(Handle* handle, const double* rhs, double* solution,
                             std::size_t iterations, std::size_t workspace, double* diagnostic,
                             std::size_t* work) {
  try {
    response::GmresOptions options;
    options.relative_tolerance = 2e-11;
    options.absolute_tolerance = 1e-13;
    options.restart = 20;
    options.max_iterations = iterations;
    options.max_workspace_bytes = workspace;
    const auto resources = handle->response->resources(options);
    const auto result = handle->response->solve({rhs, handle->response->dimension()},
                                                {solution, handle->response->dimension()}, options);
    diagnostic[0] = result.residual_norm;
    diagnostic[1] = result.relative_residual;
    work[0] = result.iterations;
    work[1] = result.operator_actions;
    work[2] = result.preconditioner_actions;
    work[3] = result.gmres_workspace_bytes;
    work[4] = result.measured_gmres_peak_bytes;
    work[5] = resources.persistent_bytes;
    work[6] = resources.action_array_bound_bytes;
    const auto xc = handle->response->xc_work();
    work[7] = xc.actions_started;
    work[8] = xc.actions_completed;
    work[9] = xc.point_visits;
    work[10] = xc.tile_reads;
    work[11] = handle->response->fock_actions();
    return static_cast<int>(result.status);
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

/** Test the public density-action contract independently of the orbital map. */
extern "C" int host_ks_density(Handle* handle, const double* input, double* output) {
  try {
    const auto count = handle->basis.nao * handle->basis.nao * handle->current.model.spins;
    handle->response->apply_density({input, count}, {output, count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_batch_resources(Handle* handle, std::size_t count, std::size_t* sizes) {
  try {
    const auto resource = handle->response->resources({}, count);
    sizes[0] = resource.persistent_bytes;
    sizes[1] = resource.action_array_bound_bytes;
    sizes[2] = resource.gmres_workspace_bytes;
    sizes[3] = resource.batch_workspace_bytes;
    sizes[4] = resource.admitted;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_batch_apply(Handle* handle, const double* input, double* output,
                                   std::size_t count, double* workspace, std::size_t elements) {
  try {
    if (count > 4) throw std::invalid_argument("probe batch is too wide");
    const auto dimension = handle->response->dimension();
    std::array<response::LinearAction, 4> actions;
    for (std::size_t index = 0; index < count; ++index)
      actions[index] = {{input + index * dimension, dimension},
                        {output + index * dimension, dimension}};
    handle->response->apply_batch({actions.data(), count}, {workspace, elements});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_batch_solve(Handle* handle, const double* rhs, double* solution,
                                   std::size_t count, std::size_t iterations, double* residuals,
                                   std::size_t* work) {
  try {
    if (count > 4) throw std::invalid_argument("probe batch is too wide");
    const auto dimension = handle->response->dimension();
    response::GmresOptions options;
    options.relative_tolerance = 0;
    options.absolute_tolerance = 1e-12;
    options.max_iterations = iterations;
    const auto before = handle->response->xc_work();
    const auto before_fock = handle->response->fock_actions();
    std::array<dft::HostKsResponseRequest, 4> requests;
    for (std::size_t index = 0; index < count; ++index)
      requests[index] = {{rhs + index * dimension, dimension},
                         {solution + index * dimension, dimension}};
    const auto results = handle->response->solve_batch({requests.data(), count}, options);
    for (std::size_t index = 0; index < count; ++index) {
      residuals[index] = results[index].residual_norm;
      work[4 * index] = static_cast<std::size_t>(results[index].status);
      work[4 * index + 1] = results[index].iterations;
      work[4 * index + 2] = results[index].operator_actions;
      work[4 * index + 3] = results[index].measured_gmres_peak_bytes;
    }
    const auto after = handle->response->xc_work();
    work[4 * count] = after.tile_reads - before.tile_reads;
    work[4 * count + 1] = after.actions_completed - before.actions_completed;
    work[4 * count + 2] = handle->response->fock_actions() - before_fock;
    work[4 * count + 3] = after.point_visits - before.point_visits;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_perturbation_bind(Handle* handle, std::size_t bytes, std::size_t terms) {
  try {
    handle->perturbation = std::make_unique<dft::HostKsPerturbation>(
        *handle->response, dft::HostKsPerturbationOptions{bytes, terms});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}

extern "C" int host_ks_perturbation_solve(Handle* handle, const double* overlap,
                                          const double* frozen, double* density, double* weights,
                                          std::size_t iterations, std::size_t workspace,
                                          double* diagnostic, std::size_t* work) {
  try {
    const auto matrix = handle->basis.nao * handle->basis.nao;
    const auto count = matrix * handle->current.model.spins;
    response::GmresOptions options;
    options.relative_tolerance = 2e-11;
    options.absolute_tolerance = 1e-13;
    options.restart = 20;
    options.max_iterations = iterations;
    options.max_workspace_bytes = workspace;
    const auto resources = handle->perturbation->resources(options);
    const auto result = handle->perturbation->solve({overlap, matrix}, {frozen, count},
                                                    {density, count}, {weights, count}, options);
    diagnostic[0] = result.physical_residual_norm;
    diagnostic[1] = result.relative_physical_residual;
    diagnostic[2] = result.response.relative_residual;
    work[0] = resources.owned_numeric_bytes;
    work[1] = resources.map_workspace_bytes;
    work[2] = resources.response_array_bound_bytes;
    work[3] = resources.gmres_workspace_bytes;
    work[4] = resources.scheduled_map_terms;
    work[5] = result.response.measured_gmres_peak_bytes;
    const auto counts = handle->perturbation->work();
    work[6] = counts.directions_started;
    work[7] = counts.directions_completed;
    work[8] = counts.map_evaluations;
    work[9] = counts.scheduled_map_terms;
    return static_cast<int>(result.status);
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
