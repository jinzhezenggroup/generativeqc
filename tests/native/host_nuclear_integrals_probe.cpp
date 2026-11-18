#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/host_hessian_aot.hpp"
#include "dft/host_nuclear_integrals.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

core::System molecule(const double* centers, unsigned angular, bool spherical, bool contracted,
                      int first_number, int first_core) {
  core::System result;
  result.atoms = {{first_number, {centers[0], centers[1], centers[2]}, first_core},
                  {1, {centers[3], centers[4], centers[5]}}};
  result.multiplicity = (first_number - first_core + 1) % 2 + 1;
  result.basis_representation = spherical ? core::kSphericalBasis : core::kCartesianBasis;
  result.shells = {{0, angular, {{.8 + .1 * angular, 1}}}, {1, 0, {{.6, 1}}}};
  if (contracted) {
    result.shells[0].primitives = {{.8 + .1 * angular, .7}, {.23 + .02 * angular, .3}};
    result.shells[1].primitives = {{.6, .7}, {.18, .3}};
  }
  std::string detail;
  if (molecule::validate_and_normalize(result, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
  return result;
}

struct Handle {
  core::System system;
  dft::AoBasis basis;
  scf::ResolvedFockBuild strategy, original;
  std::span<const double> density;
  std::size_t spins{}, mode{}, at{}, validations{};
  std::size_t changed_index{};
  double unchanged_value{};
  bool corrupted{};
  std::vector<double> replaced;
  std::unique_ptr<dft::HostKsNuclearIntegrals> response;
  std::span<const double> weighted_density;
  std::vector<double> authenticated_density, authenticated_weighted_density;
  std::unique_ptr<dft::HostKsNuclearIntegralCurvature> curvature;
  dft::HostNuclearEriSchedule eri_schedule{dft::HostNuclearEriSchedule::SymmetryUnique};
  dft::HostNuclearFirstSchedule first_schedule{dft::HostNuclearFirstSchedule::Direction};
  bool column_curvature{};
  bool matrix_curvature{};
  double* active_right{};
  double* active_density_right{};
  double* active_weighted_right{};

  Handle(const double* centers, unsigned angular, bool spherical, bool contracted, int first_number,
         int first_core, std::size_t spins, const double* density)
      : Handle(molecule(centers, angular, spherical, contracted, first_number, first_core), spins,
               density) {}

  Handle(core::System frame, std::size_t spins, const double* density)
      : system(std::move(frame)),
        basis(system),
        density(density, spins * basis.nao * basis.nao),
        spins(spins) {
    strategy.spec.spin = spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
    strategy.spec.exchange.coefficient = spins == 1 ? -.1 : -.2;
    original = strategy;
  }

  void validate() {
    if (mode == 11) {
      if (active_right && ++validations >= at) {
        active_right[0] += .125;
        mode = 0;
      }
      return;
    }
    if (curvature &&
        (!std::equal(density.begin(), density.end(), authenticated_density.begin(),
                     authenticated_density.end()) ||
         !std::equal(weighted_density.begin(), weighted_density.end(),
                     authenticated_weighted_density.begin(), authenticated_weighted_density.end())))
      throw std::runtime_error("changed authenticated D/W owner");
    if (mode >= 8 && mode <= 10 && curvature && curvature->work().ao_quartets >= at) {
      auto* values = mode == 8   ? active_right
                     : mode == 9 ? active_density_right
                                 : active_weighted_right;
      if (values) values[0] += .125;
      mode = 0;
      return;
    }
    if (mode >= 8 && mode <= 10) return;
    if (mode == 7) {
      if ((curvature  ? curvature->work().ao_quartets
           : response ? response->work().ao_quartets
                      : 0) >= at)
        throw std::runtime_error("injected completed-quartet owner failure");
      return;
    }
    if (mode && ++validations >= at) {
      if (mode == 2 && replaced.empty()) {
        replaced = basis.packed;
        basis.packed.swap(replaced);
        return;
      }
      if (mode == 3) {
        std::vector<double> direction(3 * basis.natom), overlap(basis.nao * basis.nao),
            fock(density.size()), column(2 + 6 * basis.natom);
        if (curvature) {
          if (matrix_curvature) {
            std::vector<double> matrix(1 + direction.size() + direction.size() * direction.size());
            curvature->apply_matrix(matrix);
          } else if (column_curvature)
            curvature->apply_column(direction, {}, {}, column);
          else
            curvature->apply(direction, direction, std::span(fock).first(4));
        } else
          response->apply(direction, overlap, fock);
      }
      if (mode == 4) {
        strategy.spec.exchange.coefficient = 0;
        return;
      }
      if (mode == 5 || mode == 6) {
        changed_index = mode == 5 ? 3 * basis.natom : 3 * basis.natom + 2 * basis.nprimitive + 3;
        unchanged_value = basis.packed[changed_index];
        corrupted = true;
        basis.packed[changed_index] = mode == 5 ? std::numeric_limits<double>::quiet_NaN() : .5;
        return;
      }
      throw std::runtime_error("injected stale native integral owner");
    }
  }

  void bind(const char* method, const std::size_t* limits, std::size_t cadence) {
    auto planned = dft::HostKsNuclearIntegrals::plan(basis, spins, eri_schedule, first_schedule);
    if (limits) planned = {limits[0], limits[1], limits[2], limits[3]};
    auto candidate = std::make_unique<dft::HostKsNuclearIntegrals>(
        system, basis, strategy, density, method,
        dft::HostNuclearIntegralOptions{planned, cadence, eri_schedule}, [this] { validate(); },
        first_schedule);
    response = std::move(candidate);
  }

  void bind_curvature(const double* values, std::size_t count, const char* method,
                      const std::size_t* limits, std::size_t cadence, bool column = false,
                      bool matrix = false, bool projected = false) {
    const auto schedule = projected ? dft::HostNuclearIntegralCurvatureSchedule::ProjectedMatrix
                          : matrix  ? dft::HostNuclearIntegralCurvatureSchedule::Matrix
                          : column  ? dft::HostNuclearIntegralCurvatureSchedule::Column
                                    : dft::HostNuclearIntegralCurvatureSchedule::Mixed;
    auto planned = dft::HostKsNuclearIntegralCurvature::plan(basis, spins, schedule, eri_schedule);
    if (limits) planned = {limits[0], limits[1], limits[2], limits[3]};
    const auto previous = weighted_density;
    auto previous_density = authenticated_density;
    auto previous_weighted_density = authenticated_weighted_density;
    weighted_density = {values, count};
    authenticated_density.assign(density.begin(), density.end());
    if (values && count == density.size())
      authenticated_weighted_density.assign(values, values + count);
    try {
      auto candidate = std::make_unique<dft::HostKsNuclearIntegralCurvature>(
          system, basis, strategy, density, weighted_density, method,
          dft::HostNuclearIntegralOptions{planned, cadence, eri_schedule}, [this] { validate(); },
          schedule);
      curvature = std::move(candidate);
      column_curvature = column;
      matrix_curvature = matrix;
    } catch (...) {
      weighted_density = previous;
      authenticated_density = std::move(previous_density);
      authenticated_weighted_density = std::move(previous_weighted_density);
      throw;
    }
  }
};
}  // namespace

extern "C" const char* host_integral_error() { return error.c_str(); }
extern "C" void* host_integral_create(const double* centers, unsigned angular, int spherical,
                                      int contracted, int number, int core, std::size_t spins,
                                      const double* density, const char* method,
                                      std::size_t cadence) {
  try {
    auto result = std::make_unique<Handle>(centers, angular, spherical, contracted, number, core,
                                           spins, density);
    result->bind(method, nullptr, cadence);
    return result.release();
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void host_integral_destroy(Handle* handle) { delete handle; }
extern "C" void* host_integral_create_sites(const double* centers, std::size_t atoms,
                                            std::size_t spins, const double* density,
                                            const char* method, std::size_t cadence) {
  try {
    core::System frame;
    frame.multiplicity = atoms % 2 + 1;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      frame.atoms.push_back({1, {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]}});
      frame.shells.push_back({static_cast<unsigned>(atom), 0, {{.6, 1}}});
    }
    std::string detail;
    if (molecule::validate_and_normalize(frame, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail);
    auto result = std::make_unique<Handle>(std::move(frame), spins, density);
    result->bind(method, nullptr, cadence);
    return result.release();
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" int host_integral_bind(Handle* handle, const char* method, const std::size_t* limits,
                                  std::size_t cadence) {
  try {
    handle->bind(method, limits, cadence);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void* host_integral_create_mixed_sites(const double* centers, int spherical,
                                                  std::size_t spins, const double* density,
                                                  const char* method) {
  try {
    core::System frame;
    frame.multiplicity = 2;
    frame.basis_representation = spherical ? core::kSphericalBasis : core::kCartesianBasis;
    for (std::size_t atom = 0; atom < 3; ++atom)
      frame.atoms.push_back({1, {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]}});
    frame.shells = {
        {0, 0, {{.5, 1}}}, {0, 2, {{.8, .7}, {.23, .3}}}, {1, 1, {{.7, 1}}}, {2, 2, {{.6, 1}}}};
    std::string detail;
    if (molecule::validate_and_normalize(frame, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail);
    auto result = std::make_unique<Handle>(std::move(frame), spins, density);
    result->bind(method, nullptr, 64);
    return result.release();
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" int host_integral_plan_count(Handle* handle, std::size_t count) {
  try {
    auto copy = handle->basis;
    copy.nao = count;
    dft::HostKsNuclearIntegrals::plan(copy, handle->spins);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_integral_schedule(Handle* handle, unsigned schedule, const char* method,
                                      std::size_t cadence) {
  const auto previous = handle->eri_schedule;
  try {
    if (schedule > 1 || handle->curvature)
      throw std::invalid_argument("select a valid ERI schedule before binding curvature");
    handle->eri_schedule = static_cast<dft::HostNuclearEriSchedule>(schedule);
    handle->bind(method, nullptr, cadence);
    return 0;
  } catch (const std::exception& failure) {
    handle->eri_schedule = previous;
    error = failure.what();
    return -1;
  }
}
extern "C" void host_integral_sizes(Handle* handle, std::size_t* output) {
  const auto plan = handle->response->resources();
  const std::size_t values[]{handle->basis.nao,       handle->basis.natom,
                             plan.numeric_bytes,      plan.one_electron_primitives,
                             plan.eri_primitives,     plan.map_terms,
                             handle->basis.nprimitive};
  std::copy(std::begin(values), std::end(values), output);
}
extern "C" int host_integral_coordinate_mode(Handle* handle, unsigned mode, const char* method,
                                             const std::size_t* limits) {
  const auto previous = handle->first_schedule;
  try {
    if (mode > 1) throw std::invalid_argument("select a valid first-source mode");
    handle->first_schedule = static_cast<dft::HostNuclearFirstSchedule>(mode);
    handle->bind(method, limits, 64);
    return 0;
  } catch (const std::exception& failure) {
    handle->first_schedule = previous;
    error = failure.what();
    return -1;
  }
}
extern "C" int host_integral_prepare_coordinates(Handle* handle) {
  try {
    handle->response->prepare_coordinates();
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_integral_resident_work(Handle* handle, std::size_t* output) {
  const auto work = handle->response->work();
  output[0] = work.coordinate_panels_completed;
  output[1] = work.resident_applications;
}
extern "C" std::size_t host_integral_density_response_terms(Handle* handle) {
  return dft::HostKsNuclearIntegrals::density_response_terms(handle->basis, handle->spins);
}
extern "C" int host_integral_density_response(Handle* handle, const double* density,
                                              std::size_t density_count, const double* weighted,
                                              std::size_t weighted_count, double* output,
                                              std::size_t output_count, std::size_t terms) {
  try {
    handle->response->apply_density_response({density, density_count}, {weighted, weighted_count},
                                             {output, output_count}, terms);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_integral_failure(Handle* handle, std::size_t mode, std::size_t at) {
  if (handle->corrupted) {
    handle->basis.packed[handle->changed_index] = handle->unchanged_value;
    handle->corrupted = false;
  }
  handle->mode = mode;
  handle->at = at;
  if (mode == 7)
    handle->at = (handle->curvature ? handle->curvature->work().ao_quartets
                                    : handle->response->work().ao_quartets) +
                 handle->basis.nao * handle->basis.nao * handle->basis.nao * handle->basis.nao;
  handle->validations = 0;
  handle->strategy = handle->original;
  if (!handle->replaced.empty()) {
    handle->basis.packed.swap(handle->replaced);
    handle->replaced.clear();
  }
}
extern "C" void host_integral_model(Handle* handle, unsigned mode) {
  handle->strategy = handle->original;
  if (mode == 1)
    handle->strategy.spec.coulomb.approximation = scf::FockApproximation::DensityFitted;
  if (mode == 2) handle->strategy.backend = scf::FockBackend::Cuda;
  if (mode == 3) handle->strategy.spec.exchange.op = scf::FockOperator::ShortRange;
  if (mode == 4) handle->strategy.spec.exchange.coefficient = -.5;
  if (mode == 5) handle->strategy.spec.spin = static_cast<scf::FockSpin>(-1);
  if (mode == 6) handle->strategy.spec.version = 0;
  if (mode == 7) handle->strategy.precision = static_cast<scf::FockPrecision>(-1);
}
extern "C" double* host_integral_packed(Handle* handle) { return handle->basis.packed.data(); }
extern "C" void host_integral_charge(Handle* handle, int number) {
  handle->system.atoms[0].atomic_number = number;
}
extern "C" int host_integral_apply(Handle* handle, const double* direction,
                                   std::size_t direction_count, double* overlap,
                                   std::size_t overlap_count, double* fock,
                                   std::size_t fock_count) {
  try {
    handle->active_right = const_cast<double*>(direction);
    handle->response->apply({direction, direction_count}, {overlap, overlap_count},
                            {fock, fock_count});
    handle->active_right = nullptr;
    return 0;
  } catch (const std::exception& failure) {
    handle->active_right = nullptr;
    error = failure.what();
    return -1;
  }
}
extern "C" void host_integral_work(Handle* handle, std::size_t* output) {
  const auto work = handle->response->work();
  const std::size_t values[]{work.directions_started,
                             work.directions_completed,
                             work.ao_pairs,
                             work.ao_quartets,
                             work.one_electron_primitives,
                             work.eri_primitives,
                             work.projection_lanes,
                             work.map_terms};
  std::copy(std::begin(values), std::end(values), output);
}

extern "C" int host_curvature_bind(Handle* handle, const double* weighted_density,
                                   std::size_t count, const char* method, const std::size_t* limits,
                                   std::size_t cadence) {
  try {
    handle->bind_curvature(weighted_density, count, method, limits, cadence);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_curvature_resources(Handle* handle, std::size_t* output) {
  const auto plan = handle->curvature->resources();
  const std::size_t values[]{plan.numeric_bytes, plan.one_electron_primitives, plan.eri_primitives,
                             plan.map_terms};
  std::copy(std::begin(values), std::end(values), output);
}
extern "C" int host_column_bind(Handle* handle, const double* weighted_density, std::size_t count,
                                const char* method, const std::size_t* limits,
                                std::size_t cadence) {
  try {
    handle->bind_curvature(weighted_density, count, method, limits, cadence, true);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_matrix_bind(Handle* handle, const double* weighted_density, std::size_t count,
                                const char* method, const std::size_t* limits,
                                std::size_t cadence) {
  try {
    handle->bind_curvature(weighted_density, count, method, limits, cadence, false, true);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_matrix_apply(Handle* handle, double* output, std::size_t count) {
  try {
    handle->curvature->apply_matrix({output, count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_projected_matrix_bind(Handle* handle, const double* weighted_density,
                                          std::size_t count, const char* method,
                                          const std::size_t* limits, std::size_t cadence) {
  try {
    handle->bind_curvature(weighted_density, count, method, limits, cadence, false, true, true);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_projected_matrix_supported(Handle* handle) {
  return dft::HostKsNuclearIntegralCurvature::projected_matrix_supported(handle->basis,
                                                                         handle->density);
}
extern "C" std::size_t host_curvature_density_projection_terms(Handle* handle) {
  return handle->curvature->work().density_projection_terms;
}
extern "C" int host_column_apply(Handle* handle, const double* right, std::size_t right_count,
                                 double* output, std::size_t output_count) {
  try {
    handle->curvature->apply_column({right, right_count}, {}, {}, {output, output_count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_column_response(Handle* handle, const double* right, std::size_t right_count,
                                    const double* density_right, std::size_t density_count,
                                    const double* weighted_right, std::size_t weighted_count,
                                    double* output, std::size_t output_count) {
  handle->active_right = const_cast<double*>(right);
  handle->active_density_right = const_cast<double*>(density_right);
  handle->active_weighted_right = const_cast<double*>(weighted_right);
  const auto clear = [handle] {
    handle->active_right = nullptr;
    handle->active_density_right = nullptr;
    handle->active_weighted_right = nullptr;
  };
  try {
    handle->curvature->apply_column({right, right_count}, {density_right, density_count},
                                    {weighted_right, weighted_count}, {output, output_count});
    clear();
    return 0;
  } catch (const std::exception& failure) {
    clear();
    error = failure.what();
    return -1;
  }
}
extern "C" int host_curvature_plan_schedule(Handle* handle, unsigned schedule) {
  try {
    dft::HostKsNuclearIntegralCurvature::plan(
        handle->basis, handle->spins,
        static_cast<dft::HostNuclearIntegralCurvatureSchedule>(schedule));
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int host_curvature_apply(Handle* handle, const double* left, std::size_t left_count,
                                    const double* right, std::size_t right_count, double* output,
                                    std::size_t output_count) {
  try {
    handle->curvature->apply({left, left_count}, {right, right_count}, {output, output_count});
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void host_curvature_work(Handle* handle, std::size_t* output) {
  const auto work = handle->curvature->work();
  const std::size_t values[]{work.directions_started,
                             work.directions_completed,
                             work.ao_pairs,
                             work.ao_quartets,
                             work.one_electron_primitives,
                             work.eri_primitives,
                             work.projection_lanes,
                             work.map_terms};
  std::copy(std::begin(values), std::end(values), output);
}

extern "C" void host_curvature_eri_orders(Handle* handle, std::size_t* output) {
  const auto work = handle->curvature->work();
  output[0] = work.eri_value_only_primitives;
  output[1] = work.eri_first_only_primitives;
  output[2] = work.eri_mixed_primitives;
}

extern "C" void host_curvature_matrix_fields(Handle* handle, std::size_t* output) {
  const auto work = handle->curvature->work();
  output[0] = work.eri_matrix_primitives;
  output[1] = work.eri_matrix_coordinates;
  output[2] = work.eri_matrix_second_partials;
}
