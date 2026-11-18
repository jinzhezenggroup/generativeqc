#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "integrals/ecp.hpp"
#include "integrals/ecp_gaussian.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"

namespace {
using namespace generativeqc;
thread_local std::string error;

struct Handle {
  core::System system;
  integrals::EcpDirectionPlan plan;
  std::vector<core::EcpTerm> retained;
  std::size_t validations{}, failure_at{}, epoch{}, expected_epoch{};
  std::size_t column_mutation{}, column_mutation_at{};
  std::array<std::size_t, 10> report{};

  Handle(const double* centers, std::size_t atoms, const int* numbers, const int* cores,
         unsigned angular, bool spherical, bool contracted, double exponent_scale,
         const double* terms, std::size_t term_count) {
    int electrons{};
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      system.atoms.push_back({numbers[atom],
                              {centers[3 * atom], centers[3 * atom + 1], centers[3 * atom + 2]},
                              cores[atom]});
      electrons += numbers[atom] - cores[atom];
    }
    system.multiplicity = electrons % 2 + 1;
    system.basis_representation = spherical ? core::kSphericalBasis : core::kCartesianBasis;
    system.shells = {{0, angular, {{.8 + .1 * angular, 1}}}, {1, 0, {{.6, 1}}}};
    if (contracted) {
      system.shells[0].primitives = {{.8 + .1 * angular, .7}, {.23 + .02 * angular, .3}};
      system.shells[1].primitives = {{.6, .7}, {.18, .3}};
    }
    for (auto& primitive : system.shells[0].primitives) primitive.exponent *= exponent_scale;
    for (std::size_t index = 0; index < term_count; ++index) {
      const auto* term = terms + 5 * index;
      system.ecp_terms.push_back({static_cast<unsigned>(term[0]), static_cast<int>(term[1]),
                                  static_cast<unsigned>(term[2]), term[3], term[4]});
    }
    std::string detail;
    if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail);
    retained = system.ecp_terms;
    plan = integrals::ecp_direction_plan(system);
  }

  void validate() {
    ++validations;
    if (epoch != expected_epoch || system.ecp_terms != retained ||
        (failure_at && validations >= failure_at))
      throw std::runtime_error("stale ECP direction owner");
  }
};
}  // namespace

extern "C" const char* ecp_direction_error() { return error.c_str(); }
extern "C" void* ecp_direction_create(const double* centers, std::size_t atoms, const int* numbers,
                                      const int* cores, unsigned angular, int spherical,
                                      int contracted, double exponent_scale, const double* terms,
                                      std::size_t term_count) {
  try {
    return new Handle(centers, atoms, numbers, cores, angular, spherical, contracted,
                      exponent_scale, terms, term_count);
  } catch (const std::exception& failure) {
    error = failure.what();
    return nullptr;
  }
}
extern "C" void ecp_direction_destroy(Handle* handle) { delete handle; }
extern "C" void ecp_direction_sizes(Handle* handle, std::size_t* output) {
  const auto& limits = handle->plan.limits;
  const std::size_t values[]{molecule::ao_count(handle->system),
                             handle->system.atoms.size(),
                             static_cast<std::size_t>(handle->plan.backend),
                             limits.workspace_bytes,
                             limits.gaussian_kernel_blocks,
                             limits.grid_radial_layers,
                             limits.grid_ao_primitive_points,
                             limits.grid_contraction_terms};
  std::copy(std::begin(values), std::end(values), output);
}
extern "C" void ecp_direction_failure(Handle* handle, std::size_t failure_at, int epoch) {
  handle->failure_at = failure_at;
  handle->epoch = epoch;
}
extern "C" int ecp_direction_apply(Handle* handle, const double* direction,
                                   std::size_t direction_size, double* output,
                                   std::size_t output_size, const std::size_t* limits,
                                   int missing_validator) {
  try {
    const auto count = molecule::ao_count(handle->system);
    if (!output || output_size != 2 * count * count || (!direction && direction_size))
      throw std::invalid_argument("invalid ECP direction probe buffer");
    handle->validations = 0;
    auto admitted = handle->plan.limits;
    if (limits) admitted = {limits[0], limits[1], limits[2], limits[3], limits[4]};
    runtime::cpu_resource_observation = {};
    runtime::cpu_resource_observation.active = true;
    integrals::EcpDirectionData value;
    try {
      value = integrals::checked_ecp_direction(
          handle->system, {direction, direction_size}, admitted,
          missing_validator ? std::function<void()>{} : [handle] { handle->validate(); });
    } catch (...) {
      runtime::cpu_resource_observation.active = false;
      throw;
    }
    const auto measured = runtime::cpu_resource_observation;
    runtime::cpu_resource_observation.active = false;
    handle->report = {value.work.kernel_blocks,
                      value.work.radial_layers,
                      value.work.ao_primitive_points,
                      value.work.contraction_terms,
                      value.work.shell_pairs,
                      value.work.primitive_groups,
                      value.work.radial_frontiers,
                      value.work.cartesian_values,
                      static_cast<std::size_t>(measured.peak_bytes),
                      handle->validations};
    std::copy(value.local.begin(), value.local.end(), output);
    std::copy(value.nonlocal.begin(), value.nonlocal.end(), output + count * count);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void ecp_direction_work(Handle* handle, std::size_t* output) {
  std::copy(handle->report.begin(), handle->report.end(), output);
}
extern "C" int ecp_mixed_plan(Handle* handle, std::size_t* output) {
  try {
    const auto plan = integrals::gaussian_ecp_weighted_mixed_plan(handle->system);
    output[0] = plan.workspace_bytes;
    output[1] = plan.kernel_blocks;
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int ecp_mixed_apply(Handle* handle, const double* weights, std::size_t weight_count,
                               const double* left, std::size_t left_count, const double* right,
                               std::size_t right_count, const std::size_t* limits, double* output,
                               std::size_t output_count, int missing_validator) {
  try {
    if (!output || output_count != 8 || !limits || (!weights && weight_count) ||
        (!left && left_count) || (!right && right_count))
      throw std::invalid_argument("invalid ECP mixed probe buffer");
    handle->validations = 0;
    runtime::cpu_resource_observation = {};
    runtime::cpu_resource_observation.active = true;
    integrals::EcpWeightedMixed value;
    try {
      value = integrals::gaussian_ecp_weighted_mixed(
          handle->system, {weights, weight_count}, {left, left_count}, {right, right_count},
          {limits[0], limits[1]},
          missing_validator ? std::function<void()>{} : [handle] { handle->validate(); });
    } catch (...) {
      runtime::cpu_resource_observation.active = false;
      throw;
    }
    const auto measured = runtime::cpu_resource_observation;
    runtime::cpu_resource_observation.active = false;
    handle->report = {value.work.kernel_blocks,
                      0,
                      0,
                      0,
                      value.work.shell_pairs,
                      value.work.primitive_groups,
                      value.work.radial_frontiers,
                      value.work.cartesian_values,
                      static_cast<std::size_t>(measured.peak_bytes),
                      handle->validations};
    for (std::size_t part = 0; part < 2; ++part)
      std::copy(value.parts[part].begin(), value.parts[part].end(), output + 4 * part);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int ecp_direction_raw_contract(Handle* handle, const double* direction, double* output) {
  try {
    const auto raw = integrals::checked_ecp_integrals(handle->system, true);
    const auto matrix = raw.nbf * raw.nbf;
    std::fill_n(output, 2 * matrix, 0);
    for (std::size_t coordinate = 0; coordinate < raw.ncoord; ++coordinate)
      for (std::size_t index = 0; index < matrix; ++index) {
        output[index] += direction[coordinate] * raw.local_derivative[coordinate * matrix + index];
        output[matrix + index] +=
            direction[coordinate] * raw.nonlocal_derivative[coordinate * matrix + index];
      }
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int ecp_column_plan(Handle* handle, std::size_t* output) {
  try {
    const auto plan = integrals::ecp_curvature_plan(handle->system);
    const auto& limits = plan.limits;
    const std::size_t values[]{
        static_cast<std::size_t>(plan.backend), limits.workspace_bytes,
        limits.gaussian_kernel_blocks,          limits.grid_radial_layers,
        limits.grid_ao_primitive_points,        limits.grid_contraction_terms};
    std::copy(std::begin(values), std::end(values), output);
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" void ecp_column_mutation(Handle* handle, std::size_t mode, std::size_t at) {
  handle->column_mutation = mode;
  handle->column_mutation_at = at;
}
static int column_apply(Handle* handle, const double* weights, std::size_t weight_count,
                        const double* weights_right, std::size_t moving_count, const double* right,
                        std::size_t right_count, const std::size_t* limits, double* output,
                        std::size_t output_count, int missing_validator) {
  try {
    if (!output || output_count != 2 * (2 + 6 * handle->system.atoms.size()) || !limits ||
        !weights || !right)
      throw std::invalid_argument("invalid ECP column probe buffer");
    handle->validations = 0;
    const std::function<void()> validator = missing_validator ? std::function<void()>{} : [&] {
      if (handle->validations + 1 == handle->column_mutation_at) {
        if (handle->column_mutation == 1) const_cast<double*>(weights)[0] += .125;
        if (handle->column_mutation == 2) const_cast<double*>(right)[0] += .125;
        if (handle->column_mutation == 3 && weights_right && moving_count)
          const_cast<double*>(weights_right)[0] += .125;
      }
      handle->validate();
    };
    runtime::cpu_resource_observation = {};
    runtime::cpu_resource_observation.active = true;
    integrals::EcpCurvatureData value;
    try {
      value = integrals::checked_ecp_curvature(
          handle->system, {weights, weight_count}, {weights_right, moving_count},
          {right, right_count}, {limits[0], limits[1], limits[2], limits[3], limits[4]}, validator);
    } catch (...) {
      runtime::cpu_resource_observation.active = false;
      throw;
    }
    const auto measured = runtime::cpu_resource_observation;
    runtime::cpu_resource_observation.active = false;
    handle->report = {value.work.kernel_blocks,
                      value.work.radial_layers,
                      value.work.ao_primitive_points,
                      value.work.contraction_terms,
                      value.work.shell_pairs,
                      value.work.primitive_groups,
                      value.work.radial_frontiers,
                      value.work.cartesian_values,
                      static_cast<std::size_t>(measured.peak_bytes),
                      handle->validations};
    for (std::size_t part = 0; part < 2; ++part)
      std::copy(value.parts[part].begin(), value.parts[part].end(),
                output + part * value.parts[part].size());
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
extern "C" int ecp_column_apply(Handle* handle, const double* weights, std::size_t weight_count,
                                const double* right, std::size_t right_count,
                                const std::size_t* limits, double* output, std::size_t output_count,
                                int missing_validator) {
  return column_apply(handle, weights, weight_count, nullptr, 0, right, right_count, limits, output,
                      output_count, missing_validator);
}
extern "C" int ecp_column_response(Handle* handle, const double* weights, std::size_t weight_count,
                                   const double* weights_right, std::size_t moving_count,
                                   const double* right, std::size_t right_count,
                                   const std::size_t* limits, double* output,
                                   std::size_t output_count, int missing_validator) {
  return column_apply(handle, weights, weight_count, weights_right, moving_count, right,
                      right_count, limits, output, output_count, missing_validator);
}
extern "C" int ecp_direction_gate_gaps(Handle* handle, double* output) {
  try {
    const auto coarse = integrals::ecp_integrals(handle->system, 160, 32, true);
    const auto fine = integrals::ecp_integrals(handle->system, 224, 44, true);
    const std::vector<double>* first[]{&coarse.local, &coarse.nonlocal, &coarse.local_derivative,
                                       &coarse.nonlocal_derivative};
    const std::vector<double>* second[]{&fine.local, &fine.nonlocal, &fine.local_derivative,
                                        &fine.nonlocal_derivative};
    for (std::size_t part = 0; part < 4; ++part) {
      output[part] = 0;
      for (std::size_t index = 0; index < first[part]->size(); ++index)
        output[part] =
            std::max(output[part], std::abs((*first[part])[index] - (*second[part])[index]));
    }
    return 0;
  } catch (const std::exception& failure) {
    error = failure.what();
    return -1;
  }
}
