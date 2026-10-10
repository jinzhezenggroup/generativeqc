#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "api/ks_snapshot.hpp"
#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Environment {
  generativeqc_context* context{};
  generativeqc_system* system{};
  generativeqc_calculation* calculation{};

  explicit Environment(double displacement = 0.0) {
    const generativeqc_context_descriptor context_options{sizeof(generativeqc_context_descriptor),
                                                          GENERATIVEQC_ABI_VERSION, 0,
                                                          GENERATIVEQC_BACKEND_CPU_REFERENCE};
    require(generativeqc_context_create(&context_options, &context) == GENERATIVEQC_STATUS_SUCCESS,
            "failed to create native CPU context");
    const std::array<generativeqc_atom, 2> atoms{
        {{1, 0.0, 0.0, -0.7}, {1, 0.0, 0.0, 0.7 + displacement}}};
    const std::array<generativeqc_primitive, 6> primitives{{
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
    }};
    const std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
    const generativeqc_system_descriptor system_options{sizeof(generativeqc_system_descriptor),
                                                        GENERATIVEQC_ABI_VERSION,
                                                        atoms.data(),
                                                        2,
                                                        shells.data(),
                                                        2,
                                                        primitives.data(),
                                                        6,
                                                        0,
                                                        1,
                                                        GENERATIVEQC_BASIS_CARTESIAN};
    require(generativeqc_system_create(context, &system_options, &system) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "failed to create H2/STO-3G native system");
  }
  ~Environment() {
    generativeqc_calculation_destroy(calculation);
    generativeqc_system_destroy(system);
    generativeqc_context_destroy(context);
  }
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  void prepare(generativeqc_density_fitting_mode fitting) {
    generativeqc_calculation_destroy(calculation);
    calculation = nullptr;
    generativeqc_method_descriptor method{};
    method.struct_size = sizeof(method);
    method.abi_version = GENERATIVEQC_ABI_VERSION;
    method.method = GENERATIVEQC_METHOD_PBE_RKS;
    method.max_iterations = 200;
    method.diis_history = 8;
    method.energy_tolerance = 1.0e-12;
    method.density_tolerance = 1.0e-10;
    method.screening_tolerance = 1.0e-14;
    method.density_fitting_mode = fitting;
    method.density_fitting_relative_threshold = 1.0e-10;
    method.precision_mode = GENERATIVEQC_PRECISION_FP64;
    require(generativeqc_calculation_prepare(context, system, &method, &calculation) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "failed to prepare native PBE integral source owner");
  }

  double execute_energy() const {
    generativeqc_result_descriptor result{};
    result.struct_size = sizeof(result);
    result.abi_version = GENERATIVEQC_ABI_VERSION;
    require(generativeqc_calculation_execute(calculation, &result) == GENERATIVEQC_STATUS_SUCCESS &&
                result.converged && std::isfinite(result.energy),
            "native PBE SCF did not converge before stationary source read");
    return result.energy;
  }
};

constexpr std::size_t kSourceCount = 4 * 3 * 2;
constexpr std::size_t kWorkCount = 9;
constexpr std::size_t kBudget = 8u << 20;

void test_single_prepared_stationary_sources() {
  Environment environment;
  environment.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  std::array<double, kSourceCount> sources;
  std::array<std::uint64_t, kWorkCount> work;
  sources.fill(1234.0);
  work.fill(4321);

  auto read = [&](std::size_t size = kSourceCount, std::size_t budget = kBudget) {
    return generativeqc_ks_calculation_integral_sources_v1(environment.calculation, sources.data(),
                                                           size, budget, work.data(), work.size());
  };

  require(read() == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "native stationary source read accepted an unexecuted SCF frame");
  require(std::all_of(sources.begin(), sources.end(), [](double x) { return x == 1234.0; }),
          "failed stationary source preflight modified output");

  environment.execute_energy();
  require(read(kSourceCount - 1) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "native stationary source accepted an incorrectly sized output");
  require(read(kSourceCount, 32) == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
          "stationary source ignored its compact publication budget");
  require(sources.front() == 1234.0 && work.front() == 4321,
          "failed native stationary source check wrote to user buffers");

  require(read() == GENERATIVEQC_STATUS_SUCCESS,
          "verified native single-system CPU DF stationary sources failed");
  require(std::all_of(sources.begin(), sources.end(), [](double x) { return std::isfinite(x); }),
          "stationary integral sources contain nonfinite entries");
  require(std::any_of(sources.begin() + 12, sources.begin() + 18,
                      [](double x) { return std::abs(x) > 1.0e-9; }),
          "PBE DF Coulomb nuclear derivative is unexpectedly absent");
  require(std::all_of(sources.begin() + 18, sources.end(), [](double x) { return x == 0.0; }),
          "pure PBE fabricated a stationary exact-exchange derivative");
  const auto original = sources;
  require(read() == GENERATIVEQC_STATUS_SUCCESS,
          "prepared native stationary integral source replay failed");
  for (std::size_t i = 0; i < kSourceCount; ++i)
    require(std::abs(sources[i] - original[i]) <= 1.0e-9,
            "single-system stationary integral response replay changed");

  // Attempted execution invalidates the final-state generation even if the
  // public C result descriptor itself is rejected before any SCF iteration.
  generativeqc_result_descriptor bad{};
  require(generativeqc_calculation_execute(environment.calculation, &bad) ==
              GENERATIVEQC_STATUS_ABI_MISMATCH,
          "invalid result descriptor unexpectedly accepted");
  sources.fill(1234.0);
  work.fill(4321);
  require(read() == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "native stationary sources read through a revoked final-state token");
  require(sources.front() == 1234.0 && work.front() == 4321,
          "stale final-state rejection wrote partial stationary sources");

  environment.execute_energy();
  require(read() == GENERATIVEQC_STATUS_SUCCESS,
          "native stationary source read failed after a fresh SCF replay");

  environment.prepare(GENERATIVEQC_DENSITY_FITTING_NONE);
  environment.execute_energy();
  sources.fill(1234.0);
  work.fill(4321);
  require(read() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "CPU Direct stationary source silently substituted a DF response");
  require(sources.front() == 1234.0 && work.front() == 4321,
          "unsupported CPU Direct stationary source modified user buffers");
}

void test_prepared_cpu_df_pbe_analytic_forces() {
  Environment environment;
  environment.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  generativeqc_property_flags available{};
  require(generativeqc_calculation_get_supported_properties_v1(
              environment.calculation, &available) == GENERATIVEQC_STATUS_SUCCESS &&
              (available & GENERATIVEQC_PROPERTY_FORCES),
          "CPU DF-PBE prepared force capability is missing");
  std::array<double, 6> forces;
  forces.fill(1234.0);
  generativeqc_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  result.forces = forces.data();
  result.force_count = static_cast<std::uint32_t>(forces.size());
  require(generativeqc_calculation_execute(environment.calculation, &result) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              result.converged && std::isfinite(result.energy),
          "CPU DF-PBE analytic E+F did not converge");
  for (double force : forces)
    require(std::isfinite(force) && force != 1234.0, "nonfinite/missing native PBE force");
  for (unsigned axis = 0; axis < 3; ++axis)
    require(std::abs(forces[axis] + forces[3 + axis]) < 2e-5,
            "native DFT complete force violates translation invariance");

  // Independent displaced *reconverged* molecular energies, not the AO/grid
  // fixed-density derivative test in test_dft.cpp. Both include nuclear,
  // Pulay, J/K, PBE XC and the same moving Becke grid.
  constexpr double step = 5e-4;
  Environment positive(step), negative(-step);
  positive.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  negative.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  const double fd = -(positive.execute_energy() - negative.execute_energy()) / (2.0 * step);
  if (!std::isfinite(fd) || std::abs(fd - forces[5]) >= 2e-3)
    throw std::runtime_error("native CPU DF-PBE force disagrees with reconverged FD: FD=" +
                             std::to_string(fd) + " force=" + std::to_string(forces[5]));

  environment.prepare(GENERATIVEQC_DENSITY_FITTING_NONE);
  available = 0;
  require(generativeqc_calculation_get_supported_properties_v1(
              environment.calculation, &available) == GENERATIVEQC_STATUS_SUCCESS &&
              !(available & GENERATIVEQC_PROPERTY_FORCES),
          "CPU Direct must not claim unqualified native DFT forces");
  forces.fill(1234.0);
  require(generativeqc_calculation_execute(environment.calculation, &result) ==
              GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "unqualified CPU Direct DFT forces were admitted");
  require(std::all_of(forces.begin(), forces.end(), [](double x) { return x == 1234.0; }),
          "unqualified DFT execution published partial nuclear forces");
}

/** Match the native batch endpoint to the already-qualified single owner;
 * unsupported providers and invalid neighbors must not receive forces. */
void test_prepared_cpu_df_pbe_batch_forces() {
  Environment baseline;
  baseline.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  std::array<double, 6> single_forces{};
  generativeqc_result_descriptor single{};
  single.struct_size = sizeof(single);
  single.abi_version = GENERATIVEQC_ABI_VERSION;
  single.forces = single_forces.data();
  single.force_count = single_forces.size();
  require(generativeqc_calculation_execute(baseline.calculation, &single) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              single.converged,
          "single-system native DF-PBE force oracle failed");

  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = GENERATIVEQC_METHOD_PBE_RKS;
  method.max_iterations = 200;
  method.diis_history = 8;
  method.energy_tolerance = 1.0e-12;
  method.density_tolerance = 1.0e-10;
  method.screening_tolerance = 1.0e-14;
  method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  method.density_fitting_relative_threshold = 1.0e-10;
  method.precision_mode = GENERATIVEQC_PRECISION_FP64;

  const generativeqc_system* systems[]{baseline.system, baseline.system};
  generativeqc_batch* batch = nullptr;
  require(generativeqc_batch_prepare(baseline.context, systems, 2, &method,
                                     GENERATIVEQC_BATCH_ENABLE_WARM_STARTS,
                                     &batch) == GENERATIVEQC_STATUS_SUCCESS &&
              batch != nullptr,
          "failed to prepare CPU DF-PBE native batch for forces");

  std::array<double, 6> first{}, second{};
  std::array<generativeqc_batch_item_result_descriptor, 2> outputs{};
  const auto reset_outputs = [&] {
    outputs = {};
    for (auto& output : outputs) {
      output.struct_size = sizeof(output);
      output.abi_version = GENERATIVEQC_ABI_VERSION;
    }
    outputs[0].forces = first.data();
    outputs[0].force_count = first.size();
    outputs[1].forces = second.data();
    outputs[1].force_count = second.size();
  };
  const auto execute = [&](const generativeqc_batch_input_descriptor* inputs = nullptr,
                           std::uint32_t count = 0) {
    reset_outputs();
    return generativeqc_batch_execute(batch, inputs, count, outputs.data(), outputs.size());
  };
  const auto compare_reference = [&](std::size_t item, const double* forces, double energy) {
    require(outputs[item].status == GENERATIVEQC_STATUS_SUCCESS && outputs[item].converged &&
                std::isfinite(outputs[item].energy) &&
                std::abs(outputs[item].energy - energy) < 1e-7,
            "native CPU DF-PBE batch energy differs from the single owner");
    const auto& actual = item == 0 ? first : second;
    for (std::size_t k = 0; k < actual.size(); ++k)
      require(std::isfinite(actual[k]) && std::abs(actual[k] - forces[k]) < 1e-4,
              "native CPU DF-PBE batch force differs from the single owner");
  };

  require(execute() == GENERATIVEQC_STATUS_SUCCESS, "native CPU DF-PBE batch E+F execution failed");
  compare_reference(0, single_forces.data(), single.energy);
  compare_reference(1, single_forces.data(), single.energy);
  require(execute() == GENERATIVEQC_STATUS_SUCCESS,
          "native CPU DF-PBE repeated warm batch E+F failed");
  compare_reference(0, single_forces.data(), single.energy);
  compare_reference(1, single_forces.data(), single.energy);

  constexpr double displacement = 0.005;
  Environment moved(displacement);
  moved.prepare(GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE);
  std::array<double, 6> moved_forces{};
  generativeqc_result_descriptor moved_result{};
  moved_result.struct_size = sizeof(moved_result);
  moved_result.abi_version = GENERATIVEQC_ABI_VERSION;
  moved_result.forces = moved_forces.data();
  moved_result.force_count = moved_forces.size();
  require(generativeqc_calculation_execute(moved.calculation, &moved_result) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              moved_result.converged,
          "native moved single-system DF-PBE force reference failed");
  std::array<double, 6> moved_coordinates{{0., 0., -0.7, 0., 0., 0.7 + displacement}};
  std::array<generativeqc_batch_input_descriptor, 2> inputs{};
  for (auto& input : inputs) {
    input.struct_size = sizeof(input);
    input.abi_version = GENERATIVEQC_ABI_VERSION;
  }
  inputs[1].coordinates = moved_coordinates.data();
  inputs[1].coordinate_count = moved_coordinates.size();
  require(execute(inputs.data(), inputs.size()) == GENERATIVEQC_STATUS_SUCCESS,
          "native moved-geometry batch E+F execution failed");
  compare_reference(0, single_forces.data(), single.energy);
  compare_reference(1, moved_forces.data(), moved_result.energy);

  // The second item's bad shape is a local failure, never a partial force
  // publication or a reason to poison the valid neighbor's warm state.
  std::array<double, 1> invalid{{0.0}};
  inputs[1].coordinates = invalid.data();
  inputs[1].coordinate_count = invalid.size();
  first.fill(1111.);
  second.fill(2222.);
  require(execute(inputs.data(), inputs.size()) == GENERATIVEQC_STATUS_SUCCESS,
          "one invalid native DFT item incorrectly failed the whole batch");
  compare_reference(0, single_forces.data(), single.energy);
  require(
      outputs[1].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
          std::all_of(second.begin(), second.end(), [](double value) { return value == 2222.; }),
      "invalid native DFT batch neighbor published partial forces");
  require(execute() == GENERATIVEQC_STATUS_SUCCESS,
          "valid replay after a failed DFT batch neighbor did not recover");
  compare_reference(0, single_forces.data(), single.energy);
  compare_reference(1, single_forces.data(), single.energy);
  generativeqc_batch_destroy(batch);

  // CPU Direct and other unqualified methods retain the pre-existing
  // fail-closed whole-batch force behavior, without writing caller storage.
  method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  batch = nullptr;
  require(generativeqc_batch_prepare(baseline.context, systems, 2, &method, 0, &batch) ==
                  GENERATIVEQC_STATUS_SUCCESS &&
              batch != nullptr,
          "CPU Direct DFT batch negative-control preparation failed");
  reset_outputs();
  first.fill(1111.);
  second.fill(2222.);
  require(generativeqc_batch_execute(batch, nullptr, 0, outputs.data(), outputs.size()) ==
              GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "unqualified CPU Direct batch forces were admitted");
  require(std::all_of(first.begin(), first.end(), [](double x) { return x == 1111.; }) &&
              std::all_of(second.begin(), second.end(), [](double x) { return x == 2222.; }),
          "unqualified batch request published partial DFT forces");
  generativeqc_batch_destroy(batch);
}

}  // namespace

int main() {
  try {
    test_single_prepared_stationary_sources();
    test_prepared_cpu_df_pbe_analytic_forces();
    test_prepared_cpu_df_pbe_batch_forces();
    std::cout << "Native single-system and batch stationary forces: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
