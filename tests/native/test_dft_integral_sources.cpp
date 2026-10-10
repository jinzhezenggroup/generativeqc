#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

  Environment() {
    const generativeqc_context_descriptor context_options{sizeof(generativeqc_context_descriptor),
                                                          GENERATIVEQC_ABI_VERSION, 0,
                                                          GENERATIVEQC_BACKEND_CPU_REFERENCE};
    require(generativeqc_context_create(&context_options, &context) == GENERATIVEQC_STATUS_SUCCESS,
            "failed to create native CPU context");
    const std::array<generativeqc_atom, 2> atoms{{{1, 0.0, 0.0, -0.7}, {1, 0.0, 0.0, 0.7}}};
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

  void execute_energy() const {
    generativeqc_result_descriptor result{};
    result.struct_size = sizeof(result);
    result.abi_version = GENERATIVEQC_ABI_VERSION;
    require(generativeqc_calculation_execute(calculation, &result) == GENERATIVEQC_STATUS_SUCCESS &&
                result.converged && std::isfinite(result.energy),
            "native PBE SCF did not converge before stationary source read");
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
}  // namespace

int main() {
  try {
    test_single_prepared_stationary_sources();
    std::cout << "Native single-system stationary integral sources: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
