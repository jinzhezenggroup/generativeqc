#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/fock.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void checked(generativeqc_fock_plan* plan, generativeqc_status status) {
  // Read the error string after execution; a prior c_str pointer can be
  // invalidated when execution replaces the detail buffer.
  require(status == GENERATIVEQC_STATUS_SUCCESS, generativeqc_fock_plan_last_error(plan));
}

// Only public handles cross this test boundary. Release context/system and
// mutate caller inputs before evaluating the owned source under sanitizers.
void exercise(generativeqc_backend backend, bool unrestricted, bool fitted) {
  generativeqc_context_descriptor context_spec{sizeof(context_spec), GENERATIVEQC_ABI_VERSION, 0,
                                               backend};
  generativeqc_context* context = nullptr;
  require(generativeqc_context_create(&context_spec, &context) == 0, "context create");
  std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  std::array<generativeqc_primitive, 2> primitives{{{1, 1}, {1, 1}}};
  std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
  generativeqc_system_descriptor system_spec{sizeof(system_spec),  GENERATIVEQC_ABI_VERSION,
                                             atoms.data(),         2,
                                             shells.data(),        2,
                                             primitives.data(),    2,
                                             unrestricted ? 1 : 0, unrestricted ? 2U : 1U};
  generativeqc_system* system = nullptr;
  require(generativeqc_system_create(context, &system_spec, &system) == 0, "system create");
  generativeqc_fock_spec spec{
      sizeof(spec),
      GENERATIVEQC_ABI_VERSION,
      1,
      unrestricted ? GENERATIVEQC_FOCK_UNRESTRICTED : GENERATIVEQC_FOCK_RESTRICTED,
      1,
      {1, 1, GENERATIVEQC_FOCK_FULL_RANGE, 0,
       fitted ? GENERATIVEQC_FOCK_DENSITY_FITTED : GENERATIVEQC_FOCK_EXACT},
      {1, unrestricted ? -1.0 : -0.5, GENERATIVEQC_FOCK_FULL_RANGE, 0, GENERATIVEQC_FOCK_EXACT}};
  generativeqc_fock_plan* plan = nullptr;
  require(generativeqc_fock_plan_create(context, system, nullptr, &spec, nullptr, &plan) == 0,
          "plan create");
  auto invalid = spec;
  invalid.abi_version += 1;
  auto* failed = plan;
  require(generativeqc_fock_plan_create(context, system, nullptr, &invalid, nullptr, &failed) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              failed == nullptr,
          "ABI failure handle");
  // Invalid controls fail before source preparation even when a DF cutoff
  // would be mathematically irrelevant to this exact-only request.
  for (double value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
    for (bool metric : {false, true}) {
      generativeqc_fock_controls controls{sizeof(controls), GENERATIVEQC_ABI_VERSION, 1e-12, 0, 0};
      (metric ? controls.metric_relative_threshold : controls.screening_tolerance) = value;
      failed = plan;
      require(generativeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) ==
                      GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                  failed == nullptr,
              "invalid Fock controls must not publish a plan");
    }
  }
  generativeqc_fock_controls controls{sizeof(controls), GENERATIVEQC_ABI_VERSION, 1e-12, 1.0, 0};
  require(generativeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              failed == nullptr,
          "unit metric cutoff rejected");
  controls.metric_relative_threshold = 0;
  require(generativeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) == 0,
          "zero metric cutoff retains default behavior");
  generativeqc_fock_diagnostic defaulted{sizeof(defaulted), GENERATIVEQC_ABI_VERSION};
  require(generativeqc_fock_plan_diagnostic(failed, &defaulted) == 0 &&
              defaulted.metric_relative_threshold == (fitted ? 1e-10 : 0.0),
          "default metric cutoff diagnostic");
  generativeqc_fock_plan_destroy(failed);
  generativeqc_system_destroy(system);
  generativeqc_context_destroy(context);
  atoms[1].z = 70;
  primitives[0].exponent = 19;
  generativeqc_fock_diagnostic diag{sizeof(diag), GENERATIVEQC_ABI_VERSION};
  require(generativeqc_fock_plan_diagnostic(plan, &diag) == 0 && diag.nbf == 2 &&
              diag.backend == backend,
          "diagnostics after source release");

  std::array<double, 8> density{};
  std::array<double, 6> forces{};
  generativeqc_fock_scf_result out{sizeof(out), GENERATIVEQC_ABI_VERSION};
  out.density = density.data();
  out.density_count = unrestricted ? 8 : 4;
  checked(plan, generativeqc_fock_plan_solve(plan, nullptr, nullptr, 0, &out));
  const double energy = out.energy;
  out.forces = forces.data();
  out.force_count = forces.size();
  // Deliberate input/output alias is safe: a complete seed snapshot is owned.
  checked(plan,
          generativeqc_fock_plan_solve(plan, nullptr, density.data(), out.density_count, &out));
  require(out.initial_density_used && std::abs(out.energy - energy) < 1e-10, "warm result");
  require(std::abs(forces[2] + forces[5]) < 1e-10, "force invariance");
  const auto saved_density = density;
  const auto saved_forces = forces;
  const auto saved_out = out;
  generativeqc_fock_scf_controls limited{
      sizeof(limited), GENERATIVEQC_ABI_VERSION, 1, 8, 1e-10, 1e-8};
  require(generativeqc_fock_plan_solve(plan, &limited, nullptr, 0, &out) ==
              GENERATIVEQC_STATUS_NOT_CONVERGED,
          "nonconverged status");
  require(std::memcmp(&out, &saved_out, sizeof(out)) == 0 && density == saved_density &&
              forces == saved_forces,
          "failure publication");
  out.forces = density.data();
  require(generativeqc_fock_plan_solve(plan, nullptr, nullptr, 0, &out) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "overlapping SCF outputs");
  out = saved_out;

  std::array<double, 4> j{}, ka{}, kb{}, fa{}, fb{};
  generativeqc_fock_result fixed{sizeof(fixed), GENERATIVEQC_ABI_VERSION};
  fixed.matrix_count = 4;
  fixed.coulomb = j.data();
  fixed.exchange_alpha = ka.data();
  fixed.exchange_beta = unrestricted ? kb.data() : nullptr;
  fixed.fock_alpha = fa.data();
  fixed.fock_beta = unrestricted ? fb.data() : nullptr;
  fixed.gradient = forces.data();
  fixed.gradient_count = 6;
  checked(plan, generativeqc_fock_plan_evaluate(plan, density.data(), 4,
                                                unrestricted ? density.data() + 4 : nullptr,
                                                unrestricted ? 4 : 0, &fixed));
  require(std::abs(fixed.energy_one_electron + fixed.energy_two_electron + fixed.nuclear_repulsion -
                   energy) < 1e-10,
          "fixed-density energy");
  generativeqc_fock_plan_destroy(plan);
  generativeqc_fock_plan_destroy(nullptr);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto backend = argc == 2 && std::string(argv[1]) == "cuda"
                             ? GENERATIVEQC_BACKEND_CUDA
                             : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    for (bool spin : {false, true})
      for (bool fitted : {false, true}) exercise(backend, spin, fitted);
    std::cout << "public Fock lifecycle and SCF passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
