#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/stationary_integral_gradient.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Work admission runs on shape alone, before allocating any molecular ERI.
 * Unequal contractions and near-limit requests catch AO-count substitutions. */
void work_admission() {
  generativeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {{0, 0, {}}, {1, 0, {}}};
  const auto admitted = [&](std::size_t first, std::size_t second, bool pbe0,
                            std::size_t points = 1000, std::size_t tile = 256) {
    system.shells[0].primitives.assign(first, {0.8, 1.0});
    system.shells[1].primitives.assign(second, {0.6, 1.0});
    return generativeqc::scf::cpu_stationary_direct_force_work_admitted(system, pbe0, points, tile);
  };
  require(admitted(19, 18, false) && !admitted(19, 19, false),
          "PBE primitive work admission lost its two-million-record ceiling");
  require(admitted(16, 15, true) && !admitted(16, 16, true),
          "PBE0 primitive work admission lost its independent exchange charge");
  require(admitted(1, 1, false, 1000000) && !admitted(1, 1, false, 1000001),
          "Direct grid point admission lost its ceiling");
  require(!admitted(1, 1, false, 10, 0), "Direct force accepted a zero-sized XC tile");
  system.atoms.resize(11);
  require(admitted(1, 1, false, 900000) && !admitted(1, 1, false, 910000),
          "Direct Becke pair work admission lost its ceiling");
  system.shells[0].angular_momentum = 3;
  require(!admitted(1, 1, false), "Direct force advertised an unqualified f shell");
}

/** A one-atom source still owns two twelve-slot channel publications. Their
 * storage cannot be charged as molecular coordinates or omitted for zeros. */
void one_atom_capacity() {
  if (!generativeqc::integrals::has_packaged_cpu_eri_shell_derivatives()) return;
  generativeqc::core::System system;
  system.atoms = {{2, {0.1, -0.2, 0.3}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "one-atom stationary capacity fixture is invalid");
  for (const bool pbe0 : {false, true}) {
    auto spec = generativeqc::scf::make_hf_fock_spec(generativeqc::scf::FockSpin::Restricted);
    spec.derivative_order = 0;
    spec.exchange.present = pbe0;
    spec.exchange.coefficient = -0.125;
    const auto strategy =
        generativeqc::scf::resolve_fock_build(spec, generativeqc::scf::FockBackend::Cpu);
    const generativeqc::scf::PreparedFockPlan target(system, nullptr, strategy);
    const std::array<double, 1> density{1.8}, weighted{-0.4};
    const auto result = generativeqc::scf::cpu_stationary_integral_gradient(
        target, density, weighted, pbe0, 1 << 20);
    require(result.sources.size() == 12 && result.shell_quartets == 1 &&
                result.primitive_evaluations == 1,
            "one-atom source fabricated molecular storage or repeated J/K primitive work");
    require(std::all_of(result.sources.begin(), result.sources.end(),
                        [](double value) { return std::abs(value) < 1e-10; }),
            "one-atom source violates translation invariance");
    require(result.numeric_capacity >= (21 + (pbe0 ? 26 : 13)) * sizeof(double),
            "one-atom capacity omitted owned matrix/source/channel buffers");
    const auto exact = generativeqc::scf::cpu_stationary_integral_gradient(
        target, density, weighted, pbe0, result.numeric_capacity);
    require(exact.sources == result.sources, "exact capacity changed the stationary source");
    bool rejected = false;
    try {
      (void)generativeqc::scf::cpu_stationary_integral_gradient(target, density, weighted, pbe0,
                                                                result.numeric_capacity - 1);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    require(rejected, "one-atom source ignored its below-limit capacity gate");
  }
}

/** Compare portable two-spin AD/AOT against the independent retained Jet owner.
 * Unequal alpha/beta D/W and nonzero molecular motion expose spin factors and
 * exchange pairing; the recurrence and source weights are separate programs.
 */
void unrestricted_sources() {
  if (!generativeqc::integrals::has_packaged_cpu_eri_shell_derivatives()) return;
  generativeqc::core::System system;
  system.atoms = {{1, {0.12, -0.09, -0.71}}, {1, {-0.08, 0.13, 0.79}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
  std::string detail;
  require(
      generativeqc::molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
      "UKS source fixture is invalid");
  const std::vector<double> alpha{0.7, 0.1, 0.1, 0.2}, beta{0.3, -0.03, -0.03, 0.25};
  const std::vector<double> weighted_alpha{-0.4, 0.02, 0.02, -0.07},
      weighted_beta{-0.1, 0.04, 0.04, 0.12};
  for (const bool pbe0 : {false, true}) {
    auto spec = generativeqc::scf::make_hf_fock_spec(generativeqc::scf::FockSpin::Unrestricted);
    spec.derivative_order = 0;
    spec.exchange.present = pbe0;
    spec.exchange.coefficient = -0.25;
    const generativeqc::scf::PreparedFockPlan target(
        system, nullptr,
        generativeqc::scf::resolve_fock_build(spec, generativeqc::scf::FockBackend::Cpu));
    const auto actual = generativeqc::scf::cpu_stationary_integral_gradient(
        target, alpha, weighted_alpha, pbe0, 1 << 20, beta, weighted_beta);
    spec.derivative_order = 1;
    const generativeqc::scf::PreparedFockPlan reference(
        system, nullptr,
        generativeqc::scf::resolve_fock_build(spec, generativeqc::scf::FockBackend::Cpu));
    const auto two = reference.energy_derivative_components(alpha, beta);
    const auto& one = reference.one_electron();
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
      double hcore = 0, pulay = 0;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        hcore += (alpha[slot] + beta[slot]) * one.hcore_derivative[4 * coordinate + slot];
        pulay -= (weighted_alpha[slot] + weighted_beta[slot]) *
                 one.overlap_derivative[4 * coordinate + slot];
      }
      const std::array<double, 4> expected{hcore, pulay, two.coulomb[coordinate],
                                           two.exchange[coordinate]};
      for (std::size_t source = 0; source < expected.size(); ++source)
        require(std::abs(actual.sources[6 * source + coordinate] - expected[source]) < 2e-10,
                "UKS stationary AD/AOT disagrees with independent Jet source");
    }
    require(actual.shell_quartets == 16 && actual.primitive_evaluations == 16,
            "UKS duplicated J/K or spin primitive recurrence work");
    const auto exact = generativeqc::scf::cpu_stationary_integral_gradient(
        target, alpha, weighted_alpha, pbe0, actual.numeric_capacity, beta, weighted_beta);
    require(exact.sources == actual.sources, "UKS exact capacity changed sources");
    bool rejected = false;
    try {
      (void)generativeqc::scf::cpu_stationary_integral_gradient(
          target, alpha, weighted_alpha, pbe0, actual.numeric_capacity - 1, beta, weighted_beta);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    require(rejected, "UKS below-limit capacity was admitted");
    rejected = false;
    try {
      (void)generativeqc::scf::cpu_stationary_integral_gradient(target, alpha, weighted_alpha, pbe0,
                                                                1 << 20);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "UKS absent beta blocks were silently treated as RKS");
    auto nonfinite = weighted_beta;
    nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
      (void)generativeqc::scf::cpu_stationary_integral_gradient(target, alpha, weighted_alpha, pbe0,
                                                                1 << 20, beta, nonfinite);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "UKS nonfinite beta W reached source execution");
  }
}
}  // namespace

int main() {
  try {
    work_admission();
    one_atom_capacity();
    unrestricted_sources();
    std::cout << "Native Direct stationary admission/capacity: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
