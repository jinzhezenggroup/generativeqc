#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "molecule/basis.hpp"
#include "runtime/host_component_trace.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

constexpr double kTorqueTolerance = 1.0e-9;

void require(bool condition, const std::string& detail) {
  if (!condition) throw std::runtime_error(detail);
}

core::System water_cation() {
  core::System system;
  system.charge = 1;
  system.multiplicity = 2;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  system.atoms = {{8, {0.0, 0.0, 0.0}},
                  {1, {1.43233673, 0.0, 1.10715266}},
                  {1, {-1.43233673, 0.0, 1.10715266}}};
  system.shells = {
      {0,
       0,
       {{2266.1767785, -0.0053431809926},
        {340.87010191, -0.03989003923},
        {77.363135167, -0.17853911985},
        {21.47964494, -0.46427684959},
        {6.6589433124, -0.44309745172}}},
      {0, 0, {{0.80975975668, 1.0}}},
      {0, 0, {{0.25530772234, 1.0}}},
      {0,
       1,
       {{17.721504317, 0.043394573193},
        {3.863550544, 0.23094120765},
        {1.0480920883, 0.51375311064}}},
      {0, 1, {{0.27641544411, 1.0}}},
      {0, 2, {{1.2, 1.0}}},
      {1, 0, {{13.010701, 0.019682158}, {1.9622572, 0.13796524}, {0.44453796, 0.47831935}}},
      {1, 0, {{0.12194962, 1.0}}},
      {1, 1, {{0.8, 1.0}}},
      {2, 0, {{13.010701, 0.019682158}, {1.9622572, 0.13796524}, {0.44453796, 0.47831935}}},
      {2, 0, {{0.12194962, 1.0}}},
      {2, 1, {{0.8, 1.0}}},
  };
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "H2O+ fixture normalization failed: " + detail);
  require(system.electron_count == 9 && molecule::ao_count(system) == 24,
          "H2O+ fixture electron/AO count changed");
  return system;
}

scf::ScfOptions controls() {
  scf::ScfOptions options;
  options.max_iterations = 150;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  options.screening_tolerance = 1.0e-14;
  options.compute_forces = true;
  options.resolved_fock_build = scf::resolve_fock_build(
      scf::make_hf_fock_spec(scf::FockSpin::Unrestricted, scf::FockApproximation::Exact),
      scf::FockBackend::Cpu, options.screening_tolerance,
      options.density_fitting_relative_threshold);
  return options;
}

double maximum_torque(const core::System& system, const std::vector<double>& forces) {
  require(forces.size() == 3 * system.atoms.size(), "force shape changed");
  std::array<double, 3> torque{};
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    const auto& r = system.atoms[atom].position;
    const double fx = forces[3 * atom + 0];
    const double fy = forces[3 * atom + 1];
    const double fz = forces[3 * atom + 2];
    torque[0] += r[1] * fz - r[2] * fy;
    torque[1] += r[2] * fx - r[0] * fz;
    torque[2] += r[0] * fy - r[1] * fx;
  }
  return std::max({std::abs(torque[0]), std::abs(torque[1]), std::abs(torque[2])});
}

namespace trace = runtime::host_trace;
namespace obs = scf::reference::observation;
thread_local std::size_t reference_final{}, reference_iterations{};
std::size_t observe(const char* name, std::size_t n) noexcept {
  if (std::strcmp(name, "reference_eigensolve") == 0) {
    reference_final += obs::active_reason == obs::EigenReason::final_fock;
    reference_iterations += obs::active_reason == obs::EigenReason::iteration;
  }
  return trace::detail::begin(name, n);
}
void end_observe(std::size_t token, int exceptions) noexcept {
  trace::detail::end(token, exceptions);
}
const obs::Observer observer{observe, end_observe};
struct WorkObservation {
  trace::detail::State state;
  trace::detail::State* previous_state{trace::detail::active};
  const obs::Observer* previous_observer{obs::active};
  WorkObservation() {
    reference_final = reference_iterations = 0;
    trace::detail::active = &state;
    obs::active = &observer;
  }
  ~WorkObservation() {
    trace::detail::active = previous_state;
    obs::active = previous_observer;
  }
  std::size_t count(const char* name) const {
    require(state.valid && state.current == -1, "incomplete work observation");
    return std::count_if(state.regions.begin(), state.regions.end(), [&](const auto& row) {
      require(row.finished && !row.failed, "failed final-state work region");
      return std::strcmp(row.name, name) == 0;
    });
  }
};

void check_route(const char* name, bool reference, bool forces) {
  const auto system = water_cation();
  auto options = controls();
  options.compute_forces = forces;
  WorkObservation work;
  const scf::ScfResult result = reference
                                    ? scf::run_cpu_reference_fock_strategy(system, nullptr, options)
                                    : scf::run_cpu_fock_strategy(system, nullptr, options);
  require(result.converged, std::string(name) + " route did not converge");
  require(reference_iterations == (reference ? 2 * result.iterations : 0),
          std::string(name) + " iteration eigensolver isolation changed");
  const auto corrections = work.count("strict_final_correction");
  const auto focks = work.count("final_state_fock_build");
  if (!forces) {
    require(result.forces.empty() && result.fock_builds == result.iterations + 2 && focks == 0 &&
                corrections == 0 && reference_final == (reference ? 2U : 0U),
            std::string(name) + " energy-only finalization schedule changed");
    return;
  }
  const auto solves = work.count("final_state_correction_solve");
  const auto probes = work.count("final_state_fixed_point");
  const auto promotions = work.count("final_state_fixed_point_promotion");
  require(corrections > 0 && corrections <= 32 && focks == corrections + 1 &&
              solves + promotions == corrections && probes == promotions + 1 &&
              result.fock_builds == result.iterations + 1 + focks &&
              result.fock_builds <= result.iterations + 34 &&
              reference_final == (reference ? 2 * (1 + solves + probes) : 0) &&
              work.count("final_state_weighted_density") == 1,
          std::string(name) + " final-state work/count/provider mismatch");
  require(result.forces.size() == 9, std::string(name) + " route did not publish forces");
  const double torque = maximum_torque(system, result.forces);
  if (!(torque <= kTorqueTolerance)) {
    std::cerr << std::setprecision(17) << name << " max |torque| = " << torque
              << " Eh, gate = " << kTorqueTolerance << " Eh\n";
    throw std::runtime_error(std::string(name) + " route failed H2O+ torque regression");
  }
}

}  // namespace

int main() {
  try {
    for (bool forces : {true, false}) {
      check_route("reference", true, forces);
      check_route("scalar", false, forces);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "UHF final-state regression failed: " << error.what() << '\n';
    return 1;
  }
}
