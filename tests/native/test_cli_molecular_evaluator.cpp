#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cli/molecular_evaluator.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::cli;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void refuses(Operation&& operation,
             const char* message = "invalid molecular derivative binding was admitted") {
  bool failed = false;
  try {
    operation();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, message);
}

void close(std::span<const double> actual, std::span<const double> expected,
           double tolerance = 2e-10) {
  require(actual.size() == expected.size(), "molecular derivative layout changed");
  for (std::size_t index = 0; index < actual.size(); ++index)
    require(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) <= tolerance,
            "molecular derivative differs from explicit native ECP execution");
}

/** Minimal potentials exercise the real consumer, not chemistry qualification. */
struct Fixture {
  RunOptions options;
  SaddleInvocation invocation{};
  std::vector<generativeqc_atom> atoms{{11, 0, 0, 0}, {1, 0.2, -0.1, 3}};
  std::vector<double> masses{22.9897692820, 1.00782503223};
  NativeBasisData basis;
  const method_generated::Method* program{};

  explicit Fixture(bool unrestricted) {
    if (unrestricted) {
      atoms.push_back({1, 2.8, 0.3, -0.2});
      masses.push_back(1.00782503223);
    }
    options.method_name = unrestricted ? "b3lyp-d3(bj)-uks" : "b3lyp-d3(bj)-rks";
    options.multiplicity = unrestricted ? 2 : 1;
    options.max_iterations = 200;
    options.energy_tolerance = 1e-13;
    options.density_tolerance = 1e-11;
    options.grid.version = 1;
    options.grid.radial_points = 12;
    options.grid.angular_polar = 4;
    options.grid.angular_azimuth = 8;
    invocation.controls.raw_symmetry_tolerance = 1e-10;
    invocation.derivatives.response_rtol = 1e-11;
    invocation.derivatives.response_atol = 1e-12;
    invocation.derivatives.response_atol_explicit = true;
    program = method_generated::find_method(options.method_name);
    require(program && generativeqc_ks_hessian_cpu_method_v1(program->identity.data()),
            "missing linked molecular total-method closure");
    basis.representation = GENERATIVEQC_BASIS_SPHERICAL;
    for (std::uint32_t atom = 0; atom < atoms.size(); ++atom) {
      basis.primitives.push_back({atom ? 0.7 : 0.55, 1});
      basis.shells.push_back({atom, 0, atom, 1});
      basis.core_electrons.push_back(atom ? 0 : 10);
    }
    basis.ecp_terms = {{0, -1, 2, 0.8, -2}};
  }

  std::vector<double> coordinates() const {
    std::vector<double> result;
    for (const auto& atom : atoms) result.insert(result.end(), {atom.x, atom.y, atom.z});
    return result;
  }
};

struct Evaluation {
  double energy{};
  std::vector<double> gradient, hessian;
};

Evaluation explicit_batch(const Fixture& fixture, std::span<const double> positions) {
  Context context(generativeqc_context_descriptor{sizeof(generativeqc_context_descriptor),
                                                  GENERATIVEQC_ABI_VERSION, -1,
                                                  GENERATIVEQC_BACKEND_CPU_REFERENCE});
  const auto check = [&](generativeqc_status status) {
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw Error(status, generativeqc_context_get_last_detail(context.get()));
  };
  auto atoms = fixture.atoms;
  for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
    atoms[atom].x = positions[3 * atom];
    atoms[atom].y = positions[3 * atom + 1];
    atoms[atom].z = positions[3 * atom + 2];
  }
  const auto& basis = fixture.basis;
  const generativeqc_system_descriptor geometry{sizeof(geometry),
                                                GENERATIVEQC_ABI_VERSION,
                                                atoms.data(),
                                                static_cast<std::uint32_t>(atoms.size()),
                                                basis.shells.data(),
                                                static_cast<std::uint32_t>(basis.shells.size()),
                                                basis.primitives.data(),
                                                static_cast<std::uint32_t>(basis.primitives.size()),
                                                fixture.options.charge,
                                                fixture.options.multiplicity,
                                                basis.representation};
  System system(context, geometry, basis.core_electrons, basis.ecp_terms);
  generativeqc_ks_options grid{};
  grid.struct_size = sizeof(grid);
  grid.abi_version = GENERATIVEQC_ABI_VERSION;
  grid.grid_version = fixture.options.grid.version;
  grid.radial_points = fixture.options.grid.radial_points;
  grid.angular_polar = fixture.options.grid.angular_polar;
  grid.angular_azimuth = fixture.options.grid.angular_azimuth;
  grid.partition_iterations = fixture.options.grid.partition_iterations;
  grid.coincident_tolerance = fixture.options.grid.coincident_tolerance;
  grid.tile_points = fixture.options.grid.tile_points;
  methods::named_ks::electronic_options(*fixture.program, grid);
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = fixture.program->carrier;
  method.diis_history = fixture.options.diis_history;
  method.ks_options = &grid;
  method.max_iterations = *fixture.options.max_iterations;
  method.energy_tolerance = *fixture.options.energy_tolerance;
  method.density_tolerance = *fixture.options.density_tolerance;
  method.screening_tolerance = 1e-14;
  const generativeqc_named_ks_descriptor named{
      sizeof(named), GENERATIVEQC_ABI_VERSION, fixture.program->name.data(),
      fixture.program->identity.data(), fixture.invocation.derivatives.dispersion_bytes};
  const generativeqc_system* systems[]{system.get()};
  generativeqc_batch* prepared{};
  check(generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 1, &method, &named, 0,
                                               &prepared));
  std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)> batch(
      prepared, generativeqc_batch_destroy);
  Evaluation result{0, std::vector<double>(positions.size()),
                    std::vector<double>(positions.size() * positions.size())};
  generativeqc_batch_item_result_descriptor primal{};
  primal.struct_size = sizeof(primal);
  primal.abi_version = GENERATIVEQC_ABI_VERSION;
  primal.forces = result.gradient.data();
  primal.force_count = result.gradient.size();
  check(generativeqc_batch_execute(batch.get(), nullptr, 0, &primal, 1));
  check(primal.status);
  require(primal.converged, "explicit molecular ECP state did not converge");
  result.energy = primal.energy;
  for (auto& component : result.gradient) component = -component;
  generativeqc_ks_host_hessian_options_v1 controls;
  controls.expected_atoms = atoms.data();
  controls.expected_atom_count = atoms.size();
  controls.expected_charge = fixture.options.charge;
  controls.expected_multiplicity = fixture.options.multiplicity;
  controls.relative_tolerance = fixture.invocation.derivatives.response_rtol;
  controls.absolute_tolerance = fixture.invocation.derivatives.response_atol;
  controls.symmetry_absolute_tolerance = fixture.invocation.controls.raw_symmetry_tolerance;
  controls.symmetry_relative_tolerance = 0;
  generativeqc_ks_host_hessian_receipt_v1 receipt;
  check(generativeqc_ks_hessian_cpu_v1(batch.get(), 0, &controls, nullptr, 0, result.hessian.data(),
                                       result.hessian.size(), &receipt));
  return result;
}

Evaluation evaluate(NativeMolecularEvaluator& owner, std::span<const double> positions) {
  Evaluation result{0, std::vector<double>(positions.size()),
                    std::vector<double>(positions.size() * positions.size())};
  try {
    owner.energy_gradient(positions, result.energy, result.gradient);
    owner.hessian(positions, result.hessian);
  } catch (const std::exception& failure) {
    throw std::runtime_error("molecular consumer at " +
                             molecular_detail::rows(positions, 3).dump() + ": " + failure.what());
  }
  return result;
}

void same(const Evaluation& actual, const Evaluation& expected) {
  require(std::abs(actual.energy - expected.energy) < 2e-11,
          "molecular ECP consumer changed total energy");
  close(actual.gradient, expected.gradient);
  close(actual.hessian, expected.hessian);
}

void contracts(bool unrestricted) {
  Fixture fixture(unrestricted);
  const auto initial = fixture.coordinates();
  auto supplied_basis = fixture.basis;
  NativeMolecularEvaluator owner(fixture.options, fixture.atoms, fixture.masses, fixture.invocation,
                                 *fixture.program, supplied_basis);
  supplied_basis.core_electrons[0] = 0;
  supplied_basis.ecp_terms.clear();
  require(owner.uses_ecp(), "molecular ECP metadata disappeared before execution");
  const auto baseline = evaluate(owner, initial);
  same(baseline, explicit_batch(fixture, initial));
  for (const auto* stage : {"system_prepare", "batch_prepare", "scf_forces"}) {
    const auto started =
        std::find_if(owner.records.begin(), owner.records.end(), [&](const auto& event) {
          return event.at("stage") == stage && event.at("status") == "start";
        });
    const auto completed =
        std::find_if(owner.records.begin(), owner.records.end(), [&](const auto& event) {
          return event.at("stage") == stage && event.at("status") == "complete";
        });
    require(started != owner.records.end() && completed != owner.records.end(),
            "native molecular phase progress is missing");
    require(completed->at("elapsed_seconds").get<double>() >=
                    started->at("elapsed_seconds").get<double>() &&
                completed->at("phase_seconds").get<double>() >= 0,
            "native molecular phase timing is invalid");
    require(started->at("coordinates_bohr") == completed->at("coordinates_bohr"),
            "native molecular phase progress changed geometry");
  }
  auto orbitals = owner.orbital_snapshot(initial, 128ULL << 20);
  const auto saved_coefficients = orbitals.coefficients;
  require(orbitals.shape.spins == (unrestricted ? 2 : 1),
          "molecular diagnostic changed spin layout");
  const auto electrons = unrestricted ? orbitals.shape.alpha_occupied + orbitals.shape.beta_occupied
                                      : 2 * orbitals.shape.alpha_occupied;
  require(electrons == fixture.atoms.size(), "molecular snapshot lost ECP electron counts");
  std::fill(orbitals.coefficients.begin(), orbitals.coefficients.end(), 0);
  close(owner.orbital_snapshot(initial, 128ULL << 20).coefficients, saved_coefficients, 0);
  refuses([&] { owner.orbital_snapshot(initial, 1); });
  owner.accept_geometry(initial);
  auto translated = initial;
  for (double& component : translated) component += 0.125;
  const auto previous_gate = fixture.invocation.minimum_occupied_overlap;
  fixture.invocation.minimum_occupied_overlap = 1 - 1e-10;
  same(evaluate(owner, translated), baseline);
  const auto translated_evaluation =
      std::find_if(owner.records.rbegin(), owner.records.rend(), [](const auto& event) {
        return event.at("stage") == "energy_gradient" && event.at("status") == "complete";
      });
  require(translated_evaluation != owner.records.rend(), "translated state was not retained");
  close(translated_evaluation->at("root_registration_translation_bohr").get<std::vector<double>>(),
        std::vector<double>{0.125, 0.125, 0.125}, 1e-14);
  fixture.invocation.minimum_occupied_overlap = previous_gate;
  auto trial = initial;
  trial[5] += 0.025;
  const auto changed = evaluate(owner, trial);
  same(changed, explicit_batch(fixture, trial));
  close(owner.orbital_snapshot(initial, 128ULL << 20).coefficients, saved_coefficients, 0);
  require(std::abs(changed.energy - baseline.energy) > 1e-7,
          "molecular ECP consumer ignored trial geometry");
  std::vector<double> output(initial.size(), -781.25);
  double energy = -781.25;
  auto remote = initial;
  remote[5] += 6;
  fixture.invocation.minimum_occupied_overlap = 1 - 1e-10;
  refuses([&] { owner.energy_gradient(remote, energy, output); },
          "strict occupied-root gate admitted an internal bond displacement");
  fixture.invocation.minimum_occupied_overlap = previous_gate;
  require(energy == -781.25 && std::all_of(output.begin(), output.end(),
                                           [](double value) { return value == -781.25; }),
          "failed occupied-root trial published energy/gradient");
  same(evaluate(owner, initial), baseline);
  std::vector<double> matrix(baseline.hessian.size(), -781.25);
  owner.energy_gradient(initial, energy, output);
  refuses([&] { owner.hessian(trial, matrix); });
  refuses([&] { owner.orbital_snapshot(trial, 128ULL << 20); });
  refuses([&] { owner.hessian({}, matrix); });
  refuses([&] { owner.energy_gradient({}, energy, output); });
  require(std::all_of(matrix.begin(), matrix.end(), [](double value) { return value == -781.25; }),
          "failed molecular Hessian binding published a result");
  owner.discard_current();
  refuses([&] { owner.hessian(initial, matrix); });
  same(evaluate(owner, initial), baseline);
  owner.energy_gradient(initial, energy, output);
  fixture.invocation.derivatives.max_eri_primitives = 1;
  refuses([&] { owner.hessian(initial, matrix); });
  require(std::all_of(matrix.begin(), matrix.end(), [](double value) { return value == -781.25; }),
          "resource-limited molecular Hessian published a result");
  owner.discard_current();
  fixture.invocation.derivatives.max_eri_primitives = 100'000'000;
  same(evaluate(owner, initial), baseline);
  fixture.invocation.root_bytes = 1;
  std::fill(output.begin(), output.end(), -781.25);
  energy = -781.25;
  refuses([&] { owner.energy_gradient(initial, energy, output); });
  require(energy == -781.25 && std::all_of(output.begin(), output.end(),
                                           [](double value) { return value == -781.25; }),
          "resource-limited occupied-root export published a result");
  fixture.invocation.root_bytes = 128ULL << 20;
  same(evaluate(owner, initial), baseline);
  require(owner.work.at("nuclear_response_solves").get<std::size_t>() > 0 &&
              owner.work.at("d3_pair_visits").get<std::size_t>() > 0,
          "molecular consumer omitted response/correction work");
  Fixture failed_fixture(unrestricted);
  failed_fixture.options.max_iterations = 1;
  NativeMolecularEvaluator failed_owner(failed_fixture.options, failed_fixture.atoms,
                                        failed_fixture.masses, failed_fixture.invocation,
                                        *failed_fixture.program, failed_fixture.basis);
  std::fill(output.begin(), output.end(), -781.25);
  energy = -781.25;
  refuses([&] { failed_owner.energy_gradient(initial, energy, output); });
  require(energy == -781.25 && std::all_of(output.begin(), output.end(),
                                           [](double value) { return value == -781.25; }),
          "failed SCF phase published energy/gradient");
  require(failed_owner.records.back().at("stage") == "scf_forces" &&
              failed_owner.records.back().at("status") == "failed" &&
              !failed_owner.records.back().at("detail").get<std::string>().empty(),
          "failed native SCF phase lost its progress diagnostic");
}
}  // namespace

int main() {
  try {
    contracts(false);
    contracts(true);
    std::cout << "native molecular ECP frame/root/Hessian contracts passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
