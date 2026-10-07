#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/minao.hpp"
#include "scf/mean_field.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/reference/observation.hpp"

namespace {
using namespace generativeqc;
using namespace scf::initial_guess;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class F>
void invalid(F&& action) {
  try {
    action();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid preliminary request was accepted");
}
core::System water() {
  core::System s;
  s.multiplicity = 1;
  s.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  s.atoms = {{8, {0, 0, 0}}, {1, {0, -1.43233673, 1.10715266}}, {1, {0, 1.43233673, 1.10715266}}};
  s.shells = {
      {0, 0, {{130.70932, .15432897}, {23.808861, .53532814}, {6.4436083, .44463454}}},
      {0, 0, {{5.0331513, -.09996723}, {1.1695961, .39951283}, {.3803890, .70011547}}},
      {0, 1, {{5.0331513, .15591627}, {1.1695961, .60768372}, {.3803890, .39195739}}},
      {1, 0, {{3.425250914, .1543289673}, {.6239137298, .5353281423}, {.168855404, .4446345422}}},
      {2, 0, {{3.425250914, .1543289673}, {.6239137298, .5353281423}, {.168855404, .4446345422}}},
  };
  std::string detail;
  require(molecule::validate_and_normalize(s, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "bad fixture");
  return s;
}
void matched(const scf::ScfResult& a, const scf::ScfResult& b) {
  require(a.converged && b.converged, "SCF did not converge");
  require(std::abs(a.energy - b.energy) < 1e-8, "target energy changed");
  require(a.density.size() == b.density.size(), "target density shape changed");
  for (size_t i = 0; i < a.density.size(); ++i)
    require(std::abs(a.density[i] - b.density[i]) < 1e-6, "target SCF branch changed");
}
std::size_t observed_preparation_calls = 0;
std::size_t observe_preparation(const char*, std::size_t) noexcept {
  return ++observed_preparation_calls;
}
void finish_observation(std::size_t, int) noexcept {}
void check() {
  require(!preliminary_options(nullptr), "default must remain disabled");
  generativeqc_initial_guess_options descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.kind = GENERATIVEQC_INITIAL_GUESS_HF;
  auto parsed = preliminary_options(&descriptor);
  require(parsed && parsed->max_iterations == 32, "bounded descriptor defaults changed");
  descriptor.max_iterations = 65;
  invalid([&] { preliminary_options(&descriptor); });
  descriptor.max_iterations = 0;
  descriptor.energy_tolerance = std::numeric_limits<double>::quiet_NaN();
  invalid([&] { preliminary_options(&descriptor); });
  descriptor.energy_tolerance = 0;
  descriptor.radial_points = 8;
  invalid([&] { preliminary_options(&descriptor); });
  descriptor.radial_points = 0;
  descriptor.kind = GENERATIVEQC_INITIAL_GUESS_MINAO;
  parsed = preliminary_options(&descriptor);
  require(parsed && parsed->kind == PreliminaryKind::Minao, "MINAO descriptor was not parsed");
  descriptor.radial_points = 8;
  invalid([&] { preliminary_options(&descriptor); });
  descriptor.radial_points = 0;
  descriptor.kind = static_cast<generativeqc_initial_guess_kind>(99);
  invalid([&] { preliminary_options(&descriptor); });

  const auto system = water();
  auto spec = scf::make_global_hybrid_fock_spec(scf::FockSpin::Restricted, .25);
  scf::PreparedFockPlan plan(system, nullptr, scf::resolve_fock_build(spec, scf::FockBackend::Cpu));
  dft::AoBasis basis(system);
  dft::MolecularGrid grid(system, {1, 16, 8, 16, 3, 1e-12});
  scf::ScfOptions controls;
  controls.compute_forces = false;
  controls.precision_mode = GENERATIVEQC_PRECISION_FP64;
  controls.semilocal_exchange_scale = .75;
  const auto baseline = scf::run_pbe_rks(plan, basis, grid, controls);
  require(!baseline.preliminary_guess.requested_kind, "default ran preliminary SCF");
  require(admit_preliminary_density(plan, baseline.density) == baseline.density,
          "valid candidate was repaired instead of transferred");
  auto bad_density = baseline.density;
  for (auto& value : bad_density) value *= 1.1;
  invalid([&] { admit_preliminary_density(plan, bad_density); });
  bad_density = baseline.density;
  bad_density[1] += 1e-3;
  invalid([&] { admit_preliminary_density(plan, bad_density); });
  bad_density = baseline.density;
  bad_density[0] = std::numeric_limits<double>::quiet_NaN();
  invalid([&] { admit_preliminary_density(plan, bad_density); });
  for (auto kind : {PreliminaryKind::HartreeFock, PreliminaryKind::Lda}) {
    controls.preliminary_guess = PreliminaryOptions{};
    controls.preliminary_guess->kind = kind;
    const auto seeded = scf::run_pbe_rks(plan, basis, grid, controls);
    matched(baseline, seeded);
    require(
        seeded.preliminary_guess.outcome == PreliminaryOutcome::Used && seeded.initial_density_used,
        "preliminary provider was not used");
    require(seeded.preliminary_guess.target_attempts == 1 &&
                seeded.preliminary_guess.preliminary_iterations > 0,
            "preliminary work census missing");
    require(seeded.preliminary_guess.work_counters_complete, "completed solve has partial census");
    const auto warm = scf::run_pbe_rks(plan, basis, grid, controls, &baseline.density);
    matched(baseline, warm);
    require(warm.preliminary_guess.outcome == PreliminaryOutcome::ExplicitDensity &&
                warm.preliminary_guess.preliminary_iterations == 0 &&
                warm.preliminary_guess.preparation_seconds == 0,
            "explicit density lost priority");
    const auto exact_cap = preliminary_numeric_capacity(system, *controls.preliminary_guess);
    controls.preliminary_guess->maximum_numeric_bytes = exact_cap;
    const auto boundary = scf::run_pbe_rks(plan, basis, grid, controls);
    require(boundary.preliminary_guess.outcome == PreliminaryOutcome::Used,
            "exact numeric cap was not admitted");
    controls.preliminary_guess->maximum_numeric_bytes = exact_cap - 1;
    const auto below = scf::run_pbe_rks(plan, basis, grid, controls);
    require(below.preliminary_guess.outcome == PreliminaryOutcome::BudgetSkipped,
            "one byte below numeric cap was admitted");
    controls.preliminary_guess->maximum_numeric_bytes = 1;
    const auto budget = scf::run_pbe_rks(plan, basis, grid, controls);
    matched(baseline, budget);
    require(budget.preliminary_guess.outcome == PreliminaryOutcome::BudgetSkipped &&
                !budget.initial_density_used,
            "budget did not keep core fallback");
    controls.preliminary_guess->maximum_numeric_bytes = 256U << 20;
    controls.preliminary_guess->max_iterations = 1;
    const auto failed = scf::run_pbe_rks(plan, basis, grid, controls);
    matched(baseline, failed);
    require(failed.preliminary_guess.outcome == PreliminaryOutcome::PreparationFailed &&
                !failed.initial_density_used,
            "failed preparation did not keep core fallback");
  }

  controls.preliminary_guess = PreliminaryOptions{};
  controls.preliminary_guess->kind = PreliminaryKind::Minao;
  const auto& ints = plan.one_electron();
  const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
  const auto raw_minao = minao_density(system, ints, x);
  require(raw_minao.source_aos == 7 && raw_minao.source_primitives > 0,
          "water MINAO source inventory changed");
  require(std::abs(raw_minao.source_electrons - 10.0) < 1e-12,
          "water MINAO atomic occupations changed");
  require(raw_minao.density.size() == ints.nbf * ints.nbf &&
              std::isfinite(raw_minao.projected_electrons),
          "MINAO projection shape/trace is invalid");
  invalid([&] {
    admit_preliminary_density(plan, normalized_warm_density(system, ints, raw_minao.density));
  });
  const auto admitted_minao = admissible_minao_density(system, ints, x, raw_minao.density);
  require(admit_preliminary_density(plan, admitted_minao) == admitted_minao,
          "MINAO construction did not satisfy the unchanged strict seed gate");
  require(std::abs(raw_minao.projected_electrons - 9.860917687841592) < 1e-10,
          "raw pinned MINAO projection changed");
  const auto minao = scf::run_pbe_rks(plan, basis, grid, controls);
  matched(baseline, minao);
  require(minao.preliminary_guess.outcome == PreliminaryOutcome::Used &&
              minao.preliminary_guess.preliminary_iterations == 0 &&
              minao.preliminary_guess.preliminary_fock_builds == 0 && minao.initial_density_used,
          "MINAO did not remain a zero-Fock initial-density provider");
  const auto minao_cap = preliminary_numeric_capacity(system, *controls.preliminary_guess);
  require(minao_cap > 0 && minao_cap < (256U << 20), "MINAO capacity is not bounded");
  controls.preliminary_guess->maximum_numeric_bytes = minao_cap;
  const auto minao_boundary = scf::run_pbe_rks(plan, basis, grid, controls);
  matched(baseline, minao_boundary);
  require(minao_boundary.preliminary_guess.outcome == PreliminaryOutcome::Used,
          "exact MINAO numeric cap was not admitted");
  controls.preliminary_guess->maximum_numeric_bytes = minao_cap - 1;
  scf::reference::observation::Observer observer{observe_preparation, finish_observation};
  scf::reference::observation::active = &observer;
  PreliminaryDiagnostic skipped_diagnostic;
  const auto skipped =
      prepare_preliminary_density(plan, *controls.preliminary_guess, skipped_diagnostic);
  scf::reference::observation::active = nullptr;
  require(!skipped && observed_preparation_calls == 0 &&
              skipped_diagnostic.outcome == PreliminaryOutcome::BudgetSkipped &&
              skipped_diagnostic.preparation_numeric_capacity == minao_cap,
          "MINAO budget was not checked before numeric preparation");
  const auto minao_budget = scf::run_pbe_rks(plan, basis, grid, controls);
  matched(baseline, minao_budget);
  require(minao_budget.preliminary_guess.outcome == PreliminaryOutcome::BudgetSkipped &&
              !minao_budget.initial_density_used,
          "MINAO budget did not keep core fallback");
  controls.preliminary_guess->maximum_numeric_bytes = 256U << 20;

  unsigned calls = 0;
  // Explicit control-flow doubles, not numerical evidence: discarded attempts
  // must never publish success or leave a proposed density attached to retry.
  auto retried = run_with_preliminary_guess(plan, controls, nullptr, 0, [&](const auto* seed) {
    ++calls;
    require((calls == 1) == (seed != nullptr), "core retry retained candidate seed");
    auto r = baseline;
    r.converged = calls == 2;
    return r;
  });
  require(calls == 2 && retried.converged && retried.preliminary_guess.target_attempts == 2 &&
              retried.preliminary_guess.outcome == PreliminaryOutcome::TargetRetried &&
              retried.preliminary_guess.discarded_target_iterations == baseline.iterations,
          "nonconvergence retry was not bounded/counted");
  calls = 0;
  retried = run_with_preliminary_guess(plan, controls, nullptr, 0, [&](const auto*) {
    if (++calls == 1) throw std::runtime_error("synthetic numerical failure");
    return baseline;
  });
  require(calls == 2 && !retried.preliminary_guess.work_counters_complete,
          "exception census was certified");
  calls = 0;
  try {
    (void)run_with_preliminary_guess(plan, controls, nullptr, 0,
                                     [&](const auto*) -> scf::ScfResult {
                                       ++calls;
                                       throw std::bad_alloc();
                                     });
    throw std::runtime_error("allocation failure swallowed");
  } catch (const std::bad_alloc&) {
    require(calls == 1, "allocation failure retried");
  }
  auto force = controls;
  force.compute_forces = true;
  validate_preliminary_target(system, plan.strategy(), force);
  force.preliminary_guess->kind = PreliminaryKind::HartreeFock;
  invalid([&] { validate_preliminary_target(system, plan.strategy(), force); });
  auto spin = system;
  spin.multiplicity = 3;
  invalid([&] { validate_preliminary_target(spin, plan.strategy(), controls); });
  auto ecp = system;
  ecp.atoms[0].ecp_core = 2;
  invalid([&] { validate_preliminary_target(ecp, plan.strategy(), controls); });
  auto mixed = controls;
  mixed.precision_mode = GENERATIVEQC_PRECISION_AUTO;
  invalid([&] { validate_preliminary_target(system, plan.strategy(), mixed); });

  // The exact same optional policy is shared by the ordinary RHF target.
  spec = scf::make_hf_fock_spec(scf::FockSpin::Restricted);
  spec.derivative_order = 0;
  scf::PreparedFockPlan hf(system, nullptr, scf::resolve_fock_build(spec, scf::FockBackend::Cpu));
  controls.resolved_fock_build = hf.strategy();
  controls.preliminary_guess.reset();
  const auto cold_hf = scf::run_prepared_fock_strategy(hf, controls);
  controls.preliminary_guess = PreliminaryOptions{};
  matched(cold_hf, scf::run_prepared_fock_strategy(hf, controls));
}
}  // namespace
int main() {
  try {
    check();
    std::cout << "preliminary SCF contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
