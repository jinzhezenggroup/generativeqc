#include "scf/preliminary_guess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/minao.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/solver/proposal_control.hpp"

namespace generativeqc::scf::initial_guess {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::optional<std::vector<double>> prepare_impl(const PreparedFockPlan& target,
                                                const PreliminaryOptions& policy,
                                                PreliminaryDiagnostic& diagnostic) {
  const auto& system = target.system();
  diagnostic.preparation_numeric_capacity = preliminary_numeric_capacity(system, policy);
  if (diagnostic.preparation_numeric_capacity > policy.maximum_numeric_bytes) {
    diagnostic.outcome = PreliminaryOutcome::BudgetSkipped;
    return std::nullopt;
  }
  if (policy.kind == PreliminaryKind::Minao) {
    const auto& ints = target.one_electron();
    const auto x = reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
    auto projected = minao_density(system, ints, x);
    auto density = normalized_warm_density(system, ints, projected.density);
    diagnostic.preliminary_iterations = 0;
    diagnostic.preliminary_fock_builds = 0;
    diagnostic.outcome = PreliminaryOutcome::Used;
    return admit_preliminary_density(target, std::move(density));
  }
  // The CPU integral provider admits g, but the LDA AO-grid primitive only
  // admits through f. Decline before building an unusable extra integral owner.
  if (policy.kind == PreliminaryKind::Lda &&
      std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    diagnostic.outcome = PreliminaryOutcome::PreparationFailed;
    return std::nullopt;
  }
  FockBuildSpec spec = make_hf_fock_spec(FockSpin::Restricted, FockApproximation::Exact);
  spec.derivative_order = 0;
  if (policy.kind == PreliminaryKind::Lda) spec.exchange.present = false;
  const PreparedFockPlan source(system, nullptr, resolve_fock_build(spec, FockBackend::Cpu));
  // Build fresh controls: no target functional coefficients, target hooks,
  // warm-state publication, reference export, or recursive preparation policy.
  ScfOptions controls;
  controls.compute_forces = false;
  controls.precision_mode = GENERATIVEQC_PRECISION_FP64;
  controls.max_iterations = policy.max_iterations;
  controls.diis_history = policy.diis_history;
  controls.energy_tolerance = policy.energy_tolerance;
  controls.density_tolerance = policy.density_tolerance;
  controls.resolved_fock_build = source.strategy();
  ScfResult preliminary;
  if (policy.kind == PreliminaryKind::HartreeFock) {
    preliminary = run_prepared_fock_strategy(source, controls);
  } else {
    const dft::AoBasis basis(system);
    const dft::MolecularGrid grid(
        system, {1, policy.radial_points, policy.angular_polar, policy.angular_azimuth, 3, 1e-12});
    preliminary =
        run_curated_semilocal_ks(source, basis, grid, controls, dft::SemilocalFamily::Lda, 1);
  }
  diagnostic.preliminary_iterations = preliminary.iterations;
  diagnostic.preliminary_fock_builds = preliminary.fock_builds;
  if (!preliminary.converged) {
    diagnostic.outcome = PreliminaryOutcome::PreparationFailed;
    return std::nullopt;
  }
  // Same AO basis/geometry: do not repair an invalid source charge or
  // symmetry before admission. Metric transport is a different contract.
  auto density = admit_preliminary_density(target, std::move(preliminary.density));
  diagnostic.outcome = PreliminaryOutcome::Used;
  return density;
}
}  // namespace

std::optional<std::vector<double>> prepare_preliminary_density(
    const PreparedFockPlan& target, const PreliminaryOptions& policy,
    PreliminaryDiagnostic& diagnostic) {
  return prepare_impl(target, policy, diagnostic);
}

std::vector<double> admit_preliminary_density(const PreparedFockPlan& target,
                                              std::vector<double> density) {
  solver::validate_seed(target.one_electron().overlap, density, target.one_electron().nbf,
                        {static_cast<unsigned>(target.system().electron_count)}, 2.0);
  return density;
}

void validate_preliminary_options(const PreliminaryOptions& options) {
  if (options.kind != PreliminaryKind::HartreeFock && options.kind != PreliminaryKind::Lda &&
      options.kind != PreliminaryKind::Minao)
    throw std::invalid_argument("unknown preliminary initial-density provider");
  if (!options.max_iterations || options.max_iterations > 64 || !options.diis_history ||
      options.diis_history > 16 || !(options.energy_tolerance > 0) ||
      !std::isfinite(options.energy_tolerance) || !(options.density_tolerance > 0) ||
      !std::isfinite(options.density_tolerance) || !options.maximum_numeric_bytes ||
      options.maximum_numeric_bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("invalid bounded preliminary SCF controls");
  if (options.kind == PreliminaryKind::Lda &&
      (options.radial_points < 2 || options.radial_points > 32 || options.angular_polar < 2 ||
       options.angular_polar > 16 || options.angular_azimuth < 4 || options.angular_azimuth > 32))
    throw std::invalid_argument("preliminary LDA grid exceeds the admitted coarse-grid bounds");
}

std::optional<PreliminaryOptions> preliminary_options(
    const generativeqc_initial_guess_options* descriptor) {
  if (!descriptor) return std::nullopt;
  if (descriptor->struct_size < sizeof(*descriptor) ||
      descriptor->abi_version != GENERATIVEQC_ABI_VERSION)
    throw std::invalid_argument("preliminary SCF descriptor ABI mismatch");
  PreliminaryOptions options;
  options.kind = static_cast<PreliminaryKind>(descriptor->kind);
  if (descriptor->max_iterations) options.max_iterations = descriptor->max_iterations;
  if (descriptor->diis_history) options.diis_history = descriptor->diis_history;
  if (descriptor->energy_tolerance != 0) options.energy_tolerance = descriptor->energy_tolerance;
  if (descriptor->density_tolerance != 0) options.density_tolerance = descriptor->density_tolerance;
  if (descriptor->maximum_numeric_bytes) {
    if (descriptor->maximum_numeric_bytes > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("preliminary SCF capacity exceeds size_t");
    options.maximum_numeric_bytes = static_cast<std::size_t>(descriptor->maximum_numeric_bytes);
  }
  if (descriptor->radial_points) options.radial_points = descriptor->radial_points;
  if (descriptor->angular_polar) options.angular_polar = descriptor->angular_polar;
  if (descriptor->angular_azimuth) options.angular_azimuth = descriptor->angular_azimuth;
  if ((options.kind == PreliminaryKind::HartreeFock || options.kind == PreliminaryKind::Minao) &&
      (descriptor->radial_points || descriptor->angular_polar || descriptor->angular_azimuth))
    throw std::invalid_argument("HF/MINAO initial guesses do not accept a grid");
  validate_preliminary_options(options);
  return options;
}

std::size_t preliminary_numeric_capacity(const core::System& system,
                                         const PreliminaryOptions& options) {
  validate_preliminary_options(options);
  const auto checked_add = [](std::size_t a, std::size_t b) { return runtime::size_add(a, b); };
  const auto checked_mul = [](std::size_t a, std::size_t b) { return runtime::size_mul(a, b); };
  const auto n = molecule::ao_count(system);
  const auto n2 = checked_mul(n, n);
  if (options.kind == PreliminaryKind::Minao) {
    const auto source_n = minao_source_ao_count(system);
    const auto source_primitives = minao_source_primitive_count(system);
    // X + output density, two rectangular projection buffers, occupations and
    // raw source primitive pairs. Target S/Hcore/provider storage is retained
    // by the immutable target owner and intentionally excluded here.
    auto doubles = checked_add(checked_mul(2, n2), checked_mul(2, checked_mul(n, source_n)));
    doubles = checked_add(doubles, source_n);
    auto bytes = checked_mul(sizeof(double), doubles);
    bytes = checked_add(bytes, checked_mul(2 * sizeof(double), source_primitives));
    if (bytes > static_cast<std::uint64_t>(INT64_MAX))
      throw std::overflow_error("MINAO capacity exceeds portable int64 scope");
    return bytes;
  }
  const auto cartesian = molecule::cartesian_ao_count(system);
  const auto c2 = checked_mul(cartesian, cartesian);
  std::size_t primitives = 0;
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 4)
      throw std::invalid_argument("preliminary SCF capacity is qualified through g shells");
    primitives = checked_add(primitives, shell.primitives.size());
  }
  // Three numeric ERI copies cover Cartesian generation/unpacking and a
  // public spherical tensor. The independent byte cap excludes Jet object
  // headers, just like its documented numeric-payload scope. ResourceBudget
  // also reserves the complete existing HF/KS host inventories, including
  // those headers and retained target storage, through the public planner.
  auto bytes = checked_add(16U << 20, checked_mul(24, checked_mul(c2, c2)));
  bytes = checked_add(bytes, checked_mul(8, checked_mul(n2, 192 + 2 * options.diis_history)));
  // Primitive arrays are shared by Cartesian AoViews. The LDA peak has at
  // most four 16*P numeric copies (source/grid Systems, packed AO primitives,
  // and temporary basis validation), covered by 128*P independently of P.
  auto source = checked_add(system.atoms.size(), system.shells.size());
  source = checked_add(source, checked_add(primitives, checked_add(cartesian, n)));
  bytes = checked_add(bytes, checked_mul(128, source));
  if (options.kind == PreliminaryKind::Lda) {
    const auto points =
        checked_mul(system.atoms.size(),
                    checked_mul(options.radial_points,
                                checked_mul(options.angular_polar, options.angular_azimuth)));
    bytes = checked_add(bytes, checked_mul(64, points));
    bytes = checked_add(bytes, checked_mul(8, checked_mul(n, std::min<std::size_t>(points, 256))));
    bytes = checked_add(bytes, checked_mul(256, options.max_iterations));
    bytes = checked_add(bytes, checked_mul(16, options.radial_points + options.angular_polar));
  }
  if (bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::overflow_error("preliminary SCF capacity exceeds portable int64 scope");
  return bytes;
}

void validate_preliminary_target(const core::System& system, const ResolvedFockBuild& strategy,
                                 const ScfOptions& options) {
  if (!options.preliminary_guess) return;
  validate_preliminary_options(*options.preliminary_guess);
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || term.approximation == FockApproximation::Exact;
  };
  const bool common =
      strategy.spec.spin == FockSpin::Restricted &&
      options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64) == GENERATIVEQC_PRECISION_FP64 &&
      exact(strategy.spec.coulomb) && exact(strategy.spec.exchange) &&
      system.ecp_terms.empty() &&
      std::none_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; }) &&
      system.multiplicity == 1 && system.electron_count > 0 && system.electron_count % 2 == 0;
  if (options.preliminary_guess->kind == PreliminaryKind::Minao) {
    const bool supported_elements =
        std::all_of(system.atoms.begin(), system.atoms.end(),
                    [](const auto& atom) { return atom.atomic_number >= 1 && atom.atomic_number <= 18; });
    if (!common || !supported_elements)
      throw std::invalid_argument(
          "MINAO requires an FP64 all-electron restricted H-Ar exact endpoint");
    return;
  }
  if (!common || strategy.backend != FockBackend::Cpu || options.compute_forces)
    throw std::invalid_argument(
        "preliminary SCF requires a CPU FP64 all-electron restricted exact energy endpoint");
}

ScfResult run_with_preliminary_guess(const PreparedFockPlan& target, const ScfOptions& options,
                                     const std::vector<double>* initial_density,
                                     std::size_t retained_target_bytes, const TargetSolve& solve) {
  if (!options.preliminary_guess) return solve(initial_density);
  validate_preliminary_target(target.system(), target.strategy(), options);
  PreliminaryDiagnostic diagnostic;
  diagnostic.requested_kind = static_cast<std::uint32_t>(options.preliminary_guess->kind);
  diagnostic.target_attempts = 1;
  if (initial_density) {
    diagnostic.outcome = PreliminaryOutcome::ExplicitDensity;
    auto result = solve(initial_density);
    result.preliminary_guess = diagnostic;
    return result;
  }
  std::optional<std::vector<double>> density;
  const auto started = Clock::now();
  try {
    runtime::CpuRetainedCapacity retained(retained_target_bytes);
    density = prepare_preliminary_density(target, *options.preliminary_guess, diagnostic);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception&) {
    diagnostic.outcome = PreliminaryOutcome::PreparationFailed;
    diagnostic.work_counters_complete = false;
  }
  diagnostic.preparation_seconds = elapsed(started);
  if (!density) {
    auto result = solve(nullptr);
    result.preliminary_guess = diagnostic;
    return result;
  }
  try {
    runtime::CpuRetainedCapacity retained_seed(runtime::vector_bytes(*density));
    auto result = solve(&*density);
    if (result.converged) {
      result.preliminary_guess = diagnostic;
      return result;
    }
    diagnostic.discarded_target_iterations = result.iterations;
    diagnostic.discarded_target_fock_builds = result.fock_builds;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::runtime_error&) {
    // One numerical retry only; a repeated target failure remains visible.
    diagnostic.work_counters_complete = false;
  }
  density.reset();
  diagnostic.outcome = PreliminaryOutcome::TargetRetried;
  diagnostic.target_attempts = 2;
  auto result = solve(nullptr);
  result.preliminary_guess = diagnostic;
  return result;
}

}  // namespace generativeqc::scf::initial_guess
