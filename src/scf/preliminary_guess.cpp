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
#include "scf/initial_guess/overlap.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/solver/proposal_control.hpp"

namespace generativeqc::scf::initial_guess {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/** One native numeric-payload formula for both actual preliminary admission
 * and the Python/CLI planner's shape-only query. The MINAO element tables stay
 * exclusively in integrals/minao_basis.cpp. No target SCF is prepared here. */
std::size_t minao_numeric_capacity_shape(std::size_t n, std::size_t atom_count,
                                         std::size_t source_n, std::size_t source_primitives) {
  if (!n || !atom_count || !source_n)
    throw std::invalid_argument("MINAO capacity requires nonempty AO/atom topology");
  const auto checked_add = [](std::size_t a, std::size_t b) { return runtime::size_add(a, b); };
  const auto checked_mul = [](std::size_t a, std::size_t b) { return runtime::size_mul(a, b); };
  const auto n2 = checked_mul(n, n);
  // X/raw/output and the strict validator's eigensolver and matrix copies,
  // two cross-overlap buffers, source occupations and linear eigen workspace.
  auto doubles = checked_add(checked_mul(16, n2), checked_mul(2, checked_mul(n, source_n)));
  doubles = checked_add(doubles, checked_add(checked_mul(8, n), source_n));
  auto bytes = checked_mul(sizeof(double), doubles);
  // Sparse angular views, both Jet center arrays and primitive reservations;
  // exclude the target's owned integrals, allocator headers and process RSS.
  bytes = checked_add(bytes, checked_mul(512, checked_add(n, source_n)));
  bytes = checked_add(bytes, checked_mul(256, atom_count));
  bytes = checked_add(bytes, checked_mul(2 * sizeof(double), source_primitives));
  bytes = checked_add(bytes, 8192);
  if (bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::overflow_error("MINAO capacity exceeds portable int64 scope");
  return bytes;
}

std::optional<std::vector<double>> prepare_impl(const PreparedFockPlan& target,
                                                const PreliminaryOptions& policy,
                                                PreliminaryDiagnostic& diagnostic,
                                                const EigenOperation& eigen) {
  const auto& system = target.system();
  // The CPU integral provider admits g, but the LDA AO-grid primitive only
  // admits through f. Decline before building an unusable extra integral owner.
  if (policy.kind == PreliminaryKind::Lda &&
      std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    diagnostic.outcome = PreliminaryOutcome::PreparationFailed;
    return std::nullopt;
  }
  diagnostic.preparation_numeric_capacity = preliminary_numeric_capacity(system, policy);
  if (diagnostic.preparation_numeric_capacity > policy.maximum_numeric_bytes) {
    diagnostic.outcome = PreliminaryOutcome::BudgetSkipped;
    return std::nullopt;
  }
  if (policy.kind == PreliminaryKind::Minao) {
    const auto& ints = target.one_electron();
    const auto x = symmetric_overlap(ints.overlap, ints.nbf, eigen);
    auto projected = minao_density(system, ints, x);
    auto density = admissible_minao_density(system, ints, x, projected.density, eigen);
    diagnostic.preliminary_iterations = 0;
    diagnostic.preliminary_fock_builds = 0;
    density = admit_preliminary_density(target, std::move(density), eigen);
    diagnostic.outcome = PreliminaryOutcome::Used;
    return density;
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

std::optional<std::vector<double>> prepare_preliminary_density(const PreparedFockPlan& target,
                                                               const PreliminaryOptions& policy,
                                                               PreliminaryDiagnostic& diagnostic,
                                                               const EigenOperation& eigen) {
  return prepare_impl(target, policy, diagnostic, eigen);
}

std::vector<double> admit_preliminary_density(const PreparedFockPlan& target,
                                              std::vector<double> density,
                                              const EigenOperation& eigen) {
  solver::validate_seed(target.one_electron().overlap, density, target.one_electron().nbf,
                        {static_cast<unsigned>(target.system().electron_count)}, 2.0, eigen);
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
  if (options.kind == PreliminaryKind::Minao)
    return minao_numeric_capacity_shape(n, system.atoms.size(), minao_source_ao_count(system),
                                        minao_source_primitive_count(system));
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

std::size_t preliminary_minao_numeric_capacity(std::size_t target_aos,
                                               const std::int32_t* atomic_numbers,
                                               std::size_t atom_count) {
  if (!target_aos || !atomic_numbers || !atom_count ||
      atom_count > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("invalid MINAO capacity query topology");
  // The source's exact occupied-ANO counts come from the same native table as
  // actual MINAO preparation. No synthetic target basis or integral tensors.
  core::System elements;
  elements.atoms.reserve(atom_count);
  for (std::size_t index = 0; index < atom_count; ++index) {
    const auto z = atomic_numbers[index];
    if (z < 1 || z > 18)
      throw std::invalid_argument("MINAO initial guess is currently qualified for H-Ar");
    elements.atoms.push_back({z, {0.0, 0.0, 0.0}, 0});
  }
  return minao_numeric_capacity_shape(target_aos, atom_count, minao_source_ao_count(elements),
                                      minao_source_primitive_count(elements));
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
      exact(strategy.spec.coulomb) && exact(strategy.spec.exchange) && system.ecp_terms.empty() &&
      std::none_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; }) &&
      system.multiplicity == 1 && system.electron_count > 0 && system.electron_count % 2 == 0;
  if (options.preliminary_guess->kind == PreliminaryKind::Minao) {
    const bool supported_elements = std::all_of(
        system.atoms.begin(), system.atoms.end(),
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
