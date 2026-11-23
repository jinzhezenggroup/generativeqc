#include "scf/preliminary_guess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

#include "dft/xc.hpp"
#include "generativeqc/generativeqc.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/hf_host_resources.hpp"
#include "scf/initial_guess/automatic_policy.hpp"
#include "scf/initial_guess/huckel.hpp"
#include "scf/initial_guess/huckel_capacity.hpp"
#include "scf/initial_guess/minao.hpp"
#include "scf/initial_guess/overlap.hpp"
#include "scf/ks_host_resources.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/solver/proposal_control.hpp"

namespace generativeqc::scf::initial_guess {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/** Shared compact and prepared target gate. The population cannot stand in for
 * the target spin: an unrestricted singlet still needs a different seed. */
bool preliminary_target_topology_eligible(bool restricted, bool has_ecp, std::int64_t electrons,
                                          unsigned multiplicity) {
  return restricted && !has_ecp && automatic_minao_population_eligible(electrons, multiplicity);
}

bool preliminary_target_common_eligible(const core::System& system,
                                        const ResolvedFockBuild& strategy,
                                        const ScfOptions& options) {
  const auto exact = [](const FockTermSpec& term) {
    return !term.present || term.approximation == FockApproximation::Exact;
  };
  // Application XC warmup must not silently replace the strict native seed by
  // Hcore. Projection/admission remain FP64; ordinary AUTO and HF/LDA
  // preliminary solves keep their existing precision boundary.
  const auto precision = options.precision_mode.value_or(GENERATIVEQC_PRECISION_FP64);
  const bool application_seed =
      precision == GENERATIVEQC_PRECISION_AUTO && strategy.backend == FockBackend::Cuda &&
      options.application_precision_budget[0] > 0.0 && options.preliminary_guess &&
      (options.preliminary_guess->kind == PreliminaryKind::Minao ||
       options.preliminary_guess->kind == PreliminaryKind::Huckel);
  return (precision == GENERATIVEQC_PRECISION_FP64 || application_seed) &&
         (options.preliminary_guess->kind == PreliminaryKind::Huckel ||
          (exact(strategy.spec.coulomb) && exact(strategy.spec.exchange))) &&
         preliminary_target_topology_eligible(
             strategy.spec.spin == FockSpin::Restricted,
             !system.ecp_terms.empty() ||
                 std::any_of(system.atoms.begin(), system.atoms.end(),
                             [](const auto& atom) { return atom.ecp_core != 0; }),
             system.electron_count, system.multiplicity);
}

bool minao_elements_eligible(const core::System& system) {
  return std::all_of(system.atoms.begin(), system.atoms.end(), [](const auto& atom) {
    return automatic_minao_element_eligible(atom.atomic_number);
  });
}

/** The CPU integral provider admits g, but the coarse LDA source uses AO grid
 * primitives through f. Decline before constructing a separate source owner. */
constexpr bool lda_source_supports_angular(unsigned angular) { return angular <= 3; }

bool lda_source_supported(const core::System& system) {
  return std::none_of(system.shells.begin(), system.shells.end(), [](const auto& shell) {
    return !lda_source_supports_angular(shell.angular_momentum);
  });
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
                                                const EigenOperation& eigen,
                                                const MatrixProductOperation& products) {
  const auto& system = target.system();
  if (policy.kind == PreliminaryKind::Huckel)
    return prepare_huckel_initial_density(system, target.one_electron(), policy, diagnostic, eigen,
                                          products);
  if (policy.kind == PreliminaryKind::Lda && !lda_source_supported(system)) {
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
    // Only the strict validator borrows device products; the native occupied
    // ANO construction and its independent ensemble admission stay unchanged.
    solver::validate_seed(ints.overlap, density, ints.nbf,
                          {static_cast<unsigned>(system.electron_count)}, 2.0, eigen, products);
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

std::optional<std::vector<double>> prepare_preliminary_density(
    const PreparedFockPlan& target, const PreliminaryOptions& policy,
    PreliminaryDiagnostic& diagnostic, const EigenOperation& eigen,
    const MatrixProductOperation& products) {
  return prepare_impl(target, policy, diagnostic, eigen, products);
}

std::vector<double> admit_preliminary_density(const PreparedFockPlan& target,
                                              std::vector<double> density,
                                              const EigenOperation& eigen) {
  solver::validate_seed(target.one_electron().overlap, density, target.one_electron().nbf,
                        {static_cast<unsigned>(target.system().electron_count)}, 2.0, eigen);
  return density;
}

std::optional<std::vector<double>> prepare_huckel_initial_density(
    const core::System& system, const integrals::IntegralData& target,
    const PreliminaryOptions& policy, PreliminaryDiagnostic& diagnostic,
    const EigenOperation& eigen, const MatrixProductOperation& products) {
  validate_preliminary_options(policy);
  if (policy.kind != PreliminaryKind::Huckel)
    throw std::invalid_argument("Huckel preparation requires its explicit provider kind");
  diagnostic.preparation_numeric_capacity = preliminary_numeric_capacity(system, policy);
  if (diagnostic.preparation_numeric_capacity > policy.maximum_numeric_bytes) {
    diagnostic.outcome = PreliminaryOutcome::BudgetSkipped;
    return std::nullopt;
  }
  auto candidate = huckel_density(system, target, policy, diagnostic);
  solver::validate_seed(target.overlap, candidate.density, target.nbf,
                        {static_cast<unsigned>(system.electron_count)}, 2.0, eigen, products);
  diagnostic.outcome = PreliminaryOutcome::Used;
  return std::move(candidate.density);
}

void validate_preliminary_options(const PreliminaryOptions& options) {
  if (options.kind != PreliminaryKind::HartreeFock && options.kind != PreliminaryKind::Lda &&
      options.kind != PreliminaryKind::Minao && options.kind != PreliminaryKind::Huckel)
    throw std::invalid_argument("unknown preliminary initial-density provider");
  if (options.automatic_cpu && options.kind != PreliminaryKind::Minao &&
      options.kind != PreliminaryKind::Huckel)
    throw std::invalid_argument("automatic CPU policy requires the MINAO or Huckel provider");
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
  options.automatic_cpu = descriptor->kind == GENERATIVEQC_INITIAL_GUESS_AUTO_CPU ||
                          descriptor->kind == GENERATIVEQC_INITIAL_GUESS_AUTO_HUCKEL_CPU;
  options.kind = descriptor->kind == GENERATIVEQC_INITIAL_GUESS_AUTO_CPU ? PreliminaryKind::Minao
                 : descriptor->kind == GENERATIVEQC_INITIAL_GUESS_AUTO_HUCKEL_CPU
                     ? PreliminaryKind::Huckel
                     : static_cast<PreliminaryKind>(descriptor->kind);
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
  if ((options.kind == PreliminaryKind::HartreeFock || options.kind == PreliminaryKind::Minao ||
       options.kind == PreliminaryKind::Huckel) &&
      (descriptor->radial_points || descriptor->angular_polar || descriptor->angular_azimuth))
    throw std::invalid_argument("HF/MINAO/Huckel initial guesses do not accept a grid");
  validate_preliminary_options(options);
  return options;
}

namespace {
CpuPreliminaryBasisShape cpu_preliminary_basis_shape(const core::System& system) {
  CpuPreliminaryBasisShape shape{system.atoms.size(),
                                 molecule::ao_count(system),
                                 molecule::cartesian_ao_count(system),
                                 system.shells.size(),
                                 0,
                                 0};
  for (const auto& shell : system.shells) {
    shape.primitives = runtime::size_add(shape.primitives, shell.primitives.size());
    shape.angular = std::max(shape.angular, shell.angular_momentum);
  }
  return shape;
}

std::size_t preliminary_scf_numeric_capacity(const CpuPreliminaryBasisShape& shape,
                                             const PreliminaryOptions& options) {
  const auto checked_add = [](std::size_t a, std::size_t b) { return runtime::size_add(a, b); };
  const auto checked_mul = [](std::size_t a, std::size_t b) { return runtime::size_mul(a, b); };
  const auto n = shape.orbitals;
  const auto n2 = checked_mul(n, n);
  const auto cartesian = shape.cartesian;
  const auto c2 = checked_mul(cartesian, cartesian);
  if (shape.angular > 4)
    throw std::invalid_argument("preliminary SCF capacity is qualified through g shells");
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
  auto source = checked_add(shape.atoms, shape.shells);
  source = checked_add(source, checked_add(shape.primitives, checked_add(cartesian, n)));
  bytes = checked_add(bytes, checked_mul(128, source));
  if (options.kind == PreliminaryKind::Lda) {
    const auto points = checked_mul(
        shape.atoms, checked_mul(options.radial_points,
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
}  // namespace

std::size_t preliminary_numeric_capacity(const core::System& system,
                                         const PreliminaryOptions& options) {
  validate_preliminary_options(options);
  if (options.kind == PreliminaryKind::Huckel)
    return huckel_numeric_capacity(huckel_capacity_shape(system), options.diis_history);
  if (options.kind == PreliminaryKind::Minao)
    return minao_numeric_capacity_shape(molecule::ao_count(system), system.atoms.size(),
                                        minao_source_ao_count(system),
                                        minao_source_primitive_count(system));
  return preliminary_scf_numeric_capacity(cpu_preliminary_basis_shape(system), options);
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
  const bool common = preliminary_target_common_eligible(system, strategy, options);
  if (options.preliminary_guess->kind == PreliminaryKind::Minao ||
      options.preliminary_guess->kind == PreliminaryKind::Huckel) {
    if (!common || !minao_elements_eligible(system))
      throw std::invalid_argument(
          "MINAO/Huckel requires an FP64 all-electron restricted H-Ar endpoint; MINAO is "
          "exact-only");
    return;
  }
  if (!common || strategy.backend != FockBackend::Cpu || options.compute_forces)
    throw std::invalid_argument(
        "preliminary SCF requires a CPU FP64 all-electron restricted exact energy endpoint");
}

namespace {
template <class NumericCapacity>
CpuPreliminaryInventory cpu_preliminary_inventory(const CpuPreliminaryBasisShape& shape,
                                                  const PreliminaryOptions& policy,
                                                  NumericCapacity numeric_capacity) {
  CpuPreliminaryInventory result;
  result.maximum_numeric_bytes = policy.maximum_numeric_bytes;
  if (policy.kind == PreliminaryKind::Lda && !lda_source_supports_angular(shape.angular)) {
    result.disposition = CpuPreliminaryInventory::Disposition::SourceDeclined;
    return result;
  }
  result.required_numeric_bytes = numeric_capacity();
  if (result.required_numeric_bytes > result.maximum_numeric_bytes) {
    result.disposition = CpuPreliminaryInventory::Disposition::BudgetSkipped;
    return result;
  }
  const auto orbitals = shape.orbitals;
  result.seed_bytes = runtime::size_mul(sizeof(double), runtime::size_mul(orbitals, orbitals));
  if (result.required_numeric_bytes < result.seed_bytes)
    throw std::logic_error("preliminary capacity does not cover its seed");
  result.workspace_bytes = result.required_numeric_bytes - result.seed_bytes;
  if (policy.kind == PreliminaryKind::Minao || policy.kind == PreliminaryKind::Huckel)
    return result;

  std::uint64_t source_peak = 0;
  if (policy.kind == PreliminaryKind::HartreeFock) {
    CpuHfHostFleetResources fleet;
    fleet.append({{shape.atoms, orbitals, shape.cartesian, 1, shape.angular, policy.diis_history, 0,
                   0, 0, 0},
                  shape.shells,
                  shape.primitives,
                  0,
                  0});
    source_peak = fleet.inventory()[2];
  } else {
    const auto grid_points =
        runtime::size_mul(runtime::size_mul(shape.atoms, policy.radial_points),
                          runtime::size_mul(policy.angular_polar, policy.angular_azimuth));
    CpuKsHostFleetResources fleet;
    fleet.append({shape.atoms, orbitals, shape.cartesian, shape.shells, shape.primitives, 1,
                  shape.angular, 0, grid_points, policy.radial_points, policy.angular_polar, 256,
                  policy.diis_history, policy.max_iterations, false, false, 0});
    source_peak = fleet.inventory()[14];
  }
  result.workspace_bytes = std::max(result.workspace_bytes, source_peak);
  return result;
}
}  // namespace

CpuPreliminaryInventory cpu_direct_preliminary_inventory(const core::System& system,
                                                         const PreliminaryOptions& policy,
                                                         bool rhf_forces) {
  if (policy.automatic_cpu)
    throw std::invalid_argument("CPU preliminary inventory requires an explicit guess policy");
  ScfOptions controls;
  controls.compute_forces = rhf_forces;
  controls.preliminary_guess = policy;
  const auto strategy =
      resolve_fock_build(make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu);
  validate_preliminary_target(system, strategy, controls);
  const auto shape = cpu_preliminary_basis_shape(system);
  return cpu_preliminary_inventory(shape, policy, [&] {
    if (policy.kind == PreliminaryKind::Minao || policy.kind == PreliminaryKind::Huckel)
      return preliminary_numeric_capacity(system, policy);
    return preliminary_scf_numeric_capacity(shape, policy);
  });
}

CpuPreliminaryInventory cpu_direct_preliminary_inventory(
    const CpuPreliminaryBasisShape& shape, const PreliminaryOptions& policy,
    std::span<const std::int32_t> atomic_numbers, std::span<const HuckelShellShape> shell_shapes,
    int representation, std::int64_t electrons, unsigned multiplicity, bool has_ecp,
    unsigned spin_channels) {
  validate_preliminary_options(policy);
  if (!shape.atoms || shape.atoms != atomic_numbers.size() || !shape.orbitals ||
      shape.cartesian < shape.orbitals || !shape.shells || shape.shells > shape.orbitals ||
      shape.primitives < shape.shells || electrons < 0 || !multiplicity ||
      static_cast<std::uint64_t>(electrons) / 2 > shape.orbitals ||
      (spin_channels != 1 && spin_channels != 2) ||
      (representation != core::kCartesianBasis && representation != core::kSphericalBasis) ||
      (policy.kind == PreliminaryKind::Huckel ? shell_shapes.size() != shape.shells
                                              : !shell_shapes.empty()))
    throw std::invalid_argument("invalid compact CPU preliminary topology");
  const bool common =
      preliminary_target_topology_eligible(spin_channels == 1, has_ecp, electrons, multiplicity);
  if (policy.automatic_cpu) {
    if (!common || !std::ranges::all_of(atomic_numbers, automatic_minao_element_eligible))
      return {0, policy.maximum_numeric_bytes, 0, 0,
              CpuPreliminaryInventory::Disposition::TargetDeclined};
  } else if (!common) {
    throw std::invalid_argument(
        "CPU preliminary inventory requires an all-electron restricted target");
  }
  return cpu_preliminary_inventory(shape, policy, [&] {
    if (policy.kind == PreliminaryKind::Minao)
      return preliminary_minao_numeric_capacity(shape.orbitals, atomic_numbers.data(), shape.atoms);
    if (policy.kind == PreliminaryKind::Huckel) {
      unsigned angular = 0;
      std::size_t cartesian = 0;
      const auto huckel = huckel_capacity_shape(
          shape.atoms, shape.shells, representation,
          [&](std::size_t index) { return atomic_numbers[index]; },
          [&](std::size_t index) {
            const auto shell = shell_shapes[index];
            angular = std::max(angular, shell.angular_momentum);
            cartesian = runtime::size_add(
                cartesian, molecule::shell_ao_count(shell.angular_momentum, core::kCartesianBasis));
            return shell;
          });
      if (huckel.target_aos != shape.orbitals || huckel.primitives != shape.primitives ||
          cartesian != shape.cartesian || angular != shape.angular)
        throw std::invalid_argument("Huckel shell rows disagree with preliminary topology");
      return huckel_numeric_capacity(huckel, policy.diis_history);
    }
    return preliminary_scf_numeric_capacity(shape, policy);
  });
}

void resolve_automatic_preliminary_guess(const core::System& system,
                                         const ResolvedFockBuild& strategy, ScfOptions& options) {
  if (!options.preliminary_guess || !options.preliminary_guess->automatic_cpu) return;
  validate_preliminary_options(*options.preliminary_guess);
  if (strategy.backend != FockBackend::Cpu)
    throw std::invalid_argument("automatic per-system initial guess is CPU-only");
  if (!preliminary_target_common_eligible(system, strategy, options) ||
      !minao_elements_eligible(system)) {
    options.preliminary_guess.reset();
    return;
  }
  options.preliminary_guess->automatic_cpu = false;
}

ScfResult run_with_preliminary_guess(const PreparedFockPlan& target, const ScfOptions& options,
                                     const std::vector<double>* initial_density,
                                     std::size_t retained_target_bytes, const TargetSolve& solve,
                                     const EigenOperation& eigen) {
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
    density = prepare_preliminary_density(target, *options.preliminary_guess, diagnostic, eigen);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const generativeqc::Error& error) {
    if (error.status() != GENERATIVEQC_STATUS_NUMERICAL_FAILURE) throw;
    diagnostic.outcome = PreliminaryOutcome::PreparationFailed;
    diagnostic.work_counters_complete = false;
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
  CoulombExchangeWork discarded_work;
  try {
    runtime::CpuRetainedCapacity retained_seed(runtime::vector_bytes(*density));
    auto result = solve(&*density);
    if (result.converged) {
      result.preliminary_guess = diagnostic;
      return result;
    }
    diagnostic.discarded_target_iterations = result.iterations;
    diagnostic.discarded_target_fock_builds = result.fock_builds;
    discarded_work = result.dft_diagnostic.coulomb_exchange_work;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const generativeqc::Error& error) {
    if (error.status() != GENERATIVEQC_STATUS_NUMERICAL_FAILURE) throw;
    diagnostic.work_counters_complete = false;
  } catch (const std::runtime_error&) {
    // One numerical retry only; a repeated target failure remains visible.
    diagnostic.work_counters_complete = false;
  }
  density.reset();
  diagnostic.outcome = PreliminaryOutcome::TargetRetried;
  diagnostic.target_attempts = 2;
  auto result = solve(nullptr);
  // The narrow census spans both target attempts, but not preliminary-method
  // preparation. An exception leaves discarded_work unknown and revokes it.
  result.dft_diagnostic.coulomb_exchange_work.add(discarded_work);
  result.preliminary_guess = diagnostic;
  return result;
}

}  // namespace generativeqc::scf::initial_guess
