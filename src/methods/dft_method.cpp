#include "methods/dft_method.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "dft/ao_grid.hpp"
#include "dft/application_precision.hpp"
#include "dft/cuda_ks_final_validation_policy.hpp"
#include "dft/cuda_spd_force_policy.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/grid.hpp"
#include "dft/mixed_rsh_df_sources.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/semilocal_family.hpp"
#include "dft/xc.hpp"
#include "generated_method_parameters.hpp"
#include "generativeqc/generativeqc.hpp"
#include "methods/batch_capabilities.hpp"
#include "methods/dft_method_owner_port.hpp"
#include "methods/dft_method_preparation.hpp"
#include "methods/exact_density_response_cache.hpp"
#include "methods/execution_failure.hpp"
#include "methods/geometry.hpp"
#include "molecule/basis.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "runtime/df_progress_trace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_precision.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/stationary_integral_gradient.hpp"
#include "scf/types.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#endif

namespace generativeqc::methods::detail {
namespace {

std::uint64_t next_cpu_ks_owner() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS owner identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

Result adapt_result(scf::ScfResult native, generativeqc_backend backend) {
  Result result;
  result.preliminary_guess = native.preliminary_guess;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.physical_residual_rms = native.physical_residual_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  result.fock_builds = native.fock_builds;
  result.precision = native.precision;
  result.incremental_direct_jk = native.incremental_direct_jk;
  result.precision_work = std::move(native.precision_work);
  native.dft_diagnostic.fock_builds = native.fock_builds;
  native.dft_diagnostic.initial_density_used = native.initial_density_used;
  // Move the snapshot instead of retaining another max-iteration history.
  result.ks_diagnostic = std::move(native.dft_diagnostic);
  return result;
}

/** The method's global ledger supplies the budget. Size the common direct
 * source explicitly so its standalone default cap is not a second KS limit. */
unsigned ks_direct_derivative_order(const scf::ResolvedFockBuild& strategy,
                                    generativeqc_backend backend,
                                    const NativeKsExecutionPlan& plan) noexcept {
  // Automatic CUDA KS is energy-only. Do not retain unqualified derivative
  // sources outside its exact J-only resource inventory.
  if (plan.automatic_program) return 0;
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA && strategy.backend == scf::FockBackend::Cuda) {
    const auto exact = [](const scf::FockTermSpec& term) {
      return term.present && term.approximation == scf::FockApproximation::Exact;
    };
    if (exact(strategy.spec.coulomb) || exact(strategy.spec.exchange)) return 1;
  }
#else
  (void)strategy;
  (void)backend;
#endif
  return 0;
}

/** Retain the existing generated DF response only for the semilocal CUDA J
 * domain. Hybrid/RSH/nonlocal promotion remains a separate provider contract. */
unsigned ks_fitted_derivative_order(const scf::ResolvedFockBuild& strategy,
                                    generativeqc_backend backend) noexcept {
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA && strategy.backend == scf::FockBackend::Cuda &&
      strategy.spec.coulomb.present &&
      strategy.spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
      !strategy.spec.exchange.present)
    return 1;
#else
  (void)strategy;
  (void)backend;
#endif
  return 0;
}

enum class KsProviderPurpose { scf, derivative_only };

/** Keep SCF's established optional J allowance separate from a lazy force lease.
 * Derivative order alone cannot distinguish these owners: the primary SCF
 * provider also retains first-order resources for the eventual force. */
std::size_t ks_provider_bytes(const core::System& system, generativeqc_backend backend,
                              unsigned direct_derivative_order = 0,
                              KsProviderPurpose purpose = KsProviderPurpose::scf) {
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA) {
    std::size_t primitives = 0;
    for (const auto& shell : system.shells)
      primitives = runtime::add_capacity(primitives, shell.primitives.size());
    // The lazy first-order LR owner competes with composite grid response
    // under one capacity cap. Use its known Cartesian size rather than the
    // shape-only 2*nao envelope, which can otherwise starve grid concurrency.
    // The primary provider needs its established allowance even when it also
    // owns derivatives; tightening that cap can silently evict its MD-J source.
    // A lazy exchange-only derivative lease never consumes MD-J values.
    // Reserving that optional Coulomb engine here can reject the complete
    // force before allocation, even though the bounded derivative fits. Its
    // core capacity remains charged; only SCF retains the optional J margin.
    const auto cartesian_nao =
        purpose == KsProviderPurpose::derivative_only ? molecule::cartesian_ao_count(system) : 0U;
    return scf::cuda_direct_coulomb_device_bytes(
        1, molecule::ao_count(system), system.atoms.size(), system.shells.size(), primitives,
        direct_derivative_order, purpose == KsProviderPurpose::scf, cartesian_nao);
  }
#else
  (void)direct_derivative_order;
  (void)purpose;
#endif
  return 0;
}

/** An explicit DF envelope covers both independent radial owners. Zero keeps
 * automatic admission; never turn a small explicit half into that sentinel. */
std::size_t ks_fitted_provider_budget(const scf::ScfOptions& options,
                                      const NativeKsExecutionPlan& plan, bool long_range) {
  const auto bytes = options.density_fitting_memory_budget_bytes;
  if (!plan.density_fitted_range_exchange || !bytes ||
      options.resolved_fock_build->backend != scf::FockBackend::Cuda)
    return bytes;
  if (bytes < 2U) throw std::bad_alloc();
  const auto primary = bytes / 2U;
  return long_range ? bytes - primary : primary;
}

#if GENERATIVEQC_HAS_CUDA
KsTransportDiagnostic adapt_transfers(const dft::CudaKsTransfers& value) {
  return {value.setup_h2d_bytes,
          value.density_h2d_bytes,
          value.scalar_d2h_bytes,
          value.matrix_d2h_bytes,
          value.final_state_d2h_bytes,
          value.final_state_reads,
          value.synchronizations,
          value.iterations,
          value.occupation_stabilized_proposals};
}

void add_transfers(dft::CudaKsTransfers& target, const dft::CudaKsTransfers& value) {
  const auto add = [](std::uint64_t& destination, std::uint64_t increment) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - destination)
      throw std::overflow_error("CUDA KS transport counter overflow");
    destination += increment;
  };
  add(target.setup_h2d_bytes, value.setup_h2d_bytes);
  add(target.density_h2d_bytes, value.density_h2d_bytes);
  add(target.scalar_d2h_bytes, value.scalar_d2h_bytes);
  add(target.matrix_d2h_bytes, value.matrix_d2h_bytes);
  add(target.final_state_d2h_bytes, value.final_state_d2h_bytes);
  add(target.final_state_reads, value.final_state_reads);
  add(target.resident_final_validations, value.resident_final_validations);
  add(target.synchronizations, value.synchronizations);
  add(target.iterations, value.iterations);
  add(target.occupation_stabilized_proposals, value.occupation_stabilized_proposals);
}
#endif

/** Own auxiliary shells and rebind centers when a batch item moves. The source
 * owner copies the result, so descriptor/temporary system lifetimes never leak
 * into a prepared calculation. Native callers supply an already decoded System
 * without an opaque C ABI handle; that explicit source must take precedence.
 * Only omission of both forms requests the orbital basis as the fit basis. */
std::optional<core::System> ks_auxiliary_template(const generativeqc_method_descriptor& descriptor,
                                                  const core::System* native_auxiliary) {
  if (native_auxiliary != nullptr) return *native_auxiliary;
  if (descriptor.density_fitting_auxiliary_basis)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "auxiliary basis must be resolved at the API boundary");
  return std::nullopt;
}

void validate_ks_auxiliary_geometry(const core::System& system,
                                    const std::optional<core::System>& auxiliary) {
  if (!auxiliary) return;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number ||
        auxiliary->atoms[i].position != system.atoms[i].position)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis must share the initial system geometry");
}

std::optional<core::System> ks_auxiliary_for_system(const core::System& system,
                                                    std::optional<core::System> auxiliary) {
  if (!auxiliary) return auxiliary;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i) {
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis atom ordering differs");
    auxiliary->atoms[i].position = system.atoms[i].position;
  }
  return auxiliary;
}

/** Backend selection must precede materialization: constructing the reference
 * grid and then uploading it hides cubic host work in CUDA preparation. */
dft::MolecularGrid ks_molecular_grid(const core::System& system, dft::GridSpec spec,
                                     generativeqc_backend backend, int device, bool retain_device) {
  if (backend == GENERATIVEQC_BACKEND_CUDA) {
#if GENERATIVEQC_HAS_CUDA
    return dft::MolecularGrid::from_cuda(system, spec, device, retain_device);
#else
    throw std::runtime_error("CUDA quadrature is unavailable in this build");
#endif
  }
  return dft::MolecularGrid(system, spec);
}

scf::FockOccupiedProjectionReservation ks_fitted_projection_reservation(
    const core::System& system, const NativeKsExecutionPlan& execution_plan,
    const scf::ResolvedFockBuild& strategy) {
  // This method owns an integer restricted determinant. Fixed-density Fock
  // callers do not acquire this promise from the same dimensions or system.
  const auto& spec = strategy.spec;
  // A range-separated composition may still reserve the ordinary full-range
  // fitted primary. Its independent long-range correction neither consumes
  // nor mutates this occupied projection, so range_exchange is not itself a veto.
  if (execution_plan.spin_channels != 1 || strategy.backend != scf::FockBackend::Cuda ||
      spec.spin != scf::FockSpin::Restricted || spec.derivative_order != 0 ||
      !spec.coulomb.present || !spec.exchange.present ||
      spec.coulomb.approximation != scf::FockApproximation::DensityFitted ||
      spec.exchange.approximation != scf::FockApproximation::DensityFitted ||
      spec.exchange.op != scf::FockOperator::FullRange)
    return {};
  const auto [alpha, beta] = scf::initial_guess::spin_occupations(system);
  if (system.electron_count <= 0 || !alpha || alpha != beta || system.multiplicity != 1 ||
      alpha > molecule::ao_count(system))
    throw std::invalid_argument("KS projection reservation requires valid restricted occupations");
  return {alpha};
}

#if GENERATIVEQC_HAS_CUDA
bool cuda_rsh_provider_compatible(const scf::PreparedFockPlan& provider,
                                  const scf::ResolvedFockBuild& correction,
                                  const scf::PreparedFockPlan* range_provider) noexcept {
  const auto& primary = provider.strategy();
  const auto& primary_spec = primary.spec;
  const auto& correction_spec = correction.spec;
  const auto primary_binding = scf::prepared_cuda_fock_binding(provider);
  const bool correction_compatible =
      correction.backend == scf::FockBackend::Cuda && correction_spec.spin == primary_spec.spin &&
      correction_spec.derivative_order == 0 && !correction_spec.coulomb.present &&
      correction_spec.exchange.present &&
      (correction_spec.exchange.approximation == scf::FockApproximation::Exact ||
       (correction_spec.exchange.approximation == scf::FockApproximation::DensityFitted &&
        primary_spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
        correction.metric_relative_threshold == primary.metric_relative_threshold)) &&
      correction_spec.exchange.op == scf::FockOperator::LongRange &&
      std::isfinite(correction_spec.exchange.omega) && correction_spec.exchange.omega > 0.0 &&
      std::isfinite(correction_spec.exchange.coefficient) &&
      correction.screening_tolerance == primary.screening_tolerance;
  if (!primary_binding || primary.backend != scf::FockBackend::Cuda ||
      primary_spec.derivative_order != 0 || !primary_spec.coulomb.present ||
      primary_spec.coulomb.coefficient != 1.0 ||
      primary_spec.coulomb.op != scf::FockOperator::FullRange ||
      primary_spec.coulomb.omega != 0.0 || !correction_compatible)
    return false;

  const bool fitted_primary =
      primary_spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
      (!primary_spec.exchange.present ||
       (primary_spec.exchange.approximation == scf::FockApproximation::DensityFitted &&
        primary_spec.exchange.op == scf::FockOperator::FullRange &&
        primary_spec.exchange.omega == 0.0));
  if (fitted_primary) {
    if (!range_provider || range_provider->strategy() != correction ||
        !range_provider->matches_system(provider.system()))
      return false;
    const auto range_binding = scf::prepared_cuda_fock_binding(*range_provider);
    return range_binding && range_binding.device_id == primary_binding.device_id &&
           range_binding.nbf == primary_binding.nbf;
  }

  const bool exact_primary =
      primary_spec.coulomb.approximation == scf::FockApproximation::Exact &&
      (!primary_spec.exchange.present ||
       (primary_spec.exchange.approximation == scf::FockApproximation::Exact &&
        primary_spec.exchange.op == scf::FockOperator::FullRange &&
        primary_spec.exchange.omega == 0.0));
  return exact_primary && range_provider == nullptr;
}
#endif

/** One immutable CUDA admission rule for metadata and complete force consumers.
 * Availability is deliberately context-specific, not a method-global force bit.
 * Optional compiled bindings, precise Fock coefficients, representation and
 * FP64 final states are prerequisites. Strict-density AUTO may narrow the
 * gradient; budget-Fock AUTO reuses only the FP64 force body. Neither route
 * weakens the final token checks; other contexts retain their consumer. */
const dft::CudaStationaryBinding* native_cuda_stationary_binding(
    const NativeKsExecutionPlan& execution_plan, const scf::ScfOptions& options,
    const dft::GridSpec& grid, bool all_electron, std::size_t maximum_angular, std::size_t aos,
    std::size_t atoms, bool qualify_through_d = false) noexcept {
  const bool application =
      dft::admit_stationary_force_controls(options.energy_tolerance, options.density_tolerance,
                                           options.screening_tolerance, false) &&
      dft::admit_application_gradient(options.application_precision_budget, aos, atoms,
                                      maximum_angular, execution_plan.spin_channels == 1);
  const bool budget_force =
      options.precision_mode == GENERATIVEQC_PRECISION_AUTO && options.resolved_fock_build &&
      dft::admit_stationary_force_controls(options.energy_tolerance, options.density_tolerance,
                                           options.screening_tolerance, true) &&
      dft::admit_application_fock(
          options.application_precision_budget, aos, atoms, maximum_angular,
          execution_plan.spin_channels == 1 &&
              options.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused,
          all_electron,
          scf::cuda_fock_precision_capabilities(*options.resolved_fock_build).coulomb) &&
      dft::admit_native_stationary_gradient_shape(aos, atoms, maximum_angular,
                                                  execution_plan.spin_channels == 1);
  // Above the retained-pair regime, the incumbent phased strict consumer is
  // faster than native inline cooperative FP64. Do not slow the comparator
  // merely to make both precision modes use an identical execution schedule.
  const bool native_shape =
      atoms <= 32 && dft::admit_native_stationary_gradient_shape(aos, atoms, maximum_angular,
                                                                 execution_plan.spin_channels == 1);
  if (!all_electron ||
      maximum_angular > ((qualify_through_d || native_shape || application) ? 2U : 1U) ||
      grid.partition_iterations != 3 ||
      (options.precision_mode != GENERATIVEQC_PRECISION_FP64 && !application && !budget_force) ||
      options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
      !options.resolved_fock_build ||
      options.resolved_fock_build->backend != scf::FockBackend::Cuda ||
      execution_plan.range_exchange || execution_plan.nonlocal_correlation ||
      execution_plan.d4_correction)
    return {};
  const auto& spec = options.resolved_fock_build->spec;
  const auto admitted_term = [](const scf::FockTermSpec& term) {
    return !term.present || (term.approximation == scf::FockApproximation::Exact &&
                             term.op == scf::FockOperator::FullRange && term.omega == 0.0);
  };
  if (spec.derivative_order != 0 || !spec.coulomb.present || spec.coulomb.coefficient != 1.0 ||
      !admitted_term(spec.coulomb) || !admitted_term(spec.exchange) ||
      execution_plan.spin_channels != (spec.spin == scf::FockSpin::Restricted ? 1u : 2u))
    return {};
  // Convert the authenticated Fock density convention back to the MethodIR
  // energy weight before matching the generated complete model contract.
  const double exchange =
      spec.exchange.present
          ? -spec.exchange.coefficient * (execution_plan.spin_channels == 1 ? 2.0 : 1.0)
          : 0.0;
  const dft::CudaStationaryModel model{ks_xc_functional_code(execution_plan),
                                       ks_scf_domain_version(execution_plan),
                                       execution_plan.spin_channels,
                                       grid.partition_iterations,
                                       options.semilocal_exchange_scale,
                                       options.semilocal_correlation_scale,
                                       exchange};
  const auto* binding = dft::cuda_stationary_binding(model);
  // Newly packaged split-hybrid graphs are diagnostic candidates until their
  // complete shape is qualified. Merely linking one must not promote the
  // historical s/p strict consumer or an unqualified unrestricted peer.
  if (execution_plan.generated_split_hybrid && !qualify_through_d &&
      (!native_shape || !binding || !dft::stationary_gradient_model_qualification(*binding, aos)))
    return {};
  if (application) {
    if (!binding ||
        !dft::application_gradient_qualification(*binding, options.application_precision_budget,
                                                 aos, atoms, maximum_angular))
      return {};
  } else if ((maximum_angular > 1 && !qualify_through_d) &&
             (!binding || !dft::stationary_gradient_model_qualification(*binding, aos)))
    return {};
  return binding;
}

/** Retained scientific state for one item. Execution policy belongs to
 * KsPreparedBatch for both public single calculations and ragged batches. */
class KsPreparedState final : public KsSingleOwnerPort {
 public:
  KsPreparedState(Capabilities capabilities, core::System system,
                  NativeKsExecutionPlan execution_plan, scf::ScfOptions options, dft::GridSpec grid,
                  generativeqc_backend backend, int device,
                  const std::optional<core::System>& auxiliary)
      : capabilities_(capabilities),
        system_(std::move(system)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        backend_(backend),
        fock_(system_, auxiliary ? &*auxiliary : nullptr, *options_.resolved_fock_build, device,
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE
                  ? ks_provider_bytes(system_, backend,
                                      ks_direct_derivative_order(*options_.resolved_fock_build,
                                                                 backend, execution_plan_))
                  : ks_fitted_provider_budget(options_, execution_plan_, false),
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE
                  ? ks_direct_derivative_order(*options_.resolved_fock_build, backend,
                                               execution_plan_)
                  : 0U,
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
                      execution_plan_.automatic_program
                  ? 0U
                  : 1U,
              ks_fitted_projection_reservation(system_, execution_plan_,
                                               *options_.resolved_fock_build)),
        basis_(system_),
        grid_(ks_molecular_grid(
            system_, grid, backend_, device,
            options_.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused)) {
    if (options_.nonlocal_grid_spec && *options_.nonlocal_grid_spec != grid_.spec()) {
      nonlocal_grid_ = std::make_unique<dft::MolecularGrid>(ks_molecular_grid(
          system_, *options_.nonlocal_grid_spec, backend_, device,
          options_.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused));
      options_.nonlocal_grid = nonlocal_grid_.get();
    }
    scf::initial_guess::resolve_automatic_preliminary_guess(system_, fock_.strategy(), options_);
    options_.retain_ks_state = backend_ != GENERATIVEQC_BACKEND_CUDA;
    if (execution_plan_.range_exchange) prepare_range_exchange(device);
#if GENERATIVEQC_HAS_CUDA
    if (backend_ == GENERATIVEQC_BACKEND_CUDA && execution_plan_.range_exchange &&
        (!range_strategy_ ||
         !cuda_rsh_provider_compatible(fock_, *range_strategy_, range_correction_.get())))
      throw MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "CUDA range-separated KS requires compatible primary and Direct LR providers");
#endif
    if (execution_plan_.nonlocal_correlation) prepare_nonlocal(device);
#if GENERATIVEQC_HAS_CUDA
    if (backend_ == GENERATIVEQC_BACKEND_CUDA) {
      const auto& metadata = dft::semilocal_family_metadata(execution_plan_.semilocal_family);
      if (execution_plan_.nonlocal_correlation && metadata.cuda_nonlocal_correlation &&
          options_.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::DeviceFused)
        throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                          "public CUDA nonlocal KS requires device-fused XC/nonlocal execution");
      const auto* range = range_strategy_ ? &*range_strategy_ : nullptr;
      const auto domain = metadata.molecular_nonlocal_domain
                              ? dft::nlc::Vv10DensityDomain::MolecularV1
                              : dft::nlc::Vv10DensityDomain::StrictPositive;
      cuda_ = std::make_unique<dft::CudaKsPlan>(
          fock_, basis_, grid_, options_, ks_xc_functional_code(execution_plan_),
          options_.xc_tile_points, range, nonlocal_.get(), domain, dft::CudaXcPreparationBudget{},
          range_correction_.get());
    }
#endif
    if (execution_plan_.d4_correction) prepare_d4(device);
    runtime::sample_cpu_capacity(host_numeric_capacity());
  }

  [[nodiscard]] const std::optional<scf::initial_guess::PreliminaryOptions>& initial_guess_policy()
      const noexcept {
    return options_.preliminary_guess;
  }

  /** Force capability belongs to the immutable prepared context, never the
   * method-global manifest. CPU fitted/Direct PBE/PBE0 and RKS B3LYP retain
   * their existing derivative admission; CUDA requires a complete compiled
   * s/p Direct graph with a matching linked binding and valid bounded capacity.
   * Corrections and graphs without a complete provider inventory remain fail-closed. */
  [[nodiscard]] bool supports_native_gga_force() const noexcept {
    if (backend_ == GENERATIVEQC_BACKEND_CUDA) {
      const auto* binding = admitted_cuda_stationary_binding();
      if (!binding) return false;
      try {
        (void)dft::cuda_stationary_capacity(*binding, basis_.natom, basis_.nao, basis_.nprimitive,
                                            grid_.point_count(), options_.xc_tile_points);
        return true;
      } catch (...) {
        return false;
      }
    }
    if (backend_ != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
        (options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
         options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE) ||
        execution_plan_.automatic_program || execution_plan_.generated_split_hybrid ||
        execution_plan_.range_exchange || execution_plan_.nonlocal_correlation ||
        execution_plan_.d4_correction || !system_.ecp_terms.empty() ||
        std::any_of(system_.atoms.begin(), system_.atoms.end(),
                    [](const auto& atom) { return atom.ecp_core != 0; }))
      return false;
    if (options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE &&
        (!integrals::has_packaged_cpu_eri_shell_derivatives() ||
         std::any_of(system_.shells.begin(), system_.shells.end(),
                     [](const auto& shell) { return shell.angular_momentum > 2; })))
      return false;
    const auto& spec = fock_.strategy().spec;
    if (execution_plan_.spin_channels != (spec.spin == scf::FockSpin::Restricted ? 1u : 2u))
      return false;
    const auto approximation = options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE
                                   ? scf::FockApproximation::Exact
                                   : scf::FockApproximation::DensityFitted;
    const auto admitted_term = [approximation](const scf::FockTermSpec& term) {
      return !term.present || (term.approximation == approximation &&
                               term.op == scf::FockOperator::FullRange && term.omega == 0.0);
    };
    if (!spec.coulomb.present || spec.coulomb.coefficient != 1.0 || !admitted_term(spec.coulomb) ||
        !admitted_term(spec.exchange))
      return false;
    // Normalize the spin-specific Fock K coefficient to the scientific hybrid
    // fraction. Only explicitly qualified manifest profiles are admitted.
    const double exchange_fraction =
        spec.exchange.present
            ? -spec.exchange.coefficient * (execution_plan_.spin_channels == 1 ? 2.0 : 1.0)
            : 0.0;
    return dft::semilocal_family_cpu_stationary_force_qualified(
               execution_plan_.semilocal_family, execution_plan_.spin_channels,
               spec.exchange.present, exchange_fraction, options_.semilocal_exchange_scale,
               options_.semilocal_correlation_scale) &&
           (options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
            scf::cpu_stationary_direct_force_work_admitted(
                system_, spec.exchange.present, grid_.point_count(), options_.xc_tile_points));
  }

  [[nodiscard]] generativeqc_property_flags supported_properties() const noexcept {
    return capabilities_.supported_properties |
           (supports_native_gga_force() ? GENERATIVEQC_PROPERTY_FORCES : 0u);
  }

  const core::System& system() const noexcept { return system_; }
  const std::vector<scf::CudaDensityFittingMetricDiagnostic>& fitted_diagnostics() const noexcept {
    return fock_.diagnostic().fitted;
  }

  /** Explicit retained vectors; object metadata and transient setup are not
   * inferred from this lower-bound observation. Grid/basis buffers are owned. */
  std::size_t host_numeric_capacity() const noexcept {
    auto bytes =
        runtime::add_capacity(fock_.cpu_observation_capacity(),
                              runtime::vector_capacities(basis_.packed, grid_.points(),
                                                         grid_.weights(), grid_.owners(), warm_));
    bytes = runtime::add_capacity(bytes, integral_response_cache_.bytes());
    if (nonlocal_grid_)
      bytes = runtime::add_capacity(
          bytes, runtime::vector_capacities(nonlocal_grid_->points(), nonlocal_grid_->weights(),
                                            nonlocal_grid_->owners()));
    if (range_correction_)
      bytes = runtime::add_capacity(bytes, range_correction_->cpu_observation_capacity());
#if GENERATIVEQC_HAS_CUDA
    if (range_derivative_correction_)
      bytes =
          runtime::add_capacity(bytes, range_derivative_correction_->cpu_observation_capacity());
#endif
    if (cpu_physical_)
      for (const auto* matrices : {&cpu_physical_->density, &cpu_physical_->fock})
        for (const auto& matrix : *matrices)
          bytes = runtime::add_capacity(bytes, runtime::vector_bytes(matrix));
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) bytes = runtime::add_capacity(bytes, cuda_->resources().retained_host_numeric_bytes);
#endif
    if (d4_) {
      const auto& resources = d4_->resources();
      bytes = runtime::add_capacity(bytes, static_cast<std::size_t>(resources.plan_host_bytes));
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.execution_host_bytes));
    }
    if (nonlocal_) {
      const auto& resources = nonlocal_->resources();
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.host_workspace_bytes));
    }
    return bytes;
  }

  /** Explicit output/rebuild export. Ordinary CUDA replays keep this on device. */
  std::vector<double> warm_density() {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->warm_density();
#endif
    return warm_;
  }

  void clear_warm_start() noexcept {
    warm_.clear();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) cuda_->clear_warm_start();
#endif
  }

  std::optional<std::vector<double>> prepare_cold_initial_guess(
      scf::initial_guess::PreliminaryDiagnostic& diagnostic) const {
    if (!options_.preliminary_guess) return std::nullopt;
    scf::initial_guess::validate_preliminary_target(system_, fock_.strategy(), options_);
    diagnostic.requested_kind = static_cast<std::uint32_t>(options_.preliminary_guess->kind);
    diagnostic.target_attempts = 1;
    const auto started = std::chrono::steady_clock::now();
    try {
      runtime::CpuRetainedCapacity retained(host_numeric_capacity());
      scf::initial_guess::EigenOperation eigen;
#if GENERATIVEQC_HAS_CUDA
      // The prepared CUDA owner already charges its idle solver workspace.
      // Borrow it synchronously rather than running CPU Jacobi for a CUDA seed.
      if (cuda_) eigen = cuda_->seed_eigen_operation();
#endif
      auto density = scf::initial_guess::prepare_preliminary_density(
          fock_, *options_.preliminary_guess, diagnostic, eigen);
      diagnostic.preparation_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return density;
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const generativeqc::Error& error) {
      if (error.status() != GENERATIVEQC_STATUS_NUMERICAL_FAILURE) throw;
      diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::PreparationFailed;
      diagnostic.work_counters_complete = false;
      diagnostic.preparation_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return std::nullopt;
    } catch (const std::exception&) {
      diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::PreparationFailed;
      diagnostic.work_counters_complete = false;
      diagnostic.preparation_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return std::nullopt;
    }
  }

  void invalidate_final_state() noexcept {
    cpu_physical_.reset();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) cuda_->invalidate_final_state();
#endif
  }

#if GENERATIVEQC_HAS_CUDA
  dft::CudaKsPlan* cuda_plan() noexcept { return cuda_.get(); }
#endif

  generativeqc_status final_state_token(dft::CudaKsFinalStateToken& token,
                                        std::string& detail) const override {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->final_state_token(token, detail);
#endif
    token = {};
    if (!cpu_physical_) {
      detail = "CPU KS owner has no successful current final state";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    token = {1, cpu_physical_->identity};
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density,
                                       dft::VerifiedKsFinalState& state,
                                       std::string& detail) override {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->read_final_state(expected, compute_weighted_density, state, detail);
#endif
    state = {};
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (expected != current) {
      detail = "CPU KS final-state token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    // The SCF frame predates the last F[D] rebuild. Diagonalize that actual
    // retained physical F here; do not relabel the lagged orbital energies.
    // Explicit export costs one eigen solve and validation, zero Fock builds.
    const auto& ints = fock_.one_electron();
    const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
    std::vector<scf::reference::EigenResult> spins;
    spins.reserve(current.identity.model.spins);
    for (const auto& fock : cpu_physical_->fock)
      spins.push_back(scf::reference::generalized_eigen(fock, x, ints.nbf));
    dft::KsFinalStateCandidate candidate{current.identity,
                                         current.identity.determinant.factor.density_generation,
                                         true, std::move(spins)};
    scf::solver::FinalStateLimits limits{options_.density_tolerance, options_.energy_tolerance, 0,
                                         true};
    if (!dft::validate_ks_final_state(current.identity, ints.overlap, ints.hcore, *cpu_physical_,
                                      candidate, limits, compute_weighted_density, state, detail)) {
      invalidate_final_state();
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status resident_orbital_frame(const dft::CudaKsFinalStateToken& expected,
                                             dft::CudaKsResidentOrbitalFrameBinding& binding,
                                             std::string& detail, bool prepare) {
    binding = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      if (prepare) {
        dft::KsFinalStateDiagnostic diagnostic;
        return cuda_->prepare_resident_final_orbital_frame(expected, binding, diagnostic, detail);
      }
      return cuda_->resident_final_orbital_frame(expected, binding, detail);
    }
#endif
    detail = "resident orbital frame requires a native CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status resident_fock_sources(const dft::CudaKsFinalStateToken& expected,
                                            dft::CudaKsResidentFockBinding& binding,
                                            std::string& detail) {
    binding = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->resident_final_fock_sources(expected, binding, detail);
#endif
    detail = "resident Fock sources require a native CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status resident_grid_reference(const dft::CudaKsFinalStateToken& expected,
                                              dft::CudaKsResidentGridBinding& binding,
                                              std::string& detail) {
    binding = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->resident_final_grid(expected, binding, detail);
#endif
    detail = "resident basis/grid requires a native CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status resident_density(const dft::CudaKsFinalStateToken& expected, int& device,
                                       const double*& alpha, const double*& beta,
                                       std::size_t& matrix_elements, unsigned& spins,
                                       void*& source_stream, std::string& detail) {
    device = -1;
    alpha = nullptr;
    beta = nullptr;
    matrix_elements = 0;
    spins = 0;
    source_stream = nullptr;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident final density requires a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentDensityBinding binding;
    const auto status = cuda_->resident_final_density(expected, binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!binding) {
      detail = "CUDA KS resident final-density binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = binding.device_id;
    alpha = binding.alpha;
    beta = binding.beta;
    matrix_elements = binding.matrix_elements;
    spins = binding.spins;
    source_stream = binding.stream;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident final density requires a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status fixed_density_profile(const dft::CudaKsFinalStateToken& expected,
                                            KsFixedDensityProfile& profile, std::string& detail) {
    profile = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      dft::CudaKsFixedDensityProfile native;
      const auto status = cuda_->profile_fixed_density_components(expected, native, detail);
      if (status == GENERATIVEQC_STATUS_SUCCESS) {
        profile.milliseconds = native.milliseconds;
        profile.present_mask = native.present_mask;
      }
      return status;
    }
#endif
    detail = "fixed-density component profiling requires a CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status resident_grid(const dft::CudaKsFinalStateToken& expected, int& device,
                                    const double*& points, const double*& weights,
                                    const double*& atomic_weights, std::size_t& point_count,
                                    std::string& detail) {
    device = -1;
    points = nullptr;
    weights = nullptr;
    atomic_weights = nullptr;
    point_count = 0;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident molecular grid requires a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    // Reuse the exact final-density lease as the zero-transfer token gate. The
    // grid is immutable for this prepared KS owner and survives for its lifetime.
    dft::CudaKsResidentDensityBinding density_binding;
    const auto status = cuda_->resident_final_density(expected, density_binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto grid = grid_.cuda_view();
    if (density_binding && !grid &&
        options_.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::HostUnfused) {
      detail = "host-unfused KS owner has no resident molecular grid";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (!density_binding || !grid || grid.device != density_binding.device_id) {
      detail = "CUDA KS resident molecular-grid binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = grid.device;
    points = grid.points;
    weights = grid.weights;
    atomic_weights = grid.atomic_weights;
    point_count = grid.point_count;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident molecular grid requires a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status resident_nonlocal_features(const dft::CudaKsFinalStateToken& expected,
                                                 int& device, const double*& density,
                                                 const double*& gradient, std::size_t& point_count,
                                                 void*& source_stream, std::string& detail) {
    device = -1;
    density = nullptr;
    gradient = nullptr;
    point_count = 0;
    source_stream = nullptr;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident nonlocal features require a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentNonlocalFeaturesBinding binding;
    const auto status = cuda_->resident_final_nonlocal_features(expected, binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!binding) {
      detail = "CUDA KS resident nonlocal feature binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = binding.device_id;
    density = binding.density;
    gradient = binding.gradient;
    point_count = binding.point_count;
    source_stream = binding.stream;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident nonlocal features require a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status read_derivative_state(const dft::CudaKsFinalStateToken& expected,
                                            KsDerivativeSnapshot& output, std::string& detail) {
    output = {};
    dft::VerifiedKsFinalState state;
#if GENERATIVEQC_HAS_CUDA
    const auto before = cuda_ ? cuda_->transfers() : dft::CudaKsTransfers{};
#endif
    const auto status = read_final_state(expected, true, state, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto sources_status = read_derivative_sources(expected, output, detail);
    if (sources_status != GENERATIVEQC_STATUS_SUCCESS) return sources_status;
    output.state = std::move(state);
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      const auto after = cuda_->transfers();
      output.export_d2h_bytes = after.final_state_d2h_bytes - before.final_state_d2h_bytes;
      output.export_reads = after.final_state_reads - before.final_state_reads;
      output.export_synchronizations = after.synchronizations - before.synchronizations;
    }
#endif
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status read_derivative_sources(const dft::CudaKsFinalStateToken& expected,
                                              KsDerivativeSnapshot& output, std::string& detail) {
    output = {};
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (current != expected) {
      detail = "KS derivative sources have a stale final-state token";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    // These are the provider's actual metric and the collocation/grid sources
    // used by this immutable KS owner, not caller-supplied identity labels.
    output.system = system_;
    output.overlap = fock_.one_electron().overlap;
    output.packed_basis = basis_.packed;
    output.points = grid_.points();
    output.weights = grid_.weights();
    output.grid_owners = grid_.owners();
    // Both backends collocate this owner's exact host-built quadrature. Export
    // its raw measures directly; dividing partitioned weights loses tail data.
    output.atomic_weights = grid_.atomic_weights();
    output.resident_grid = grid_.cuda_view();
    if (nonlocal_grid_) {
      output.nonlocal_points = nonlocal_grid_->points();
      output.nonlocal_weights = nonlocal_grid_->weights();
      output.nonlocal_grid_owners = nonlocal_grid_->owners();
      output.nonlocal_atomic_weights = nonlocal_grid_->atomic_weights();
      output.nonlocal_resident_grid = nonlocal_grid_->cuda_view();
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status density_fitted_integral_gradient(
      const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      bool allow_mixed_rsh_primary = false,
      std::optional<std::size_t> one_electron_device_maximum_bytes = std::nullopt) {
    output.clear();
    work = {};
    if (options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
        !system_.ecp_terms.empty() ||
        ((execution_plan_.range_exchange || execution_plan_.nonlocal_correlation) &&
         !allow_mixed_rsh_primary) ||
        (allow_mixed_rsh_primary && !execution_plan_.range_exchange)) {
      detail =
          "density-fitted stationary derivatives require qualified all-electron full-range primary";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto& strategy = fock_.strategy();
    const auto fitted_term = [](const scf::FockTermSpec& term) {
      return !term.present || (term.approximation == scf::FockApproximation::DensityFitted &&
                               term.op == scf::FockOperator::FullRange);
    };
    if (!fitted_term(strategy.spec.coulomb) || !fitted_term(strategy.spec.exchange)) {
      detail = "density-fitted stationary derivative provider identity mismatch";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsFinalStateToken current;
    auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (current != expected) {
      detail = "density-fitted stationary derivative token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto spins = expected.identity.model.spins;
    const auto& one = fock_.one_electron();
    const auto matrix_elements = one.nbf * one.nbf;
    const auto coordinates = 3 * system_.atoms.size();
    if (spins < 1 || spins > 2 || density.size() != spins || weighted_density.size() != spins) {
      detail = "density-fitted stationary D/W shape mismatch";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    for (unsigned spin = 0; spin < spins; ++spin)
      if (density[spin].size() != matrix_elements ||
          weighted_density[spin].size() != matrix_elements) {
        detail = "density-fitted stationary D/W AO shape mismatch";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
    // H'/Pulay/J'/K' vectors coexist with the four-source publication buffer.
    // This allowance covers only compact publication, not DF-provider scratch.
    if (!maximum_bytes || coordinates > maximum_bytes / (8U * sizeof(double))) {
      detail = "density-fitted stationary derivative publication exceeds its budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }

    std::vector<double> hcore, pulay;
    std::size_t one_electron_device_bytes = 0;
    std::size_t one_electron_h2d_bytes = 0;
    std::size_t one_electron_d2h_bytes = 0;
    bool resident_one_electron = false;
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      scf::OneElectronGradientResources one_electron;
      dft::CudaKsResidentStationaryWeightsBinding resident_weights;
      const auto resident_status =
          cuda_->resident_final_stationary_weights(expected, resident_weights, detail);
      if (resident_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!resident_weights || resident_weights.device_id != expected.identity.model.device ||
            resident_weights.matrix_elements != matrix_elements ||
            resident_weights.spins != spins) {
          detail = "CUDA DF stationary D/W binding is incompatible with the final KS state";
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
        }
        // The standalone paired bridge supports the same generated schedules
        // as the prepared Direct owner. Honor its existing execution policy;
        // forcing thread-per-AO here serializes every attraction-center sum.
        const auto one_status = scf::execute_cuda_stationary_one_electron_pair(
            resident_weights.device_id, system_, {}, {},
            scf::cuda_policy::one_electron_derivative_mapping_requested(),
            one_electron_device_maximum_bytes.value_or(maximum_bytes), hcore, pulay, detail,
            &one_electron, resident_weights.density, resident_weights.weighted_density);
        if (one_status == GENERATIVEQC_STATUS_SUCCESS) {
          resident_one_electron = true;
          one_electron_device_bytes = one_electron.device_bytes;
          one_electron_h2d_bytes = one_electron.host_to_device_bytes;
          one_electron_d2h_bytes = one_electron.device_to_host_bytes;
        } else if (one_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                   one_status != GENERATIVEQC_STATUS_OUT_OF_MEMORY) {
          return one_status;
        }
      } else if (resident_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        return resident_status;
      }
    }
#endif
    if (!resident_one_electron) {
      // CPU and bounded CUDA fallback preserve the established exact host
      // contraction. CUDA only reaches this branch when the optional resident
      // one-electron consumer cannot be admitted; J/K response ownership is
      // unaffected.
      fock_.ensure_one_electron_derivatives();
      if (one.ncoord != coordinates ||
          one.hcore_derivative.size() != coordinates * matrix_elements ||
          one.overlap_derivative.size() != coordinates * matrix_elements) {
        detail = "density-fitted stationary one-electron derivative shape mismatch";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      hcore.assign(coordinates, 0.0);
      pulay.assign(coordinates, 0.0);
      for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
        const auto* dh = one.hcore_derivative.data() + coordinate * matrix_elements;
        const auto* ds = one.overlap_derivative.data() + coordinate * matrix_elements;
        for (unsigned spin = 0; spin < spins; ++spin)
          for (std::size_t item = 0; item < matrix_elements; ++item) {
            hcore[coordinate] += density[spin][item] * dh[item];
            pulay[coordinate] -= weighted_density[spin][item] * ds[item];
          }
      }
      detail.clear();
    }

    const std::vector<double> empty;
    scf::FockEnergyDerivativeComponents two;
    std::optional<scf::CudaDfBorrowedResponseDensity> response_density;
    std::optional<scf::CudaDfBorrowedFittedProjection> fitted_projection;
#if GENERATIVEQC_HAS_CUDA
    if (cuda_ && spins == 1) {
      dft::CudaKsResidentDensityBinding lease;
      std::string lease_detail;
      const auto lease_status = cuda_->resident_final_density(expected, lease, lease_detail);
      if (lease_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!lease || lease.spins != 1 || lease.matrix_elements != matrix_elements ||
            lease.device_id != expected.identity.model.device) {
          detail = "CUDA KS returned an incompatible final density lease for DF response";
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
        }
        response_density.emplace(scf::CudaDfBorrowedResponseDensity{
            lease.device_id, lease.alpha, lease.matrix_elements, lease.stream});
      } else if (lease_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        detail = lease_detail;
        return lease_status;
      }
    }
    if (cuda_ && spins == 1 && strategy.spec.exchange.present) {
      dft::CudaKsResidentFittedProjectionBinding lease;
      std::string lease_detail;
      const auto lease_status =
          cuda_->resident_final_fitted_projection(expected, lease, lease_detail);
      if (lease_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!lease) {
          detail = "CUDA KS returned an invalid final fitted projection lease";
          return GENERATIVEQC_STATUS_INTERNAL_ERROR;
        }
        fitted_projection.emplace(scf::CudaDfBorrowedFittedProjection{
            lease.device_id, lease.occupied_coefficients, lease.projection, lease.nbf, lease.naux,
            lease.rank, lease.stream});
      } else if (lease_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                 lease_status != GENERATIVEQC_STATUS_INVALID_ARGUMENT) {
        detail = lease_detail;
        return lease_status;
      }
    }
#endif
    try {
      two = response_density ? fock_.energy_derivative_components_with_cuda_df_state(
                                   density[0], empty, &*response_density,
                                   fitted_projection ? &*fitted_projection : nullptr)
            : fitted_projection
                ? fock_.energy_derivative_components_with_fitted_projection(density[0], empty,
                                                                            *fitted_projection)
                : fock_.energy_derivative_components(density[0], spins == 2 ? density[1] : empty);
    } catch (const std::bad_alloc&) {
      detail = "density-fitted stationary response exceeded the prepared resource budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& error) {
      detail = error.what();
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      detail = error.what();
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    if (two.coulomb.size() != coordinates || two.exchange.size() != coordinates) {
      detail = "density-fitted stationary J/K derivative source shape mismatch";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    std::vector<double> candidate;
    candidate.reserve(4 * coordinates);
    std::size_t publication_remaining_bytes = maximum_bytes;
    for (const auto* values : {&hcore, &pulay, &two.coulomb, &two.exchange, &candidate}) {
      if (values->capacity() > publication_remaining_bytes / sizeof(double)) {
        detail = "density-fitted stationary derivative publication exceeds its budget";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      }
      publication_remaining_bytes -= values->capacity() * sizeof(double);
    }
    const auto publication_peak_bytes = maximum_bytes - publication_remaining_bytes;
    candidate.insert(candidate.end(), hcore.begin(), hcore.end());
    candidate.insert(candidate.end(), pulay.begin(), pulay.end());
    candidate.insert(candidate.end(), two.coulomb.begin(), two.coulomb.end());
    candidate.insert(candidate.end(), two.exchange.begin(), two.exchange.end());
    if (!std::all_of(candidate.begin(), candidate.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "density-fitted stationary derivative source is nonfinite";
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    // H'/S' derivative tensors remain retained host provider data. CUDA may
    // instead contract the exact final resident D/W with the paired device
    // consumer; slots 2/4/5 report that consumer's device peak and actual
    // metadata/output movement. DF J/K response scratch/transfers remain
    // separate and are not inferred from spin dimensions.
    work[0] = fock_.diagnostic().device_bytes;
    work[2] = one_electron_device_bytes;
    work[3] = publication_peak_bytes;
    work[4] = one_electron_h2d_bytes;
    work[5] = one_electron_d2h_bytes;
    output = std::move(candidate);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  /** Reuse the exact retained response providers without claiming a complete
   * analytic force. The validated single-system snapshot supplies D/W and the
   * live token proves the SCF Hamiltonian and basis have not changed. */
  generativeqc_status prepared_integral_gradient(
      const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) override {
    output.clear();
    work = {};
    if (!system_.ecp_terms.empty() ||
        std::any_of(system_.atoms.begin(), system_.atoms.end(),
                    [](const auto& atom) { return atom.ecp_core != 0; }) ||
        execution_plan_.range_exchange || execution_plan_.nonlocal_correlation ||
        execution_plan_.d4_correction) {
      detail = "native stationary integral sources require uncorrected all-electron full-range KS";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (current != expected) {
      detail = "native stationary integral source token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    if (options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE)
      return density_fitted_integral_gradient(expected, density, weighted_density, output,
                                              maximum_bytes, work, detail);
#if GENERATIVEQC_HAS_CUDA
    if (cuda_)
      return cuda_integral_gradient(expected, output, maximum_bytes, work, detail, &density,
                                    &weighted_density);
#endif
    if (!supports_native_gga_force()) {
      detail = "CPU Direct KS stationary integral sources require a qualified GGA owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto spins = expected.identity.model.spins;
    if (density.size() != spins || weighted_density.size() != spins || (spins != 1 && spins != 2)) {
      detail = "CPU Direct stationary D/W spin shape mismatch";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const std::span<const double> empty;
    auto candidate = scf::cpu_stationary_integral_gradient(
        fock_, density[0], weighted_density[0], maximum_bytes,
        spins == 2 ? std::span<const double>(density[1]) : empty,
        spins == 2 ? std::span<const double>(weighted_density[1]) : empty);
    work[3] = candidate.numeric_capacity;
    output = std::move(candidate.sources);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  /** Authenticate the restricted exact full-range derivative source. Pure J
   * uses the same spatial recurrence, including when a separate D4/gCP owner
   * contributes geometry corrections. Their forces do not change this source. */
  bool qualified_spatial_force_default(
      const scf::DirectForcePrecisionRequest* precision = nullptr) const {
    const auto& strategy = fock_.strategy();
    const auto& spec = strategy.spec;
    const bool pure_coulomb = !spec.exchange.present;
    const bool global_hybrid =
        spec.exchange.present &&
        dft::restricted_global_hybrid_exchange_coefficient(spec.exchange.coefficient) &&
        spec.exchange.approximation == scf::FockApproximation::Exact &&
        spec.exchange.op == scf::FockOperator::FullRange && !execution_plan_.d4_correction;
    const bool direct_full_range =
        (pure_coulomb || global_hybrid) && strategy.backend == scf::FockBackend::Cuda &&
        spec.derivative_order == 0 && spec.coulomb.present && spec.coulomb.coefficient == 1.0 &&
        spec.coulomb.approximation == scf::FockApproximation::Exact &&
        spec.coulomb.op == scf::FockOperator::FullRange &&
        options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE &&
        !execution_plan_.range_exchange && !range_strategy_ &&
        !execution_plan_.nonlocal_correlation && system_.ecp_terms.empty() &&
        std::none_of(system_.atoms.begin(), system_.atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; });
    // This schedule contracts the authenticated final density. Its arithmetic
    // follows the force invocation, not the earlier SCF proposal precision or
    // availability of a fused stationary binding. Python's strict fallback and
    // a mixed-force retry therefore retain the same FP64 source optimization.
    const bool strict_force = !precision || (!precision->mixed_weighted &&
                                             !precision->mixed_geometry && !precision->mixed_grid);
    return dft::spatial_spd_force_default_eligible(
        fock_.diagnostic().nbf, execution_plan_.spin_channels, direct_full_range, strict_force);
  }

  generativeqc_status cuda_full_range_integral_derivatives(
      const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::string& detail) {
#if GENERATIVEQC_HAS_CUDA
    output.clear();
    if (!cuda_ || !system_.ecp_terms.empty() || range_strategy_ ||
        options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE) {
      detail = "CUDA full-range stationary shell derivatives require an all-electron Direct owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto derivative_source = scf::prepared_cuda_direct_derivative_binding(fock_);
    if (!derivative_source) {
      detail = "prepared CUDA Fock owner has no retained Direct shell derivative lease";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentDensityBinding resident_density;
    auto status = cuda_->resident_final_density(expected, resident_density, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_density || resident_density.device_id != derivative_source.device_id ||
        resident_density.matrix_elements != derivative_source.nbf * derivative_source.nbf) {
      detail = "CUDA full-range stationary density is incompatible with the Direct shell owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    status = scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
        fock_, resident_density.alpha, resident_density.beta, resident_density.matrix_elements,
        output, detail, true, qualified_spatial_force_default());
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      output.clear();
      return status;
    }
    const auto coordinates = 3 * system_.atoms.size();
    if (output.size() != 2 * coordinates) {
      output.clear();
      detail = "prepared CUDA full-range shell derivative source shape mismatch";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)expected;
    (void)output;
    detail = "CUDA full-range stationary shell derivatives are unavailable in this build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status cuda_prepare_force_memory(const dft::CudaKsFinalStateToken& expected,
                                                std::size_t additional_device_bytes,
                                                std::array<std::uint64_t, 4>& receipt,
                                                std::string& detail) {
    receipt = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_)
      return cuda_->prepare_force_memory(expected, additional_device_bytes, receipt, detail);
#else
    (void)expected;
    (void)additional_device_bytes;
#endif
    detail = "CUDA force-memory handoff requires a CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status cuda_integral_gradient_additional_device_bytes(
      const dft::CudaKsFinalStateToken& expected, std::size_t& bytes, std::string& detail) {
    bytes = 0;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_ || backend_ != GENERATIVEQC_BACKEND_CUDA) {
      detail = "CUDA integral derivative capacity requires a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (current != expected) {
      detail = "CUDA integral derivative capacity token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    if (execution_plan_.range_exchange &&
        options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE) {
      if (!range_strategy_ || !range_correction_ || !range_correction_->matches_system(system_) ||
          range_correction_->strategy() != *range_strategy_ ||
          fock_.strategy().backend != scf::FockBackend::Cuda ||
          fock_.strategy().spec.coulomb.approximation != scf::FockApproximation::DensityFitted) {
        detail = "mixed RSH-DF derivative capacity lacks its resolved CUDA providers";
        return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      }
      // Keep cold and warm admission identical: this is the same conservative
      // capacity used before constructing the separate first-order LR owner.
      bytes = execution_plan_.density_fitted_range_exchange
                  ? 0U
                  : ks_provider_bytes(system_, backend_, 1U, KsProviderPurpose::derivative_only);
    }
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)expected;
    detail = "CUDA integral derivative capacity requires CUDA";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  /** Stationary bridge for a full-range DF primary and an independent LR
   * owner (exact by default, fitted only by explicit opt-in). Admission separately qualifies the
   * complete semilocal/nonlocal/moving-grid inventory; this source bridge is not a C force API.
   *
   * Only legacy Direct LR needs a separate lazy derivative owner. Fitted LR
   * retains its own bounded response capability with its forward metric. */
  generativeqc_status mixed_rsh_df_integral_gradient(
      const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density,
      const std::vector<scf::reference::Matrix>* cached_weighted_density,
      std::span<const double> cpu_lr_correction = {}) {
    output.clear();
    work = {};
    // AUTO retains its descriptor spelling after resolving a CUDA DF primary.
    // Backend/approximation identities below, not that spelling, own execution.
    if (!range_strategy_ || !range_correction_ ||
        (backend_ != GENERATIVEQC_BACKEND_CPU_REFERENCE && backend_ != GENERATIVEQC_BACKEND_CUDA) ||
        options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
        options_.precision_mode != GENERATIVEQC_PRECISION_FP64 || !execution_plan_.range_exchange ||
        !system_.ecp_terms.empty() || !range_correction_->matches_system(system_) ||
        range_correction_->strategy() != *range_strategy_) {
      detail = "mixed RSH-DF stationary sources require live strict-FP64 providers";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto& primary = fock_.strategy();
    const auto& p = primary.spec;
    const auto& correction = *range_strategy_;
    const auto& c = correction.spec;
    const double spin_factor = p.spin == scf::FockSpin::Restricted ? -0.5 : -1.0;
    const double full_coefficient = p.exchange.present ? p.exchange.coefficient : 0.0;
    const auto provider_backend =
        backend_ == GENERATIVEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    const bool matched =
        primary.backend == provider_backend && correction.backend == provider_backend &&
        primary.screening_tolerance == correction.screening_tolerance && p.spin == c.spin &&
        p.derivative_order == 0 && c.derivative_order == 0 && p.coulomb.present &&
        p.coulomb.coefficient == 1.0 && p.coulomb.op == scf::FockOperator::FullRange &&
        p.coulomb.approximation == scf::FockApproximation::DensityFitted &&
        (!p.exchange.present ||
         (p.exchange.op == scf::FockOperator::FullRange &&
          p.exchange.approximation == scf::FockApproximation::DensityFitted)) &&
        !c.coulomb.present && c.exchange.present &&
        (c.exchange.approximation == scf::FockApproximation::Exact ||
         (execution_plan_.density_fitted_range_exchange &&
          c.exchange.approximation == scf::FockApproximation::DensityFitted &&
          correction.metric_relative_threshold == primary.metric_relative_threshold)) &&
        c.exchange.op == scf::FockOperator::LongRange &&
        c.exchange.omega == execution_plan_.range_omega && c.exchange.omega > 0.0 &&
        full_coefficient == spin_factor * execution_plan_.short_range_exchange &&
        c.exchange.coefficient == spin_factor * (execution_plan_.long_range_exchange -
                                                 execution_plan_.short_range_exchange) &&
        std::isfinite(full_coefficient) && std::isfinite(c.exchange.coefficient) &&
        c.exchange.coefficient != 0.0 &&
        expected.identity.model.spins == (p.spin == scf::FockSpin::Restricted ? 1U : 2U);
    if (!matched) {
      detail = "mixed RSH-DF primary/LR correction have incompatible resolved Fock identities";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto nc = 3 * system_.atoms.size();
    // Four primary rows, one LR row and five result rows can coexist.
    if (!maximum_bytes || nc > maximum_bytes / (10U * sizeof(double))) {
      detail = "mixed RSH-DF compact source publication exceeds its host budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    if ((cached_density == nullptr) != (cached_weighted_density == nullptr)) {
      detail = "mixed RSH-DF cached D/W must be provided together";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::VerifiedKsFinalState exported;
    if (!cached_density) {
      const auto status = read_final_state(expected, true, exported, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      cached_density = &exported.density;
      cached_weighted_density = &exported.weighted_density;
    }
    if (execution_plan_.density_fitted_range_exchange) {
      if (!cpu_lr_correction.empty()) {
        detail = "full RI-K response must not consume a Direct LR correction";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      std::vector<double> fitted_four;
      auto status =
          density_fitted_integral_gradient(expected, *cached_density, *cached_weighted_density,
                                           fitted_four, maximum_bytes, work, detail, true);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      const auto& alpha = cached_density->at(0);
      const scf::reference::Matrix beta =
          cached_density->size() == 2 ? cached_density->at(1) : scf::reference::Matrix{};
      const auto fitted_lr = range_correction_->retained_energy_derivative(alpha, beta);
      status = dft::compose_mixed_rsh_df_integral_sources(
          fitted_four, fitted_lr, nc, full_coefficient, c.exchange.coefficient, output, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      work[3] = std::max<std::uint64_t>(work[3], 10U * nc * sizeof(double));
      dft::CudaKsFinalStateToken current;
      status = final_state_token(current, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS || current != expected) {
        output.clear();
        detail = "RI-K source token changed during response";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      return GENERATIVEQC_STATUS_SUCCESS;
    }
    if (backend_ == GENERATIVEQC_BACKEND_CPU_REFERENCE) {
      // CPU executes the compiler-owned ordered LR contraction in bounded
      // tiles. Only its compact weighted result crosses this private bridge;
      // provider identity and DF/SR/LR recomposition are shared with CUDA.
      if (cpu_lr_correction.size() != nc) {
        detail = "mixed RSH-DF CPU source requires one compact LR correction";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      std::vector<double> fitted_four;
      auto status =
          density_fitted_integral_gradient(expected, *cached_density, *cached_weighted_density,
                                           fitted_four, maximum_bytes, work, detail, true);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      status = dft::compose_mixed_rsh_df_integral_sources(fitted_four, cpu_lr_correction, nc,
                                                          full_coefficient, c.exchange.coefficient,
                                                          output, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      work[3] = std::max<std::uint64_t>(work[3], 10U * nc * sizeof(double));
      dft::CudaKsFinalStateToken current;
      status = final_state_token(current, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS || current != expected) {
        output.clear();
        detail = "mixed RSH-DF CPU source token changed during response";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    }
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_ || !cpu_lr_correction.empty()) {
      detail = "mixed RSH-DF CUDA source requires its resident density owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::CudaKsResidentDensityBinding resident;
    auto status = cuda_->resident_final_density(expected, resident, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto nbf = fock_.diagnostic().nbf;
    if (!resident || resident.device_id != expected.identity.model.device ||
        resident.matrix_elements != nbf * nbf || resident.spins != expected.identity.model.spins ||
        !resident.stream) {
      detail = "mixed RSH-DF resident density disagrees with the live KS token";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }

    // The independent value-side range owner remains untouched; the
    // derivative-only lease is a separate, lazy provider with identical
    // operator/omega/spin/geometry and a charged direct-workspace budget.
    // This derivative owner was not paid for by the energy-only SCF plan.
    // Charge it on every call, including reuse, before creating any storage;
    // maximum_bytes independently bounds compact host and additional device use.
    const auto derivative_device_bytes =
        range_derivative_correction_
            ? range_derivative_correction_->diagnostic().device_bytes
            : ks_provider_bytes(system_, backend_, 1U, KsProviderPurpose::derivative_only);
    std::size_t primary_device_bytes{};
    status = dft::mixed_rsh_df_primary_device_allowance(maximum_bytes, derivative_device_bytes,
                                                        primary_device_bytes, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!range_derivative_correction_) {
      try {
        range_derivative_correction_ = std::make_unique<scf::PreparedFockPlan>(
            system_, nullptr, *range_strategy_, resident.device_id, derivative_device_bytes, 1U);
      } catch (const std::bad_alloc&) {
        detail = "mixed RSH-DF LR derivative source exceeded its provider budget";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      } catch (const std::exception& error) {
        detail = error.what();
        return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
      }
    }
    const auto lr_binding =
        scf::prepared_cuda_direct_derivative_binding(*range_derivative_correction_);
    if (!lr_binding || lr_binding.device_id != resident.device_id || lr_binding.nbf != nbf ||
        range_derivative_correction_->strategy() != correction ||
        !range_derivative_correction_->matches_system(system_) ||
        lr_binding.retained_device_bytes > derivative_device_bytes) {
      detail = "mixed RSH-DF independent LR source lacks a matching first-order lease";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }

    std::vector<double> fitted_four;
    status = density_fitted_integral_gradient(expected, *cached_density, *cached_weighted_density,
                                              fitted_four, maximum_bytes, work, detail, true,
                                              primary_device_bytes);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;

    // The KS final D is produced on its own stream. The isolated LR gradient
    // kernel must wait on that exact producer before dereferencing resident D.
    // An event records the dependency without a host/device-wide synchronize.
    auto cuda_status = cudaSetDevice(resident.device_id);
    if (cuda_status != cudaSuccess) {
      detail = std::string("mixed RSH-DF device selection: ") + cudaGetErrorString(cuda_status);
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    cudaEvent_t density_ready{};
    cuda_status = cudaEventCreateWithFlags(&density_ready, cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      detail = std::string("mixed RSH-DF density event: ") + cudaGetErrorString(cuda_status);
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    const auto producer = static_cast<cudaStream_t>(resident.stream);
    const auto recorded = cudaEventRecord(density_ready, producer);
    const auto waited = recorded == cudaSuccess
                            ? cudaStreamWaitEvent(lr_binding.stream, density_ready, 0)
                            : recorded;
    (void)cudaEventDestroy(density_ready);
    if (waited != cudaSuccess) {
      detail = std::string("mixed RSH-DF LR density dependency: ") + cudaGetErrorString(waited);
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }

    std::vector<double> direct_lr;
    status = scf::execute_prepared_cuda_direct_long_range_derivatives_device(
        *range_derivative_correction_, resident.alpha, resident.beta, resident.matrix_elements,
        direct_lr, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = dft::compose_mixed_rsh_df_integral_sources(
        fitted_four, direct_lr, nc, full_coefficient, c.exchange.coefficient, output, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    // Slot 2 is the mixed bridge's additional-device peak: its lazy LR
    // owner coexists with one-electron staging. DF J/K response resources
    // remain outside this deliberately partial nine-slot resource scope.
    if (work[2] > maximum_bytes || lr_binding.retained_device_bytes > maximum_bytes - work[2]) {
      output.clear();
      detail = "mixed RSH-DF additional derivative storage exceeds its device budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    // Slot 0 remains the primary SCF-owned bytes; do not double-charge LR.
    work[2] += lr_binding.retained_device_bytes;
    work[3] = std::max<std::uint64_t>(work[3], 10U * nc * sizeof(double));
    dft::CudaKsFinalStateToken current;
    status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS || current != expected) {
      output.clear();
      detail = "mixed RSH-DF final-state source token changed during response";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)expected;
    (void)output;
    (void)maximum_bytes;
    (void)work;
    (void)cached_density;
    (void)cached_weighted_density;
    detail = "mixed RSH-DF stationary sources require CUDA";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status cuda_integral_gradient(
      const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density = nullptr,
      const std::vector<scf::reference::Matrix>* cached_weighted_density = nullptr,
      bool combined_two_electron = false) {
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_ || !system_.ecp_terms.empty()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const bool fitted = options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE;
    if (fitted && execution_plan_.range_exchange && !combined_two_electron)
      return mixed_rsh_df_integral_gradient(expected, output, maximum_bytes, work, detail,
                                            cached_density, cached_weighted_density);
    if (fitted && ks_fitted_derivative_order(fock_.strategy(), backend_) == 0) {
      detail =
          "CUDA density-fitted stationary derivatives require the qualified semilocal DF-J domain";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const bool range_exchange = execution_plan_.range_exchange;
    if (combined_two_electron && (fitted || range_exchange)) {
      detail = "combined stationary two-electron derivative requires full-range Direct sources";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (range_exchange != range_strategy_.has_value()) {
      detail = "CUDA stationary integral gradient has inconsistent range-exchange ownership";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if ((cached_density == nullptr) != (cached_weighted_density == nullptr)) {
      detail = "cached CUDA stationary D/W must be supplied together";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    // Only the already exported, token-bound Direct RSH density is a key.
    // This experiment introduces no new host density export and never reuses W.
    const auto* reuse_policy = std::getenv("GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE");
    const bool reuse_response = reuse_policy && std::string_view(reuse_policy) == "1" && !fitted &&
                                range_exchange && cached_density &&
                                !std::getenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK");
    const auto response_cache_limit = std::min<std::size_t>(64U << 20, maximum_bytes / 4);
    if (!reuse_response || integral_response_cache_.bytes() > response_cache_limit)
      integral_response_cache_.clear();
    const auto transfers_before = cuda_->transfers();
    dft::VerifiedKsFinalState exported_state;
    generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
    if (!cached_density && fitted) {
      status = read_final_state(expected, true, exported_state, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      cached_density = &exported_state.density;
      cached_weighted_density = &exported_state.weighted_density;
    }
    const auto& model = expected.identity.model;
    const auto device = model.device;
    const auto bytes = maximum_bytes - integral_response_cache_.bytes();
    const auto derivative_source = scf::prepared_cuda_direct_derivative_binding(fock_);
    if (!fitted && !derivative_source) {
      detail = "CUDA integral gradient requires a retained prepared Direct derivative source";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto spins = model.spins;
    const auto nbf = fock_.diagnostic().nbf;
    const auto matrix_elements = nbf * nbf;
    if (!nbf) {
      detail = "CUDA stationary derivative source has an empty AO basis";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    const auto valid_cached = [&](const auto& blocks) {
      return blocks.size() == spins &&
             std::all_of(blocks.begin(), blocks.end(),
                         [&](const auto& matrix) { return matrix.size() == matrix_elements; });
    };
    if (cached_density &&
        (!valid_cached(*cached_density) || !valid_cached(*cached_weighted_density))) {
      detail = "cached CUDA stationary D/W has an incompatible spin or AO shape";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    if (!fitted && (derivative_source.device_id != device || derivative_source.nbf != nbf)) {
      detail = "CUDA stationary Direct derivative source disagrees with the live KS owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::CudaKsResidentDensityBinding resident_density;
    status = cuda_->resident_final_density(expected, resident_density, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_density || resident_density.device_id != device ||
        resident_density.matrix_elements != matrix_elements || resident_density.spins != spins) {
      detail = "CUDA stationary derivative density is incompatible with the prepared Fock owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::CudaKsResidentStationaryWeightsBinding resident_weights;
    if (cached_density) {
      status = cuda_->resident_final_stationary_weights(expected, resident_weights, detail);
    } else {
      // Direct sources need only resident D/W. The native owner authenticates
      // and validates them without manufacturing a detached host snapshot.
      dft::KsFinalStateDiagnostic resident_diagnostic;
      status = cuda_->prepare_resident_final_stationary_weights(expected, resident_weights,
                                                                resident_diagnostic, detail);
    }
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_weights || resident_weights.device_id != device ||
        resident_weights.matrix_elements != matrix_elements || resident_weights.spins != spins) {
      detail = "CUDA stationary one-electron D/W is incompatible with the prepared Fock owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto transfers_after = cuda_->transfers();
    // Both Direct and DF sources belong to the already-budgeted SCF owner.
    // Report retained provider bytes without charging them again to the
    // stationary consumer's additional-device allowance.
    work = {fitted ? fock_.diagnostic().device_bytes : derivative_source.retained_device_bytes,
            0,
            0,
            0,
            0,
            0,
            transfers_after.final_state_d2h_bytes - transfers_before.final_state_d2h_bytes,
            transfers_after.final_state_reads - transfers_before.final_state_reads,
            transfers_after.synchronizations - transfers_before.synchronizations};
    scf::OneElectronGradientResources one;
    const auto nc = 3 * system_.atoms.size();
    std::vector<double> candidate;
    candidate.reserve((combined_two_electron ? 3 : range_exchange ? 5 : 4) * nc);
    std::vector<double> hcore, pulay, value;
    status = GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (!fitted)
      status = scf::execute_prepared_cuda_stationary_one_electron_pair(
          fock_.cuda_direct_source(), resident_weights.density, resident_weights.weighted_density,
          resident_weights.matrix_elements, bytes, hcore, pulay, detail, &one);
    if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
      status = scf::execute_cuda_stationary_one_electron_pair(
          device, system_, {}, {}, scf::cuda_policy::one_electron_derivative_mapping_requested(),
          bytes, hcore, pulay, detail, &one, resident_weights.density,
          resident_weights.weighted_density);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      const bool retry_without_cache =
          status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && integral_response_cache_.bytes() != 0;
      integral_response_cache_.clear();
      if (retry_without_cache)
        return cuda_integral_gradient(expected, output, maximum_bytes, work, detail, cached_density,
                                      cached_weighted_density, combined_two_electron);
      return status;
    }
    work[2] = std::max<std::uint64_t>(work[2], one.device_bytes);
    work[3] = std::max<std::uint64_t>(work[3], one.host_numeric_bytes);
    work[4] += one.host_to_device_bytes;
    work[5] += one.device_to_host_bytes;
    candidate.insert(candidate.end(), hcore.begin(), hcore.end());
    candidate.insert(candidate.end(), pulay.begin(), pulay.end());
    if (fitted) {
      try {
        value = spins == 1 ? fock_.retained_energy_derivative(cached_density->at(0))
                           : fock_.retained_energy_derivative(cached_density->at(0),
                                                              cached_density->at(1));
      } catch (const std::bad_alloc&) {
        detail = "CUDA density-fitted stationary response exceeded its retained budget";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      } catch (const std::invalid_argument& error) {
        detail = error.what();
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      } catch (const std::exception& error) {
        detail = error.what();
        return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      }
      if (value.size() != nc) {
        detail = "CUDA density-fitted stationary response returned the wrong coordinate shape";
        return GENERATIVEQC_STATUS_INTERNAL_ERROR;
      }
      candidate.insert(candidate.end(), value.begin(), value.end());
      // Semilocal DF is deliberately J-only. Keep the stationary runtime's
      // canonical four-source layout without manufacturing exchange work.
      candidate.insert(candidate.end(), nc, 0.0);
    } else {
      runtime::df_progress::Scope response_trace("ks_integral_response", "cuda_completed");
      const bool hit = reuse_response && integral_response_cache_.matches(*cached_density, 3 * nc,
                                                                          response_cache_limit);
      if (hit) {
        const auto retained = integral_response_cache_.response();
        value.assign(retained.begin(), retained.end());
        status = GENERATIVEQC_STATUS_SUCCESS;
      } else {
        integral_response_cache_.clear();
        status = range_exchange
                     ? scf::execute_prepared_cuda_direct_rsh_energy_derivatives_device(
                           fock_, *range_strategy_, resident_density.alpha, resident_density.beta,
                           resident_density.matrix_elements, value, detail)
                     : scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
                           fock_, resident_density.alpha, resident_density.beta,
                           resident_density.matrix_elements, value, detail, !combined_two_electron,
                           qualified_spatial_force_default());
        if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
        // Reserve room for both transient one-electron work and publication;
        // refusal or allocation failure leaves the ordinary computed value.
        const auto publication_bytes = 8 * (candidate.capacity() + value.capacity());
        const auto used_host = work[3] + publication_bytes;
        const auto available = used_host < maximum_bytes ? maximum_bytes - used_host : 0;
        if (reuse_response && value.size() == 3 * nc)
          integral_response_cache_.store(*cached_density, value,
                                         std::min<std::uint64_t>(response_cache_limit, available));
      }
      runtime::df_progress::Scope::number("density_key_hit", hit);
      runtime::df_progress::Scope::number("two_electron_response_evaluations", hit ? 0 : 1);
      runtime::df_progress::Scope::number("retained_host_bytes", integral_response_cache_.bytes());
      response_trace.finish(hit ? "exact_density_reuse" : "computed");
      candidate.insert(candidate.end(), value.begin(), value.end());
    }
    // Slot 1 is retained source host preparation, separate from the transient
    // one-electron peak in slot 3. The prepared owner reports this same slab.
    work[1] = integral_response_cache_.bytes();
    output = std::move(candidate);
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status native_cuda_stationary_gradient(
      const dft::CudaKsFinalStateToken& expected, std::size_t maximum_device_bytes,
      std::vector<double>& output, dft::CudaStationaryResources& resources, std::string& detail,
      scf::DirectForcePrecisionRequest* precision = nullptr) {
    resources = {};
    detail.clear();
#if GENERATIVEQC_HAS_CUDA
    const auto current = [&] {
      dft::CudaKsFinalStateToken token;
      std::string token_detail;
      return cuda_ &&
             cuda_->final_state_token(token, token_detail) == GENERATIVEQC_STATUS_SUCCESS &&
             token == expected;
    };
    if (!current()) {
      detail = "native CUDA stationary gradient requires the exact live final-state token";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    // Private probes can qualify complete d-shell providers. Public requests
    // must first pass the immutable application-budget/model admission above.
    const auto* binding = admitted_cuda_stationary_binding(precision != nullptr);
    if (!binding) {
      detail = "native CUDA stationary profile is unavailable for this scientific model";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    try {
      if (stationary_controller_ &&
          stationary_controller_->resources().device_upper_bytes > maximum_device_bytes) {
        detail = "retained native CUDA stationary controller exceeds the new device allowance";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      }
      // The controller's shape plan includes grid/provider scratch. Charge only
      // storage not already retained on a replay, before creating any force owner.
      const auto capacity = dft::cuda_stationary_capacity(
          *binding, basis_.natom, basis_.nao, basis_.nprimitive, grid_.point_count(),
          options_.xc_tile_points, precision && precision->mixed_grid);
      const auto envelope = std::min(maximum_device_bytes, capacity.cached_device_bytes);
      const auto retained =
          stationary_controller_ ? stationary_controller_->resources().device_upper_bytes : 0;
      std::array<std::uint64_t, 4> memory_receipt{};
      auto memory_status = cuda_->prepare_force_memory(
          expected, envelope > retained ? envelope - retained : 0, memory_receipt, detail);
      if (memory_status != GENERATIVEQC_STATUS_SUCCESS) return memory_status;
      if (!stationary_controller_) {
        stationary_controller_ = std::make_unique<dft::CudaStationaryController>(
            system_, basis_, grid_, *binding, expected.identity.model.device,
            options_.xc_tile_points, maximum_device_bytes);
      }
      dft::CudaKsResidentDensityBinding density;
      auto status = cuda_->resident_final_density(expected, density, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      dft::CudaKsResidentStationaryWeightsBinding weights;
      dft::KsFinalStateDiagnostic diagnostic;
      status =
          cuda_->prepare_resident_final_stationary_weights(expected, weights, diagnostic, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      status =
          stationary_controller_->gradient(density, weights, fock_, current, output, detail,
                                           qualified_spatial_force_default(precision), precision);
      if (status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE && precision &&
          (precision->mixed_geometry || precision->mixed_grid)) {
        // Failed geometry never publishes a partial result. Reset every source
        // and retry the complete gradient once in FP64 at the SAME final token.
        // Driver, token and resource failures must not masquerade as rounding.
        ++precision->geometry_retries;
        detail.clear();
        status = stationary_controller_->gradient(density, weights, fock_, current, output, detail,
                                                  qualified_spatial_force_default(), nullptr);
      }
      if (status == GENERATIVEQC_STATUS_SUCCESS)
        resources = stationary_controller_->resources();
      else
        stationary_controller_.reset();
      return status;
    } catch (const std::bad_alloc&) {
      detail = "native CUDA stationary host/device allocation or admission failed";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    } catch (const generativeqc::Error& failure) {
      stationary_controller_.reset();
      detail = failure.what();
      return failure.status();
    } catch (const std::invalid_argument& failure) {
      detail = failure.what();
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    } catch (const std::exception& failure) {
      stationary_controller_.reset();
      detail = failure.what();
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
#else
    detail = "native CUDA stationary gradient requires a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  const dft::CudaStationaryBinding* admitted_cuda_stationary_binding(
      bool qualify_through_d = false) const noexcept {
    if (backend_ != GENERATIVEQC_BACKEND_CUDA) return {};
    const bool all_electron = system_.ecp_terms.empty() &&
                              std::none_of(system_.atoms.begin(), system_.atoms.end(),
                                           [](const auto& atom) { return atom.ecp_core != 0; });
    std::size_t maximum_angular = 0;
    for (const auto& shell : system_.shells)
      maximum_angular = std::max(maximum_angular, std::size_t{shell.angular_momentum});
    return native_cuda_stationary_binding(execution_plan_, options_, grid_.spec(), all_electron,
                                          maximum_angular, basis_.nao, basis_.natom,
                                          qualify_through_d);
  }

  /** Both capacity planning and execution consume the same budget/model gate.
   * In particular, strict SCF does not imply strict derivative arithmetic. */
  std::uint64_t application_gradient_qualification() const noexcept {
    if (!dft::admit_stationary_force_controls(options_.energy_tolerance, options_.density_tolerance,
                                              options_.screening_tolerance, false))
      return 0;
    const auto* binding = admitted_cuda_stationary_binding();
    if (!binding) return 0;
    std::size_t maximum_angular = 0;
    for (const auto& shell : system_.shells)
      maximum_angular = std::max(maximum_angular, std::size_t{shell.angular_momentum});
    return dft::application_gradient_qualification(*binding, options_.application_precision_budget,
                                                   basis_.nao, basis_.natom, maximum_angular);
  }

  generativeqc_status native_cuda_stationary_capacity(dft::CudaStationaryCapacity& capacity,
                                                      std::string& detail) const {
    const auto* binding = admitted_cuda_stationary_binding();
    if (!binding) {
      detail = "native CUDA stationary capacity is unavailable for this prepared context";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    try {
      const auto candidate = dft::cuda_stationary_capacity(
          *binding, basis_.natom, basis_.nao, basis_.nprimitive, grid_.point_count(),
          options_.xc_tile_points, application_gradient_qualification() != 0);
      capacity = candidate;
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      detail = "native CUDA stationary capacity metadata allocation failed";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
      detail = error.what();
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
  }

  generativeqc_status native_cuda_stationary_limit(std::size_t maximum_device_bytes,
                                                   std::string& detail) {
    dft::CudaStationaryCapacity capacity;
    const auto status = native_cuda_stationary_capacity(capacity, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (maximum_device_bytes < capacity.minimum_device_bytes) {
      detail = "native CUDA stationary allowance is below the complete consumer minimum";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
#if GENERATIVEQC_HAS_CUDA
    if (stationary_controller_ &&
        stationary_controller_->resources().device_upper_bytes > maximum_device_bytes) {
      detail = "retained native CUDA stationary controller exceeds the new item allowance";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
#endif
    stationary_device_allowance_ = maximum_device_bytes;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  /** The native single-system and batch entry points must consume identical
   * token-qualified stationary derivative sources and one complete force sink.
   * This is intentionally not a second batch-specific DFT force equation. */
  void add_prepared_native_forces(Result& result) {
    if (!supports_native_gga_force())
      throw MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "native DFT analytic forces are unqualified for this prepared context (#2151)");
    if (!result.convergence.converged) return;
    dft::CudaKsFinalStateToken token;
    dft::VerifiedKsFinalState frame;
    std::string detail;
    auto status = final_state_token(token, detail);
    if (backend_ == GENERATIVEQC_BACKEND_CUDA) {
      dft::CudaStationaryCapacity capacity;
      if (status == GENERATIVEQC_STATUS_SUCCESS)
        status = native_cuda_stationary_capacity(capacity, detail);
      std::vector<double> gradient;
      dft::CudaStationaryResources resources;
      scf::DirectForcePrecisionRequest precision;
      const auto qualification = application_gradient_qualification();
      const bool application = qualification != 0;
      precision.mixed_weighted = application;
      precision.mixed_geometry = application;
      precision.mixed_grid = application;
      if (status == GENERATIVEQC_STATUS_SUCCESS)
        status = native_cuda_stationary_gradient(
            token, stationary_device_allowance_.value_or(capacity.cached_device_bytes), gradient,
            resources, detail, application ? &precision : nullptr);
      if (status != GENERATIVEQC_STATUS_SUCCESS)
        throw MethodError(status, detail.empty() ? "native CUDA DFT force unavailable" : detail);
      for (auto& value : gradient) value = -value;
      result.forces.swap(gradient);
      if (application) {
        result.precision_work.gradient = {1, precision.mixed_weighted, precision.geometry_points,
                                          precision.grid_points, precision.geometry_retries};
        result.precision_work.gradient_qualification = qualification;
      }
      return;
    }
    if (status == GENERATIVEQC_STATUS_SUCCESS)
      status = read_final_state(token, true, frame, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw MethodError(status, detail.empty() ? "native DFT final state unavailable" : detail);

    auto retained_bytes = host_numeric_capacity();
    for (const auto* matrices : {&frame.density, &frame.fock, &frame.weighted_density})
      for (const auto& matrix : *matrices)
        retained_bytes = runtime::add_capacity(retained_bytes, runtime::vector_bytes(matrix));
    for (const auto& orbital : frame.orbitals)
      retained_bytes = runtime::add_capacity(
          retained_bytes, runtime::vector_capacities(orbital.values, orbital.vectors));
    if (result.ks_diagnostic)
      retained_bytes = runtime::add_capacity(retained_bytes,
                                             runtime::vector_bytes(result.ks_diagnostic->history));
    runtime::CpuRetainedCapacity retained(retained_bytes);

    std::vector<double> integrals;
    std::array<std::uint64_t, 9> work{};
    // Bound owned source/weight buffers separately from the retained SCF owner
    // and DF response providers. This is not an allocator/recurrence census.
    constexpr std::size_t kSourcePublicationBytes = 64ull * 1024ull * 1024ull;
    status = prepared_integral_gradient(token, frame.density, frame.weighted_density, integrals,
                                        kSourcePublicationBytes, work, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS)
      throw MethodError(status, detail.empty() ? "native DFT integral force unavailable" : detail);
    runtime::CpuRetainedCapacity retained_integrals(runtime::vector_bytes(integrals));

    const auto xc = dft::stationary_curated_xc_derivative(
        basis_, grid_, frame.density, options_.xc_tile_points, execution_plan_.semilocal_family,
        options_.semilocal_exchange_scale, options_.semilocal_correlation_scale);
    const std::size_t ncoord = 3 * system_.atoms.size();
    if (integrals.size() != 4 * ncoord || xc.gradient.size() != ncoord)
      throw MethodError(GENERATIVEQC_STATUS_INTERNAL_ERROR,
                        "native DFT stationary-force source shape mismatch");
    std::vector<double> gradient = xc.gradient;
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate)
      for (std::size_t source = 0; source < 4; ++source)
        gradient[coordinate] += integrals[source * ncoord + coordinate];
    molecule::add_nuclear_repulsion_gradient(system_, gradient);
    dft::CudaKsFinalStateToken current;
    status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS || current != token)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "native DFT final-state force token changed");
    result.forces.resize(ncoord);
    runtime::sample_cpu_capacity(runtime::vector_capacities(xc.gradient, gradient, result.forces));
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
      if (!std::isfinite(gradient[coordinate]))
        throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "nonfinite complete native DFT nuclear gradient");
      result.forces[coordinate] = -gradient[coordinate];
    }
  }

  void apply_d4(Result& result) {
    if (!d4_) return;
    std::vector<double> coordinates;
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms)
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    const std::uint8_t active = 1;
    const std::uint8_t want_gradient = 0;
    std::vector<dft::dispersion::D4Status> statuses;
    std::vector<double> components, gradients, charges;
    std::string detail;
    const auto status =
        d4_->execute(coordinates, std::span(&active, 1), std::span(&want_gradient, 1), statuses,
                     components, gradients, charges, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS || statuses.size() != 1 ||
        statuses[0] != dft::dispersion::D4Status::success || components.size() != 2)
      throw MethodError(
          status == GENERATIVEQC_STATUS_SUCCESS ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE : status,
          detail.empty() ? "PBE-D4 correction failed" : detail);
    result.energy += components[0] + components[1];
  }

  /** CPU solve used by the common scheduler; CUDA is advanced there per stream. */
  scf::ScfResult run_cpu(const std::vector<double>* initial_density, bool reuse_warm,
                         bool update_warm, bool allow_preliminary = true) {
    if (backend_ != GENERATIVEQC_BACKEND_CPU_REFERENCE)
      throw std::logic_error("CPU KS solve requires a CPU prepared owner");
    invalidate_final_state();
    if (cpu_epoch_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS solve epoch exhausted");
    ++cpu_epoch_;
    const auto* seed =
        initial_density ? initial_density : (reuse_warm && !warm_.empty() ? &warm_ : nullptr);
    // The CPU driver already samples its provider/grid. Add only the retained
    // last-good density, which coexists with its current/proposed densities.
    runtime::CpuRetainedCapacity retained_warm(runtime::vector_bytes(warm_));
    auto execution_options = options_;
    if (!allow_preliminary) execution_options.preliminary_guess.reset();
    scf::ScfResult native;
    if (execution_plan_.automatic_program) {
      native = ks_is_unrestricted(execution_plan_)
                   ? scf::run_semilocal_uks(fock_, basis_, grid_, execution_options,
                                            *execution_plan_.automatic_program, seed)
               : execution_plan_.automatic_restricted_program
                   ? scf::run_semilocal_rks(fock_, basis_, grid_, execution_options,
                                            *execution_plan_.automatic_restricted_program, seed)
                   : scf::run_semilocal_rks(fock_, basis_, grid_, execution_options,
                                            *execution_plan_.automatic_program, seed);
    } else if (execution_plan_.generated_split_hybrid)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "generated split-global-hybrid CPU KS is unavailable");
    else if (dft::semilocal_family_uses_molecular_nonlocal_domain(
                 execution_plan_.semilocal_family)) {
      if (!range_correction_ || !nonlocal_)
        throw std::runtime_error("WB97M-V requires both range-exchange and nonlocal owners");
      native = ks_is_unrestricted(execution_plan_)
                   ? scf::run_wb97mv_uks(fock_, *range_correction_, basis_, grid_,
                                         execution_options, *nonlocal_, seed)
                   : scf::run_wb97mv_rks(fock_, *range_correction_, basis_, grid_,
                                         execution_options, *nonlocal_, seed);
    } else if (execution_plan_.range_exchange) {
      if (!range_correction_)
        throw std::runtime_error("KS range-exchange correction owner is missing");
      native = ks_is_unrestricted(execution_plan_)
                   ? scf::run_pbe_rsh_uks(fock_, *range_correction_, basis_, grid_,
                                          execution_options, seed, nonlocal_.get())
                   : scf::run_pbe_rsh_rks(fock_, *range_correction_, basis_, grid_,
                                          execution_options, seed, nonlocal_.get());
    } else if (execution_plan_.semilocal_family == dft::SemilocalFamily::B3lyp)
      native = ks_is_unrestricted(execution_plan_)
                   ? scf::run_b3lyp_uks(fock_, basis_, grid_, execution_options, seed)
                   : scf::run_b3lyp_rks(fock_, basis_, grid_, execution_options, seed);
    else if (execution_plan_.semilocal_family == dft::SemilocalFamily::Pbe && nonlocal_)
      native =
          ks_is_unrestricted(execution_plan_)
              ? scf::run_pbe_uks_nonlocal(fock_, basis_, grid_, execution_options, seed, *nonlocal_)
              : scf::run_pbe_rks_nonlocal(fock_, basis_, grid_, execution_options, seed,
                                          *nonlocal_);
    else
      native = scf::run_curated_semilocal_ks(fock_, basis_, grid_, execution_options,
                                             execution_plan_.semilocal_family,
                                             execution_plan_.spin_channels, seed);
    // This owner has immutable model/geometry/spin identity. Only successful
    // executions may replace its compatible last-good density; DIIS is fresh.
    if (native.converged && options_.retain_ks_state) {
      const auto spins = execution_plan_.spin_channels;
      const auto matrix = fock_.one_electron().nbf * fock_.one_electron().nbf;
      if (native.density.size() != spins * matrix ||
          native.ks_physical_fock.size() != spins * matrix ||
          (spins == 2 && native.dft_diagnostic.occupations.size() != spins))
        throw std::runtime_error("CPU KS retained spin-state shape mismatch");
      std::vector<scf::reference::Matrix> densities, focks;
      densities.reserve(spins);
      focks.reserve(spins);
      for (unsigned spin = 0; spin < spins; ++spin) {
        const auto begin = spin * matrix;
        densities.emplace_back(native.density.begin() + begin,
                               native.density.begin() + begin + matrix);
        focks.emplace_back(native.ks_physical_fock.begin() + begin,
                           native.ks_physical_fock.begin() + begin + matrix);
      }
      std::vector<std::size_t> occupied;
      if (spins == 1)
        occupied = {static_cast<std::size_t>(system_.electron_count / 2)};
      else
        occupied.assign(native.dft_diagnostic.occupations.begin(),
                        native.dft_diagnostic.occupations.end());
      dft::KsFinalStateIdentity identity;
      identity.determinant = {
          {cpu_owner_, 1, 1, 1}, cpu_epoch_, fock_.strategy(), std::move(occupied)};
      identity.model = {1,
                        ks_scf_domain_version(execution_plan_),
                        grid_.spec(),
                        options_.xc_tile_points,
                        ks_xc_functional_code(execution_plan_),
                        spins,
                        -1,
                        cpu_owner_,
                        options_.semilocal_exchange_scale,
                        options_.semilocal_correlation_scale};
      if (range_correction_) identity.model.range_correction = range_correction_->strategy();
      if (nonlocal_) identity.model.nonlocal_correlation = nonlocal_->parameters();
      identity.model.nonlocal_grid = options_.nonlocal_grid_spec;
      if (dft::semilocal_family_uses_molecular_nonlocal_domain(execution_plan_.semilocal_family))
        identity.model.nonlocal_density_domain = dft::nlc::Vv10DensityDomain::MolecularV1;
      dft::KsPhysicalState physical{identity,
                                    true,
                                    std::move(densities),
                                    std::move(focks),
                                    native.dft_diagnostic.components,
                                    native.energy,
                                    native.dft_diagnostic.physical_residual};
      cpu_physical_ = std::move(physical);
      if (seed && spins == 2) {
        // A warm UKS proposal may pass the SCF step gate while its latest
        // physical F[D] frame still fails the stricter derivative export.
        // Certify that frame before publishing success or replacing the last
        // good seed. Rejection uses the existing single cold retry in the
        // batch owner, never a hidden solve during snapshot/force export.
        dft::VerifiedKsFinalState verified;
        std::string detail;
        const dft::CudaKsFinalStateToken token{1, cpu_physical_->identity};
        if (read_final_state(token, false, verified, detail) != GENERATIVEQC_STATUS_SUCCESS) {
          native.converged = false;
          native.ks_physical_fock.clear();
        }
      }
    }
    if (native.converged && update_warm) warm_ = std::move(native.density);
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return native;
  }

 private:
  void prepare_range_exchange(int device) {
    const auto spin = ks_is_unrestricted(execution_plan_) ? scf::FockSpin::Unrestricted
                                                          : scf::FockSpin::Restricted;
    const auto fock_backend =
        backend_ == GENERATIVEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    range_strategy_ = scf::resolve_fock_build(
        scf::make_rsh_correction_fock_spec(
            spin, execution_plan_.short_range_exchange, execution_plan_.long_range_exchange,
            execution_plan_.range_omega,
            execution_plan_.density_fitted_range_exchange ? scf::FockApproximation::DensityFitted
                                                          : scf::FockApproximation::Exact),
        fock_backend, options_.screening_tolerance, options_.density_fitting_relative_threshold);
    const bool fitted_primary =
        fock_.strategy().spec.coulomb.approximation == scf::FockApproximation::DensityFitted;
    if (fock_backend == scf::FockBackend::Cpu || fitted_primary) {
      const auto budget = execution_plan_.density_fitted_range_exchange
                              ? ks_fitted_provider_budget(options_, execution_plan_, true)
                          : fock_backend == scf::FockBackend::Cuda
                              ? ks_provider_bytes(system_, backend_, 0U)
                              : 0U;
      const auto* auxiliary =
          execution_plan_.density_fitted_range_exchange ? fock_.auxiliary_system() : nullptr;
      range_correction_ = std::make_unique<scf::PreparedFockPlan>(
          system_, auxiliary, *range_strategy_, device, budget, 0U,
          execution_plan_.density_fitted_range_exchange ? 1U : 0U);
    }
  }

  void prepare_nonlocal(int device) {
    const auto& nonlocal_grid = options_.nonlocal_grid ? *options_.nonlocal_grid : grid_;
    if (nonlocal_grid.point_count() > std::numeric_limits<std::uint32_t>::max())
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                        "KS grid exceeds the VV10 public point-count domain");
    generativeqc_status status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    nonlocal_ = dft::nlc::Vv10Plan::prepare(
        backend_, device, static_cast<std::uint32_t>(nonlocal_grid.point_count()),
        static_cast<std::uint32_t>(options_.xc_tile_points), execution_plan_.nonlocal_parameters,
        execution_plan_.nonlocal_maximum_bytes, detail, status);
    if (!nonlocal_)
      throw MethodError(
          status, detail.empty() ? "self-consistent nonlocal plan preparation failed" : detail);
  }

  void prepare_d4(int device) {
    const auto source = ::generativeqc::generated::method_parameters::pbeD4();
    dft::dispersion::D4Parameters parameters{dft::dispersion::D4ReferenceModel::eeq,
                                             source.s6,
                                             source.s8,
                                             source.s9,
                                             source.a1,
                                             source.a2,
                                             source.cn_cutoff,
                                             source.pair_cutoff,
                                             source.atm_cutoff,
                                             source.ga,
                                             source.gc};
    std::vector<std::uint32_t> offsets{0, static_cast<std::uint32_t>(system_.atoms.size())};
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> coordinates;
    atomic_numbers.reserve(system_.atoms.size());
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms) {
      atomic_numbers.push_back(atom.atomic_number);
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    }
    generativeqc_status status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    d4_ = dft::dispersion::D4Plan::prepare(
        backend_, device, std::move(offsets), std::move(atomic_numbers),
        std::vector<double>{static_cast<double>(system_.charge)}, std::move(coordinates),
        parameters, dft::dispersion::D4EEQProfile::standard, 256ull * 1024ull * 1024ull, detail,
        status);
    if (!d4_)
      throw MethodError(status,
                        detail.empty() ? "PBE-D4 production plan preparation failed" : detail);
  }

  Capabilities capabilities_;
  core::System system_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  generativeqc_backend backend_;
  scf::PreparedFockPlan fock_;
  std::optional<scf::ResolvedFockBuild> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_;
#if GENERATIVEQC_HAS_CUDA
  std::unique_ptr<scf::PreparedFockPlan> range_derivative_correction_;
#endif
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::unique_ptr<dft::MolecularGrid> nonlocal_grid_;
  std::vector<double> warm_;
  ExactDensityResponseCache integral_response_cache_;
  const std::uint64_t cpu_owner_{next_cpu_ks_owner()};
  std::uint64_t cpu_epoch_{};
  std::optional<dft::KsPhysicalState> cpu_physical_;
  std::unique_ptr<dft::dispersion::D4Plan> d4_;
  std::unique_ptr<dft::nlc::Vv10Plan> nonlocal_;
  std::optional<std::size_t> stationary_device_allowance_;
#if GENERATIVEQC_HAS_CUDA
  std::unique_ptr<dft::CudaKsPlan> cuda_;
  std::unique_ptr<dft::CudaStationaryController> stationary_controller_;
#endif
};

/** Independent native KS owners, with ordinary-stream round-robin CUDA work.
 * Geometry is rebuilt per item, while model/basis/charge/spin remain immutable.
 * No HF graph or Python calculation loop participates in this schedule. */
class KsPreparedBatch final : public PreparedBatch,
                              public KsBatchOwnerPort,
                              public BatchWarmStarts,
                              public BatchKsTransportDiagnostics,
                              public BatchDfMetricDiagnostics {
 public:
  KsPreparedBatch(Capabilities capabilities, std::vector<core::System> systems,
                  NativeKsExecutionPlan execution_plan, scf::ScfOptions options, dft::GridSpec grid,
                  generativeqc_backend backend, int device, bool warm_enabled,
                  std::optional<core::System> auxiliary)
      : PreparedBatch(backend, device),
        capabilities_(capabilities),
        systems_(std::move(systems)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        grid_spec_(std::move(grid)),
        backend_(backend),
        device_(device),
        warm_enabled_(warm_enabled),
        auxiliary_(std::move(auxiliary)),
        items_(systems_.size()) {
    for (std::size_t i = 0; i < size(); ++i) {
      runtime::CpuRetainedCapacity neighbors(host_numeric_capacity());
      items_[i].plan = make_plan(systems_[i]);
      items_[i].prepared_properties = items_[i].plan->supported_properties();
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  std::optional<generativeqc_property_flags> supported_properties(
      std::size_t index) const noexcept override {
    if (index >= items_.size()) return std::nullopt;
    return items_[index].prepared_properties;
  }

  void invalidate_result() override {
    for (auto& item : items_)
      if (item.plan) item.plan->invalidate_final_state();
  }

  std::vector<BatchItemResult> execute_impl(const Coordinates& coordinates,
                                            bool compute_forces) override {
    return execute_items(coordinates, compute_forces, {});
  }

  std::vector<BatchItemResult> execute_selected_forces_impl(
      const Coordinates& coordinates, std::span<const std::uint8_t> forces) override {
    return execute_items(coordinates, false, forces);
  }

  /** Selection changes only derivative consumers. All items share the existing
   * SCF scheduler, warm policy, state generations and error isolation. */
  std::vector<BatchItemResult> execute_items(const Coordinates& coordinates, bool compute_forces,
                                             std::span<const std::uint8_t> selected_forces) {
    const auto wants_forces = [&](std::size_t index) {
      return selected_forces.empty() ? compute_forces : selected_forces[index] != 0;
    };
    // Every requested force still needs its exact prepared capability. An
    // energy-only neighbor does not grant or revoke another item's admission.
    bool force_unavailable = false;
    for (std::size_t i = 0; i < size(); ++i)
      force_unavailable |=
          wants_forces(i) && !(items_[i].prepared_properties & GENERATIVEQC_PROPERTY_FORCES);
    if (force_unavailable) {
      const char* issue = execution_plan_.automatic_program ? "#1122"
                          : dft::semilocal_family_requires_tau(execution_plan_.semilocal_family)
                              ? "#164"
                              : "#163";
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        std::string(ks_semilocal_family_name(execution_plan_)) +
                            " KS nuclear gradients are tracked separately in issue " + issue +
                            "; this prepared native DFT force context is unqualified (#2151)");
    }
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("KS batch coordinates do not match system count");
    for (auto& item : items_) item.failure = nullptr;
    std::vector<BatchItemResult> results(size());
    std::vector<scf::CoulombExchangeWork> target_coulomb_exchange_work(
        size(), scf::CoulombExchangeWork::empty());
    std::vector<bool> ready(size(), false);
    std::vector<std::optional<std::vector<double>>> preliminary_seeds(size());
    std::vector<scf::initial_guess::PreliminaryDiagnostic> preliminary_diagnostics(size());
    std::vector<bool> preliminary_requested(size(), false);
    // Allocate source-geometry metadata before launching any item. The success
    // path can then publish its last-good identity without a coordinate copy.
    std::vector<scf::HfWarmState> candidates(size());
    for (std::size_t i = 0; i < size(); ++i) {
      auto& result = results[i];
      try {
        auto target = geometry::bind_batch_item(systems_[i], coordinates, i, result, backend_, i);
        auto& item = items_[i];
        candidates[i].coordinates = geometry::coordinates(target);
        if (!item.plan || geometry::coordinates(item.plan->system()) != candidates[i].coordinates) {
          // Preserve the last GOOD seed before freeing its device owner. This
          // explicit rebuild download is never part of routine SCF iterations.
          materialize_warm(i);
#if GENERATIVEQC_HAS_CUDA
          if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr)
            add_transfers(item.retired_transfers, cuda->transfers());
#endif
          item.plan.reset();
          item.resident_warm = false;
          item.plan = make_plan(target, item.stationary_device_allowance);
        }
        result.warm_start_used = warm_enabled_ && item.warm.has_value();
#if GENERATIVEQC_HAS_CUDA
        if (item.plan->cuda_plan() && item.plan->initial_guess_policy()) {
          auto& diagnostic = preliminary_diagnostics[i];
          preliminary_requested[i] = true;
          diagnostic.requested_kind =
              static_cast<std::uint32_t>(item.plan->initial_guess_policy()->kind);
          diagnostic.target_attempts = 1;
          if (result.warm_start_used) {
            diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::ExplicitDensity;
          } else {
            preliminary_seeds[i] = item.plan->prepare_cold_initial_guess(diagnostic);
          }
        }
#endif
        ready[i] = true;
      } catch (...) {
        items_[i].failure = std::current_exception();
        result.status = item_exception_status();
      }
    }

    const auto finish = [&](std::size_t i, scf::ScfResult native) {
      auto& result = results[i];
      auto& observed = target_coulomb_exchange_work[i];
      observed.add(native.dft_diagnostic.coulomb_exchange_work);
      native.dft_diagnostic.coulomb_exchange_work = observed;
      if (preliminary_requested[i] && !native.preliminary_guess.requested_kind)
        native.preliminary_guess = preliminary_diagnostics[i];
      result.calculation = adapt_result(std::move(native), backend_);
      if (wants_forces(i) && result.calculation.convergence.converged)
        items_[i].plan->add_prepared_native_forces(result.calculation);
      items_[i].plan->apply_d4(result.calculation);
      const auto& calculation = result.calculation;
      result.status = calculation.convergence.converged ? GENERATIVEQC_STATUS_SUCCESS
                                                        : GENERATIVEQC_STATUS_NOT_CONVERGED;
      if (calculation.convergence.converged && warm_enabled_ && warm_updates_) {
        auto& state = candidates[i];
        state.energy = calculation.energy;
        state.energy_change = calculation.convergence.energy_change;
        state.density_rms = calculation.convergence.residual_rms;
        state.iterations = calculation.convergence.iterations;
        items_[i].warm = std::move(state);
        items_[i].resident_warm = true;
      }
    };

    // A rejected/nonconverged warm solve gets one cold retry. CUDA retries
    // retain the same per-item scheduler; a failed neighbor never halts it.
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      std::vector<bool> running(size(), false);
      for (std::size_t i = 0; i < size(); ++i) {
        auto& result = results[i];
        // Retry only seed-related failures. Resource/driver failures preserve
        // their first status and leave the last-good seed for explicit replay.
        const bool seed_failure = retryable_seed_failure(result.status, items_[i].failure);
        const bool preliminary_seeded = preliminary_seeds[i].has_value();
        const bool first_seeded = result.warm_start_used || preliminary_seeded;
        if (!ready[i] || (attempt && (!first_seeded || !seed_failure))) continue;
        if (attempt) {
          result.warm_start_fallback = true;
          // Release the failed attempt's exported history before starting another
          // solve, preserving the two-history resource bound.
          result.calculation.ks_diagnostic.reset();
          if (preliminary_requested[i]) {
            auto& diagnostic = preliminary_diagnostics[i];
            diagnostic.target_attempts = 2;
            diagnostic.discarded_target_iterations = result.calculation.convergence.iterations;
            diagnostic.discarded_target_fock_builds = result.calculation.fock_builds;
            diagnostic.work_counters_complete = result.status == GENERATIVEQC_STATUS_NOT_CONVERGED;
            if (preliminary_seeded)
              diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::TargetRetried;
          }
          preliminary_seeds[i].reset();
        }
        auto& item = items_[i];
        item.failure = nullptr;
        const bool reuse = !attempt && result.warm_start_used;
        const auto* seed =
            reuse && !item.resident_warm
                ? &item.warm->density
                : (!attempt && preliminary_seeds[i] ? &*preliminary_seeds[i] : nullptr);
        try {
#if GENERATIVEQC_HAS_CUDA
          if (auto* cuda = item.plan->cuda_plan()) {
            if (!result.target_fock_work) result.target_fock_work.emplace();
            result.target_fock_work->begin_attempt();
            cuda->set_warm_start_updates(warm_enabled_ && warm_updates_);
            cuda->begin(seed, reuse && item.resident_warm && !preliminary_seeds[i],
                        !attempt && !reuse && preliminary_seeds[i]
                            ? dft::CudaKsSeedOrigin::NativeColdGuess
                            : dft::CudaKsSeedOrigin::External);
            running[i] = true;
            continue;
          }
#endif
          runtime::CpuRetainedCapacity neighbors(host_numeric_capacity(i));
          const auto discarded_iterations = result.calculation.convergence.iterations;
          const auto discarded_fock_builds = result.calculation.fock_builds;
          const bool discarded_complete = result.status == GENERATIVEQC_STATUS_NOT_CONVERGED;
          auto native = item.plan->run_cpu(seed, reuse && item.resident_warm,
                                           warm_enabled_ && warm_updates_, !attempt);
          if (attempt && item.plan->initial_guess_policy()) {
            auto& diagnostic = native.preliminary_guess;
            diagnostic.requested_kind =
                static_cast<std::uint32_t>(item.plan->initial_guess_policy()->kind);
            diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::ExplicitDensity;
            diagnostic.target_attempts = 2;
            diagnostic.discarded_target_iterations = discarded_iterations;
            diagnostic.discarded_target_fock_builds = discarded_fock_builds;
            diagnostic.work_counters_complete = discarded_complete;
          }
          finish(i, std::move(native));
        } catch (...) {
          target_coulomb_exchange_work[i].complete = false;
          if (result.calculation.ks_diagnostic)
            result.calculation.ks_diagnostic->coulomb_exchange_work.complete = false;
          items_[i].failure = std::current_exception();
          result.status = item_exception_status();
        }
      }
#if GENERATIVEQC_HAS_CUDA
      while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
        // Submit ALL active streams before synchronizing any scalar record.
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            items_[i].plan->cuda_plan()->enqueue_iteration();
          } catch (...) {
            target_coulomb_exchange_work[i].complete = false;
            if (results[i].calculation.ks_diagnostic)
              results[i].calculation.ks_diagnostic->coulomb_exchange_work.complete = false;
            items_[i].failure = std::current_exception();
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            auto* cuda = items_[i].plan->cuda_plan();
            if (cuda->finish_iteration()) continue;
            running[i] = false;
            if (cuda->failed())
              throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS physical evaluation failed");
            finish(i, cuda->result(false));
            results[i].target_fock_work->finish_attempt(cuda->completed_fock_builds());
          } catch (...) {
            target_coulomb_exchange_work[i].complete = false;
            if (results[i].calculation.ks_diagnostic)
              results[i].calculation.ks_diagnostic->coulomb_exchange_work.complete = false;
            items_[i].failure = std::current_exception();
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
      }
#endif
    }
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return results;
  }

  /** Run one item with the exact batch schedule, preserving scalar error semantics.
   * Nonconvergence returns diagnostics; other failures remain exceptions. */
  Result execute_single(bool compute_forces) {
    if (size() != 1) throw std::logic_error("single calculation requires exactly one KS item");
    auto results = execute({}, compute_forces);
    auto& result = results.front();
    if (result.status != GENERATIVEQC_STATUS_SUCCESS &&
        result.status != GENERATIVEQC_STATUS_NOT_CONVERGED) {
      if (items_.front().failure) std::rethrow_exception(items_.front().failure);
      throw MethodError(result.status, "prepared KS calculation failed");
    }
    return std::move(result.calculation);
  }

  /** Borrow the sole scientific port; the facade never replaces this geometry. */
  KsSingleOwnerPort& single_owner() const { return *items_.at(0).plan; }

  void clear_warm_starts() override {
    for (auto& item : items_) {
      item.warm.reset();
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    const std::size_t spins = execution_plan_.spin_channels;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
      throw std::invalid_argument("KS warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    materialize_warm(index);
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("KS seed restore requires a matching warm-enabled batch");
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      const auto& state = *states[i];
      if (state.density.size() != warm_density_size(i) ||
          !geometry::valid_coordinates(state.coordinates, systems_[i]) || state.iterations < 0 ||
          !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
          !std::isfinite(state.density_rms) || state.density_rms < 0)
        throw std::invalid_argument("invalid KS seed dimensions or diagnostics");
      auto source = systems_[i];
      geometry::set_coordinates(source, state.coordinates);
      scf::initial_guess::EigenOperation eigen;
#if GENERATIVEQC_HAS_CUDA
      if (auto* cuda = items_[i].plan ? items_[i].plan->cuda_plan() : nullptr)
        eigen = cuda->seed_eigen_operation();
#endif
      // This common validation reads only source S and checks the shared
      // spin-density convention. An idle CUDA plan supplies the ordinary
      // eigensolver without overwriting its final/warm state; source coordinates
      // still determine S, including changed-geometry checkpoints. The shared
      // validator preserves its legacy near-symmetric input fallback.
      scf::validate_hf_warm_density(
          source,
          ks_is_unrestricted(execution_plan_) ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF,
          state.density, eigen);
    }
    // All source-metric validation precedes the no-throw commit. Missing
    // entries preserve neighbors, including their resident density ownership.
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      auto& item = items_[i];
      item.warm.swap(states[i]);
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic(std::size_t index) const override {
    const auto& item = items_.at(index);
#if GENERATIVEQC_HAS_CUDA
    if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr) {
      auto cumulative = item.retired_transfers;
      add_transfers(cumulative, cuda->transfers());
      return adapt_transfers(cumulative);
    }
#endif
    return std::nullopt;
  }

  generativeqc_status final_state_token(std::size_t index, dft::CudaKsFinalStateToken& token,
                                        std::string& detail) const override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->final_state_token(token, detail);
    token = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status read_final_state(std::size_t index,
                                       const dft::CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density,
                                       dft::VerifiedKsFinalState& state,
                                       std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_final_state(expected, compute_weighted_density, state,
                                                  detail);
    state = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status native_cuda_stationary_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      std::size_t maximum_device_bytes, std::vector<double>& output,
      dft::CudaStationaryResources& resources, std::string& detail,
      scf::DirectForcePrecisionRequest* precision = nullptr) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->native_cuda_stationary_gradient(
          expected, maximum_device_bytes, output, resources, detail, precision);
    resources = {};
    detail = "KS batch item has no native stationary owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status native_cuda_stationary_capacity(std::size_t index,
                                                      dft::CudaStationaryCapacity& capacity,
                                                      std::string& detail) const {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->native_cuda_stationary_capacity(capacity, detail);
    detail = "KS batch item has no native stationary owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status native_cuda_stationary_limit(std::size_t index,
                                                   std::size_t maximum_device_bytes,
                                                   std::string& detail) {
    if (index >= items_.size() || !items_[index].plan) {
      detail = "native CUDA stationary allowance requires a live prepared item";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto status =
        items_[index].plan->native_cuda_stationary_limit(maximum_device_bytes, detail);
    if (status == GENERATIVEQC_STATUS_SUCCESS)
      items_[index].stationary_device_allowance = maximum_device_bytes;
    return status;
  }

  generativeqc_status resident_orbital_frame(std::size_t index,
                                             const dft::CudaKsFinalStateToken& expected,
                                             dft::CudaKsResidentOrbitalFrameBinding& binding,
                                             std::string& detail, bool prepare) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_orbital_frame(expected, binding, detail, prepare);
    binding = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_fock_sources(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected,
                                            dft::CudaKsResidentFockBinding& binding,
                                            std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_fock_sources(expected, binding, detail);
    binding = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_grid_reference(std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              dft::CudaKsResidentGridBinding& binding,
                                              std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_grid_reference(expected, binding, detail);
    binding = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_density(std::size_t index,
                                       const dft::CudaKsFinalStateToken& expected, int& device,
                                       const double*& alpha, const double*& beta,
                                       std::size_t& matrix_elements, unsigned& spins,
                                       void*& source_stream, std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_density(expected, device, alpha, beta, matrix_elements,
                                                  spins, source_stream, detail);
    device = -1;
    alpha = nullptr;
    beta = nullptr;
    matrix_elements = 0;
    spins = 0;
    source_stream = nullptr;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status fixed_density_profile(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected,
                                            KsFixedDensityProfile& profile,
                                            std::string& detail) override {
    profile = {};
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->fixed_density_profile(expected, profile, detail);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_grid(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                    int& device, const double*& points, const double*& weights,
                                    const double*& atomic_weights, std::size_t& point_count,
                                    std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_grid(expected, device, points, weights, atomic_weights,
                                               point_count, detail);
    device = -1;
    points = nullptr;
    weights = nullptr;
    atomic_weights = nullptr;
    point_count = 0;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_nonlocal_features(std::size_t index,
                                                 const dft::CudaKsFinalStateToken& expected,
                                                 int& device, const double*& density,
                                                 const double*& gradient, std::size_t& point_count,
                                                 void*& source_stream,
                                                 std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_nonlocal_features(expected, device, density, gradient,
                                                            point_count, source_stream, detail);
    device = -1;
    density = nullptr;
    gradient = nullptr;
    point_count = 0;
    source_stream = nullptr;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status read_derivative_state(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected,
                                            KsDerivativeSnapshot& output,
                                            std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_derivative_state(expected, output, detail);
    output = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status read_derivative_sources(std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              KsDerivativeSnapshot& output,
                                              std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_derivative_sources(expected, output, detail);
    output = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_prepare_force_memory(std::size_t index,
                                                const dft::CudaKsFinalStateToken& expected,
                                                std::size_t additional_device_bytes,
                                                std::array<std::uint64_t, 4>& receipt,
                                                std::string& detail) override {
    receipt = {};
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_prepare_force_memory(expected, additional_device_bytes,
                                                           receipt, detail);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_integral_gradient_additional_device_bytes(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::size_t& bytes,
      std::string& detail) override {
    bytes = 0;
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_integral_gradient_additional_device_bytes(expected, bytes,
                                                                                detail);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status density_fitted_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->density_fitted_integral_gradient(
          expected, density, weighted_density, output, maximum_bytes, work, detail);
    output.clear();
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_full_range_integral_derivatives(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_full_range_integral_derivatives(expected, output, detail);
    output.clear();
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status mixed_density_fitted_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density,
      std::span<const double> lr_correction, std::vector<double>& output, std::size_t maximum_bytes,
      std::array<std::uint64_t, 9>& work, std::string& detail) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->mixed_rsh_df_integral_gradient(expected, output, maximum_bytes,
                                                                work, detail, &density,
                                                                &weighted_density, lr_correction);
    output.clear();
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density = nullptr,
      const std::vector<scf::reference::Matrix>* cached_weighted_density = nullptr,
      bool combined_two_electron = false) override {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_integral_gradient(
          expected, output, maximum_bytes, work, detail, cached_density, cached_weighted_density,
          combined_two_electron);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    std::vector<scf::CudaDensityFittingMetricDiagnostic> result;
    for (std::size_t index = 0; index < items_.size(); ++index)
      if (items_[index].plan)
        for (auto diagnostic : items_[index].plan->fitted_diagnostics()) {
          // KS assigns one provider/bucket per input slot, including equal shapes.
          diagnostic.bucket_id = index;
          diagnostic.system_index = index;
          result.push_back(diagnostic);
        }
    return result;
  }

 private:
  /** All other owners remain alive while one CPU item executes. The selected
   * item's externally materialized seed is also distinct from its plan. */
  std::size_t host_numeric_capacity(std::size_t exclude_plan = SIZE_MAX) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < items_.size(); ++i) {
      const auto& item = items_[i];
      if (item.plan && i != exclude_plan)
        bytes = runtime::add_capacity(bytes, item.plan->host_numeric_capacity());
      if (item.warm)
        bytes = runtime::add_capacity(
            bytes, runtime::vector_capacities(item.warm->density, item.warm->coordinates));
    }
    return bytes;
  }

  struct Item {
    // Admission belongs to the immutable prepared model/basis/backend, not a
    // replaceable geometry owner. A failed rebuild must not revoke it and
    // prevent the next force replay from reconstructing this item's plan.
    generativeqc_property_flags prepared_properties{};
    std::optional<std::size_t> stationary_device_allowance;
    // A single-calculation adapter rethrows the original terminal failure so
    // C ABI status mapping and context detail survive the ragged scheduler.
    std::exception_ptr failure;
    std::unique_ptr<KsPreparedState> plan;
    // Density is materialized only for explicit output, import, or rebuilding
    // an owner. Empty density with resident_warm=true is a valid lazy snapshot.
    mutable std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
#if GENERATIVEQC_HAS_CUDA
    dft::CudaKsTransfers retired_transfers;
#endif
  };
  std::unique_ptr<KsPreparedState> make_plan(
      const core::System& system,
      std::optional<std::size_t> stationary_device_allowance = {}) const {
    auto plan = std::make_unique<KsPreparedState>(capabilities_, system, execution_plan_, options_,
                                                  grid_spec_, backend_, device_,
                                                  ks_auxiliary_for_system(system, auxiliary_));
    if (stationary_device_allowance) {
      std::string detail;
      const auto status = plan->native_cuda_stationary_limit(*stationary_device_allowance, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) throw MethodError(status, detail);
    }
    return plan;
  }
  void materialize_warm(std::size_t i) const {
    const auto& item = items_.at(i);
    if (item.warm && item.warm->density.empty() && item.resident_warm)
      item.warm->density = item.plan->warm_density();
  }

  Capabilities capabilities_;
  std::vector<core::System> systems_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  dft::GridSpec grid_spec_;
  generativeqc_backend backend_;
  int device_;
  bool warm_enabled_, warm_updates_{true};
  std::optional<core::System> auxiliary_;
  std::vector<Item> items_;
};

}  // namespace

Result adapt_dft_result(scf::ScfResult native, generativeqc_backend backend) {
  return adapt_result(std::move(native), backend);
}

generativeqc_status dft_cuda_stationary_capacity(const generativeqc_method_descriptor& descriptor,
                                                 std::span<const std::uint64_t> shape,
                                                 dft::CudaStationaryCapacity& capacity,
                                                 std::string& detail) {
  if (shape.size() != 7 || !shape[0] || !shape[1] || !shape[2] ||
      std::any_of(shape.begin(), shape.end(),
                  [](auto value) { return value > std::numeric_limits<std::size_t>::max(); }))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    NativeKsExecutionPlan execution_plan;
    auto options = dft_options(descriptor, GENERATIVEQC_BACKEND_CUDA, execution_plan);
    const auto grid = ks_grid_options(descriptor, options, execution_plan);
    const auto* binding = native_cuda_stationary_binding(execution_plan, options, grid,
                                                         shape[5] == 0 && shape[6] == 0, shape[4],
                                                         shape[1], shape[0]);
    if (!binding) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    // This shape-only query has no element inventory. V4's exact count is
    // checked by the prepared grid/controller; never invent a uniform count.
    if (grid.version != 4 &&
        runtime::size_mul(shape[0], dft::grid_points_per_atom(grid)) != shape[3])
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto candidate = dft::cuda_stationary_capacity(
        *binding, shape[0], shape[1], shape[2], shape[3], options.xc_tile_points,
        dft::admit_stationary_force_controls(options.energy_tolerance, options.density_tolerance,
                                             options.screening_tolerance, false) &&
            dft::application_gradient_qualification(*binding, options.application_precision_budget,
                                                    shape[1], shape[0], shape[4]) != 0);
    capacity = candidate;
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const MethodError& error) {
    detail = error.what();
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}

namespace {
std::unique_ptr<KsPreparedBatch> prepare_ks_batch(const Capabilities& capabilities,
                                                  core::ContextState& context,
                                                  std::vector<core::System> systems,
                                                  const generativeqc_method_descriptor& descriptor,
                                                  generativeqc_batch_flags flags,
                                                  const core::System* auxiliary_basis) {
  if ((flags & ~GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "KS batches support warm starts but not HF-specific profiling flags");
#if !GENERATIVEQC_HAS_CUDA
  if (context.requested_backend == GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  NativeKsExecutionPlan execution_plan;
  auto options = dft_options(descriptor, context.requested_backend, execution_plan);
  for (const auto& system : systems) {
    validate_ks_spin_state(execution_plan, system);
    if (options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
        execution_plan.semilocal_family != dft::SemilocalFamily::Wb97mv &&
        (!system.ecp_terms.empty() ||
         std::any_of(system.atoms.begin(), system.atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; })))
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT ECP density fitting is not qualified");
  }
  const auto auxiliary = ks_auxiliary_template(descriptor, auxiliary_basis);
  if (!systems.empty()) validate_ks_auxiliary_geometry(systems.front(), auxiliary);
  auto grid = ks_grid_options(descriptor, options, execution_plan);
  return std::make_unique<KsPreparedBatch>(
      capabilities, std::move(systems), execution_plan, std::move(options), std::move(grid),
      context.requested_backend, context.device_id,
      (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0, auxiliary);
}

/** Single-system ABI/SDK view of the same native scheduler used by Python.
 * Keep repeated-call warm starts, token identity and diagnostics in the batch;
 * this adapter owns no scientific buffers and performs no SCF/force work. */
class KsPreparedCalculation final : public PreparedCalculation, public KsSingleOwnerPort {
 public:
  KsPreparedCalculation(Capabilities capabilities, std::size_t atoms, generativeqc_backend backend,
                        int device, std::unique_ptr<KsPreparedBatch> owner)
      : PreparedCalculation(backend, device),
        capabilities_(capabilities),
        atoms_(atoms),
        owner_(std::move(owner)) {}

  std::size_t atom_count() const noexcept override { return atoms_; }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  generativeqc_property_flags supported_properties() const noexcept override {
    return *owner_->supported_properties(0);
  }
  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    return owner_->execution_resources(0);
  }
  void invalidate_result() override { owner_->invalidate_result(); }
  std::optional<KsTransportDiagnostic> ks_transport_diagnostic() const override {
    return owner_->ks_transport_diagnostic(0);
  }
  generativeqc_status final_state_token(dft::CudaKsFinalStateToken& token,
                                        std::string& detail) const override {
    return owner_->single_owner().final_state_token(token, detail);
  }
  generativeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density,
                                       dft::VerifiedKsFinalState& state,
                                       std::string& detail) override {
    return owner_->single_owner().read_final_state(expected, compute_weighted_density, state,
                                                   detail);
  }
  generativeqc_status prepared_integral_gradient(
      const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) override {
    return owner_->single_owner().prepared_integral_gradient(expected, density, weighted_density,
                                                             output, maximum_bytes, work, detail);
  }

 protected:
  Result execute_impl(bool compute_forces) override {
    return owner_->execute_single(compute_forces);
  }
  void reset_execution_impl() override { owner_->reset_execution(); }

 private:
  Capabilities capabilities_;
  std::size_t atoms_;
  std::unique_ptr<KsPreparedBatch> owner_;
};
}  // namespace

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* auxiliary_basis) {
  // Retained native calculations historically reuse their last good density.
  // Python singlepoint remains cold by constructing a fresh owner per call.
  auto owner = prepare_ks_batch(capabilities, context, {system}, descriptor,
                                GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, auxiliary_basis);
  return std::make_unique<KsPreparedCalculation>(capabilities, system.atoms.size(),
                                                 context.requested_backend, context.device_id,
                                                 std::move(owner));
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const generativeqc_method_descriptor& descriptor,
                                                 generativeqc_batch_flags flags,
                                                 const core::System* auxiliary_basis) {
  return prepare_ks_batch(capabilities, context, std::move(systems), descriptor, flags,
                          auxiliary_basis);
}

}  // namespace generativeqc::methods::detail
