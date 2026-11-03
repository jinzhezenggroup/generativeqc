#include "methods/uccsd_method.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>

#include "cc/uccsd_force.hpp"
#include "cc/uccsd_solver.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "methods/ump2_method.hpp"
#include "methods/unrestricted_df.hpp"
#include "methods/unrestricted_integrals.hpp"
#include "methods/unrestricted_response.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_interaction_source.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "posthf/ump2_force.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {
namespace {

class UccsdPrepared final : public PreparedCalculation {
 public:
  UccsdPrepared(Capabilities capabilities, core::System system, scf::ScfOptions reference_options,
                cc::SolverOptions options, std::size_t reference_capacity, double mp2_threshold,
                bool cuda, int device, UnrestrictedFitting fitting)
      : PreparedCalculation(cuda ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE,
                            device),
        capabilities_(capabilities),
        system_(std::move(system)),
        reference_options_(reference_options),
        options_(options),
        reference_capacity_(reference_capacity),
        mp2_threshold_(mp2_threshold),
        cuda_(cuda),
        device_(device),
        fitting_(std::move(fitting)) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_;
  }

  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return performance_;
  }

  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    performance_.reset();
  }

  Result execute_impl(bool compute_forces) override {
    return execute_components_impl(compute_forces, 0);
  }

  Result execute_components_impl(bool compute_forces, unsigned components) override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    performance_.reset();
    if ((components & ~(kHfComponent | kMp2Component)) ||
        (components && capabilities_.method != GENERATIVEQC_METHOD_UCCSD_T))
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "unrestricted companions require UCCSD(T)");
    try {
      const bool triples = capabilities_.method == GENERATIVEQC_METHOD_UCCSD_T;
      const auto derivative_capacity = cuda_ && compute_forces
                                           ? posthf::one_electron_response_capacity(system_)
                                           : posthf::source_scratch_bytes;
      if (compute_forces) {
        const auto nbf = molecule::ao_count(system_);
        const auto alpha =
            static_cast<std::size_t>((system_.electron_count + system_.multiplicity - 1) / 2);
        cc::unrestricted_force_capacity(
            nbf, alpha, system_.electron_count - alpha, options_, triples, reference_capacity_,
            fitting_.auxiliary && !fitting_.full_reference, derivative_capacity);
      }
      const auto started = std::chrono::steady_clock::now();
      const auto strategy = scf::resolve_fock_build(
          fitting_.reference_spec(compute_forces),
          cuda_ ? scf::FockBackend::Cuda : scf::FockBackend::Cpu, 0, fitting_.threshold);
      scf::PreparedFockPlan reference_plan(
          system_, fitting_.full_reference ? &*fitting_.auxiliary : nullptr, strategy,
          cuda_ ? device_ : -1, cuda_ ? options_.max_bytes : 0,
          cuda_ && !fitting_.full_reference && compute_forces && (components & kHfComponent) ? 1
                                                                                             : 0,
          fitting_.full_reference && compute_forces && (components & kHfComponent) ? 1 : 0);
      auto reference_options = reference_options_;
      reference_options.compute_forces =
          (cuda_ || fitting_.full_reference) && compute_forces && (components & kHfComponent);
      reference_options.density_fitting_relative_threshold = fitting_.threshold;
      reference_options.resolved_fock_build = reference_plan.strategy();
      auto reference_result = warm_reference::solve_reference(
          nullptr, scf::FockSpin::Unrestricted, nullptr,
          [&](const std::vector<double>* seed) {
            return scf::run_prepared_fock_strategy(reference_plan, reference_options, seed);
          },
          "UHF did not produce a converged physical unrestricted CC reference");
      const auto reference_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      const auto& reference = *reference_result.unrestricted_reference;
      auto retained_reference_capacity = posthf::checked_add(
          reference_capacity_, cuda_ ? reference_plan.diagnostic().peak_device_bytes : 0);
      reference_result.density.clear();
      reference_result.density.shrink_to_fit();
      scf::PreparedFockInteractionSourceView reference_source(reference_plan);
      std::optional<posthf::DensityFittedInteractionSource> fitted;
      if (fitting_.auxiliary) {
        if (retained_reference_capacity >= options_.max_bytes)
          throw std::length_error("unrestricted DF reference exceeds endpoint budget");
        fitted.emplace(reference_source, *fitting_.auxiliary, fitting_.threshold,
                       std::min(fitting_.budget, options_.max_bytes - retained_reference_capacity),
                       cuda_, device_, compute_forces, unrestricted_integral_backend());
        retained_reference_capacity =
            posthf::checked_add(retained_reference_capacity, fitted->numeric_capacity_bytes());
      }
      const integrals::ElectronInteractionSource& source =
          fitted ? static_cast<const integrals::ElectronInteractionSource&>(*fitted)
                 : reference_source;
      const auto* independent_reference =
          fitted && !fitting_.full_reference ? &reference_source : nullptr;
      if (compute_forces)
        cc::unrestricted_force_capacity(reference.nbf, reference.nocc[0], reference.nocc[1],
                                        options_, triples, retained_reference_capacity,
                                        independent_reference != nullptr, derivative_capacity);
      std::optional<response::SpinResponseProblem> orbital_problem;
      if (compute_forces && (reference.nocc[0] * (reference.nbf - reference.nocc[0]) +
                             reference.nocc[1] * (reference.nbf - reference.nocc[1])))
        orbital_problem.emplace(unrestricted_response_problem(reference, reference_plan,
                                                              fitting_.reference_spec(true)));
      Result result;
      if (components & kHfComponent) {
        result.hf_component.emplace();
        result.hf_component->energy = reference.energy;
        if (compute_forces)
          result.hf_component->forces =
              cuda_ || fitting_.full_reference
                  ? std::move(reference_result.forces)
                  : unrestricted_reference_force_cpu(reference, reference_plan, options_.max_bytes);
      }
      if (components & kMp2Component) {
        if (retained_reference_capacity >= options_.max_bytes)
          throw std::length_error("UMP2 companion reference/source exceeds endpoint budget");
        const auto phase_budget = options_.max_bytes - retained_reference_capacity;
        // Finish the bounded MP2 phase before CC allocates its solver workspace.
        // Both phases use this exact physical UHF frame and interaction owner.
        const auto corr = mp2::conventional_unrestricted_energy(reference, source, phase_budget,
                                                                mp2_threshold_, cuda_, device_);
        result.mp2_component.emplace();
        auto& companion = *result.mp2_component;
        companion.energy =
            reference.energy + corr.channels[0] + corr.channels[1] + corr.channels[2];
        std::optional<mp2::ConventionalForceResult> companion_force;
        if (compute_forces) {
          companion_force = mp2::unrestricted_force_cpu(
              reference, orbital_problem ? &*orbital_problem : nullptr, source, phase_budget,
              mp2_threshold_, cuda_, device_, fitted ? &*fitted : nullptr, independent_reference,
              unrestricted_integral_backend());
          companion.forces = std::move(companion_force->forces);
        }
        if (!std::isfinite(companion.energy))
          throw std::runtime_error("nonfinite UMP2 companion energy");
        companion.correlation.emplace();
        auto& diagnostic = *companion.correlation;
        diagnostic.struct_size = sizeof(diagnostic);
        diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
        diagnostic.reference_energy = reference.energy;
        diagnostic.reference_residual = reference.commutator_residual;
        diagnostic.opposite_spin_energy = corr.channels[2];
        diagnostic.same_spin_energy = corr.channels[0] + corr.channels[1];
        diagnostic.minimum_absolute_denominator = corr.minimum_denominator;
        diagnostic.numeric_capacity_bytes =
            posthf::checked_add(retained_reference_capacity, corr.numeric_capacity_bytes);
        diagnostic.energy_tile_count = corr.tiles;
        diagnostic.mo_host_staging = cuda_ ? 1 : 0;
        if (companion_force) {
          const auto& force = *companion_force;
          diagnostic.response_iterations = force.response.iterations;
          diagnostic.response_restarts = force.response.restarts;
          diagnostic.response_absolute_residual = force.response.residual_norm;
          diagnostic.response_relative_residual = force.response.relative_residual;
          diagnostic.response_workspace_bytes = force.response.workspace_bytes;
          diagnostic.measured_response_workspace_peak_bytes =
              force.response.measured_workspace_peak_bytes;
          diagnostic.response_workspace_allocation_count =
              force.response.workspace_allocation_count;
          diagnostic.derivative_workspace_bytes = force.derivative_workspace_bytes;
          diagnostic.planned_endpoint_peak_bytes = std::max<std::uint64_t>(
              diagnostic.numeric_capacity_bytes,
              posthf::checked_add(retained_reference_capacity, force.planned_endpoint_peak_bytes));
          diagnostic.numeric_capacity_bytes = diagnostic.planned_endpoint_peak_bytes;
          diagnostic.force_provenance_flags = fitted && cuda_ ? 0x1 : 0x7;
        }
        std::copy_n(mp2::generated::kUmp2InventoryHash, 64, diagnostic.equation_hash);
      }
      const auto solved =
          cc::solve_unrestricted(reference, source, options_, triples, retained_reference_capacity,
                                 compute_forces, cuda_, device_);
      if (!solved.converged)
        throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED,
                          "UCCSD physical residual/replay convergence gates did not pass");
      // UHF owns the nuclear repulsion; do not add it a second time here.
      result.energy = reference.energy + solved.correlation_energy + solved.triples_energy;
      if (!std::isfinite(result.energy))
        throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "nonfinite unrestricted CC total energy");
      result.convergence = {
          solved.iterations, solved.energy_change,
          std::max(solved.replay_singles_residual, solved.replay_doubles_residual), true};
      result.physical_residual_rms = reference.commutator_residual;
      result.fock_builds = reference_result.fock_builds;
      result.executed_backend =
          cuda_ ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
      generativeqc_correlation_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(diagnostic);
      diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
      diagnostic.reference_energy = reference.energy;
      diagnostic.reference_residual = reference.commutator_residual;
      diagnostic.mo_host_staging = cuda_ ? 1 : 0;
      diagnostic.minimum_absolute_denominator = solved.minimum_denominator;
      diagnostic.numeric_capacity_bytes = solved.numeric_capacity_bytes;
      if (compute_forces) {
        auto force = cc::unrestricted_force(
            reference, source, solved, options_, triples, retained_reference_capacity,
            orbital_problem ? &*orbital_problem : nullptr, cuda_, device_,
            fitted ? &*fitted : nullptr, independent_reference, unrestricted_integral_backend());
        result.forces = std::move(force.forces);
        diagnostic.numeric_capacity_bytes = force.numeric_capacity_bytes;
        diagnostic.planned_endpoint_peak_bytes = force.numeric_capacity_bytes;
        diagnostic.response_iterations = force.orbital.iterations;
        diagnostic.response_restarts = force.orbital.restarts;
        diagnostic.response_absolute_residual = force.orbital.residual_norm;
        diagnostic.response_relative_residual = force.orbital.relative_residual;
        diagnostic.response_workspace_bytes = force.orbital.workspace_bytes;
        diagnostic.measured_response_workspace_peak_bytes =
            force.orbital.measured_workspace_peak_bytes;
        diagnostic.response_workspace_allocation_count = force.orbital.workspace_allocation_count;
        diagnostic.force_provenance_flags = fitted && cuda_ ? 0x1 : 0x7;
      }
      if (result.mp2_component)
        diagnostic.numeric_capacity_bytes =
            std::max(diagnostic.numeric_capacity_bytes,
                     result.mp2_component->correlation->numeric_capacity_bytes);
      diagnostic.planned_endpoint_peak_bytes =
          std::max(diagnostic.planned_endpoint_peak_bytes, diagnostic.numeric_capacity_bytes);
      diagnostic.ccsd_iterations = solved.iterations;
      diagnostic.ccsd_correlation_energy = solved.correlation_energy;
      diagnostic.ccsd_energy_change = solved.energy_change;
      diagnostic.ccsd_singles_residual_max = solved.singles_residual;
      diagnostic.ccsd_doubles_residual_max = solved.doubles_residual;
      diagnostic.ccsd_replay_singles_residual_max = solved.replay_singles_residual;
      diagnostic.ccsd_replay_doubles_residual_max = solved.replay_doubles_residual;
      std::copy_n(cc::unrestricted_equation_hash(), 64, diagnostic.equation_hash);
      std::copy_n(cc::unrestricted_equation_hash(), 64, diagnostic.ccsd_replay_equation_hash);
      if (triples) {
        diagnostic.ccsd_t_triples_energy = solved.triples_energy;
        diagnostic.ccsd_t_virtual_triples = solved.triples_tiles;
        diagnostic.ccsd_t_workspace_bytes = solved.triples_workspace_bytes;
        std::copy_n(cc::unrestricted_triples_hash(), 64, diagnostic.ccsd_t_equation_hash);
      }
      CcPerformanceDiagnostic performance;
      performance.reference_seconds = reference_seconds;
      performance.problem_seconds = solved.problem_seconds;
      performance.provider_seconds = solved.provider_work.provider_seconds;
      performance.source_seconds = solved.provider_work.source_seconds;
      performance.solver_seconds = solved.solver_seconds;
      performance.iteration_seconds = solved.iteration_seconds;
      performance.replay_seconds = solved.replay_seconds;
      performance.update_seconds = solved.update_seconds;
      performance.diis_seconds = solved.diis_seconds;
      performance.triples_seconds = solved.triples_seconds;
      performance.source_scans = solved.provider_work.source_scans;
      performance.source_reads = solved.provider_work.source_reads;
      performance.source_values = solved.provider_work.source_values;
      performance.transform_fmas = solved.provider_work.transform_fmas;
      performance.transform_stages = solved.provider_work.transform_stages;
      performance.mo_blocks = solved.provider_work.mo_blocks;
      performance.iteration_graph_calls = solved.graph_calls - solved.replay_calls;
      performance.replay_graph_calls = solved.replay_calls;
      performance.update_calls = solved.update_calls;
      last_ = diagnostic;
      performance_ = performance;
      return result;
    } catch (const MethodError&) {
      throw;
    } catch (const std::length_error& error) {
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY, error.what());
    } catch (const std::runtime_error& error) {
      throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, error.what());
    }
  }

 private:
  Capabilities capabilities_;
  core::System system_;
  scf::ScfOptions reference_options_;
  cc::SolverOptions options_;
  std::size_t reference_capacity_{};
  double mp2_threshold_{};
  bool cuda_{};
  int device_{};
  UnrestrictedFitting fitting_;
  mutable std::mutex mutex_;
  std::optional<generativeqc_correlation_diagnostic> last_;
  std::optional<CcPerformanceDiagnostic> performance_;
};

}  // namespace

generativeqc_status validate_uccsd_system(generativeqc_method method, const core::System& system,
                                          std::string& detail) {
  const auto status = validate_ump2_system(method, system, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) {
    if (const auto position = detail.find("UMP2"); position != std::string::npos)
      detail.replace(position, 4, "unrestricted CC");
    return status;
  }
  if (std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 2; })) {
    detail = "native unrestricted CC is qualified for s/p/d references";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_uccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* auxiliary) {
  if (context.requested_backend != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
      context.requested_backend != GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native unrestricted CC requires an explicit CPU/CUDA backend; no fallback");
  if (descriptor.ks_options || descriptor.initial_guess || descriptor.ccsd_frozen_core ||
      descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64 ||
      descriptor.screening_tolerance != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native unrestricted CC requires unscreened real FP64 without "
                      "KS, initial-guess or frozen-core controls");
  if (descriptor.correlation_memory_budget_bytes > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("unrestricted CC memory budget exceeds signed-64-bit capacity");
  cc::SolverOptions options;
  options.max_bytes = descriptor.correlation_memory_budget_bytes
                          ? descriptor.correlation_memory_budget_bytes
                          : 256ULL << 20;
  if (descriptor.ccsd_max_iterations) options.max_iterations = descriptor.ccsd_max_iterations;
  options.diis_size = descriptor.ccsd_diis_history;
  if (descriptor.ccsd_energy_tolerance) options.energy_tolerance = descriptor.ccsd_energy_tolerance;
  if (descriptor.ccsd_residual_tolerance)
    options.residual_tolerance = descriptor.ccsd_residual_tolerance;
  if (descriptor.ccsd_denominator_threshold)
    options.denominator_threshold = descriptor.ccsd_denominator_threshold;
  options.damping = descriptor.ccsd_damping;
  options.level_shift = descriptor.ccsd_level_shift;
  cc::validate_options(options);
  const auto mp2_threshold =
      descriptor.mp2_denominator_threshold != 0 ? descriptor.mp2_denominator_threshold : 1e-10;
  if (!std::isfinite(mp2_threshold) || mp2_threshold <= 0)
    throw std::invalid_argument("invalid UMP2 companion denominator threshold");
  if (!std::isfinite(descriptor.energy_tolerance) || descriptor.energy_tolerance < 0 ||
      !std::isfinite(descriptor.density_tolerance) || descriptor.density_tolerance < 0)
    throw std::invalid_argument("invalid unrestricted CC reference convergence threshold");
  scf::ScfOptions reference_options;
  reference_options.max_iterations = descriptor.max_iterations ? descriptor.max_iterations : 100;
  reference_options.diis_history = descriptor.diis_history ? descriptor.diis_history : 8;
  reference_options.energy_tolerance =
      descriptor.energy_tolerance > 0 ? std::min(descriptor.energy_tolerance, 1e-11) : 1e-11;
  reference_options.density_tolerance =
      descriptor.density_tolerance > 0 ? std::min(descriptor.density_tolerance, 1e-11) : 1e-11;
  reference_options.screening_tolerance = 0;
  reference_options.compute_forces = false;
  reference_options.export_physical_reference = true;
  reference_options.reference_memory_budget_bytes = options.max_bytes;
  auto fitting = unrestricted_fitting(descriptor, auxiliary, options.max_bytes);
  const auto reference_capacity =
      fitting.reference_capacity(system, reference_options.diis_history);
  const auto nbf = molecule::ao_count(system);
  const auto alpha =
      static_cast<std::size_t>((system.electron_count + system.multiplicity - 1) / 2);
  const auto beta = static_cast<std::size_t>(system.electron_count) - alpha;
  try {
    cc::unrestricted_capacity(nbf, alpha, beta, options,
                              capabilities.method == GENERATIVEQC_METHOD_UCCSD_T,
                              reference_capacity);
  } catch (const std::length_error& error) {
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY, error.what());
  }
  return std::make_unique<UccsdPrepared>(capabilities, system, reference_options, options,
                                         reference_capacity, mp2_threshold,
                                         context.requested_backend == GENERATIVEQC_BACKEND_CUDA,
                                         context.device_id, std::move(fitting));
}

}  // namespace generativeqc::methods::detail
