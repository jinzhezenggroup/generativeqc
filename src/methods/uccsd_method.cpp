#include "methods/uccsd_method.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>

#include "cc/uccsd_solver.hpp"
#include "methods/ump2_method.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {
namespace {

class UccsdPrepared final : public PreparedCalculation {
 public:
  UccsdPrepared(Capabilities capabilities, core::System system, scf::ScfOptions reference_options,
                cc::SolverOptions options, std::size_t reference_capacity)
      : capabilities_(capabilities),
        system_(std::move(system)),
        reference_options_(reference_options),
        options_(options),
        reference_capacity_(reference_capacity) {}

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
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    performance_.reset();
    if (compute_forces)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "native unrestricted CC analytic forces are not yet available");
    try {
      const auto started = std::chrono::steady_clock::now();
      const auto strategy = scf::resolve_fock_build(
          scf::make_hf_fock_spec(scf::FockSpin::Unrestricted), scf::FockBackend::Cpu, 0);
      scf::PreparedFockPlan reference_plan(system_, nullptr, strategy);
      auto reference_options = reference_options_;
      reference_options.resolved_fock_build = reference_plan.strategy();
      auto reference_result = scf::run_prepared_fock_strategy(reference_plan, reference_options);
      if (!reference_result.converged || !reference_result.unrestricted_reference)
        throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED,
                          "UHF did not produce a converged physical unrestricted CC reference");
      const auto reference_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      const auto& reference = *reference_result.unrestricted_reference;
      scf::PreparedFockInteractionSourceView source(reference_plan);
      const bool triples = capabilities_.method == GENERATIVEQC_METHOD_UCCSD_T;
      const auto solved =
          cc::solve_unrestricted(reference, source, options_, triples, reference_capacity_);
      if (!solved.converged)
        throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED,
                          "UCCSD physical residual/replay convergence gates did not pass");
      Result result;
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
      result.executed_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
      generativeqc_correlation_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(diagnostic);
      diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
      diagnostic.reference_energy = reference.energy;
      diagnostic.reference_residual = reference.commutator_residual;
      diagnostic.minimum_absolute_denominator = solved.minimum_denominator;
      diagnostic.numeric_capacity_bytes = solved.numeric_capacity_bytes;
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
  if (molecule::ao_count(system) > 12 ||
      std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 2; })) {
    detail =
        "native unrestricted CC is qualified for all-electron s/p/d references through twelve AOs";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_uccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor) {
  if (context.requested_backend != GENERATIVEQC_BACKEND_CPU_REFERENCE)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native unrestricted CC currently requires the CPU backend; no fallback");
  if (descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
      descriptor.density_fitting_auxiliary_basis ||
      descriptor.density_fitting_memory_budget_bytes || descriptor.ks_options ||
      descriptor.initial_guess || descriptor.ccsd_frozen_core ||
      descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64 ||
      descriptor.screening_tolerance != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native unrestricted CC requires unscreened all-electron real FP64 without "
                      "DF, KS, initial-guess or frozen-core controls");
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
  const auto reference_capacity =
      posthf::uhf_reference_capacity(system, reference_options.diis_history, true);
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
                                         reference_capacity);
}

}  // namespace generativeqc::methods::detail
