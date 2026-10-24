#include "methods/ump2_method.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>

#include "methods/batch_capabilities.hpp"
#include "methods/correlated_batch.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "methods/unrestricted_response.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "posthf/ump2_force.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {

scf::FockBuildSpec ump2_reference_fock_spec() {
  auto spec = scf::make_hf_fock_spec(scf::FockSpin::Unrestricted);
  spec.derivative_order = 0;
  return spec;
}

namespace {

class Ump2Prepared final : public PreparedCalculation, public ReferenceSeededCalculation {
 public:
  Ump2Prepared(Capabilities caps, core::System system, scf::ScfOptions options, std::size_t budget,
               std::size_t reference_capacity, double threshold)
      : caps_(caps),
        system_(std::move(system)),
        options_(options),
        budget_(budget),
        reference_capacity_(reference_capacity),
        threshold_(threshold) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return caps_; }
  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_;
  }
  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
  }

  Result execute_seeded_reference(bool compute_forces, const scf::HfWarmState* seed, bool* fallback,
                                  std::optional<scf::HfWarmState>* candidate) override {
    return execute_with_seed(compute_forces, seed, fallback, candidate);
  }

  Result execute_impl(bool compute_forces) override {
    return execute_with_seed(compute_forces, nullptr, nullptr, nullptr);
  }

  Result execute_with_seed(bool compute_forces, const scf::HfWarmState* initial_state,
                           bool* warm_start_fallback,
                           std::optional<scf::HfWarmState>* retained_warm_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    if (warm_start_fallback) *warm_start_fallback = false;
    try {
      const auto warm_capacity = warm_reference::reservation_bytes(
          system_, initial_state, retained_warm_state != nullptr, 2);
      if (warm_capacity >= budget_ || reference_capacity_ > budget_ - warm_capacity)
        throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                          "UMP2 warm state and UHF reference exceed numeric memory budget");
      const auto phase_budget = budget_ - warm_capacity;
      const auto strategy =
          scf::resolve_fock_build(ump2_reference_fock_spec(), scf::FockBackend::Cpu, 0);
      scf::PreparedFockPlan reference_plan(system_, nullptr, strategy);
      auto execution = options_;
      execution.resolved_fock_build = reference_plan.strategy();
      execution.reference_memory_budget_bytes = phase_budget;
      const auto run_reference = [&](const std::vector<double>* seed) {
        return scf::run_prepared_fock_strategy(reference_plan, execution, seed);
      };
      auto hf = warm_reference::solve_reference(
          initial_state ? &initial_state->density : nullptr, scf::FockSpin::Unrestricted,
          warm_start_fallback, run_reference,
          "UHF did not produce a validated physical UMP2 reference");
      const auto& reference = *hf.unrestricted_reference;
      if (retained_warm_state)
        *retained_warm_state =
            warm_reference::capture(system_, std::move(hf.density), hf.energy, hf.energy_change,
                                    hf.density_rms, static_cast<int>(hf.iterations));
      hf.density.clear();
      hf.density.shrink_to_fit();
      scf::PreparedFockInteractionSourceView source(reference_plan);
      const auto corr =
          mp2::conventional_unrestricted_energy(reference, source, phase_budget, threshold_);
      Result result;
      result.energy = reference.energy + corr.channels[0] + corr.channels[1] + corr.channels[2];
      if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite UMP2 total energy");
      std::optional<mp2::ConventionalForceResult> force;
      if (compute_forces) {
        std::optional<response::SpinResponseProblem> response_problem;
        if (reference.nocc[0] * (reference.nbf - reference.nocc[0]) +
            reference.nocc[1] * (reference.nbf - reference.nocc[1]))
          response_problem.emplace(unrestricted_response_problem(reference, reference_plan));
        force =
            mp2::unrestricted_force_cpu(reference, response_problem ? &*response_problem : nullptr,
                                        source, phase_budget, threshold_);
        result.forces = std::move(force->forces);
      }
      result.convergence = {hf.iterations, hf.energy_change, hf.density_rms, hf.converged};
      result.fock_builds = hf.fock_builds;
      result.executed_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
      generativeqc_correlation_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(diagnostic);
      diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
      diagnostic.reference_energy = reference.energy;
      diagnostic.opposite_spin_energy = corr.channels[2];
      diagnostic.same_spin_energy = corr.channels[0] + corr.channels[1];
      diagnostic.minimum_absolute_denominator = corr.minimum_denominator;
      diagnostic.reference_residual = reference.commutator_residual;
      diagnostic.numeric_capacity_bytes = posthf::checked_add(
          std::max(reference_capacity_, corr.numeric_capacity_bytes), warm_capacity);
      diagnostic.energy_tile_count = corr.tiles;
      // The generated inventory hash binds all three spin-labelled TensorIR programs.
      std::copy_n(mp2::generated::kUmp2InventoryHash, 64, diagnostic.equation_hash);
      if (force) {
        diagnostic.response_iterations = force->response.iterations;
        diagnostic.response_restarts = force->response.restarts;
        diagnostic.response_absolute_residual = force->response.residual_norm;
        diagnostic.response_relative_residual = force->response.relative_residual;
        diagnostic.response_workspace_bytes = force->response.workspace_bytes;
        diagnostic.measured_response_workspace_peak_bytes =
            force->response.measured_workspace_peak_bytes;
        diagnostic.response_workspace_allocation_count = force->response.workspace_allocation_count;
        diagnostic.derivative_workspace_bytes = force->derivative_workspace_bytes;
        diagnostic.planned_endpoint_peak_bytes = std::max<std::uint64_t>(
            diagnostic.numeric_capacity_bytes,
            posthf::checked_add(force->planned_endpoint_peak_bytes, warm_capacity));
        diagnostic.numeric_capacity_bytes = diagnostic.planned_endpoint_peak_bytes;
        diagnostic.force_provenance_flags = 0x7;
        constexpr char response_hash[] = "uhf-canonical-response-v1";
        std::copy_n(response_hash, sizeof(response_hash), diagnostic.response_operator_hash);
      }
      last_ = diagnostic;
      return result;
    } catch (const std::length_error& error) {
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY, error.what());
    }
  }

 private:
  Capabilities caps_;
  core::System system_;
  scf::ScfOptions options_;
  std::size_t budget_{}, reference_capacity_{};
  double threshold_{};
  mutable std::mutex mutex_;
  std::optional<generativeqc_correlation_diagnostic> last_;
};

}  // namespace

generativeqc_status validate_ump2_system(generativeqc_method, const core::System& system,
                                         std::string& detail) {
  if (system.shells.empty()) {
    detail = "UMP2 requires an explicit Gaussian orbital basis";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!system.ecp_terms.empty()) {
    detail = "UMP2 with ECP is unavailable";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (std::any_of(system.shells.begin(), system.shells.end(),
                  [](const auto& shell) { return shell.angular_momentum > 3; })) {
    detail = "UMP2 native conventional source supports shells through f";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  const auto n = molecule::ao_count(system);
  if (system.electron_count <= 0 || system.multiplicity == 0 ||
      static_cast<std::uint64_t>(system.multiplicity) >
          static_cast<std::uint64_t>(system.electron_count) + 1 ||
      (static_cast<std::uint64_t>(system.electron_count) + system.multiplicity - 1) % 2) {
    detail = "UMP2 requires consistent UHF electron and spin occupations";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto alpha = static_cast<std::size_t>(
      (static_cast<std::uint64_t>(system.electron_count) + system.multiplicity - 1) / 2);
  if (alpha > static_cast<std::size_t>(system.electron_count)) {
    detail = "UMP2 multiplicity exceeds the available electron count";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto beta = static_cast<std::size_t>(system.electron_count) - alpha;
  if (alpha > n || beta > n || (alpha == n && beta == n)) {
    detail = "UMP2 reference has invalid spin occupations or no virtual space";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  return GENERATIVEQC_STATUS_SUCCESS;
}

std::unique_ptr<PreparedCalculation> prepare_ump2_calculation(
    const Capabilities& caps, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System*) {
  if (context.requested_backend != GENERATIVEQC_BACKEND_CPU_REFERENCE)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "UMP2 requires the CPU backend");
  if (descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE ||
      descriptor.density_fitting_auxiliary_basis || descriptor.density_fitting_memory_budget_bytes)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "UMP2 RI/DF is unavailable");
  if (descriptor.ks_options || descriptor.initial_guess)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "UMP2 KS/preliminary initial-guess controls are unavailable");
  if (descriptor.ccsd_frozen_core)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "UMP2 frozen core is unavailable");
  if (descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "UMP2 requires real FP64");
  if (descriptor.screening_tolerance != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "UMP2 requires unscreened conventional integrals");
  const auto budget = descriptor.correlation_memory_budget_bytes
                          ? descriptor.correlation_memory_budget_bytes
                          : 256ULL << 20;
  if (budget > static_cast<std::uint64_t>(INT64_MAX))
    throw std::invalid_argument("UMP2 budget exceeds signed-64-bit numeric capacity");
  const auto threshold =
      descriptor.mp2_denominator_threshold != 0 ? descriptor.mp2_denominator_threshold : 1e-10;
  if (!std::isfinite(threshold) || threshold <= 0)
    throw std::invalid_argument("invalid UMP2 denominator threshold");
  if (!std::isfinite(descriptor.energy_tolerance) || descriptor.energy_tolerance < 0 ||
      !std::isfinite(descriptor.density_tolerance) || descriptor.density_tolerance < 0)
    throw std::invalid_argument("invalid UMP2 reference convergence threshold");
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations ? descriptor.max_iterations : 100;
  options.diis_history = descriptor.diis_history ? descriptor.diis_history : 8;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0 ? std::min(descriptor.energy_tolerance, 1e-11) : 1e-11;
  options.density_tolerance =
      descriptor.density_tolerance > 0 ? std::min(descriptor.density_tolerance, 1e-11) : 1e-11;
  options.screening_tolerance = 0;
  options.compute_forces = false;
  options.export_physical_reference = true;
  options.reference_memory_budget_bytes = budget;
  const auto reference_capacity =
      posthf::uhf_reference_capacity(system, options.diis_history, true);
  if (reference_capacity > budget)
    throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                      "UMP2 complete UHF reference exceeds numeric memory budget");
  return std::make_unique<Ump2Prepared>(caps, system, options, budget, reference_capacity,
                                        threshold);
}

std::unique_ptr<PreparedBatch> prepare_ump2_batch(const Capabilities& capabilities,
                                                  core::ContextState& context,
                                                  std::vector<core::System> systems,
                                                  const generativeqc_method_descriptor& descriptor,
                                                  generativeqc_batch_flags flags,
                                                  const core::System*) {
  if (flags & ~GENERATIVEQC_BATCH_ENABLE_WARM_STARTS)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "UMP2 batch supports warm starts only; profiling is unavailable");
  // The batch owner clears borrowed pointers before storing its descriptor.
  // Reject their scientific requests first so they cannot become no-ops.
  if (descriptor.density_fitting_auxiliary_basis || descriptor.ks_options)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "UMP2 batch rejects auxiliary-basis and KS options");
  return prepare_correlated_batch(capabilities, context, std::move(systems), descriptor,
                                  (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0,
                                  prepare_ump2_calculation, "UMP2", GENERATIVEQC_METHOD_UHF, true,
                                  false);
}

}  // namespace generativeqc::methods::detail
