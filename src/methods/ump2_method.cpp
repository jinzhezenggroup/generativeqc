#include "methods/ump2_method.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>

#include "methods/correlated_warm_reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/ump2_cpu_generated.hpp"
#include "posthf/ump2_energy.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {
namespace {

class Ump2Prepared final : public PreparedCalculation {
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

  Result execute(bool compute_forces) override {
    return execute_with_seed(compute_forces, nullptr, nullptr, nullptr);
  }

  Result execute_with_seed(bool compute_forces, const scf::HfWarmState* initial_state,
                           bool* warm_start_fallback,
                           std::optional<scf::HfWarmState>* retained_warm_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    if (warm_start_fallback) *warm_start_fallback = false;
    if (compute_forces)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "UMP2 forces are unavailable");
    try {
      const auto warm_capacity = warm_reference::reservation_bytes(
          system_, initial_state, retained_warm_state != nullptr, 2);
      if (warm_capacity >= budget_ || reference_capacity_ > budget_ - warm_capacity)
        throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                          "UMP2 warm state and UHF reference exceed numeric memory budget");
      const auto phase_budget = budget_ - warm_capacity;
      const auto strategy = scf::resolve_fock_build(
          scf::make_hf_fock_spec(scf::FockSpin::Unrestricted), scf::FockBackend::Cpu, 0);
      scf::PreparedFockPlan reference_plan(system_, nullptr, strategy);
      auto execution = options_;
      execution.resolved_fock_build = reference_plan.strategy();
      execution.reference_memory_budget_bytes = phase_budget;
      const auto run_reference = [&](const scf::HfWarmState* state) {
        return scf::run_prepared_fock_strategy(reference_plan, execution,
                                               state ? &state->density : nullptr);
      };
      scf::ScfResult hf;
      if (initial_state) {
        bool cold = false;
        try {
          hf = run_reference(initial_state);
        } catch (...) {
          cold = true;
          if (warm_start_fallback) *warm_start_fallback = true;
          hf = run_reference(nullptr);
        }
        if (!cold && (!hf.converged || !hf.unrestricted_reference)) {
          hf = {};
          if (warm_start_fallback) *warm_start_fallback = true;
          hf = run_reference(nullptr);
        }
      } else {
        hf = run_reference(nullptr);
      }
      if (!hf.converged || !hf.unrestricted_reference)
        throw MethodError(GENERATIVEQC_STATUS_NOT_CONVERGED,
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

generativeqc_status ump2_item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

class Ump2PreparedBatch final : public PreparedBatch {
 public:
  Ump2PreparedBatch(Capabilities capabilities, core::ContextState& context,
                    std::vector<core::System> systems,
                    const generativeqc_method_descriptor& descriptor, bool warm_enabled)
      : capabilities_(capabilities),
        context_(&context),
        systems_(std::move(systems)),
        warm_enabled_(warm_enabled),
        warm_states_(systems_.size()) {
    descriptor_ = descriptor;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_ump2_calculation(capabilities_, *context_, system, descriptor_));
      owner_coordinates_.push_back(warm_reference::coordinates(system));
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }
  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }
  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("UMP2 batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& item = results[index];
      item.bucket_id = index;
      item.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      item.calculation.executed_backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
      try {
        auto target = systems_[index];
        auto target_coordinates = warm_reference::coordinates(target);
        if (!coordinates.empty() && coordinates[index]) {
          if (!warm_reference::valid_coordinates(*coordinates[index], target))
            throw std::invalid_argument("invalid UMP2 batch item coordinates");
          target_coordinates = *coordinates[index];
          warm_reference::set_coordinates(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          owners_[index] = prepare_ump2_calculation(capabilities_, *context_, target, descriptor_);
          owner_coordinates_[index] = std::move(target_coordinates);
        }
        const bool seeded = warm_enabled_ && warm_states_[index].has_value();
        item.warm_start_used = seeded;
        bool fallback = false;
        std::optional<scf::HfWarmState> next;
        auto& owner = static_cast<Ump2Prepared&>(*owners_[index]);
        item.calculation =
            owner.execute_with_seed(compute_forces, seeded ? &*warm_states_[index] : nullptr,
                                    &fallback, warm_enabled_ && warm_updates_ ? &next : nullptr);
        item.warm_start_fallback = fallback;
        if (next) warm_states_[index].swap(next);
        item.status = GENERATIVEQC_STATUS_SUCCESS;
      } catch (...) {
        item.status = ump2_item_exception_status();
      }
    }
    return results;
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const override {
    return owners_.at(index)->correlation_diagnostic();
  }
  void clear_warm_starts() override {
    for (auto& state : warm_states_) state.reset();
  }
  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    return posthf::checked_mul(2, posthf::checked_mul(n, n));
  }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    return warm_states_.at(index);
  }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("checkpoint restore requires a warm-enabled UMP2 batch");
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index])
        warm_reference::validate_checkpoint(systems_[index], *states[index], "UMP2",
                                            GENERATIVEQC_METHOD_UHF);
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index]) warm_states_[index].swap(states[index]);
  }
  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    return {};
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  Capabilities capabilities_;
  core::ContextState* context_{};
  std::vector<core::System> systems_;
  bool warm_enabled_{}, warm_updates_{true};
  std::vector<std::optional<scf::HfWarmState>> warm_states_;
  generativeqc_method_descriptor descriptor_{};
  std::vector<std::unique_ptr<PreparedCalculation>> owners_;
  std::vector<std::vector<double>> owner_coordinates_;
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
    const generativeqc_method_descriptor& descriptor) {
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
                                                  generativeqc_batch_flags flags) {
  if (flags & ~GENERATIVEQC_BATCH_ENABLE_WARM_STARTS)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "UMP2 batch supports warm starts only; profiling is unavailable");
  return std::make_unique<Ump2PreparedBatch>(capabilities, context, std::move(systems), descriptor,
                                             (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0);
}

}  // namespace generativeqc::methods::detail
