#include "methods/rccsdt_method.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cc/rccsdt_force.hpp"
#include "cc/triples_cuda.hpp"
#include "generated_rccsdt_cpu.hpp"
#include "methods/batch_capabilities.hpp"
#include "methods/correlated_cuda_reference.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "methods/rccsd_method.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_energy.hpp"
#include "posthf/mp2_force.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace generativeqc::methods::detail {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("RCCSD(T) size overflow");
  return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD(T) size overflow");
  return a * b;
}

generativeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

class RccsdtPrepared final : public PreparedCalculation {
 public:
  RccsdtPrepared(Capabilities capabilities, runtime::ExecutionContext execution,
                 core::System system, const generativeqc_method_descriptor& descriptor)
      : PreparedCalculation(execution.backend(), execution.device_id()),
        capabilities_(capabilities),
        execution_(std::move(execution)),
        system_(std::move(system)) {
    descriptor_ = descriptor;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    // Publish a coherent snapshot while another caller may execute the owner.
    std::lock_guard<std::mutex> lock(mutex_);
    return execution_.resources();
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_;
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_performance_;
  }

  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_.reset();
    last_performance_.reset();
  }

  Result execute_impl(bool compute_forces) override {
    return execute_with_reference_seed(compute_forces, nullptr, nullptr, nullptr);
  }
  Result execute_components_impl(bool compute_forces, unsigned components) override {
    if (components & ~(kHfComponent | kMp2Component))
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "unknown RCCSD(T) companion");
    return execute_with_reference_seed(compute_forces, nullptr, nullptr, nullptr, components);
  }

  void transfer_cuda_reference_plan_to(RccsdtPrepared& target) noexcept {
    target.cuda_reference_plan_ = std::move(cuda_reference_plan_);
  }

  Result execute_with_reference_seed(bool compute_forces, const scf::HfWarmState* initial_state,
                                     bool* warm_start_fallback,
                                     std::optional<scf::HfWarmState>* retained_warm_state,
                                     unsigned components = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime::df_progress::Scope component_trace("ccsdt_endpoint");
    last_.reset();
    last_performance_.reset();
    const auto warm_capacity =
        warm_reference::reservation_bytes(system_, initial_state, retained_warm_state != nullptr);
    auto state = run_rccsd_native_state(
        execution_, system_, descriptor_, &cpu_exact_plan_,
        initial_state ? &initial_state->density : nullptr, warm_start_fallback, warm_capacity,
        nullptr, false, true, execution_.cuda_requested() ? cuda_reference_plan_.slot() : nullptr);
    last_ = state.diagnostic;
    last_performance_ = state.performance;
    if (state.solved.status == cc::SolveStatus::NumericalFailure)
      throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, state.solved.reason);
    if (!state.solved.converged()) return std::move(state.result);

    try {
      const auto retained_capacity = [&] {
        auto retained = checked_add(state.problem.reference_retained_bytes,
                                    cc::problem_host_bytes(state.problem));
        retained = checked_add(
            retained, checked_mul(state.eps_o.capacity() + state.eps_v.capacity(), sizeof(double)));
        retained = checked_add(
            retained,
            checked_mul(state.solved.t1.capacity() + state.solved.t2.capacity(), sizeof(double)));
        return retained;
      };
      auto retained = retained_capacity();
      const double denominator_threshold =
          descriptor_.ccsd_denominator_threshold ? descriptor_.ccsd_denominator_threshold : 1e-10;
      double triples_energy = 0.0;
      double triples_minimum_denominator = std::numeric_limits<double>::infinity();
      std::size_t triples_virtual_count = 0;
      std::size_t triples_workspace_bytes = 0;

      const auto triples_started = std::chrono::steady_clock::now();
      const auto phase_budget = checked_add(state.budget, state.reference_execution_plan_bytes);
      const auto update_reference_reservation = [&](std::size_t budget) {
        state.budget = budget;
        state.reference_execution_plan_bytes = cuda_reference_plan_.retained_numeric_bytes();
        state.reference_execution_plan_device_bytes = cuda_reference_plan_.owned_device_bytes();
        state.diagnostic.reference_execution_plan_owned_device_bytes =
            state.reference_execution_plan_device_bytes;
      };
      run_with_rccsd_reference_source(state, [&] {
        retained = retained_capacity();
        run_with_cuda_reference_budget(
            cuda_reference_plan_.slot(), phase_budget, [&](std::size_t budget) {
              update_reference_reservation(budget);
              if (retained >= budget)
                throw std::length_error(
                    "RCCSD(T) retained CC state exhausts the correlation memory budget");
              if (execution_.cuda_requested()) {
#if GENERATIVEQC_HAS_CUDA
                const auto triples = cc::triples::evaluate_cuda(
                    state.problem.nocc, state.problem.nvir, state.problem.ovvv.data(),
                    state.problem.ovoo.data(), state.problem.ovov.data(), state.problem.fov.data(),
                    state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(),
                    state.eps_v.data(), denominator_threshold, state.budget - retained,
                    execution_.device_id());
                triples_energy = triples.energy;
                triples_minimum_denominator = triples.minimum_absolute_denominator;
                triples_virtual_count = triples.virtual_triples;
                triples_workspace_bytes = triples.workspace_bytes;
#else
          throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                            "native RCCSD(T) CUDA owner is not compiled in this library");
#endif
              } else {
                const auto triples = cc::triples::generated::evaluate(
                    state.problem.nocc, state.problem.nvir, state.problem.ovvv.data(),
                    state.problem.ovoo.data(), state.problem.ovov.data(), state.problem.fov.data(),
                    state.solved.t1.data(), state.solved.t2.data(), state.eps_o.data(),
                    state.eps_v.data(), denominator_threshold, state.budget - retained);
                triples_energy = triples.energy;
                triples_minimum_denominator = triples.minimum_absolute_denominator;
                triples_virtual_count = triples.virtual_triples;
                triples_workspace_bytes = triples.workspace_bytes;
              }
            });
      });

      const double triples_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - triples_started).count();
      auto diagnostic = state.diagnostic;
      auto performance = state.performance;
      performance.triples_seconds = triples_seconds;
      diagnostic.minimum_absolute_denominator =
          std::min(diagnostic.minimum_absolute_denominator, triples_minimum_denominator);
      diagnostic.numeric_capacity_bytes = std::max<std::uint64_t>(
          diagnostic.numeric_capacity_bytes,
          checked_add(
              checked_add(retained, triples_workspace_bytes),
              checked_add(state.external_reservation_bytes, state.reference_execution_plan_bytes)));
      diagnostic.ccsd_t_triples_energy = triples_energy;
      diagnostic.ccsd_t_virtual_triples = triples_virtual_count;
      diagnostic.ccsd_t_workspace_bytes = triples_workspace_bytes;
      if (execution_.cuda_requested()) {
        diagnostic.correlation_owned_device_bytes = std::max<std::uint64_t>(
            diagnostic.correlation_owned_device_bytes, triples_workspace_bytes);
        execution_.observe_workspace_peak(runtime::ExecutionMemorySpace::Device,
                                          triples_workspace_bytes);
        execution_.observe_numeric_peak(
            runtime::ExecutionMemorySpace::Device,
            checked_add(state.reference_execution_plan_device_bytes, triples_workspace_bytes));
      } else {
        execution_.observe_workspace_peak(runtime::ExecutionMemorySpace::Host,
                                          triples_workspace_bytes);
      }
      std::copy_n(cc::triples::generated::inventory_hash,
                  std::min<std::size_t>(64, std::strlen(cc::triples::generated::inventory_hash)),
                  diagnostic.ccsd_t_equation_hash);
      state.result.energy = state.solved.total_energy + triples_energy;
      if (compute_forces) {
        if (!state.reference)
          throw std::runtime_error("RCCSD(T) force owner lost the converged RHF reference");
        const auto force_denominator_threshold =
            descriptor_.ccsd_denominator_threshold ? descriptor_.ccsd_denominator_threshold : 1e-10;
        // Reuse the same prepared exact interaction owner for the force Hamiltonian.
        // CUDA sources publish raw ERI tiles directly into the existing bounded MO transform.
        std::unique_ptr<posthf::RawSource> force_raw_source;
        std::optional<scf::PreparedFockInteractionSourceView> force_prepared_source;
        const integrals::ElectronInteractionSource* force_source = nullptr;
        if (state.reference_interaction_source) {
          force_source = state.reference_interaction_source.get();
        } else if (cpu_exact_plan_) {
          force_prepared_source.emplace(*cpu_exact_plan_);
          force_source = &*force_prepared_source;
        }

        constexpr std::size_t kCudaDerivativeStageBudget = 64ULL << 20;
        auto force = run_with_rccsd_reference_source(state, [&] {
          return run_with_cuda_reference_budget(
              cuda_reference_plan_.slot(), phase_budget, [&](std::size_t budget) {
                if (!state.reference_interaction_source && !cpu_exact_plan_)
                  force_source = force_raw_source.get();
                if (!force_source) {
                  force_raw_source = std::make_unique<posthf::RawSource>(system_);
                  force_source = force_raw_source.get();
                }
                update_reference_reservation(budget);
                return execution_.cuda_requested()
                           ? cc::rccsdt_force_cuda(
                                 system_, *force_source, *state.reference, state.problem,
                                 state.solved, state.eps_o, state.eps_v, state.budget,
                                 execution_.device_id(),
                                 std::min(state.budget, kCudaDerivativeStageBudget),
                                 force_denominator_threshold)
                           : cc::rccsdt_force_cpu(system_, *force_source, *state.reference,
                                                  state.problem, state.solved, state.eps_o,
                                                  state.eps_v, state.budget,
                                                  force_denominator_threshold);
              });
        });
        diagnostic.reference_execution_plan_owned_device_bytes =
            state.reference_execution_plan_device_bytes;
        if (execution_.cuda_requested()) {
          if (!force.lambda.cuda_actions || !force.lambda.owned_device_bytes)
            throw std::runtime_error(
                "RCCSD(T) CUDA force did not execute generated Lambda actions on device");
          if (!force.cuda_response_actions || !force.response_owned_device_bytes)
            throw std::runtime_error(
                "RCCSD(T) CUDA force replayed Hamiltonian/orbital response on host");
          execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Device,
                                          checked_add(state.reference_execution_plan_device_bytes,
                                                      force.lambda.owned_device_bytes));
          execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Device,
                                          checked_add(state.reference_execution_plan_device_bytes,
                                                      force.response_owned_device_bytes));
          diagnostic.correlation_owned_device_bytes =
              std::max<std::uint64_t>(diagnostic.correlation_owned_device_bytes,
                                      std::max<std::uint64_t>(force.lambda.owned_device_bytes,
                                                              force.response_owned_device_bytes));
        }
        state.result.forces = std::move(force.forces);
        diagnostic.response_iterations = force.orbital_response.iterations;
        diagnostic.response_restarts = force.orbital_response.restarts;
        diagnostic.response_absolute_residual =
            std::max(force.orbital_response.residual_norm, force.independent_orbital_residual);
        diagnostic.response_relative_residual = force.orbital_response.relative_residual;
        diagnostic.response_workspace_bytes = force.orbital_response.workspace_bytes;
        diagnostic.measured_response_workspace_peak_bytes =
            force.orbital_response.measured_workspace_peak_bytes;
        diagnostic.response_workspace_allocation_count =
            force.orbital_response.workspace_allocation_count;
        const auto force_capacity = checked_add(
            force.numeric_capacity_bytes,
            checked_add(state.external_reservation_bytes, state.reference_execution_plan_bytes));
        diagnostic.planned_endpoint_peak_bytes =
            std::max<std::uint64_t>(diagnostic.numeric_capacity_bytes, force_capacity);
        // Bits 0-2 retain the qualified analytic-response provenance. Bit 3
        // records that the final conventional nuclear derivative consumer ran on CUDA.
        diagnostic.force_provenance_flags = execution_.cuda_requested() ? 0xf : 0x7;
        diagnostic.numeric_capacity_bytes =
            std::max<std::uint64_t>(diagnostic.numeric_capacity_bytes, force_capacity);
        execution_.observe_numeric_peak(runtime::ExecutionMemorySpace::Host, force_capacity);
        execution_.observe_workspace_peak(runtime::ExecutionMemorySpace::Host,
                                          force.orbital_response.workspace_bytes);
        std::copy_n(force.response_operator_hash.c_str(),
                    std::min<std::size_t>(64, force.response_operator_hash.size()),
                    diagnostic.response_operator_hash);
      }
      if (components) {
        // CC response is complete. Retire amplitudes/Fock/denominator storage,
        // but admit a symmetry-complete cache for the companion's MO consumers.
        std::array<std::vector<double>, 6> cached_blocks{
            std::move(state.problem.ovov), std::move(state.problem.oovv),
            std::move(state.problem.ovoo), std::move(state.problem.ovvv),
            std::move(state.problem.oooo), std::move(state.problem.vvvv)};
        bool cache_available = (components & kMp2Component) != 0;
        if (!cache_available)
          for (auto& block : cached_blocks) std::vector<double>().swap(block);
        const auto& ovov = cached_blocks[0];
        const std::array<posthf::BorrowedMOBlock, 6> borrowed_blocks{{
            {{false, true, false, true}, cached_blocks[0]},
            {{false, false, true, true}, cached_blocks[1]},
            {{false, true, false, false}, cached_blocks[2]},
            {{false, true, true, true}, cached_blocks[3]},
            {{false, false, false, false}, cached_blocks[4]},
            {{true, true, true, true}, cached_blocks[5]},
        }};
        const auto cached_bytes = [&] {
          std::size_t bytes = 0;
          for (const auto& block : cached_blocks)
            bytes = checked_add(bytes, checked_mul(block.capacity(), sizeof(double)));
          return bytes;
        };
        const auto reference_bytes = state.problem.reference_retained_bytes;
        state.problem = {};
        state.problem.reference_retained_bytes = reference_bytes;
        state.solved = {};
        std::vector<double>().swap(state.eps_o);
        std::vector<double>().swap(state.eps_v);
        std::unique_ptr<posthf::RawSource> raw_source;
        std::optional<scf::PreparedFockInteractionSourceView> prepared_source;
        const integrals::ElectronInteractionSource* source = nullptr;
        std::size_t companion_outputs = checked_mul(state.result.forces.capacity(), sizeof(double));
        const auto run_companion_once = [&](auto&& operation) {
          return run_with_rccsd_reference_source(state, [&] {
            return run_with_cuda_reference_budget(
                cuda_reference_plan_.slot(), phase_budget, [&](std::size_t budget) {
                  update_reference_reservation(budget);
                  const auto retained_blocks = cached_bytes();
                  const auto external = checked_add(companion_outputs, retained_blocks);
                  if (external >= budget)
                    throw std::length_error("RCCSD(T) companion outputs exhaust memory budget");
                  if (state.reference_interaction_source) {
                    source = state.reference_interaction_source.get();
                  } else if (cpu_exact_plan_) {
                    prepared_source.emplace(*cpu_exact_plan_);
                    source = &*prepared_source;
                  } else {
                    if (!raw_source) raw_source = std::make_unique<posthf::RawSource>(system_);
                    source = raw_source.get();
                  }
                  auto value = operation(budget - external);
                  return value;
                });
          });
        };
        const auto run_companion = [&](auto&& operation) {
          try {
            return run_companion_once(operation);
          } catch (const std::length_error&) {
            if (!cache_available) throw;
          } catch (const std::bad_alloc&) {
            if (!cache_available) throw;
          }
          // Optional transformed residency never displaces a bounded replay.
          // Numerical failures cannot enter this resource-only fallback.
          for (auto& block : cached_blocks) std::vector<double>().swap(block);
          cache_available = false;
          return run_companion_once(operation);
        };
        if (components & kHfComponent) {
          state.result.hf_component.emplace();
          state.result.hf_component->energy = state.reference->energy;
          if (compute_forces) {
            auto force = run_companion([&](std::size_t budget) {
              return mp2::reference_force(*state.reference, *source, budget,
                                          execution_.cuda_requested(), execution_.device_id());
            });
            diagnostic.numeric_capacity_bytes = std::max<std::uint64_t>(
                diagnostic.numeric_capacity_bytes,
                checked_add(
                    force.planned_endpoint_peak_bytes,
                    checked_add(state.external_reservation_bytes,
                                checked_add(state.reference_execution_plan_bytes,
                                            checked_add(companion_outputs, cached_bytes())))));
            state.result.hf_component->forces = std::move(force.forces);
            companion_outputs = checked_add(
                companion_outputs,
                checked_mul(state.result.hf_component->forces.capacity(), sizeof(double)));
          }
        }
        if (components & kMp2Component) {
          const auto mp2_threshold =
              descriptor_.mp2_denominator_threshold ? descriptor_.mp2_denominator_threshold : 1e-10;
          const auto energy = run_companion([&](std::size_t budget) {
            return mp2::conventional_energy(*state.reference, *source, budget, mp2_threshold, 8,
                                            execution_.cuda_requested(), execution_.device_id(),
                                            ovov);
          });
          state.result.mp2_component.emplace();
          runtime::df_progress::Scope::number("mp2_companion_energy_source_reads",
                                              energy.provider_work.source_reads);
          runtime::df_progress::Scope::number("mp2_companion_energy_transform_fmas",
                                              energy.provider_work.transform_fmas);
          runtime::df_progress::Scope::number("mp2_companion_ovov_reused", ovov.empty() ? 0 : 1);
          state.result.mp2_component->energy =
              state.reference->energy + energy.opposite_spin + energy.same_spin;
          auto& mp2_diagnostic = state.result.mp2_component->correlation.emplace();
          mp2_diagnostic.struct_size = sizeof(mp2_diagnostic);
          mp2_diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
          mp2_diagnostic.reference_energy = state.reference->energy;
          mp2_diagnostic.opposite_spin_energy = energy.opposite_spin;
          mp2_diagnostic.same_spin_energy = energy.same_spin;
          mp2_diagnostic.minimum_absolute_denominator = energy.minimum_denominator;
          mp2_diagnostic.reference_residual = state.reference->commutator_residual;
          mp2_diagnostic.energy_tile_count = energy.tiles;
          mp2_diagnostic.mo_host_staging = execution_.cuda_requested() ? 1 : 0;
          mp2_diagnostic.correlation_owned_device_bytes = energy.metrics.owned_device_bytes;
          mp2_diagnostic.correlation_provider_retained_bytes =
              energy.metrics.provider_retained_bytes;
          mp2_diagnostic.mo_transfer_bytes = energy.mo_transfer_bytes;
          const auto companion_reservation = [&] {
            return checked_add(state.external_reservation_bytes,
                               checked_add(state.reference_execution_plan_bytes,
                                           checked_add(companion_outputs, cached_bytes())));
          };
          mp2_diagnostic.numeric_capacity_bytes =
              checked_add(energy.numeric_capacity_bytes, companion_reservation());
          std::copy_n(energy.equation_hash, 64, mp2_diagnostic.equation_hash);
          diagnostic.numeric_capacity_bytes =
              std::max(diagnostic.numeric_capacity_bytes, mp2_diagnostic.numeric_capacity_bytes);
          if (compute_forces) {
            response::GmresOptions options;
            options.relative_tolerance = 1e-10;
            options.absolute_tolerance = 1e-12;
            options.restart = 30;
            options.max_iterations = 200;
            auto force = run_companion([&](std::size_t budget) {
              options.max_workspace_bytes = budget;
              const auto blocks = cache_available
                                      ? std::span<const posthf::BorrowedMOBlock>(borrowed_blocks)
                                      : std::span<const posthf::BorrowedMOBlock>{};
              return execution_.cuda_requested()
                         ? mp2::conventional_force_cuda(*state.reference, *source, budget,
                                                        mp2_threshold, 1e-10, options,
                                                        execution_.device_id(), ovov, blocks)
                         : mp2::conventional_force_cpu(*state.reference, *source, budget,
                                                       mp2_threshold, 1e-10, options, ovov, blocks);
            });
            diagnostic.numeric_capacity_bytes = std::max<std::uint64_t>(
                diagnostic.numeric_capacity_bytes,
                checked_add(force.planned_endpoint_peak_bytes, companion_reservation()));
            runtime::df_progress::Scope::number("mp2_companion_force_blocks_reused",
                                                cache_available ? 1 : 0);
            mp2_diagnostic.response_iterations = force.response.iterations;
            mp2_diagnostic.response_restarts = force.response.restarts;
            mp2_diagnostic.response_absolute_residual = force.response.residual_norm;
            mp2_diagnostic.response_relative_residual = force.response.relative_residual;
            mp2_diagnostic.response_workspace_bytes = force.response.workspace_bytes;
            mp2_diagnostic.measured_response_workspace_peak_bytes =
                force.response.measured_workspace_peak_bytes;
            mp2_diagnostic.response_workspace_allocation_count =
                force.response.workspace_allocation_count;
            mp2_diagnostic.derivative_workspace_bytes = force.derivative_workspace_bytes;
            mp2_diagnostic.planned_endpoint_peak_bytes =
                checked_add(force.planned_endpoint_peak_bytes, companion_reservation());
            mp2_diagnostic.numeric_capacity_bytes = std::max(
                mp2_diagnostic.numeric_capacity_bytes, mp2_diagnostic.planned_endpoint_peak_bytes);
            mp2_diagnostic.force_provenance_flags = 0x7;
            constexpr char response_hash[] = "rhf-canonical-response-v1";
            std::copy_n(response_hash, sizeof(response_hash),
                        mp2_diagnostic.response_operator_hash);
            state.result.mp2_component->forces = std::move(force.forces);
          }
        }
        diagnostic.planned_endpoint_peak_bytes =
            std::max(diagnostic.planned_endpoint_peak_bytes, diagnostic.numeric_capacity_bytes);
        diagnostic.measured_endpoint_peak_bytes = 0;
      }
      if (retained_warm_state && state.reference)
        *retained_warm_state =
            warm_reference::capture(system_, *state.reference, state.reference_energy_change,
                                    state.reference_density_rms, state.reference_iterations);
      static_assert(std::is_nothrow_move_constructible_v<Result>);
      if (scf::reclaim_rhf_cuda_reference_plan(cuda_reference_plan_.slot(),
                                               state.reference_interaction_source))
        diagnostic.reference_execution_plan_owned_device_bytes =
            cuda_reference_plan_.owned_device_bytes();
      last_ = diagnostic;
      last_performance_ = performance;
      return std::move(state.result);
    } catch (const std::length_error& error) {
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY, error.what());
    } catch (const std::bad_alloc&) {
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                        "RCCSD(T) triples workspace allocation failed");
    }
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::System system_;
  generativeqc_method_descriptor descriptor_{};
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  CorrelatedCudaReferencePlan cuda_reference_plan_;
  std::optional<generativeqc_correlation_diagnostic> last_;
  std::optional<CcPerformanceDiagnostic> last_performance_;
  mutable std::mutex mutex_;
};

class RccsdtPreparedBatch final : public PreparedBatch,
                                  public BatchWarmStarts,
                                  public BatchCorrelationDiagnostics,
                                  public BatchCcPerformanceDiagnostics {
 public:
  RccsdtPreparedBatch(Capabilities capabilities, core::ContextState& context,
                      std::vector<core::System> systems,
                      const generativeqc_method_descriptor& descriptor, bool warm_starts_enabled)
      : PreparedBatch(context.requested_backend, context.device_id),
        capabilities_(capabilities),
        execution_(context),
        context_(&context),
        systems_(std::move(systems)),
        warm_starts_enabled_(warm_starts_enabled),
        warm_states_(systems_.size()) {
    descriptor_ = descriptor;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(systems_.size());
    owner_coordinates_.reserve(systems_.size());
    for (const auto& system : systems_) {
      owners_.push_back(prepare_rccsdt_calculation(capabilities_, *context_, system, descriptor_));
      owner_coordinates_.push_back(warm_reference::coordinates(system));
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }

  std::vector<BatchItemResult> execute_impl(const Coordinates& coordinates,
                                            bool compute_forces) override {
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("RCCSD(T) batch coordinates do not match system count");

    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& result = results[index];
      result.bucket_id = 0;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      result.calculation.executed_backend = execution_.backend();
      try {
        auto target = systems_[index];
        auto target_coordinates = warm_reference::coordinates(target);
        if (!coordinates.empty() && coordinates[index]) {
          if (!warm_reference::valid_coordinates(*coordinates[index], target))
            throw std::invalid_argument("invalid RCCSD(T) batch item coordinates");
          target_coordinates = *coordinates[index];
          warm_reference::set_coordinates(target, target_coordinates);
        }
        if (target_coordinates != owner_coordinates_[index]) {
          auto candidate =
              prepare_rccsdt_calculation(capabilities_, *context_, target, descriptor_);
          auto& current_owner = static_cast<RccsdtPrepared&>(*owners_[index]);
          auto& candidate_owner = static_cast<RccsdtPrepared&>(*candidate);
          current_owner.transfer_cuda_reference_plan_to(candidate_owner);
          owners_[index] = std::move(candidate);
          owner_coordinates_[index] = std::move(target_coordinates);
        }
        const bool has_warm_state = warm_starts_enabled_ && warm_states_[index].has_value();
        result.warm_start_used = has_warm_state;
        bool warm_start_fallback = false;
        std::optional<scf::HfWarmState> next_warm_state;
        auto& owner = static_cast<RccsdtPrepared&>(*owners_[index]);
        result.calculation = owner.execute_with_reference_seed(
            compute_forces, has_warm_state ? &*warm_states_[index] : nullptr, &warm_start_fallback,
            warm_starts_enabled_ && warm_start_updates_enabled_ ? &next_warm_state : nullptr);
        result.warm_start_fallback = warm_start_fallback;
        if (next_warm_state) warm_states_[index].swap(next_warm_state);
        result.status = result.calculation.convergence.converged
                            ? GENERATIVEQC_STATUS_SUCCESS
                            : GENERATIVEQC_STATUS_NOT_CONVERGED;
      } catch (...) {
        result.status = item_exception_status();
      }
    }
    return results;
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const override {
    if (index >= owners_.size())
      throw std::invalid_argument("correlation diagnostic batch index is out of range");
    return owners_[index]->correlation_diagnostic();
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic(
      std::size_t index) const override {
    if (index >= owners_.size())
      throw std::invalid_argument("CC performance diagnostic batch index is out of range");
    return owners_[index]->cc_performance_diagnostic();
  }

  void clear_warm_starts() override {
    for (auto& state : warm_states_) state.reset();
  }
  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n)
      throw std::invalid_argument("RCCSD(T) warm density dimensions overflow");
    return posthf::checked_mul(n, n);
  }
  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    return warm_states_.at(index);
  }
  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_starts_enabled_ || states.size() != size())
      throw std::invalid_argument(
          "checkpoint restore requires a matching warm-enabled RCCSD(T) batch");
    for (std::size_t index = 0; index < size(); ++index) {
      if (!states[index]) continue;
      const auto& state = *states[index];
      warm_reference::validate_checkpoint(systems_[index], state, "RCCSD(T)");
    }
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index]) warm_states_[index].swap(states[index]);
  }
  void set_warm_start_updates(bool enabled) override { warm_start_updates_enabled_ = enabled; }
  runtime::ExecutionResourceSnapshot execution_resources(
      std::size_t index) const noexcept override {
    if (index >= owners_.size() || !owners_[index]) return {};
    return owners_[index]->execution_resources();
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::ContextState* context_{};
  std::vector<core::System> systems_;
  bool warm_starts_enabled_{};
  bool warm_start_updates_enabled_{true};
  std::vector<std::optional<scf::HfWarmState>> warm_states_;
  generativeqc_method_descriptor descriptor_{};
  std::vector<std::unique_ptr<PreparedCalculation>> owners_;
  std::vector<std::vector<double>> owner_coordinates_;
};

}  // namespace

generativeqc_status validate_rccsdt_system(generativeqc_method method, const core::System& system,
                                           std::string& detail) {
  return validate_rccsd_system(method, system, detail);
}

std::unique_ptr<PreparedCalculation> prepare_rccsdt_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor) {
  runtime::ExecutionContext execution(context);
  if (execution.backend() != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
      execution.backend() != GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD(T) requires an explicit CPU or CUDA backend");
#if !GENERATIVEQC_HAS_CUDA
  if (execution.cuda_requested())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native RCCSD(T) CUDA owner is not compiled in this library");
#endif
  return std::make_unique<RccsdtPrepared>(capabilities, std::move(execution), system, descriptor);
}

std::unique_ptr<PreparedBatch> prepare_rccsdt_batch(
    const Capabilities& capabilities, core::ContextState& context,
    std::vector<core::System> systems, const generativeqc_method_descriptor& descriptor,
    generativeqc_batch_flags flags) {
  const bool warm_starts_enabled = (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0;
  constexpr auto supported_flags =
      static_cast<generativeqc_batch_flags>(GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
  if ((flags & ~supported_flags) != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "RCCSD(T) prepared batches support only the warm-start compatibility flag");
  if (systems.empty())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "RCCSD(T) batch is empty");

  const auto nbf = molecule::ao_count(systems.front());
  const auto nocc = static_cast<std::size_t>(systems.front().electron_count / 2);
  for (const auto& system : systems) {
    if (molecule::ao_count(system) != nbf ||
        static_cast<std::size_t>(system.electron_count / 2) != nocc)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "RCCSD(T) prepared batch requires one homogeneous (nocc,nvir) shape; split "
                        "ragged groups");
  }
  return std::make_unique<RccsdtPreparedBatch>(capabilities, context, std::move(systems),
                                               descriptor, warm_starts_enabled);
}

}  // namespace generativeqc::methods::detail
