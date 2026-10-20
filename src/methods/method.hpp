#ifndef GENERATIVEQC_METHODS_METHOD_HPP
#define GENERATIVEQC_METHODS_METHOD_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"
#include "runtime/compiled_execution_region.hpp"
#include "runtime/execution_context.hpp"
#include "scf/precision_work.hpp"
#include "scf/target_fock_work.hpp"
#include "scf/types.hpp"

namespace generativeqc::methods {

/** Registry metadata used by the public capability query and method factory. */
struct Capabilities {
  generativeqc_method method{};
  generativeqc_method_family family{};
  generativeqc_property_flags supported_properties{};
  bool available{};
  bool supports_batch{};
};

/** Method-neutral convergence diagnostics published through the legacy ABI. */
struct Convergence {
  unsigned iterations{};
  double energy_change{};
  double residual_rms{};  // Historical internal name for the legacy density-update RMS.
  bool converged{};
};

/** Complete before optional instantiation, including on Clang/libstdc++ where
 * a nested aggregate's default-construction trait can be queried too early. */
struct ResultCompanion {
  double energy{};
  std::vector<double> forces;
  std::optional<generativeqc_correlation_diagnostic> correlation;
};

/** Common calculation result. Method-specific retained state stays in the plan. */
struct Result {
  double energy{};
  std::vector<double> forces;
  Convergence convergence;
  generativeqc_backend executed_backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
  /** Owner-observed physical Fock evaluations; zero means unavailable. CUDA
   * batch publication additionally requires a complete KS operator census. */
  std::size_t fock_builds{};
  /** How the requested precision policy resolved in the executed backend. */
  scf::PrecisionProvenance precision{};
  /** #990 Direct-J/K execution/work record; zeroed for unrelated methods. */
  scf::IncrementalDirectJkDiagnostic incremental_direct_jk{};
  /** Ordered work remains explicitly incomplete until an execution owner
   * instruments every event/operator for the returned attempt. */
  scf::PrecisionWork precision_work{};
  /** Optional physical commutator at the returned density; zero is a valid
   * measured value, while absence means the method does not report it. */
  std::optional<double> physical_residual_rms;
  /** Completed KS solve, moved through the adapter and ABI handle. Keeping a
   * single exported history owner preserves the method resource bound. */
  std::optional<dft::ScfDiagnostic> ks_diagnostic;
  scf::initial_guess::PreliminaryDiagnostic preliminary_guess;
  /** Optional synchronous CBS companions, evaluated from this exact reference.
   * They are absent unless requested and never survive a failed transaction. */
  using Companion = ResultCompanion;
  std::optional<Companion> hf_component, mp2_component;
};

inline constexpr unsigned kHfComponent = 1;
inline constexpr unsigned kMp2Component = 2;

/** Method-neutral copy of the cumulative native CUDA KS movement ledger. */
struct CcPerformanceDiagnostic {
  double reference_seconds{};
  double problem_seconds{};
  double provider_seconds{};
  double source_seconds{};
  double solver_seconds{};
  double iteration_seconds{};
  double replay_seconds{};
  double update_seconds{};
  double diis_seconds{};
  double triples_seconds{};
  std::uint64_t source_scans{};
  std::uint64_t source_reads{};
  std::uint64_t source_values{};
  std::uint64_t transform_fmas{};
  std::uint64_t transform_stages{};
  std::uint64_t mo_blocks{};
  std::uint64_t cuda_transform_calls{};
  std::uint64_t cuda_batch_calls{};
  std::uint64_t iteration_graph_calls{};
  std::uint64_t replay_graph_calls{};
  std::uint64_t update_calls{};
  std::uint64_t generated_error_checks{};
  std::uint64_t diis_gram_calls{};
  std::uint64_t diis_coefficient_calls{};
  std::uint64_t diis_combine_calls{};
  // Private versioned precision telemetry; do not extend the frozen public
  // performance descriptor or infer execution from a requested Fast policy.
  std::uint64_t ccsd_fp32_requested{}, ccsd_fp32_executed{}, ccsd_fp32_refused{};
  std::uint64_t ccsd_fp32_evaluations{}, ccsd_fp32_gemm_calls{}, ccsd_fp32_cast_elements{};
};

struct KsTransportDiagnostic {
  std::uint64_t setup_h2d_bytes{};
  std::uint64_t density_h2d_bytes{};
  std::uint64_t scalar_d2h_bytes{};
  std::uint64_t matrix_d2h_bytes{};
  /** Internal #163 handoff accounting. The legacy public ABI continues to
   * report these bytes
   * inside matrix_d2h_bytes without adding fields. */
  std::uint64_t final_state_d2h_bytes{};
  std::uint64_t final_state_reads{};
  std::uint64_t synchronizations{};
  std::uint64_t iterations{};
  std::uint64_t occupation_stabilized_proposals{};
};

struct BatchItemResult {
  generativeqc_status status{GENERATIVEQC_STATUS_INTERNAL_ERROR};
  Result calculation;
  std::size_t bucket_id{};
  bool warm_start_used{};
  bool warm_start_fallback{};
  /** Present only for an owner that inventories all physical target attempts.
   * An unavailable total must not fall back to the returned attempt's count. */
  std::optional<scf::TargetFockWork> target_fock_work;
};

using Coordinates = std::vector<std::optional<std::vector<double>>>;

/** Prepared single-system method execution, independent of the public C ABI.
 * Instances own mutable execution/cache state and are not concurrently
 * reentrant. The caller must serialize execution and destruction per instance.
 * A method may share a context-owned workspace only with an explicit execution
 * lock covering its full execute/snapshot transaction. Immutable scientific
 * controls alone do not make mutable execution state safe to share.
 */
class PreparedCalculation {
 public:
  explicit PreparedCalculation(generativeqc_backend backend = GENERATIVEQC_BACKEND_CPU_REFERENCE,
                               int device = 0);
  PreparedCalculation(const PreparedCalculation&) = delete;
  PreparedCalculation& operator=(const PreparedCalculation&) = delete;
  virtual ~PreparedCalculation() = default;
  [[nodiscard]] virtual std::size_t atom_count() const noexcept = 0;
  [[nodiscard]] virtual const Capabilities& capabilities() const noexcept = 0;
  /** Context-qualified property bits of this immutable prepared plan. The
   * registry reports method-wide capabilities, never every backend/basis. */
  [[nodiscard]] virtual generativeqc_property_flags supported_properties() const noexcept {
    return capabilities().supported_properties;
  }
  /** Method-neutral execution-resource high waters. Zero means the owner has
   * not supplied a measurement for that category; it must not be guessed. */
  [[nodiscard]] virtual runtime::ExecutionResourceSnapshot execution_resources() const noexcept {
    return {};
  }
  /** Execute only the requested output work. Energy and convergence
   * diagnostics are always produced; forces are opt-in per execution. */
  Result execute(bool compute_forces, unsigned components = 0);
  /** The original owner supplies scientific identity/leases and budget admission.
   * This gate reports the endpoint transaction, not buffer/cache hit counts. */
  [[nodiscard]] const runtime::CompiledExecutionRegion& execution_lifecycle() const noexcept {
    return lifecycle_;
  }
  /** Revoke exports before recovery. Native owners retain only their existing
   * last-good warm state; reset never manufactures a scientific generation. */
  void reset_execution();
  void close_execution();
  /** Revoke publication if the ABI copy-out/validation transaction fails. */
  void reject_execution() noexcept;
  virtual void invalidate_result() {}
  [[nodiscard]] virtual std::optional<KsTransportDiagnostic> ks_transport_diagnostic() const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic()
      const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const {
    return std::nullopt;
  }

 protected:
  virtual Result execute_components_impl(bool compute_forces, unsigned components) {
    if (components) throw std::invalid_argument("method does not support correlated companions");
    return execute_impl(compute_forces);
  }
  virtual Result execute_impl(bool compute_forces) = 0;
  virtual void reset_execution_impl() {}

 private:
  runtime::CompiledExecutionRegion lifecycle_;
};

/** Prepared ragged execution. Method families choose their own batching policy.
 * As for PreparedCalculation, serialize all calls and destruction per instance;
 * diagnostics and warm-state access must not race execution or replacement.
 */
class PreparedBatch {
 public:
  explicit PreparedBatch(generativeqc_backend backend = GENERATIVEQC_BACKEND_CPU_REFERENCE,
                         int device = 0);
  PreparedBatch(const PreparedBatch&) = delete;
  PreparedBatch& operator=(const PreparedBatch&) = delete;
  virtual ~PreparedBatch() = default;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  /** Internal per-item immutable prepared capabilities, not the method-global
   * registry. Unsupported batch owners return no value, not a fabricated bit. */
  [[nodiscard]] virtual std::optional<generativeqc_property_flags> supported_properties(
      std::size_t index) const noexcept {
    (void)index;
    return std::nullopt;
  }
  /** Output selection is per replay; retained scientific controls stay immutable.
   * A false force request skips response evaluation for the complete fleet. */
  std::vector<BatchItemResult> execute(const Coordinates& coordinates, bool compute_forces = true);
  /** Select force work per original input slot under the same lifecycle gate.
   * Each byte is 0 or 1. Owners without this optional execution policy reject
   * it explicitly; ordinary whole-fleet execution retains its existing contract. */
  std::vector<BatchItemResult> execute_selected_forces(const Coordinates& coordinates,
                                                       std::span<const std::uint8_t> forces);
  [[nodiscard]] const runtime::CompiledExecutionRegion& execution_lifecycle() const noexcept {
    return lifecycle_;
  }
  void reset_execution();
  void close_execution();
  void reject_execution() noexcept;
  /** Revoke exported-state eligibility even when API validation rejects a replay. */
  virtual void invalidate_result() {}
  [[nodiscard]] virtual runtime::ExecutionResourceSnapshot execution_resources(
      std::size_t index) const noexcept {
    (void)index;
    return {};
  }

 protected:
  virtual std::vector<BatchItemResult> execute_impl(const Coordinates& coordinates,
                                                    bool compute_forces) = 0;
  virtual std::vector<BatchItemResult> execute_selected_forces_impl(
      const Coordinates& coordinates, std::span<const std::uint8_t> forces);

 private:
  std::vector<BatchItemResult> execute_request(
      const Coordinates& coordinates, bool compute_forces,
      std::optional<std::span<const std::uint8_t>> selected_forces);
  runtime::CompiledExecutionRegion lifecycle_;
};

/** Exception carrying an exact public status across the C++ method boundary. */
class MethodError final : public std::runtime_error {
 public:
  MethodError(generativeqc_status status, const std::string& message)
      : std::runtime_error(message), status_(status) {}

  [[nodiscard]] generativeqc_status status() const noexcept { return status_; }

 private:
  generativeqc_status status_;
};

[[nodiscard]] const Capabilities* find_capabilities(generativeqc_method method) noexcept;

std::unique_ptr<PreparedCalculation> prepare_calculation(
    core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor,
    const core::System* auxiliary_basis = nullptr);

std::unique_ptr<PreparedBatch> prepare_batch(core::ContextState& context,
                                             std::vector<core::System> systems,
                                             const generativeqc_method_descriptor& descriptor,
                                             generativeqc_batch_flags flags,
                                             const core::System* auxiliary_basis = nullptr);

}  // namespace generativeqc::methods

#endif
