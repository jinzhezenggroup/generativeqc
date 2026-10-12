#ifndef GENERATIVEQC_METHODS_BATCH_CAPABILITIES_HPP
#define GENERATIVEQC_METHODS_BATCH_CAPABILITIES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "methods/method.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/warm_state.hpp"

namespace generativeqc::methods {

// Optional batch facets keep warm-state and backend profiling out of the
// method-neutral PreparedBatch contract. ABI consumers use checked cross-casts.
struct DirectShellClassProfileEntry {
  std::uint64_t shell_quartets{};
  std::uint64_t tiles{};
  std::uint64_t ao_quartets{};
  std::uint64_t primitive_quartets{};
};

/** Final-density statistics for the resident scalar PPPS force queue. */
struct DirectPppsQueueProfile {
  static constexpr std::size_t kBlockSizeCount = 4;
  static constexpr std::size_t kOrientationCount = 2;
  static constexpr std::size_t kPrimitivePairBucketCount = 65;

  std::uint64_t descriptor_slots{};
  std::uint64_t non_empty_descriptors{};
  std::uint64_t empty_descriptors{};
  std::uint64_t tasks{};
  std::uint64_t primitive_work{};
  std::uint32_t ket_count_min{};
  std::uint32_t ket_count_median{};
  std::uint32_t ket_count_p90{};
  std::uint32_t ket_count_p99{};
  std::uint32_t ket_count_max{};
  std::array<double, kBlockSizeCount> lane_efficiency{};
  double primitive_warp_efficiency{};
  std::array<double, kBlockSizeCount> task_tail_imbalance{};
  std::array<double, kBlockSizeCount> primitive_tail_imbalance{};
  std::array<std::uint64_t, kOrientationCount> orientation_tasks{};
  std::array<std::uint64_t, kOrientationCount> orientation_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> bra_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> bra_primitive_work{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> ket_primitive_tasks{};
  std::array<std::uint64_t, kPrimitivePairBucketCount> ket_primitive_work{};
};

/** Method-neutral record of one CUDA bucket's setup-time eigensolver choice. */
struct EigensolverDiagnostic {
  std::uint32_t bucket_id{};
  std::uint32_t ordinary_family{};
  std::uint32_t graph_family{};
  std::uint32_t selection_source{};
  std::uint64_t matrix_dimension{};
  std::uint64_t physical_system_count{};
  std::uint64_t solver_batch_count{};
  bool api_eligible{};
  std::uint32_t api_reason{};
  std::uint64_t matrix_batch_product{};
  std::uint32_t probe_failure_stage{};
  std::uint64_t device_workspace_bytes{};
  std::uint64_t host_workspace_bytes{};
  std::uint64_t available_device_bytes{};
  std::int32_t device_id{-1};
  std::array<std::uint8_t, 16> device_uuid{};
  std::array<char, 256> device_name{};
  std::int32_t compute_capability_major{};
  std::int32_t compute_capability_minor{};
  std::int32_t cuda_runtime_version{};
  std::int32_t cuda_driver_version{};
  std::int32_t cusolver_version{};
  std::int32_t cuda_error{};
  std::int32_t cusolver_error{};
  bool ordinary_execution_passed{};
  bool graph_capture_passed{};
  bool host_graph_replay_passed{};
  bool device_tail_replay_passed{};
  bool graph_eligible{};
  double maximum_eigenvalue_error{};
  double maximum_residual{};
  double maximum_orthogonality_error{};
};

/** Method-neutral per-iteration inactive-eigensolver measurement. */
struct InactiveEigensolverProfileEntry {
  std::uint32_t bucket_id{};
  std::uint32_t iteration{};
  std::uint32_t family{};
  std::uint32_t physical_system_count{};
  std::uint32_t solver_batch_count{};
  std::uint32_t active_physical_count{};
  std::uint32_t active_solver_count{};
  std::uint64_t solver_elapsed_nanoseconds{};
  std::uint32_t inactive_input_nonfinite_count{};
  std::uint32_t inactive_submission_nonfinite_count{};
  std::uint32_t inactive_info_nonzero_count{};
  std::uint32_t inactive_touch_flags{};
  bool provider_invoked{};
};

/** Methods without an SCF seed need not implement checkpointing. */
class BatchWarmStarts {
 public:
  virtual ~BatchWarmStarts() = default;
  virtual void clear_warm_starts() = 0;
  [[nodiscard]] virtual std::size_t warm_density_size(std::size_t index) const = 0;
  [[nodiscard]] virtual const std::optional<scf::HfWarmState>& warm_state(
      std::size_t index) const = 0;
  virtual void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) = 0;
  virtual void set_warm_start_updates(bool enabled) = 0;
};

/** CUDA direct J/K queue and resident PPPS profiles. */
class BatchDirectQueueProfiles {
 public:
  virtual ~BatchDirectQueueProfiles() = default;
  [[nodiscard]] virtual std::optional<std::vector<DirectShellClassProfileEntry>>
  last_direct_shell_class_profile() const = 0;
  [[nodiscard]] virtual std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile()
      const = 0;
};

/** CUDA eigensolver selection and inactive-lane profiles. */
class BatchEigensolverProfiles {
 public:
  virtual ~BatchEigensolverProfiles() = default;
  [[nodiscard]] virtual std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const = 0;
  [[nodiscard]] virtual std::vector<InactiveEigensolverProfileEntry>
  last_inactive_eigensolver_profile() const = 0;
};

/** CUDA density-fitting metric profiles. */
class BatchDfMetricDiagnostics {
 public:
  virtual ~BatchDfMetricDiagnostics() = default;
  [[nodiscard]] virtual std::vector<scf::CudaDensityFittingMetricDiagnostic>
  last_density_fitting_metric_diagnostics() const = 0;
};

class BatchKsTransportDiagnostics {
 public:
  virtual ~BatchKsTransportDiagnostics() = default;
  [[nodiscard]] virtual std::optional<KsTransportDiagnostic> ks_transport_diagnostic(
      std::size_t index) const = 0;
};

class BatchCorrelationDiagnostics {
 public:
  virtual ~BatchCorrelationDiagnostics() = default;
  [[nodiscard]] virtual std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const = 0;
};

class BatchCcPerformanceDiagnostics {
 public:
  virtual ~BatchCcPerformanceDiagnostics() = default;
  [[nodiscard]] virtual std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic(
      std::size_t index) const = 0;
};

}  // namespace generativeqc::methods

#endif
