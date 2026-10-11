#pragma once

#include <memory>
#include <string>

#include "scf/cuda/direct_jk_source_policy.hpp"
#include "scf/cuda_direct_jk_device.hpp"

namespace generativeqc::scf::cuda_execution {

/** Method-neutral, phase-local lease of immutable full-range FP64 ERIs.
 * The common Direct provider remains the scientific owner. Consumers assemble
 * their own HF/KS equations and retire captured borrowers before this lease.
 * The caller-owned stream outlives destruction; no source survives into a
 * subsequent solve, geometry or derivative consumer.
 */
class DirectJkSourceLease {
 public:
  explicit DirectJkSourceLease(cudaStream_t stream) : stream_(stream) {}
  ~DirectJkSourceLease();
  DirectJkSourceLease(const DirectJkSourceLease&) = delete;
  DirectJkSourceLease& operator=(const DirectJkSourceLease&) = delete;

  /** Optional resource/capability refusal returns SUCCESS with active()==false.
   * Driver and numerical errors propagate; no partially built source is used.
   * required is the complete mandatory consumer admission, not free VRAM.
   * Forced selection never bypasses capacity/capability gates.
   * own_outputs=false borrows the consumer's raw J/K slots; compensation and
   * audits remain lease-owned, and enqueue must receive those consumer slots.
   * auto_policy defaults to the unchanged RHF generic/f-shell domain. The KS
   * screened policy is opt-in and does not turn the iteration limit into an
   * expected reuse count.
   * bucket_mask defaults to complete residency. Partial masks share the full
   * source's immutable Schwarz layout; uncached buckets remain exact Direct.
   */
  generativeqc_status prepare(
      int device, const std::vector<core::System>& systems, unsigned maximum_iterations,
      std::size_t direct_nbf, std::size_t required, std::size_t budget, const char* selection_value,
      const char* trace_name, bool cold_phase = false, bool generic_fock = false,
      double screening = 0.0, std::size_t value_ceiling = 8ULL << 30, bool own_outputs = true,
      DirectJkSourceAutoPolicy auto_policy = DirectJkSourceAutoPolicy::generic_f,
      std::uint32_t bucket_mask = (1U << 28) - 1);
  bool active() const noexcept { return direct_ != nullptr; }
  std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
  std::size_t value_bytes() const noexcept { return value_bytes_; }
  const std::string& reason() const noexcept { return reason_; }

  /** Enqueue raw restricted J/K without applying scientific coefficients.
   * Preserve the caller's Schwarz threshold on each contraction. A null error
   * slot uses the lease's own slot. No allocation, upload or success-path fence
   * occurs; caller outputs must outlive the final audit.
   */
  generativeqc_status enqueue(FockBuildSpec spec, const double* density, double* coulomb,
                              double* exchange, int* numerical_error, double screening,
                              std::string& detail);
  /** Fence finite/census validation before publishing the last action.
   * At the frozen Schwarz threshold every uncached value must be evaluated.
   * A stricter replay predicate may skip recurrences but not source visits.
   */
  generativeqc_status audit();
  /** Completed physical work, not host graph-capture calls. */
  void observe_completed(std::size_t physical_focks) const noexcept;

  /** Optional lease-owned raw outputs; KS borrows its already admitted slots. */
  double* coulomb() const noexcept { return scratch_ && output_elements_ ? scratch_ : nullptr; }
  double* exchange() const noexcept {
    return scratch_ && output_elements_ ? scratch_ + matrix_elements_ : nullptr;
  }
  std::size_t matrix_elements() const noexcept { return matrix_elements_; }
  std::size_t value_count() const noexcept { return value_count_; }
  /** Stream-completed construction, including a later publication refusal. */
  std::size_t completed_value_count() const noexcept { return completed_values_; }
  const void* source_identity() const noexcept { return direct_.get(); }
  /** Explicit live device storage; capacity_bytes also includes host setup. */
  std::size_t device_bytes() const noexcept {
    return active() ? cuda_direct_jk_plan_diagnostic(direct_.get()).device_bytes +
                          (output_elements_ + correction_elements_) * sizeof(double) + sizeof(int) +
                          2 * sizeof(std::uint64_t)
                    : 0;
  }

 private:
  void release_values() noexcept;
  cudaStream_t stream_{};
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> direct_{
      nullptr, destroy_cuda_direct_jk_plan};
  double* scratch_{};
  int* error_{};
  int* last_error_{};
  std::uint64_t* census_{};
  std::size_t matrix_elements_{}, output_elements_{}, correction_elements_{};
  std::size_t capacity_bytes_{}, value_bytes_{}, value_count_{}, source_value_count_{};
  std::size_t completed_values_{};
  double source_screening_{}, last_screening_{};
  bool prepared_{};
  const char* selection_{"auto"};
  std::string reason_{"ineligible"};
};

}  // namespace generativeqc::scf::cuda_execution
