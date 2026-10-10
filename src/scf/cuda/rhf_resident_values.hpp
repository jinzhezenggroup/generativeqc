#pragma once

#include <memory>
#include <string>

#include "scf/cuda/rhf_graph.hpp"
#include "scf/cuda_direct_jk_device.hpp"

namespace generativeqc::scf::cuda_execution {

/** One immutable geometry's optional exact source, owned only by an RHF call.
 * This is a schedule of the common compensated Direct provider, not a second
 * RHF/ERI implementation. Captured borrowers die before values and scratch;
 * nothing survives into correlation or a later geometry execution.
 */
class RhfResidentValues {
 public:
  explicit RhfResidentValues(cudaStream_t stream) : stream_(stream) {}
  ~RhfResidentValues();
  RhfResidentValues(const RhfResidentValues&) = delete;
  RhfResidentValues& operator=(const RhfResidentValues&) = delete;

  /** Optional resource/capability refusal returns SUCCESS with active()==false.
   * Driver and numerical errors propagate; no partially built source is used.
   * required is the complete mandatory RHF numeric admission, not free VRAM.
   */
  generativeqc_status prepare(int device, const std::vector<core::System>& systems,
                              unsigned maximum_iterations, std::size_t direct_nbf,
                              std::size_t required, std::size_t budget, bool cold_reference = false,
                              bool generic_fock = false);
  bool active() const noexcept { return direct_ != nullptr; }
  std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
  std::size_t value_bytes() const noexcept { return value_bytes_; }
  const std::string& reason() const noexcept { return reason_; }

  /** All matrices are public-AO row/column-major symmetric RHF matrices.
   * The existing provider returns raw J/K; the shared HF assembly applies
   * coefficients once. No allocation, upload or success-path fence occurs.
   */
  generativeqc_status enqueue(const double* density, const double* hcore, double* fock);
  /** Fence the last action's finite audit before publishing the reference. */
  generativeqc_status audit();
  /** Completed physical work, not host graph-capture calls. */
  void observe_completed(std::size_t physical_focks) const noexcept;

  RhfIterationGraphs graphs;

 private:
  void release_values() noexcept;
  cudaStream_t stream_{};
  std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> direct_{
      nullptr, destroy_cuda_direct_jk_plan};
  double* scratch_{};
  int* error_{};
  std::uint64_t* census_{};
  std::size_t matrix_elements_{}, correction_elements_{};
  std::size_t capacity_bytes_{}, value_bytes_{}, value_count_{};
  bool prepared_{};
  const char* selection_{"auto"};
  std::string reason_{"ineligible"};
};

}  // namespace generativeqc::scf::cuda_execution
