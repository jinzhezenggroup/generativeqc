#pragma once

#include "scf/cuda/direct_jk_source_lease.hpp"
#include "scf/cuda/rhf_graph.hpp"

namespace generativeqc::scf::cuda_execution {

/** RHF-specific assembly and graph borrower of the common phase source.
 * Drain and retire local graphs before releasing the source on every exit.
 */
class RhfResidentValues {
 public:
  explicit RhfResidentValues(cudaStream_t stream) : stream_(stream), source_(stream) {}
  ~RhfResidentValues();
  RhfResidentValues(const RhfResidentValues&) = delete;
  RhfResidentValues& operator=(const RhfResidentValues&) = delete;

  generativeqc_status prepare(int device, const std::vector<core::System>& systems,
                              unsigned maximum_iterations, std::size_t direct_nbf,
                              std::size_t required, std::size_t budget, bool cold_reference = false,
                              bool generic_fock = false);
  bool active() const noexcept { return source_.active(); }
  std::size_t capacity_bytes() const noexcept { return source_.capacity_bytes(); }
  std::size_t value_bytes() const noexcept { return source_.value_bytes(); }
  const std::string& reason() const noexcept { return source_.reason(); }
  generativeqc_status enqueue(const double* density, const double* hcore, double* fock);
  generativeqc_status audit() { return source_.audit(); }
  void observe_completed(std::size_t physical_focks) const noexcept {
    source_.observe_completed(physical_focks);
  }

  RhfIterationGraphs graphs;

 private:
  cudaStream_t stream_{};
  DirectJkSourceLease source_;
};

}  // namespace generativeqc::scf::cuda_execution
