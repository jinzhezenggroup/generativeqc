#ifndef GENERATIVEQC_XTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
#define GENERATIVEQC_XTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_scc_mixer.cuh"
#include "backends/cuda/gfn2_density.cuh"
#include "runtime/types.hpp"

namespace generativeqc::xtb::detail {

struct Gfn2CudaMixerDiagnosticSnapshot {
  std::uint64_t call_id = 0;
  std::uint64_t plan_token = 0;
  std::uint64_t attempted_receipts = 0;
  std::int32_t device_id = -1;
  // 0: bounded fallback, 1: device-tail graph, 2: device-dispatch chain.
  std::uint32_t graph_family = 0;
  bool graph_submitted = false;
  bool endpoint_completed = false;
  std::vector<cuda::Gfn2SccMixerDeviceReceipt> receipts;
};

struct Gfn2CudaDensityDiagnosticSnapshot {
  std::uint64_t call_id = 0;
  std::uint64_t plan_token = 0;
  std::uint64_t attempted_receipts = 0;
  std::uint64_t receipt_capacity = 0;
  std::uint64_t arena_bytes = 0;
  std::int32_t device_id = -1;
  std::int64_t batch_size = 0;
  std::int32_t maximum_iterations = 0;
  std::uint32_t grid_tiles = 0;
  std::uint32_t graph_family = 0;
  bool graph_submitted = false;
  bool endpoint_completed = false;
  std::vector<cuda::Gfn2DensityDeviceReceipt> receipts;
};

// Owns molecular CUDA topology, numerical arenas, solver handles and the SCC
// graph across synchronous calls. Rebuilding topology is transactional; the
// bounded SCC fallback remains available when conditional capture is unsupported.
class Gfn2CudaExecutionCache {
 public:
  Gfn2CudaExecutionCache(std::int32_t device_id, void* stream);
  ~Gfn2CudaExecutionCache();
  Gfn2CudaExecutionCache(const Gfn2CudaExecutionCache&) = delete;
  Gfn2CudaExecutionCache& operator=(const Gfn2CudaExecutionCache&) = delete;

  // Must be set before the first prepared topology. Read only after the
  // synchronous endpoint returns; a failed endpoint retains partial receipts.
  [[nodiscard]] bool enable_mixer_diagnostics() noexcept;
  [[nodiscard]] bool read_mixer_diagnostics(Gfn2CudaMixerDiagnosticSnapshot& snapshot,
                                            std::string& error) const;
  // Available only in a build configured with
  // GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS. Enable before preparation;
  // read only after the synchronous endpoint settles, including failures.
  [[nodiscard]] bool enable_density_diagnostics() noexcept;
  [[nodiscard]] bool read_density_diagnostics(Gfn2CudaDensityDiagnosticSnapshot& snapshot,
                                              std::string& error) const;

 private:
  friend generativeqc_xtb_status_t execute_restricted_gfn2_cuda_impl(Gfn2CudaExecutionCache&,
                                                               const generativeqc_xtb_batch_t&,
                                                               const generativeqc_xtb_compute_options_t&,
                                                               generativeqc_xtb_batch_result_t&,
                                                               std::string&);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Completes geometry refresh, SCC, energy/forces and result publication before
// returning. Failure before the output commit leaves caller outputs untouched.
[[nodiscard]] generativeqc_xtb_status_t execute_restricted_gfn2_cuda(
    Gfn2CudaExecutionCache& cache, const generativeqc_xtb_batch_t& batch,
    const generativeqc_xtb_compute_options_t& options, generativeqc_xtb_batch_result_t& result,
    std::string& error);

}  // namespace generativeqc::xtb::detail
#endif
