#ifndef GENERATIVEQC_XTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
#define GENERATIVEQC_XTB_RUNTIME_GFN2_CUDA_EXECUTION_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "runtime/types.hpp"

namespace generativeqc::xtb::detail {

// Owns molecular CUDA topology, numerical arenas, solver handles and the SCC
// graph across synchronous calls. Rebuilding topology is transactional; the
// bounded SCC fallback remains available when conditional capture is unsupported.
class Gfn2CudaExecutionCache {
 public:
  Gfn2CudaExecutionCache(std::int32_t device_id, void* stream);
  ~Gfn2CudaExecutionCache();
  Gfn2CudaExecutionCache(const Gfn2CudaExecutionCache&) = delete;
  Gfn2CudaExecutionCache& operator=(const Gfn2CudaExecutionCache&) = delete;

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
