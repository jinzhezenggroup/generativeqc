#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_wb97mv_df_response.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}
namespace generativeqc::dft {

/** Bind fitted-primary AD to the actual accepted RKS/UKS WB97M-V CUDA owners.
 * The adapter borrows live D/W/eigenvalues and prepares raw public A through the
 * actual DF integral source, preserving both basis representations. Retained A
 * and all algebra scratch share one explicit device-byte bound. No primary J/K
 * scratch or occupied-factor projection is borrowed or revoked.
 *
 * apply consumes raw dA/dV supplied by a separate nuclear-integral AD owner;
 * it is not a molecular Hessian endpoint. The batch must outlive this exclusive
 * consumer. Repeated actions allocate nothing and reauthenticate the complete
 * model, accepted state, raw integral source and metric-domain leases.
 */
class CudaKsWb97mvDfResponse {
 public:
  CudaKsWb97mvDfResponse();
  ~CudaKsWb97mvDfResponse();
  CudaKsWb97mvDfResponse(const CudaKsWb97mvDfResponse&) = delete;
  CudaKsWb97mvDfResponse& operator=(const CudaKsWb97mvDfResponse&) = delete;
  CudaKsWb97mvDfResponse(CudaKsWb97mvDfResponse&&) = delete;
  CudaKsWb97mvDfResponse& operator=(CudaKsWb97mvDfResponse&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const CudaWb97mvDfOptions& options, std::string& detail);
  cudaError_t apply(const double* delta_density, const double* delta_three_center,
                    const double* delta_metric, const CudaWb97mvDfOutput& output,
                    cudaStream_t stream) noexcept;
  CudaWb97mvDfShape shape() const noexcept;
  /** Includes the adapter's retained raw A, not just algebra scratch. */
  CudaWb97mvDfWork work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
