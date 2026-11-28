#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_wb97mv_df_sweep.hpp"
#include "integrals/cuda_df_nuclear_response.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}
namespace generativeqc::dft {

/** Actual native RKS/UKS WB97M-V DF raw nuclear source. Integer topology comes
 * from the normalized prepared orbital/auxiliary Systems; all numerical basis,
 * normalization and g-capable public expansions are borrowed from the actual
 * immutable CUDA DF integral source. The complete model and accepted state are
 * reauthenticated at preparation, each pass and before publication.
 *
 * The two counted passes surround method-owned inverse/weight AD and CPKS;
 * this adapter does not silently replace fitted primary J/K with exact ERIs or
 * claim a stationary molecular Hessian. The batch must outlive the exclusive
 * nonmovable adapter. Actions allocate no work buffers or use Python/JIT/CPU
 * numerical contraction, and preserve ordinary J/K/occupied projection leases.
 */
class CudaKsWb97mvDfNuclear {
 public:
  CudaKsWb97mvDfNuclear();
  ~CudaKsWb97mvDfNuclear();
  CudaKsWb97mvDfNuclear(const CudaKsWb97mvDfNuclear&) = delete;
  CudaKsWb97mvDfNuclear& operator=(const CudaKsWb97mvDfNuclear&) = delete;
  CudaKsWb97mvDfNuclear(CudaKsWb97mvDfNuclear&&) = delete;
  CudaKsWb97mvDfNuclear& operator=(CudaKsWb97mvDfNuclear&&) = delete;
  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const integrals::CudaDfNuclearOptions& options, std::string& detail);
  /** Optional setup before any action; no allocation when storage does not fit. */
  cudaError_t retain_directional_source(std::size_t available);
  cudaError_t values(const double* direction, const integrals::CudaDfNuclearValues& output,
                     cudaStream_t stream) noexcept;
  cudaError_t contract(const double* direction, const integrals::CudaDfNuclearWeights& weights,
                       const integrals::CudaDfNuclearVectors& output, cudaStream_t stream) noexcept;
  cudaError_t contract_fixed_weights(const double* direction, const double* three_center,
                                     const double* metric, double* hvp,
                                     cudaStream_t stream) noexcept;
  integrals::CudaDfNuclearShape shape() const noexcept;
  const integrals::CudaDfNuclearWork& work() const noexcept;
  /** Optional dense full-range source pass; never used for an arbitrary HVP. */
  cudaError_t private_frozen_sweep(const double* charge, CudaDfFrozenTables output,
                                   std::size_t max_scratch_bytes, int* status,
                                   cudaStream_t stream) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
