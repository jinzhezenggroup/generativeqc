#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
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
  cudaError_t values(const double* direction, const integrals::CudaDfNuclearValues& output,
                     cudaStream_t stream) noexcept;
  cudaError_t contract(const double* direction, const integrals::CudaDfNuclearWeights& weights,
                       const integrals::CudaDfNuclearVectors& output, cudaStream_t stream) noexcept;
  integrals::CudaDfNuclearShape shape() const noexcept;
  const integrals::CudaDfNuclearWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
