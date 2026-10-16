#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "dft/cuda_wb97mv_nuclear_jk.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}
namespace generativeqc::dft {

/** Live KS adapter, not a complete stationary Hessian provider. Borrows the
 * actual accepted D and normalized basis, reauthenticating grid and Fock owners
 * before work/publication. Exact owners admit exact or individual radial parts;
 * DF owners admit ONLY the exact LR correction, never an exact substitute for
 * fitted primary J/K. A future fitted nuclear source must supply that primary.
 * Preparation allocates independent arena storage; actions allocate nothing.
 * Batch outlives the adapter and excludes concurrent batch access. */
class CudaKsWb97mvNuclearJk {
 public:
  CudaKsWb97mvNuclearJk();
  ~CudaKsWb97mvNuclearJk();
  CudaKsWb97mvNuclearJk(const CudaKsWb97mvNuclearJk&) = delete;
  CudaKsWb97mvNuclearJk& operator=(const CudaKsWb97mvNuclearJk&) = delete;
  CudaKsWb97mvNuclearJk(CudaKsWb97mvNuclearJk&&) = delete;
  CudaKsWb97mvNuclearJk& operator=(CudaKsWb97mvNuclearJk&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const CudaWb97mvJkOptions& options, std::string& detail);
  cudaError_t apply(const double* direction, const double* delta_density,
                    const CudaWb97mvJkOutput& output, cudaStream_t stream) noexcept;
  const CudaWb97mvJkWork& work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
