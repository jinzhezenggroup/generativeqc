#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "integrals/cuda_ecp_nuclear_response.hpp"

namespace generativeqc::methods {
class PreparedBatch;
}
namespace generativeqc::dft {

/** Residual ECP nuclear source bound to an accepted native WB97M-V state.
 * The actual immutable grid AO basis and normalized primary-Fock System own
 * basis identity and residual terms; Direct/DF and RKS/UKS share this source.
 * Terms are packed once, with all staging and generated spin-sum storage
 * charged to the explicit workspace budget. Contract uses the accepted spin
 * densities and a physical signed spin-density direction, not occupied factors.
 *
 * The batch must outlive this exclusive nonmovable synchronous adapter. Every
 * action reauthenticates the complete model, canonical frame and owner before
 * publication, allocates nothing, and leaves ordinary SCF/DF workspaces alone.
 * Numerical work uses linked compiler-generated CUDA only. This source does
 * not by itself assemble a stationary Hessian or supply an implicit CPKS solve.
 */
class CudaKsWb97mvEcpNuclear {
 public:
  CudaKsWb97mvEcpNuclear();
  ~CudaKsWb97mvEcpNuclear();
  CudaKsWb97mvEcpNuclear(const CudaKsWb97mvEcpNuclear&) = delete;
  CudaKsWb97mvEcpNuclear& operator=(const CudaKsWb97mvEcpNuclear&) = delete;
  CudaKsWb97mvEcpNuclear(CudaKsWb97mvEcpNuclear&&) = delete;
  CudaKsWb97mvEcpNuclear& operator=(CudaKsWb97mvEcpNuclear&&) = delete;
  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const integrals::CudaEcpNuclearOptions& options, std::string& detail);
  cudaError_t values(const double* direction, const integrals::CudaEcpNuclearValues& output,
                     cudaStream_t stream) noexcept;
  cudaError_t contract(const double* direction, const double* spin_direction_density,
                       const integrals::CudaEcpNuclearVectors& output,
                       cudaStream_t stream) noexcept;
  integrals::CudaEcpNuclearShape shape() const noexcept;
  const integrals::CudaEcpNuclearWork& work() const noexcept;
  std::size_t device_bytes() const noexcept;
  std::uint64_t spin_sum_terms_per_contraction() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
