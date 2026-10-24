#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "integrals/cuda_one_electron_response.hpp"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::methods {
class PreparedBatch;
}
namespace generativeqc::dft {

/** Host-only sizing from normalized native topology, including effective charges.
 * Throws on malformed topology or mandatory admission failure, just as prepare.
 * Aggregate owners can disable parallel_streams to reserve the scalar baseline
 * before allowing optional schedules to consume any remaining capacity. */
std::size_t cuda_ks_one_electron_workspace(const core::System& system,
                                           const integrals::CudaOneElectronShape& shape,
                                           const integrals::CudaOneElectronOptions& options);

struct CudaKsOneElectronWork {
  std::size_t device_bytes{};
  integrals::CudaOneElectronWork integrals;
};

/** Live KS adapter for the native compiler-AD S and T+effective-charge source.
 * Integer topology and effective charges come from the actual prepared Fock
 * owner's normalized System; numerical basis records are borrowed from its
 * canonical native grid binding. A caller cannot substitute reconstructed basis
 * records, ECP charges, device, stream or generation labels. Every source action
 * reborrows both live bindings before execution/publication. ECP core removal is
 * retained, but residual local/nonlocal ECP projectors are explicitly excluded.
 *
 * Preparation may allocate independent arena/charge storage. Actions do not
 * allocate, change devices, compile or execute Python/CPU numerical work. The
 * batch outlives this nonmovable adapter and excludes concurrent batch access.
 * Weights remain method-supplied; no spin/HF factors are inferred here. This is
 * a nuclear source component, not a complete Hessian/HVP endpoint.
 */
class CudaKsOneElectronResponse {
 public:
  CudaKsOneElectronResponse();
  ~CudaKsOneElectronResponse();
  CudaKsOneElectronResponse(const CudaKsOneElectronResponse&) = delete;
  CudaKsOneElectronResponse& operator=(const CudaKsOneElectronResponse&) = delete;
  CudaKsOneElectronResponse(CudaKsOneElectronResponse&&) = delete;
  CudaKsOneElectronResponse& operator=(CudaKsOneElectronResponse&&) = delete;

  generativeqc_status prepare(methods::PreparedBatch& batch, std::size_t item,
                              const CudaKsFinalStateToken& expected,
                              const integrals::CudaOneElectronOptions& options,
                              std::string& detail);
  cudaError_t apply(const double* direction, const integrals::CudaOneElectronWeights& weights,
                    const integrals::CudaOneElectronOutput& output, cudaStream_t stream) noexcept;
  CudaKsOneElectronWork work() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::dft
