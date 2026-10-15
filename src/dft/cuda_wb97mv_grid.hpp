#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::dft {

using CudaWb97mvGridAuthenticate = cudaError_t (*)(void*) noexcept;

struct CudaWb97mvGridShape {
  std::size_t atoms{}, primitives{}, nao{}, points{};
  unsigned spins{1};
};

struct CudaWb97mvGridOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  std::uint64_t max_pair_evaluations{16'777'216};
  std::size_t pair_tile_points{32};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
};

/** Immutable native quadrature/basis/density lease, not a detached state label.
 * Basis is the validated AoBasis normalized 16-double record format, points
 * are [point,xyz], weights [point], D [spin,AO,AO] row-major. RKS D is total;
 * UKS panels are physical alpha/beta D. All buffers are device-resident on the
 * supplied ordinary stream/device. The authenticator must check the actual
 * live owner, exact density token AND immutable basis/quadrature identity.
 * Owner/buffers/arena must survive all synchronous calls, without concurrent
 * writes. Copying this descriptor does not prove lifetime or scientific identity.
 */
struct CudaWb97mvGridReference {
  const double *basis{}, *points{}, *weights{}, *density{};
  int device_id{-1};
  cudaStream_t stream{};
  CudaWb97mvGridAuthenticate authenticate{};
  void* owner{};
};

struct CudaWb97mvGridWork {
  std::size_t workspace_bytes{}, active_nonlocal_points{};
  std::uint64_t reference_pairs{}, response_pairs{}, response_actions{};
};

/** Pure checked sizing. Pair storage and total ordered-pair WORK are separate
 * gates. Actual active-count squared is checked at preparation, before pairs.
 * The arena includes the complete dense first-jet panel and map scratch;
 * density-dependent screening is fixed by the reference on both VV10 legs.
 */
cudaError_t cuda_wb97mv_grid_workspace(const CudaWb97mvGridShape& shape,
                                       const CudaWb97mvGridOptions& options,
                                       std::size_t& bytes) noexcept;

/** Native WB97M-V semilocal + bilateral VV10 density kernel on a fixed grid.
 * Scalar gradients/JVPs and feature/potential maps are compiler AD, generated
 * at build time. Includes physical spin coupling, tau halves, both VV10 legs,
 * ordered-pair traversal and the local beta term. No allocation, interpreter,
 * runtime compilation, CPU contraction, device switch or hidden fallback.
 * Each call fences at numerical/authentication gates; only status/active-count
 * metadata reaches the host. A failed action leaves public output unchanged
 * before publication; a CUDA error during publication invalidates that output.
 * This is the grid component of induced Fock, NOT J/K or a complete Hessian.
 */
class CudaWb97mvGridResponse {
 public:
  CudaWb97mvGridResponse() = default;
  CudaWb97mvGridResponse(const CudaWb97mvGridResponse&) = delete;
  CudaWb97mvGridResponse& operator=(const CudaWb97mvGridResponse&) = delete;
  CudaWb97mvGridResponse(CudaWb97mvGridResponse&&) = delete;
  CudaWb97mvGridResponse& operator=(CudaWb97mvGridResponse&&) = delete;

  /** Precompute immutable AOs/fields/reference Vxc within the caller arena.
   * Failure revokes this instance. A retry requires an intact live reference.
   */
  cudaError_t prepare(const CudaWb97mvGridShape& shape, const CudaWb97mvGridOptions& options,
                      const CudaWb97mvGridReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  /** Apply to a finite symmetric SIGNED/indefinite D direction, with both UKS
   * blocks together. Output is [spin,AO,AO], disjoint from all retained inputs.
   */
  cudaError_t apply(const double* direction, double* output, cudaStream_t stream) noexcept;
  /** Borrow the prepared native value-side potential under the same live lease.
   * It remains internal, and must not be written or retained after revocation.
   */
  cudaError_t reference_potential(const double*& potential) noexcept;
  const CudaWb97mvGridWork& work() const noexcept { return work_; }

 private:
  cudaError_t authenticate(cudaStream_t stream) noexcept;
  cudaError_t evaluate(bool response) noexcept;
  CudaWb97mvGridShape shape_{};
  CudaWb97mvGridOptions options_{};
  CudaWb97mvGridReference reference_{};
  CudaWb97mvGridWork work_{};
  void* workspace_{};
  bool ready_{};
};

}  // namespace generativeqc::dft
