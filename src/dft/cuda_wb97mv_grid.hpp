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
  /** Tile cap, not extra VV10 work. Smaller explicit tiles retain the bounded
   * low-memory schedule; 256 supplies enough independent pairs for modern GPUs. */
  std::size_t pair_tile_points{256};
  double symmetry_absolute_tolerance{1e-12};
  double symmetry_relative_tolerance{1e-10};
  bool nuclear_response{false};
  std::uint64_t max_nuclear_affine_terms{1ULL << 34};
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
  std::uint64_t nuclear_reference_pairs{}, nuclear_response_pairs{}, nuclear_affine_terms{};
  std::uint64_t nuclear_directions{}, nuclear_contractions{};
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
  /** Opt-in moving-grid direction, including signed density motion. AO jets are
   * [4,point,AO], point motion [point,xyz]. Inputs are copied into the arena;
   * they need not survive this synchronous call. A successful call arms mixed
   * contractions and publishes [spin,AO,AO] dVxc. Any prepare/apply/direction
   * attempt expires that cache, including failed attempts. No pair traversal
   * is repeated for subsequent left-coordinate contractions. */
  cudaError_t prepare_nuclear_direction(const double* ao, const double* weights,
                                        const double* points, const double* density,
                                        double* potential, cudaStream_t stream) noexcept;
  /** Stationary left nuclear partial with zero left/mixed density motion.
   * The retained right density response participates in the mixed AO features.
   * Mixed point motion is zero for affine owner-atom coordinates. The scalar
   * output is device resident and transactional up to CUDA publication errors. */
  cudaError_t nuclear_mixed(const double* left_ao, const double* mixed_ao,
                            const double* left_weights, const double* mixed_weights,
                            const double* left_points, double* output,
                            cudaStream_t stream) noexcept;
  /** Borrow the opt-in prepared grid energy scalar under the live lease. */
  cudaError_t reference_energy(const double*& energy) noexcept;
  const CudaWb97mvGridWork& work() const noexcept { return work_; }

 private:
  cudaError_t authenticate(cudaStream_t stream) noexcept;
  cudaError_t evaluate(bool response) noexcept;
  cudaError_t evaluate_nuclear(bool response) noexcept;
  CudaWb97mvGridShape shape_{};
  CudaWb97mvGridOptions options_{};
  CudaWb97mvGridReference reference_{};
  CudaWb97mvGridWork work_{};
  void* workspace_{};
  bool ready_{};
  bool nuclear_ready_{};
};

}  // namespace generativeqc::dft
