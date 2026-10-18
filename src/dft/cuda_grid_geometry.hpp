#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc::dft {

struct CudaGridGeometryShape {
  std::size_t atoms{}, primitives{}, nao{}, points{}, points_per_atom{};
};

struct CudaGridGeometryOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  /** Bounded point-parallel Becke scratch; small explicit tiles remain valid. */
  std::size_t point_tile{1024};
  std::uint64_t max_partition_pairs{16'777'216};
  std::uint64_t max_ao_jet_values{134'217'728};
  /** Conservative preparation bound: 20*points*AO*primitives*3 polynomial
   * terms, independent of the output/storage bound. No Gaussian replay in mixed(). */
  std::uint64_t max_ao_primitive_terms{134'217'728};
  unsigned partition_iterations{3};
  double coincident_tolerance{1e-12};
  /** Dimensionless partition tolerances, before multiplying the atomic
   * measure. Far radial measures amplify ordinary switching roundoff. */
  double reference_absolute_tolerance{1e-15};
  double reference_relative_tolerance{3e-11};
};

/** Immutable normalized AoBasis and native unpruned quadrature. Owner atoms
 * follow atom/radial/polar/azimuth order, with points_per_atom consecutive
 * points per atom. Atomic measures are the actual pre-partition CUDA source,
 * never weights divided by a possibly zero partition. The caller authenticates
 * all storage, extents and grid policy on the current device/ordinary stream.
 */
struct CudaGridGeometryReference {
  const double *basis{}, *points{}, *weights{}, *atomic_weights{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
};

struct CudaGridGeometryWork {
  std::size_t workspace_bytes{};
  std::uint64_t prepared_ao_jet_values{}, prepared_partition_pairs{};
  std::uint64_t action_ao_jet_values{}, action_partition_pairs{};
  std::uint64_t attempted_actions{}, completed_actions{};
  std::uint64_t prepared_ao_primitive_term_bound{};
  std::uint64_t action_point_values{};
};

/** Pure host overflow/resource/work admission; does not initialize CUDA. */
cudaError_t cuda_grid_geometry_workspace(const CudaGridGeometryShape& shape,
                                         const CudaGridGeometryOptions& options,
                                         std::size_t& bytes) noexcept;

/** Native first/mixed-second moving Becke and AO geometry source for Hessians.
 * Spatial jets through order three and all scalar geometry/product/ratio AD are
 * generated at build time. Shared binary scaling of the four product jets
 * preserves exact-zero factors without dividing by a factor or underflowing
 * whole long products. Pair-center jets are evaluated once per action, then
 * reused across bounded point tiles. No atom-count-specific JIT or IR witness.
 *
 * Preparation and mixed() borrow one caller-owned arena, without allocations,
 * CPU numerical work, device switching, Python or runtime compilation. Finite,
 * smooth-branch and live-owner gates fence before publication. This is geometry
 * source response, not XC/VV10, CPKS or a complete stationary nuclear Hessian.
 */
class CudaGridGeometryResponse {
 public:
  CudaGridGeometryResponse() = default;
  CudaGridGeometryResponse(const CudaGridGeometryResponse&) = delete;
  CudaGridGeometryResponse& operator=(const CudaGridGeometryResponse&) = delete;
  CudaGridGeometryResponse(CudaGridGeometryResponse&&) = delete;
  CudaGridGeometryResponse& operator=(CudaGridGeometryResponse&&) = delete;

  /** Cache immutable order-three spatial jets and verify the exact partitioned
   * reference weights against the supplied atomic measure. Failure revokes. */
  cudaError_t prepare(const CudaGridGeometryShape& shape, const CudaGridGeometryOptions& options,
                      const CudaGridGeometryReference& reference, void* workspace,
                      std::size_t workspace_bytes) noexcept;
  /** Left/right are finite device [atom,xyz] directions, allowed to coincide.
   * Output AO is [left/right/mixed,spatial-prefix(4),point,AO]; weight output is
   * [left/right/mixed,point]. Both outputs are disjoint from all retained inputs,
   * directions, arena and each other. Neither is published on prepublication
   * failure; a CUDA error during the final copies invalidates BOTH outputs. */
  cudaError_t mixed(const double* left, const double* right, double* ao_output,
                    double* weight_output, cudaStream_t stream) noexcept;
  /** Borrow [20,point,AO] ordinary spatial jets under the same live lease. */
  cudaError_t spatial_jets(const double*& jets) noexcept;
  /** Borrow [left/right,point,xyz] from the last successful mixed() action.
   * Point motion is the actual owner-atom direction gathered on CUDA. The
   * next prepare/mixed attempt or live-owner revocation expires this lease. */
  cudaError_t point_directions(const double*& directions) noexcept;
  const CudaGridGeometryWork& work() const noexcept { return work_; }

 private:
  cudaError_t authenticate(cudaStream_t stream) noexcept;
  cudaError_t partition(const double* left, const double* right) noexcept;
  CudaGridGeometryShape shape_{};
  CudaGridGeometryOptions options_{};
  CudaGridGeometryReference reference_{};
  CudaGridGeometryWork work_{};
  void* workspace_{};
  bool ready_{};
  bool motion_ready_{};
};

}  // namespace generativeqc::dft
