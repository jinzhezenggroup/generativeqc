#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace generativeqc_grid_pair_geometry {
struct CenterPair;
}

namespace generativeqc::dft {

/** Maximum atom lanes in the one-warp Cartesian Becke column. Consumers
 * reserve at most three reduction rows per supported atom. */
inline constexpr std::size_t cuda_grid_cartesian_atoms = 32;
/** Upper extent of the globally backed Cartesian point-tile schedule. */
inline constexpr std::size_t cuda_grid_tiled_cartesian_atoms = 96;
/** Private transaction status bit, independent of the generic invalid-data bit. */
inline constexpr int cuda_grid_cartesian_repair_exhausted = 2;
inline constexpr int cuda_grid_geometry_invalid = 4;

struct CudaGridGeometryShape {
  std::size_t atoms{}, primitives{}, nao{}, points{}, points_per_atom{};
  /** Optional sum over AO primitive/polynomial incidences, checked against
   * device AoBasis metadata before evaluating jets. Zero uses the generic
   * AO*all-primitives*3 upper bound for callers without shell topology. */
  std::size_t ao_primitive_terms{};
  /** A verified single constant Cartesian monomial per AO can retain four
   * radial moments instead of twenty spatial values. Requires a host incidence
   * bound and device verification of the packed AO records. */
  bool radial_s{};
};

struct CudaGridGeometryOptions {
  std::size_t max_workspace_bytes{512ULL << 20};
  /** Bounded point-parallel Becke scratch; small explicit tiles remain valid. */
  std::size_t point_tile{1024};
  std::uint64_t max_partition_pairs{16'777'216};
  std::uint64_t max_ao_jet_values{134'217'728};
  /** Preparation work bound: (20 or 4)*points*AO primitive/polynomial incidences,
   * independent of the output/storage bound. Also bounds each replay pass. */
  std::uint64_t max_ao_primitive_terms{134'217'728};
  unsigned partition_iterations{3};
  double coincident_tolerance{1e-12};
  /** Dimensionless partition tolerances, before multiplying the atomic
   * measure. Far radial measures amplify ordinary switching roundoff. */
  double reference_absolute_tolerance{1e-15};
  double reference_relative_tolerance{3e-11};
  /** Zero requires resident spatial jets. Otherwise, retain a budgeted prefix
   * and replay the tail in point tiles, bounded over the requested action block. */
  std::uint64_t max_ao_replay_terms{};
  /** Disable prefix growth if a subsequent owner can use the shared capacity. */
  bool retain_partial_jets{true};
  /** Reuse the idle directional AO scratch during a private Cartesian sweep. */
  bool cache_cartesian_weights{true};
  /** Pair visits reserved for rare scaled repairs of a tiled Cartesian
   * column. Exhaustion invalidates the private transaction, not the primal. */
  std::uint64_t max_cartesian_repair_pairs{16'777'216};
};

/** Immutable normalized AoBasis and native atom-major quadrature. Legacy
 * grids use points_per_atom; version-4 grids use the retained atom_offsets
 * and Treutler atomic_radii, with points_per_atom=0. Atomic measures are the actual pre-partition
 * CUDA source, never weights divided by a possibly zero partition. The caller authenticates all
 * storage, extents and grid policy on the current device/ordinary stream.
 */
struct CudaGridGeometryReference {
  const double *basis{}, *points{}, *weights{}, *atomic_weights{};
  int device_id{-1};
  cudaStream_t stream{};
  cudaError_t (*authenticate)(void*) noexcept {};
  void* owner{};
  /** Optional native atom-prefix topology and positive Bragg partition radii.
   * Required together when points_per_atom is zero; share the owner lease. */
  const std::uint64_t* atom_offsets{};
  const double* atomic_radii{};
  /** Optional FP64 center geometry installed by the authenticated force owner. */
  const generativeqc_grid_pair_geometry::CenterPair* center_pairs{};
};

struct CudaGridGeometryWork {
  std::size_t workspace_bytes{};
  std::uint64_t prepared_ao_jet_values{}, prepared_partition_pairs{};
  std::uint64_t action_ao_jet_values{}, action_partition_pairs{};
  std::uint64_t attempted_actions{}, completed_actions{};
  std::uint64_t prepared_ao_primitive_term_bound{};
  std::uint64_t action_point_values{};
  std::uint64_t replayed_ao_primitive_term_bound{};
  std::uint64_t cartesian_partition_sweeps{};
  std::uint64_t cartesian_diagonal_sweeps{};
  std::uint64_t tiled_cartesian_sweeps{};
  std::uint64_t tiled_cartesian_repair_pair_bound{};
};

struct CudaGridGeometryMixedView {
  const double *ao{}, *weights{}, *point_directions{};
};

/** One direction: [4,point,AO] first jets, [point] weights, [point,xyz] motion. */
struct CudaGridGeometryDirectionView {
  const double *ao{}, *weights{}, *point_directions{};
};

/** Private weights plus immutable spatial jets; no directional AO panels. */
struct CudaGridGeometryCartesianView {
  const double *basis{}, *spatial{}, *weights{};
  const std::uint64_t* atom_offsets{};
  std::size_t atoms{}, points_per_atom{}, left{}, right{};
  // A single row has left/right/mixed weights. A full column stores only
  // left/mixed per row; its common right weights belong to the direction view.
  std::size_t rows{1};
  /** Non-null for [4,point,AO] radial moments; reconstruct spatial derivatives
   * using these same authenticated [point,xyz] coordinates. */
  const double* radial_points{};
};

/** Pure host overflow/resource/work admission. The optional minimum reserves
 * the smallest admissible tile (or the full resident layout if replay is disabled). */
cudaError_t cuda_grid_geometry_workspace(const CudaGridGeometryShape& shape,
                                         const CudaGridGeometryOptions& options, std::size_t& bytes,
                                         std::size_t* minimum = nullptr) noexcept;

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

  /** Verify the exact partition weights and cache immutable order-three AO
   * jets when resident storage fits. Failure revokes the lease. */
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
  /** Same action and finite/owner gates as mixed(), without copying the panels.
   * The next prepare or mixed attempt expires this view. Consume it on the
   * authenticated stream; do not retain it across a lease revocation. */
  cudaError_t mixed_view(const double* left, const double* right, CudaGridGeometryMixedView& view,
                         cudaStream_t stream) noexcept;
  /** Borrow a provisional view on the same stream without a host audit. A
   * separate initialized transaction status accumulates errors; the caller
   * must drain and check it before publishing any result derived from view. */
  cudaError_t enqueue_private_mixed_view(const double* left, const double* right,
                                         CudaGridGeometryMixedView& view, int* transaction_status,
                                         cudaStream_t stream) noexcept;
  /** Same provisional mixed view for an authenticated unit direction on both
   * sides. Publish just the diagonal weight row with the ordinary AO panels. */
  cudaError_t enqueue_private_cartesian_diagonal_view(const double* direction, std::size_t column,
                                                      CudaGridGeometryMixedView& view,
                                                      int* transaction_status,
                                                      cudaStream_t stream) noexcept;
  /** Dense unit directions constructed by the caller. Return cudaErrorNotSupported
   * before enqueue if order-three spatial jets are not fully resident. */
  cudaError_t enqueue_private_cartesian_view(const double* left, const double* right,
                                             std::size_t left_column, std::size_t right_column,
                                             CudaGridGeometryCartesianView& view,
                                             int* transaction_status, cudaStream_t stream) noexcept;
  /** Borrow all left rows for the authenticated right unit column. The
   * implicit left directions are generated by index, never read from a panel.
   * When requested, borrow its right first jets and weights for the CPKS RHS.
   * Both views remain live until the next geometry call that uses scratch.
   * If the batched fallback does not fit, return NotSupported before enqueue. */
  cudaError_t enqueue_private_cartesian_column(
      const double* right, std::size_t right_column, CudaGridGeometryCartesianView& view,
      int* transaction_status, cudaStream_t stream,
      CudaGridGeometryDirectionView* rhs = nullptr) noexcept;
  /** Admit all mixed actions of an endpoint before any partial work. */
  cudaError_t admit_actions(std::size_t actions) noexcept;
  /** Borrow [20,point,AO] spatial jets when resident and expanded. */
  cudaError_t spatial_jets(const double*& jets) noexcept;
  /** Borrow [left/right,point,xyz] from the last successful mixed() action.
   * Point motion is the actual owner-atom direction gathered on CUDA. The
   * next prepare/mixed attempt or live-owner revocation expires this lease. */
  cudaError_t point_directions(const double*& directions) noexcept;
  const CudaGridGeometryWork& work() const noexcept { return work_; }

 private:
  cudaError_t authenticate(cudaStream_t stream) noexcept;
  cudaError_t partition(const double* left, const double* right, double* output = nullptr,
                        const double* accepted = nullptr, bool reference_only = false) noexcept;
  cudaError_t evaluate_mixed(const double* left, const double* right, cudaStream_t stream,
                             int* transaction_status = nullptr, std::size_t left_column = SIZE_MAX,
                             std::size_t right_column = SIZE_MAX,
                             bool weights_only = false) noexcept;
  cudaError_t mixed_view_impl(const double* left, const double* right,
                              CudaGridGeometryMixedView& view, int* transaction_status,
                              cudaStream_t stream, std::size_t diagonal_column = SIZE_MAX) noexcept;
  void finish_mixed() noexcept;
  CudaGridGeometryShape shape_{};
  CudaGridGeometryOptions options_{};
  CudaGridGeometryReference reference_{};
  CudaGridGeometryWork work_{};
  void* workspace_{};
  std::size_t cartesian_right_{SIZE_MAX};
  bool ready_{};
  bool motion_ready_{};
};

}  // namespace generativeqc::dft
