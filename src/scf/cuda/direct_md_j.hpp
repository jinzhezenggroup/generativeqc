#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/md_j_engine_contract.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Extra allowance requested by private KS's shape-only provider estimate.
 * This is a reservation hint, not a second cap on an already budgeted owner.
 * Actual MD storage must fit the direct plan's remaining capacity; public
 * ledger-bound plans cannot spend this optional allowance. */
inline constexpr std::size_t kMdJDefaultReservation = 128U << 20;
inline constexpr double kMdJDensityErrorCap = 1e-12;

struct MdSourceTask {
  std::uint32_t bra, ket;
};

inline constexpr std::size_t kMdSourceFixedBytes =
    sizeof(double) + sizeof(std::uint32_t) + sizeof(unsigned long long);

/** One unordered shell pair. Same-shell primitive exchange is folded into the
 * cached E transforms; distinct shells retain every ordered primitive product. */
struct MdJPair {
  std::int32_t system, first, second;
  std::int32_t first_ao, second_ao, first_count, second_count;
  std::uint32_t angular, hermites, primitive_begin, primitive_end;
  std::uint32_t transform_map;
};

/** Geometry is filled on the owning stream, never by a host integral oracle. */
struct MdJPrimitive {
  std::uint32_t pair;
  std::int64_t first, second;
  std::size_t transform_offset, hermite_offset;
  double exponent;
  Vec3<double> product;
};

/** Intrusive per-replay work census, never allocated in the normal path.
 * Uniform roots are radial source evaluations; directions and summands count
 * density consumers independently. Residual roots count primitive products
 * actually contracted after shell and density admission, not queue capacity.
 * All arrays use the index 5 * bra_angular + ket_angular.
 */
struct MdJWorkCounts {
  unsigned long long uniform_roots[25]{}, uniform_directions[25]{}, uniform_summands[25]{};
  unsigned long long residual_candidates[25]{}, residual_tasks[25]{}, residual_roots[25]{};
  unsigned long long engine_tiles[25]{}, engine_screened_tiles[25]{};
};

/** Device-only replay census for complete primitive J. Prefixes are refreshed
 * after the current density bound, including every captured graph replay. */
// Finite IEEE-754 bins; unknown/infinite norms are never omitted.
// Keep the leading two mantissa bits as well as the exponent. All boundaries
// are exact nonnegative FP64 bit patterns, including the subnormal domain.
constexpr unsigned kMdJSourceNormShift = 50;
constexpr unsigned kMdJSourceNormBins = 2047 * 4;
struct MdJPrimitiveScreenState {
  double maximum_density;
  double maximum_source_norm;
  std::uint32_t active[5];
  unsigned long long source_norm_counts[kMdJSourceNormBins];
  double source_norm_tails[kMdJSourceNormBins];
};

/** Borrowed, budgeted resident transforms and density-dependent Hermite scratch.
 * A null minimum_bounds selects the unchanged normal J provider. The minimum
 * of every AO-pair Schwarz bound permits MD only for uniformly accepted shell
 * quartets; partially screened quartets keep the exact original AO mask.
 */
struct MdJView {
  MdJPair* pairs{};
  MdJPrimitive* primitives{};
  const std::int64_t *shell_offsets{}, *pair_offsets{}, *primitive_offsets{};
  std::uint32_t* ordered_primitives{};
  const std::uint16_t* transform_slots{};
  double *minimum_bounds{}, *maximum_bounds{}, *density_bounds{}, *maximum_bound{}, *transforms{},
      *density{}, *potential{};
  std::uint32_t *active_pairs{}, *active_count{};
  unsigned long long* source_cursor{};
  MdJWorkCounts* work_counts{};
  /** Freeze the schedule with the resident owner; replay never polls policy. */
  bool reciprocal{};
  /** Selected once, after the entire reference-engine workspace is admitted.
   * The original AO-pair mask and residual owner remain authoritative. */
  bool reference_engine{};
  /** Explicitly different screening contract: complete J with a cumulative
   * primitive omission allowance, rather than the legacy public-AO mask. */
  double* primitive_bounds{};
  MdJPrimitiveScreenState* primitive_screen{};
  double primitive_maximum_bound{};
  std::size_t maximum_pair_primitives{};
  MdJHierarchyStorage hierarchy_layout{};
  MdJScreenBound* hierarchy{};
  std::size_t pair_loc_offsets[6]{};
  std::size_t active_pair_count{};
  std::size_t residual_candidate_count{};
  /** Compact angular segments let each source kernel retain a fixed recurrence
   * order instead of inheriting the largest class's register/local footprint. */
  std::uint32_t active_pair_offsets[6]{};
  std::size_t pair_count{}, primitive_count{}, class_offsets[6]{};
};

/** Structural zeros share only the primitive's explicitly weighted zero slot. */
__host__ __device__ inline double md_j_transform(MdJView md, const MdJPrimitive& primitive,
                                                 const MdJPair& pair, unsigned component,
                                                 unsigned derivative) {
  const auto slot = md.transform_slots[pair.transform_map + component * pair.hermites + derivative];
  return md.transforms[primitive.transform_offset + slot];
}

/** Simplex size of all Hermite derivatives through the pair's angular order. */
__host__ __device__ constexpr unsigned md_j_hermite_count(unsigned angular) {
  return (angular + 1) * (angular + 2) * (angular + 3) / 6;
}

cudaError_t prepare_md_j(cudaStream_t stream, DeviceBatch batch, MdJView& md, const double* bounds,
                         double screening);

/** Complete the sorted primitive pair_loc map after preparation-only upload.
 * No allocation, host integral evaluation or replay-time host poll. */
cudaError_t prepare_md_j_engine_layout(cudaStream_t stream, MdJView md);

/** Refresh conservative total-density bounds before the J-only residual source.
 * K never consumes this density-dependent screening state. */
void launch_md_j_density_bounds(cudaStream_t stream, DeviceBatch batch, MdJView md,
                                std::size_t system_begin, std::size_t system_count,
                                bool unrestricted, const double* density, const double* beta);

/** Bounded source-ordered exchange and exact Coulomb.
 * Evaluate each unique shell quartet once, share primitive R/E data across
 * public-AO components, and scatter all distinct ERI orientations in FP64.
 * Nonsymmetric densities and the original per-orientation AO mask are retained.
 * Exchange requests fuse all Coulomb work into that same source traversal;
 * J-only requests compute the partially screened residual before MD projection.
 */
void launch_md_source_jk(cudaStream_t stream, DeviceBatch batch, MdJView md,
                         std::size_t system_begin, std::size_t system_count, bool want_j,
                         bool want_k, bool unrestricted, double screening, const double* bounds,
                         const double* density, const double* beta, double* coulomb,
                         double* alpha_exchange, double* beta_exchange);

/** Allocation-free FP64 density transform, Coulomb contraction and AO projection.
 * Add to the independently computed partially screened J; never touch K or
 * neighboring items. Metadata and geometry remain immutable during replay.
 */
void launch_md_j(cudaStream_t stream, DeviceBatch batch, MdJView md, std::size_t system_begin,
                 std::size_t system_count, bool unrestricted, double screening,
                 const double* density, const double* beta, double* coulomb);

}  // namespace generativeqc::scf::cuda_execution
