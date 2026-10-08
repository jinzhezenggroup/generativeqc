#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

inline constexpr std::size_t kMdJResidentCap = 128U << 20;
inline constexpr double kMdJDensityErrorCap = 1e-12;

struct MdSourceTask {
  std::uint32_t bra, ket;
};

inline constexpr std::size_t kMdSourceFixedBytes =
    sizeof(double) + sizeof(std::uint32_t) + sizeof(unsigned long long);

/** One unordered shell pair; primitive products retain their ordered contraction. */
struct MdJPair {
  std::int32_t system, first, second;
  std::int32_t first_ao, second_ao, first_count, second_count;
  std::uint32_t angular, hermites, primitive_begin, primitive_end;
};

/** Geometry is filled on the owning stream, never by a host integral oracle. */
struct MdJPrimitive {
  std::uint32_t pair;
  std::int64_t first, second;
  std::size_t transform_offset, hermite_offset;
  double exponent;
  Vec3<double> product;
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
  double *minimum_bounds{}, *maximum_bounds{}, *density_bounds{}, *maximum_bound{}, *transforms{},
      *density{}, *potential{};
  std::uint32_t *active_pairs{}, *active_count{};
  unsigned long long* source_cursor{};
  std::size_t active_pair_count{};
  std::size_t residual_candidate_count{};
  /** Compact angular segments let each source kernel retain a fixed recurrence
   * order instead of inheriting the largest class's register/local footprint. */
  std::uint32_t active_pair_offsets[6]{};
  std::size_t pair_count{}, primitive_count{}, class_offsets[6]{};
};

/** Simplex size of all Hermite derivatives through the pair's angular order. */
__host__ __device__ constexpr unsigned md_j_hermite_count(unsigned angular) {
  return (angular + 1) * (angular + 2) * (angular + 3) / 6;
}

cudaError_t prepare_md_j(cudaStream_t stream, DeviceBatch batch, MdJView& md, const double* bounds,
                         double screening);

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
