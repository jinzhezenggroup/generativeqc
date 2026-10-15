#pragma once

#include <cstddef>
#include <cstdint>

namespace generativeqc::dft {

/** Immutable basis/quadrature and accepted spin densities from one live KS
 * owner. The basis is AoBasis's normalized 16-double-record representation;
 * points are [point,xyz], weights [point], density [spin,AO,AO]. The exact
 * scalar-validated canonical frame must already be prepared. No matrices are
 * exported or copied. Reborrow before execution/publication under the original
 * token and exclusive batch lifetime; copied labels alone prove nothing.
 * This lease authenticates storage, not a particular response functional.
 */
struct CudaKsResidentGridBinding {
  int device_id{-1};
  const double *basis{}, *points{}, *weights{}, *density{};
  std::size_t atoms{}, primitives{}, nao{}, point_count{};
  unsigned spins{};
  void* stream{};
  std::uint64_t owner{}, solve_epoch{}, generation{};
  /** Optional exact pre-partition measure/policy from a retained native CUDA
   * quadrature owner. Host-built grids remain valid fixed-grid leases, but do
   * not acquire moving-grid capability by reconstructing atomic measures. */
  const double* atomic_weights{};
  std::size_t points_per_atom{};
  unsigned partition_iterations{};
  double coincident_tolerance{};
  std::uint64_t quadrature_owner{};

  bool operator==(const CudaKsResidentGridBinding&) const = default;

  explicit operator bool() const noexcept {
    return device_id >= 0 && basis && points && weights && density && atoms && primitives && nao &&
           point_count && (spins == 1 || spins == 2) && stream && owner && solve_epoch &&
           generation;
  }
};

}  // namespace generativeqc::dft
