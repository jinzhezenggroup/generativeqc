#pragma once

#include <cmath>
#include <cstdint>

namespace generativeqc::scf::cuda_execution {

inline constexpr unsigned kMdSourceTileWidth = 8;

struct MdSourceTile {
  std::uint32_t bra, ket;
};

/** Count bounded shell-pair tiles, not primitive quartets or admitted tasks. */
__host__ __device__ constexpr std::uint64_t md_source_tile_count(std::uint32_t bra_count,
                                                                 std::uint32_t ket_count,
                                                                 bool triangular) {
  const auto bra_tiles = (std::uint64_t{bra_count} + kMdSourceTileWidth - 1) / kMdSourceTileWidth;
  const auto ket_tiles = (std::uint64_t{ket_count} + kMdSourceTileWidth - 1) / kMdSourceTileWidth;
  return triangular ? bra_tiles * (bra_tiles + 1) / 2 : bra_tiles * ket_tiles;
}

/** Requires index < md_source_tile_count and therefore a nonempty tile domain.
 * The diagonal tile still applies bra >= ket to its individual candidates.
 * Correct the floating inverse so large triangular indices retain exact coverage. */
__host__ __device__ inline MdSourceTile md_source_tile(std::uint64_t index, std::uint32_t ket_count,
                                                       bool triangular) {
  std::uint64_t bra{}, ket{};
  if (triangular) {
    bra = static_cast<std::uint64_t>((sqrt(8.0 * index + 1.0) - 1.0) * 0.5);
    while (bra * (bra + 1) / 2 > index) --bra;
    while ((bra + 1) * (bra + 2) / 2 <= index) ++bra;
    ket = index - bra * (bra + 1) / 2;
  } else {
    const auto ket_tiles = (std::uint64_t{ket_count} + kMdSourceTileWidth - 1) / kMdSourceTileWidth;
    bra = index / ket_tiles;
    ket = index % ket_tiles;
  }
  return {static_cast<std::uint32_t>(bra * kMdSourceTileWidth),
          static_cast<std::uint32_t>(ket * kMdSourceTileWidth)};
}

/** Products are rounded OUTWARD by the caller: maximum/density upward,
 * minimum downward. A rejected tile must contain no incumbent residual task.
 * Zero density is explicit because infinity times zero need not be finite. */
__host__ __device__ inline bool md_source_tile_rejected(
    double maximum_product, double minimum_product, double bra_density_product,
    double ket_density_product, bool zero_density, double screening, double density_allowance) {
  return maximum_product < screening || minimum_product >= screening || zero_density ||
         (bra_density_product < density_allowance && ket_density_product < density_allowance);
}

}  // namespace generativeqc::scf::cuda_execution
