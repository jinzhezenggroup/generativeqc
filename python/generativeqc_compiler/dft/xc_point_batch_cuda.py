"""Emit bounded physical point-domain planning, independent of XC algebra.

Only AO panels survive across tiles: density-product scratch is reused before
the point submission and again by the original ordered potential contractions.
Selected AO domains are never merged. This is a scheduling-only ablation.
"""


def emit_native_xc_point_batch_plan() -> str:
    """Return a pure host selector with explicit residency and overflow bounds."""
    return r"""
CudaXcPointBatchPlan prepare_point_batch_plan(const CudaXcLayout& layout,
                                              const std::vector<std::size_t>& offsets,
                                              std::size_t requested_tiles,
                                              std::size_t device_budget) {
  if (!layout.npoint || !layout.tile_points || !layout.nao)
    throw std::invalid_argument("invalid XC point batch domain");
  const auto tile_count = 1 + (layout.npoint - 1) / layout.tile_points;
  if (layout.local_ao && (offsets.size() != tile_count + 1 || offsets.front() != 0))
    throw std::invalid_argument("invalid XC point batch maps");
  if (layout.local_ao)
    for (std::size_t tile = 0; tile < tile_count; ++tile)
      if (offsets[tile + 1] < offsets[tile] ||
          offsets[tile + 1] - offsets[tile] > layout.nao)
        throw std::invalid_argument("invalid XC point batch AO count");
  if (layout.response || layout.ao_precision != CudaXcAoPrecision::Fp64 ||
      requested_tiles < 2 || tile_count < 2 || !device_budget)
    return {};
  // All products below are bounded by the explicit byte allowance before
  // multiplication. The virtual point domain must also fit the launch ABI.
  const auto capacity = device_budget / sizeof(double);
  auto tiles = std::min({requested_tiles, tile_count,
                        std::size_t{std::numeric_limits<int>::max()} / layout.tile_points});
  for (; tiles > 1; tiles = (tiles + 1) / 2) {
    const auto points = tiles * layout.tile_points;
    const auto channels = layout.spins * layout.feature_terms;
    if (points > capacity / (2 * channels + 3)) continue;
    const auto features = points * channels;
    const auto totals = 3 * points;
    const auto available_ao = capacity - 2 * features - totals;
    std::size_t max_ao = 0;
    bool fits = true;
    for (std::size_t first = 0; first < tile_count && fits; first += tiles) {
      std::size_t ao = 0;
      for (std::size_t tile = first; tile < std::min(tile_count, first + tiles); ++tile) {
        const auto count = std::min(layout.tile_points,
                                    layout.npoint - tile * layout.tile_points);
        const auto active = layout.local_ao ? offsets[tile + 1] - offsets[tile] : layout.nao;
        if (active && count > (available_ao - ao) / layout.jets / active) {
          fits = false;
          break;
        }
        ao += count * active * layout.jets;
      }
      max_ao = std::max(max_ao, ao);
    }
    if (fits)
      return {tiles, max_ao, features, totals,
              (max_ao + 2 * features + totals) * sizeof(double)};
  }
  return {};
}
"""
