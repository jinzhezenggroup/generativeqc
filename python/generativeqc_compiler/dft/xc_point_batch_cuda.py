"""Emit bounded physical point-domain planning, independent of XC algebra.

The incumbent retains AO panels and reuses density-product scratch. An explicit
compact-contraction candidate additionally retains mapped work/local matrices.
Selected AO domains are never merged, including in the batched candidate.
"""


def emit_native_xc_point_batch_plan() -> str:
    """Return a pure host selector with explicit residency and overflow bounds."""
    return r"""
bool compact_point_batch_admitted(const CudaXcLayout& layout,
                                  const std::vector<std::size_t>& offsets,
                                  std::size_t first, std::size_t end) {
  if (end <= first || end - first < 2) return false;
  for (std::size_t tile = first; tile < end; ++tile) {
    const auto active = layout.local_ao ? offsets[tile + 1] - offsets[tile] : layout.nao;
    const auto count = std::min(layout.tile_points, layout.npoint - tile * layout.tile_points);
    if (active && (active < 32 || active > 128 || count < 32)) return false;
  }
  return true;
}

CudaXcPointBatchPlan prepare_point_batch_plan(const CudaXcLayout& layout,
                                              const std::vector<std::size_t>& offsets,
                                              std::size_t requested_tiles,
                                              std::size_t device_budget,
                                              bool compact = false) {
  if (!layout.npoint || !layout.tile_points || !layout.nao ||
      (layout.spins != 1 && layout.spins != 2) ||
      (layout.jets != 1 && layout.jets != 4) ||
      (layout.work_jets != 1 && layout.work_jets != 4) ||
      (layout.feature_terms != 1 && layout.feature_terms != 4 && layout.feature_terms != 5))
    throw std::invalid_argument("invalid XC point batch domain");
  const auto tile_count = 1 + (layout.npoint - 1) / layout.tile_points;
  if (layout.local_ao && (offsets.empty() || tile_count == std::numeric_limits<std::size_t>::max() ||
                          offsets.size() != tile_count + 1 || offsets.front() != 0))
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
  if (compact && (tile_count > device_budget / sizeof(CudaXcCompactTile) ||
                  layout.tile_points > std::size_t{65535} * 8)) compact = false;
  const auto descriptor_bytes = compact ? tile_count * sizeof(CudaXcCompactTile) : 0;
  const auto numeric_capacity = (device_budget - descriptor_bytes) / sizeof(double);
  auto tiles = std::min({requested_tiles, tile_count,
                        std::size_t{std::numeric_limits<int>::max()} / layout.tile_points});
  if (compact) tiles = std::min(tiles, std::size_t{65535} / layout.spins / layout.work_jets);
  for (; tiles > 1; tiles = (tiles + 1) / 2) {
    const auto points = tiles * layout.tile_points;
    const auto channels = layout.spins * layout.feature_terms;
    if (points > numeric_capacity / (2 * channels + 3)) continue;
    const auto features = points * channels;
    const auto totals = 3 * points;
    const auto available_ao = numeric_capacity - 2 * features - totals;
    std::size_t max_ao = 0, max_work = 0, max_potential = 0;
    std::size_t compact_groups = 0, compact_tiles = 0, compact_nonempty_tiles = 0;
    bool fits = true;
    for (std::size_t first = 0; first < tile_count && fits; first += tiles) {
      std::size_t ao = 0, work = 0, potential = 0;
      const auto end = std::min(tile_count, first + tiles);
      const bool grouped = compact && compact_point_batch_admitted(layout, offsets, first, end);
      if (grouped) {
        ++compact_groups;
        compact_tiles += end - first;
      }
      for (std::size_t tile = first; tile < std::min(tile_count, first + tiles); ++tile) {
        const auto count = std::min(layout.tile_points,
                                    layout.npoint - tile * layout.tile_points);
        const auto active = layout.local_ao ? offsets[tile + 1] - offsets[tile] : layout.nao;
        if (active && count > (available_ao - ao) / layout.jets / active) {
          fits = false;
          break;
        }
        ao += count * active * layout.jets;
        if (grouped) {
          compact_nonempty_tiles += active != 0;
          const auto left = available_ao - ao;
          if (active && (count > left / active / layout.spins / layout.work_jets ||
                         active > left / active / layout.spins)) {
            fits = false;
            break;
          }
          const auto next_work = count * active * layout.spins * layout.work_jets;
          const auto next_potential = active * active * layout.spins;
          if (work > left || next_work > left - work ||
              potential > left - work - next_work ||
              next_potential > left - work - next_work - potential) {
            fits = false;
            break;
          }
          work += next_work;
          potential += next_potential;
        }
      }
      max_ao = std::max(max_ao, ao);
      max_work = std::max(max_work, work);
      max_potential = std::max(max_potential, potential);
    }
    // Fixed slots must fit the sum of individual maxima, even if their peaks
    // belong to different ragged batches.
    if (fits && max_work <= available_ao - max_ao &&
        max_potential <= available_ao - max_ao - max_work) {
      if (compact && !compact_groups)
        return prepare_point_batch_plan(layout, offsets, requested_tiles, device_budget, false);
      return {tiles, max_ao, features, totals,
              (max_ao + 2 * features + totals + max_work + max_potential) * sizeof(double) +
                  descriptor_bytes,
              max_work, max_potential, descriptor_bytes, compact,
              compact_groups, compact_tiles, compact_nonempty_tiles};
    }
  }
  if (compact) return prepare_point_batch_plan(layout, offsets, requested_tiles, device_budget, false);
  return {};
}
"""
