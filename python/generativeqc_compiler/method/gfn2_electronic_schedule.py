"""Bounded CUDA work distribution for independent GFN2 electronic pairs.

Pair arithmetic has no cross-pair reduction: one triangular owner writes
both symmetric entries. Density retains its ordered orbital loop per pair. Splitting a system across blocks therefore preserves
its scalar program and summation order. Validation and publication remain
separate native launches, so a failing tile cannot publish a partial matrix.
"""

from __future__ import annotations

from generativeqc_compiler.common.schedule import ScheduleTopology


def gfn2_electronic_schedule() -> ScheduleTopology:
    """Describe the allocation-free, bounded matrix schedule.

    Native host descriptors expose total matrix size, not each ragged extent.
    Use the mean extent to choose a grid width, then stride over each system's
    actual device extent. A skewed batch may need more passes, but every matrix
    entry has exactly one owner regardless of the width or batch imbalance.
    """
    return ScheduleTopology(
        tiles=(128,),
        workgroup_threads=256,
        fusion="electronic-pair",
        materialization="matrix-scratch",
        reduction="none",
    )


def emit_gfn2_electronic_schedule() -> str:
    """Emit host launch policy; the one-block case is the bounded fallback."""
    schedule = gfn2_electronic_schedule()
    return f"""
inline constexpr unsigned gfn2_electronic_threads = {schedule.workgroup_threads}U;
inline constexpr unsigned gfn2_electronic_maximum_tiles = {schedule.tiles[0]}U;

// Call only after native admission has checked positive descriptor extents.
// Quotient/remainder ceilings avoid overflow near the signed 64-bit limit.
inline unsigned gfn2_electronic_matrix_tiles(
    std::int64_t matrix_elements, std::int64_t systems) noexcept {{
  if (matrix_elements <= 0 || systems <= 0) return 1U;
  const auto mean = matrix_elements / systems + (matrix_elements % systems != 0);
  const auto tiles = mean / gfn2_electronic_threads +
                     (mean % gfn2_electronic_threads != 0);
  return tiles < gfn2_electronic_maximum_tiles
             ? static_cast<unsigned>(tiles) : gfn2_electronic_maximum_tiles;
}}
"""
