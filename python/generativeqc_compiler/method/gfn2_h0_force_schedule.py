"""Bounded shell-pair scheduling for the generated H0/Pulay force consumer.

Each ordered shell pair owns a disjoint AO block. Its AO traversal and scalar
program stay ordered; atom gradients and coordination adjoints retain their
existing FP64 atomic accumulation, whose inter-pair order is unspecified.
"""

from __future__ import annotations

from generativeqc_compiler.common.schedule import ScheduleTopology


def gfn2_h0_force_schedule() -> ScheduleTopology:
    """Distribute pairs without extra storage or repeated admission scans."""
    return ScheduleTopology(
        tiles=(256,),
        workgroup_threads=128,
        fusion="h0-pulay-shell-pair",
        materialization="existing-force-scratch",
        reduction="fp64-atomic",
    )


def emit_gfn2_h0_force_schedule() -> str:
    """Choose a bounded grid from host totals; device offsets define ragged work.

    Quotient/remainder ceilings avoid overflow. Small means retain one block,
    and every system grid-strides its own extent even when batch sizes differ.
    """
    schedule = gfn2_h0_force_schedule()
    return f"""
inline constexpr unsigned gfn2_h0_force_threads = {schedule.workgroup_threads}U;
inline unsigned gfn2_h0_force_pair_tiles(
    std::int64_t pairs, std::int64_t systems) noexcept {{
  if (pairs <= 0 || systems <= 0) return 1U;
  const auto mean = pairs / systems + (pairs % systems != 0);
  const auto tiles = mean / gfn2_h0_force_threads +
                     (mean % gfn2_h0_force_threads != 0);
  return tiles < {schedule.tiles[0]} ? static_cast<unsigned>(tiles) : {schedule.tiles[0]}U;
}}
"""
