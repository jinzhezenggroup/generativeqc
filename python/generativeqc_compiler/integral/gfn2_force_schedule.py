"""CUDA admission scheduling for complete GFN2 S/D/Q force adjoints.

Validation and gradient-seed initialization retain one block per system. The
larger width shares its ten matrix-component scans across more lanes without
repeating metadata scans, introducing launches or changing force reductions.
"""

from __future__ import annotations

from generativeqc_compiler.common.schedule import ScheduleTopology


def gfn2_force_preflight_schedule() -> ScheduleTopology:
    """Use a portable block width with the existing per-system shared state."""
    return ScheduleTopology(
        workgroup_threads=256,
        fusion="integral-force-preflight",
        materialization="existing-gradient-scratch",
        reduction="finite-input-validation",
    )


def emit_gfn2_force_preflight_schedule() -> str:
    """Select from host totals without downloading individual matrix extents."""
    schedule = gfn2_force_preflight_schedule()
    return f"""
inline unsigned gfn2_force_preflight_threads(
    std::int64_t matrices, std::int64_t systems) noexcept {{
  if (matrices <= 0 || systems <= 0) return 64U;
  const auto mean = matrices / systems + (matrices % systems != 0);
  return mean <= 4096 ? 64U : {schedule.workgroup_threads}U;
}}
"""
