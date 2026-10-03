"""Bounded AES2 atom scheduling with ordered per-atom peer accumulation.

Large systems validate their input and pair cache once, then distribute the
independent target atoms across blocks. Lanes evaluate one peer chunk into
bounded shared storage; component owners consume it in the original peer order.
Small systems retain the fused single-block validation/evaluation route.
"""

from __future__ import annotations

from generativeqc_compiler.common.schedule import ScheduleTopology


def gfn2_aes2_atom_schedule() -> ScheduleTopology:
    """One warp evaluates a target atom and preserves each ordered component sum."""
    return ScheduleTopology(
        tiles=(256,),
        workgroup_threads=32,
        fusion="aes2-atom-peer",
        materialization="bounded-peer-contributions",
        reduction="ordered-peer",
    )


def emit_gfn2_aes2_schedule() -> str:
    """Emit the host choice; actual ragged atom extents remain device inputs."""
    schedule = gfn2_aes2_atom_schedule()
    return f"""
inline constexpr unsigned gfn2_aes2_peer_threads = {schedule.workgroup_threads}U;
inline unsigned gfn2_aes2_atom_tiles(
    std::int64_t atoms, std::int64_t systems) noexcept {{
  if (atoms <= 0 || systems <= 0) return 1U;
  const auto mean = atoms / systems + (atoms % systems != 0);
  if (mean <= gfn2_aes2_peer_threads) return 1U;
  return mean < {schedule.tiles[0]} ? static_cast<unsigned>(mean) : {schedule.tiles[0]}U;
}}
"""
