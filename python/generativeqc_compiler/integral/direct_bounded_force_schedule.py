"""Experimental homogeneous passes over the existing bounded force domain.

This is an execution experiment, not an integral-count or asymptotic reduction.
Each pass repeats the screening traversal but owns a disjoint angular range.
Fixed orders bypass the retained noinline, all-order derivative dispatcher; the
scientific contraction and the low/high-order fallbacks remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuda_schedule import ScheduleIR, ScheduleKind


@dataclass(frozen=True, slots=True)
class BoundedForcePass:
    """One disjoint range and its existing thread/warp contraction mapping."""

    tag: int
    minimum_order: int
    maximum_order: int
    schedule: ScheduleIR
    workers_per_cta: int

    @property
    def cta_threads(self) -> int:
        """Native bounded CTA replicates independent warp workers, not shared math."""
        return self.schedule.block_threads * self.workers_per_cta

    def accepts(self, angular_order: int) -> bool:
        """Whether the pass owns a screened shell quartet of this total order."""
        return self.minimum_order <= angular_order <= self.maximum_order


def homogeneous_bounded_force_passes() -> tuple[BoundedForcePass, ...]:
    """Partition supported through-f derivatives, retaining high-order science.

    The five screening scans deliberately avoid an O(shell**4) resident task
    list. Register/stack benefits and any endpoint gain require device evidence;
    ScheduleIR records mappings, not calibrated throughput or FLOPs.
    """
    return tuple(
        BoundedForcePass(
            tag=minimum,
            minimum_order=minimum,
            maximum_order=maximum,
            schedule=ScheduleIR(
                kind=(
                    ScheduleKind.THREAD_TASKS
                    if minimum == 0
                    else ScheduleKind.SHELL_TASK
                ),
                block_threads=256 if minimum == 0 else 32,
                component_tile=256,
                tasks_per_warp=32 if minimum == 0 else 1,
                shared_coulomb=False,
                minimum_blocks_per_sm=1,
            ),
            workers_per_cta=1 if minimum == 0 else 8,
        )
        for minimum, maximum in ((0, 3), (4, 4), (5, 5), (6, 6), (7, 12))
    )


def emit_direct_bounded_force_schedule_header() -> str:
    """Generate host pass order and device membership from one compiler policy."""
    passes = homogeneous_bounded_force_passes()
    tags = ", ".join(str(item.tag) for item in passes)
    applications = " ".join(f"APPLY({item.tag});" for item in passes)
    ranges = " ||\n           ".join(
        f"(Pass == {item.tag} && order >= {item.minimum_order}U && "
        f"order <= {item.maximum_order}U)"
        for item in passes
    )
    supported = " || ".join(f"Pass == {item.tag}" for item in passes)
    return f"""#pragma once

#include <array>

#define GENERATIVEQC_FOR_EACH_HOMOGENEOUS_FORCE_PASS(APPLY) {applications}

// Generated execution policy; recurrence and density weights remain native consumers.
namespace generativeqc::scf::cuda_execution {{

inline constexpr std::array<int, {len(passes)}> kHomogeneousBoundedForcePasses{{{tags}}};
inline constexpr unsigned kHomogeneousBoundedForceThreads = {passes[0].cta_threads}U;

template <int Pass>
struct BoundedForcePassPolicy {{
  static_assert(Pass == -1 || {supported});
  static constexpr bool scalar = Pass == -1 || Pass == 0;
  static constexpr bool warp = Pass != 0;
  static constexpr bool fixed = Pass >= 4 && Pass <= 6;

#ifdef __CUDACC__
  __host__ __device__
#endif
  static constexpr bool accepts(unsigned order) {{
    return Pass == -1 || {ranges};
  }}
}};

}}  // namespace generativeqc::scf::cuda_execution
"""
