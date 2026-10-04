"""Plan disjoint screened pages and exact-class Direct force consumers.

This schedule changes traversal/storage, not integral mathematics. A page owns
disjoint block products, retains each accepted classification across prefix and
scatter, and invokes the existing derivative consumers without another shell
screen. AO-level screening remains inside those consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuda_schedule import ScheduleIR, ScheduleKind
from .shell_spec import FUSED_SHELL_SPECS, ShellClassSpec

BLOCK_PAIR_CAPACITY = 32 * 32
MAXIMUM_PAGE_BLOCKS = 4096
TILE_BYTES = 12


@dataclass(frozen=True)
class ForcePageConsumer:
    """An exact class with scalar or warp-local recurrence ownership."""

    shell_class: int
    spec: ShellClassSpec
    schedule: ScheduleIR


def force_page_consumers() -> tuple[ForcePageConsumer, ...]:
    """Reuse canonical class IDs and the existing schedule vocabulary."""
    return tuple(
        ForcePageConsumer(
            shell_class,
            spec,
            ScheduleIR(
                kind=ScheduleKind.THREAD_TASKS
                if sum(spec.angular) <= 3
                else ScheduleKind.SHELL_TASK,
                block_threads=256,
                component_tile=spec.component_count,
                tasks_per_warp=32 if sum(spec.angular) <= 3 else 1,
                shared_coulomb=False,
                warp_size=32,
            ),
        )
        for shell_class, spec in enumerate(FUSED_SHELL_SPECS)
    )


def emit_direct_force_page_header() -> str:
    """Emit budget admission and host dispatch metadata; never probe a device."""
    consumers = force_page_consumers()
    cases = " \\\n".join(
        f"  APPLY({consumer.shell_class}, {sum(consumer.spec.angular)}, "
        f"{consumer.schedule.block_threads}, {consumer.schedule.tasks_per_warp})"
        for consumer in consumers
    )
    return f"""#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

// Compiler-owned bounded page policy. No recurrence or precision changes.
namespace generativeqc::scf::cuda_execution {{
inline constexpr std::size_t kForcePageBlockCandidates = {BLOCK_PAIR_CAPACITY};
inline constexpr unsigned kForcePageThreads = 256;
inline constexpr std::size_t kForcePageClassCount = {len(consumers)};

struct ForcePageLayout {{
  std::size_t blocks{{}}, candidates{{}}, bytes{{}};
  std::size_t input{{}}, tasks{{}}, classes{{}}, counts{{}}, offsets{{}}, writes{{}}, heads{{}};
}};

/** Admit one reusable page against the owner's remaining budget, or decline.
 * The fixed cap bounds storage independently of molecular quartet capacity.
 * Reducing a budget adds disjoint pages, never another full-domain scan.
 */
inline ForcePageLayout plan_force_page(std::size_t products, std::size_t budget) {{
  for (auto blocks = std::min<std::size_t>(products, {MAXIMUM_PAGE_BLOCKS}); blocks; blocks /= 2) {{
    ForcePageLayout layout;
    layout.blocks = blocks;
    layout.candidates = blocks * kForcePageBlockCandidates;
    layout.tasks = {TILE_BYTES} * layout.candidates;
    layout.classes = 2 * layout.tasks;
    layout.counts = layout.classes + layout.candidates;
    layout.offsets = layout.counts + 4 * kForcePageClassCount;
    layout.writes = layout.offsets + 4 * (kForcePageClassCount + 1);
    layout.heads = layout.writes + 4 * kForcePageClassCount;
    layout.bytes = layout.heads + 4 * kForcePageClassCount;
    if (layout.bytes <= budget) return layout;
  }}
  return {{}};
}}
}}  // namespace generativeqc::scf::cuda_execution

#define GENERATIVEQC_FOR_EACH_FORCE_PAGE_CLASS(APPLY) \\
{cases}
"""
