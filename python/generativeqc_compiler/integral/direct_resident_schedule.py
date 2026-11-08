"""Compiler ownership for retained persistent Direct-HF schedules.

The 128-thread/64-bra-pair baseline is accepted performance evidence, not CUDA
semantics.  Keep it in compiler policy so native runtime code consumes one
generated schedule definition while #597 can later replace the baseline through
target/workload profile selection without another handwritten CUDA constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cuda_schedule import ScheduleIR, ScheduleKind
from .shell_spec import PSSS_SPEC


@dataclass(frozen=True, slots=True)
class ResidentPsssScheduleProfile:
    """Accepted resident-PSSS execution profile."""

    schedule: ScheduleIR
    maximum_bra_primitive_pairs: int


@dataclass(frozen=True, slots=True)
class DirectPersistentScheduleProfile:
    """Persistent queue domains, not integral angular-momentum support limits.

    Defaults retain the qualified native schedules. Smaller domains send the
    remaining orders to their fixed-grid workers; extending the domains needs
    independent endpoint evidence for the higher-register-footprint workers.
    Orders zero through two retain their specialized shell-fused dispatch.
    """

    force_angular_order_count: int = 7
    fock_angular_order_count: int = 6

    def __post_init__(self) -> None:
        for name, maximum in (
            ("force_angular_order_count", 7),
            ("fock_angular_order_count", 6),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 3 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [3, {maximum}]")


def resident_psss_schedule_profile() -> ResidentPsssScheduleProfile:
    """Return the accepted baseline through the shared CUDA schedule vocabulary."""

    return ResidentPsssScheduleProfile(
        schedule=ScheduleIR(
            kind=ScheduleKind.THREAD_TASKS,
            block_threads=128,
            component_tile=PSSS_SPEC.component_count,
            tasks_per_warp=32,
            shared_coulomb=False,
            minimum_blocks_per_sm=4,
            warp_size=32,
        ),
        maximum_bra_primitive_pairs=64,
    )


def emit_direct_resident_psss_schedule_header(
    persistent: DirectPersistentScheduleProfile | None = None,
) -> str:
    """Emit the native ABI constants from compiler-owned schedule policy."""

    profile = resident_psss_schedule_profile()
    if persistent is None:
        persistent = DirectPersistentScheduleProfile()
    schedule = profile.schedule
    return f"""#pragma once

#include <cstddef>

// Generated from compiler-owned Direct-HF schedule policy. Do not edit this
// build artifact; change direct_resident_schedule.py or the selected profile.
namespace generativeqc::scf::cuda_execution {{

inline constexpr unsigned kResidentPsssThreads = {schedule.block_threads}U;
inline constexpr unsigned kResidentPsssMinimumBlocksPerSm =
    {schedule.minimum_blocks_per_sm}U;
inline constexpr std::size_t kResidentPsssMaximumBraPrimitivePairs =
    {profile.maximum_bra_primitive_pairs}U;
inline constexpr unsigned kPersistentForceAngularOrderCount =
    {persistent.force_angular_order_count}U;
inline constexpr unsigned kPersistentFockAngularOrderCount =
    {persistent.fock_angular_order_count}U;

static_assert(kResidentPsssThreads % {schedule.warp_size}U == 0U);

}}  // namespace generativeqc::scf::cuda_execution
"""
