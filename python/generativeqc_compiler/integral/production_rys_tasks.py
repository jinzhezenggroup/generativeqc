"""Task-parallel Rys-K inventory and explicitly qualified target preferences."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import (
    CAPABILITY_K_BLOCK_FOCK,
    CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
    CAPABILITY_STREAMING_FOCK,
)
from .cuda_schedule import ScheduleKind, schedule_candidates
from .ir import KernelConsumer, build_integral_ir
from .production_selection import KernelSelection
from .rys_task import task_parallel_rys_eligible

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile


def direct_rys_task_candidates(
    profile: ResolvedProductionProfile,
) -> tuple[KernelSelection, ...]:
    """Reuse exact K admission with a bounded, lane-local value producer.

    Inventory intersects existing compiled streaming consumers. Higher orders,
    unavailable classes and portable builds retain their exact incumbent; no
    molecule, method or runtime density enters source selection.
    """
    candidates = []
    for incumbent in profile.selections:
        if (
            KernelConsumer.FOCK not in incumbent.consumers
            or not incumbent.has_capability(CAPABILITY_STREAMING_FOCK)
        ):
            continue
        integral = build_integral_ir(
            incumbent.spec,
            (KernelConsumer.FOCK,),
            recurrence=f"rys{sum(incumbent.spec.angular) // 2 + 1}",
        )
        if not task_parallel_rys_eligible(integral):
            continue
        schedule = next(
            (
                item
                for item in schedule_candidates(integral, profile.target)
                if item.kind == ScheduleKind.PACKED_TASKS
            ),
            None,
        )
        if schedule is None:
            continue
        candidates.append(
            KernelSelection(
                architecture=profile.target.architecture,
                profile=profile.profile,
                spec=incumbent.spec,
                consumers=(KernelConsumer.FOCK,),
                recurrence=integral.recurrence,
                integral=integral,
                schedule=schedule,
                capabilities=frozenset(
                    (
                        CAPABILITY_STREAMING_FOCK,
                        CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
                        CAPABILITY_K_BLOCK_FOCK,
                    )
                ),
                fock_route="streaming",
                tuned=False,
            )
        )
    return tuple(candidates)


def preferred_rys_task_candidates(
    profile: ResolvedProductionProfile,
) -> tuple[KernelSelection, ...]:
    """Restrict default dispatch to the measured RTX 5090 profile and classes.

    This is target scheduling metadata, not method or molecule policy. Other
    architectures and portable profiles need their own complete-endpoint
    qualification before promotion; explicit experiments keep full capability.
    """
    if profile.target.architecture != "sm_120" or profile.profile != "sm_120":
        return ()
    preferred = {"psps", "ppps", "dsss", "dpss", "dsps", "ddss", "dsds", "dpps", "dspp"}
    return tuple(
        candidate
        for candidate in direct_rys_task_candidates(profile)
        if candidate.spec.name in preferred
    )
