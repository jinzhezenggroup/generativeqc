"""Compile optional value-only Rys candidates beside incumbent Fock kernels.

Eligibility follows the value IR and implemented CUDA decoder. These candidates
are not measured production preferences: each prepared J/K consumer explicitly
opts in, and classes absent from this inventory retain their incumbent lowering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import CAPABILITY_STREAMING_FOCK, query_integral_capability
from .cuda_schedule import ScheduleKind, schedule_candidates
from .fused_schedule import build_fused_shell_plan
from .ir import KernelConsumer, build_integral_ir
from .production_selection import KernelSelection

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile


def direct_rys_value_candidates(
    profile: ResolvedProductionProfile,
) -> tuple[KernelSelection, ...]:
    """Return legal alternatives only for compiled exact streaming consumers.

    This intersects the existing consumer inventory, rather than extending the
    runtime coverage contract or using a molecule/method shell-class allowlist.
    Root counts belong to derivative-free IR; unsupported roots and component
    mappings are explicit omissions, never recurrence substitutions.
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
        if not query_integral_capability(integral).supported:
            continue
        schedules = tuple(
            item
            for item in schedule_candidates(integral, profile.target)
            if item.kind == ScheduleKind.COMPONENT_LANES
        )
        if not schedules:
            continue
        plan = build_fused_shell_plan(
            incumbent.spec,
            integral=integral,
            target=profile.target,
            schedule=schedules[0],
        )
        candidates.append(
            KernelSelection(
                architecture=profile.target.architecture,
                profile=profile.profile,
                spec=incumbent.spec,
                consumers=(KernelConsumer.FOCK,),
                recurrence=integral.recurrence,
                integral=integral,
                schedule=plan.schedule,
                capabilities=frozenset((CAPABILITY_STREAMING_FOCK,)),
                fock_route="streaming",
                tuned=False,
            )
        )
    return tuple(candidates)
