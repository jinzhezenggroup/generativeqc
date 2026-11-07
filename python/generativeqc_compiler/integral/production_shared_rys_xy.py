"""Explicit SM90 restricted-FP64 paged Fock companions.

These are independent FOCK-only plans, never primary KernelSelections. The
manifest declares availability; HF owns the default-off runtime admission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .capabilities import CAPABILITY_SHARED_RYS_XY_FOCK
from .fused_schedule import FusedShellPlan, build_fused_shell_plan
from .ir import KernelConsumer

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile
    from .production_selection import KernelSelection


@dataclass(frozen=True, slots=True)
class SharedRysXyCompanion:
    """A capability-bound companion with the independently qualified mapping."""

    incumbent: KernelSelection
    plan: FusedShellPlan


def shared_rys_xy_companions(
    profile: ResolvedProductionProfile,
) -> tuple[SharedRysXyCompanion, ...]:
    """Do not transfer this two-class capability to other architectures."""
    if profile.target.architecture != "sm_90":
        return ()
    result = []
    for item in profile.selections:
        if (
            item.spec.name not in ("dppp", "pppp")
            or KernelConsumer.FOCK not in item.consumers
            or not item.has_capability(CAPABILITY_SHARED_RYS_XY_FOCK)
        ):
            continue
        plan = build_fused_shell_plan(
            item.spec,
            consumers=(KernelConsumer.FOCK,),
            recurrence="rys3",
            target=profile.target,
        )
        plan = build_fused_shell_plan(
            item.spec,
            integral=plan.kernel.integral,
            target=profile.target,
            schedule=replace(plan.schedule, fock_shared_rys_xy=True),
        )
        expected = 192 if item.spec.name == "dppp" else 96
        if plan.block_threads != expected:
            raise ValueError("shared Rys x/y companion mapping changed")
        result.append(SharedRysXyCompanion(item, plan))
    return tuple(result)
