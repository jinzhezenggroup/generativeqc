"""Capability-bound restricted-FP64 paged Fock companions.

These are independent FOCK-only plans, never primary KernelSelections. The
manifest declares availability; HF owns target-scoped runtime admission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .capabilities import (
    CAPABILITY_SHARED_RYS_WEIGHTED_Z_FOCK,
    CAPABILITY_SHARED_RYS_XY_FOCK,
)
from .fused_schedule import FusedShellPlan, build_fused_shell_plan
from .ir import KernelConsumer

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile
    from .production_selection import KernelSelection


@dataclass(frozen=True, slots=True)
class SharedRysXyCompanion:
    """A capability-bound companion; runtime qualification remains separate."""

    incumbent: KernelSelection
    plan: FusedShellPlan


def shared_rys_xy_companions(
    profile: ResolvedProductionProfile,
    *,
    weighted_z: bool = False,
) -> tuple[SharedRysXyCompanion, ...]:
    """Keep the fixed mapping separate from runtime and weighted-Z admission.

    Unweighted x/y follows the manifest capability and validated target plan,
    not an architecture whitelist. Native HF owns throughput-class defaults.
    The independent weighted-Z qualification remains target-specific.
    """
    architecture = profile.target.architecture
    if weighted_z and architecture != "sm_90":
        return ()
    result = []
    capability = (
        CAPABILITY_SHARED_RYS_WEIGHTED_Z_FOCK
        if weighted_z
        else CAPABILITY_SHARED_RYS_XY_FOCK
    )
    for item in profile.selections:
        if (
            item.spec.name not in ("dppp", "pppp")
            or KernelConsumer.FOCK not in item.consumers
            or not item.has_capability(capability)
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
            schedule=replace(
                plan.schedule,
                fock_shared_rys_xy=True,
                fock_shared_rys_weighted_z=weighted_z,
            ),
        )
        expected = 192 if item.spec.name == "dppp" else 96
        if plan.block_threads != expected:
            raise ValueError("shared Rys x/y companion mapping changed")
        result.append(SharedRysXyCompanion(item, plan))
    return tuple(result)
