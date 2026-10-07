"""Compile optional restricted exact-K block-contraction alternatives."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import (
    CAPABILITY_K_BLOCK_FOCK,
    CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
    CAPABILITY_STREAMING_FOCK,
)
from .cuda_schedule import ScheduleKind
from .ir import KernelConsumer
from .k_block import packed_restricted_k_block_eligible
from .production_selection import KernelSelection, _selection_integral
from .specialize import specialize_fock_integral

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile


def direct_k_block_candidates(
    profile: ResolvedProductionProfile,
) -> tuple[KernelSelection, ...]:
    """Return bounded packed K alternatives with the incumbent lane storage policy."""

    candidates = []
    for incumbent in profile.selections:
        if (
            KernelConsumer.FOCK not in incumbent.consumers
            or not incumbent.has_capability(CAPABILITY_STREAMING_FOCK)
        ):
            continue
        schedule = incumbent.fock_schedule or incumbent.schedule
        if (
            schedule.kind != ScheduleKind.PACKED_TASKS
            or not packed_restricted_k_block_eligible(incumbent.spec)
        ):
            continue
        integral = specialize_fock_integral(_selection_integral(incumbent))
        capabilities = {CAPABILITY_STREAMING_FOCK, CAPABILITY_K_BLOCK_FOCK}
        # Storage placement remains valid for the specialized value producer.
        # Do not inherit unrelated consumer/precision capabilities with it.
        if incumbent.has_capability(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK):
            capabilities.add(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK)
        candidates.append(
            KernelSelection(
                architecture=profile.target.architecture,
                profile=profile.profile,
                spec=incumbent.spec,
                consumers=(KernelConsumer.FOCK,),
                recurrence=integral.recurrence,
                integral=integral,
                schedule=schedule,
                capabilities=frozenset(capabilities),
                fock_route="streaming",
                tuned=False,
            )
        )
    return tuple(candidates)
