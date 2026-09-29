"""Prepared-execution identity for compiler-planned stationary gradients."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from generativeqc_compiler.common.prepared_execution import PreparedExecutionRequest
from generativeqc_compiler.common.provenance import canonical_hash

from .stationary_execution import (
    StationaryExecutionGraph,
    compile_stationary_execution_graph,
)
from .stationary_gradient import StationaryGradientPlan
from .stationary_lifetime import (
    StationaryLifetimePlan,
    plan_stationary_lifetimes,
)

STATIONARY_PREPARED_PLAN_VERSION = "stationary-prepared-plan-v1"


@dataclass(frozen=True, slots=True)
class StationaryPreparedPlan:
    """Scientific graph plus method-neutral lifetime decisions."""

    graph: StationaryExecutionGraph
    lifetimes: StationaryLifetimePlan
    version: str = STATIONARY_PREPARED_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.lifetimes.graph_identity != self.graph.identity:
            raise ValueError(
                "stationary prepared plan graph/lifetime identity mismatch"
            )

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "graph_identity": self.graph.identity,
            "lifetime_identity": self.lifetimes.identity,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())

    def prepared_request(
        self,
        *,
        target_identity: str,
        schedule_identity: str,
        workspace_identity: str,
        device: int = 0,
        specialization_identity: str | None = None,
        capture_identity: str | None = None,
    ) -> PreparedExecutionRequest:
        """Bind compiler planning to the shared prepared-execution lifecycle.

        The caller still owns concrete target/schedule/workspace selection.  The
        lifetime identity is folded into schedule compatibility so a retained
        owner cannot replay after residency semantics change.
        """

        effective_schedule = canonical_hash(
            {
                "schema": STATIONARY_PREPARED_PLAN_VERSION,
                "schedule_identity": schedule_identity,
                "lifetime_identity": self.lifetimes.identity,
            }
        )
        return PreparedExecutionRequest(
            kind="stationary-gradient",
            scientific_identity=self.graph.identity,
            target_identity=target_identity,
            schedule_identity=effective_schedule,
            workspace_identity=workspace_identity,
            device=device,
            specialization_identity=specialization_identity,
            capture_identity=capture_identity,
        )


def compile_stationary_prepared_plan(
    plan: StationaryGradientPlan,
) -> StationaryPreparedPlan:
    """Compile scientific dataflow and lifetime semantics without allocation."""

    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary prepared planning requires StationaryGradientPlan")
    graph = compile_stationary_execution_graph(plan)
    lifetimes = plan_stationary_lifetimes(graph)
    return StationaryPreparedPlan(graph=graph, lifetimes=lifetimes)
