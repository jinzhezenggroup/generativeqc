"""Lifetime and residency analysis for stationary execution graphs.

This pass is intentionally allocation-free.  It classifies semantic values by
producer/consumer topology so later backend planners can decide concrete bytes,
streams and spill/recompute schedules without named-functional policy.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from generativeqc_compiler.common.provenance import canonical_hash

from .stationary_execution import StationaryExecutionGraph

STATIONARY_LIFETIME_PLAN_VERSION = "stationary-lifetime-plan-v1"

DEVICE_RETAINED = "device-retained"
DEVICE_BORROWED = "device-borrowed"
HOST_PUBLISHED = "host-published"
TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class StationaryValueLifetime:
    """Method-neutral lifetime decision for one graph value."""

    name: str
    producer: str
    consumers: tuple[str, ...]
    placement: str
    reason: str

    def semantic_payload(self) -> typing.Any:
        return {
            "name": self.name,
            "producer": self.producer,
            "consumers": self.consumers,
            "placement": self.placement,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StationaryLifetimePlan:
    """Topology-derived lifetime policy independent of concrete allocation."""

    graph_identity: str
    values: tuple[StationaryValueLifetime, ...]
    version: str = STATIONARY_LIFETIME_PLAN_VERSION

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "graph_identity": self.graph_identity,
            "values": [value.semantic_payload() for value in self.values],
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())

    def value(self, name: str) -> StationaryValueLifetime:
        for value in self.values:
            if value.name == name:
                return value
        raise KeyError(name)


def plan_stationary_lifetimes(
    graph: StationaryExecutionGraph,
) -> StationaryLifetimePlan:
    """Classify graph values by observable producer/consumer requirements.

    Multi-consumer borrowable device values are retained so independent
    consumers can share one physical generation.  A single same-device consumer
    receives a borrowed lifetime with no implied host publication.  Concrete
    memory capacity can still reject retention in a later resource pass.
    """

    if not isinstance(graph, StationaryExecutionGraph):
        raise TypeError(
            "stationary lifetime planning requires StationaryExecutionGraph"
        )

    values: list[StationaryValueLifetime] = []
    for value in graph.values:
        if value.host_visible:
            placement = HOST_PUBLISHED
            reason = "public-output"
        elif value.memory_space != "device" or not value.borrowable:
            placement = TRANSIENT
            reason = "not-device-borrowable"
        elif len(value.consumers) > 1:
            placement = DEVICE_RETAINED
            reason = "shared-device-consumers"
        elif len(value.consumers) == 1:
            placement = DEVICE_BORROWED
            reason = "single-device-consumer"
        else:
            placement = TRANSIENT
            reason = "no-consumer"
        values.append(
            StationaryValueLifetime(
                name=value.name,
                producer=value.producer,
                consumers=value.consumers,
                placement=placement,
                reason=reason,
            )
        )

    return StationaryLifetimePlan(graph_identity=graph.identity, values=tuple(values))
