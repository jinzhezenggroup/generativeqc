"""Grid-feature lease contract derived from stationary execution planning."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from generativeqc_compiler.common.provenance import canonical_hash

from .stationary_lifetime import DEVICE_RETAINED
from .stationary_prepared import StationaryPreparedPlan

STATIONARY_FEATURE_LEASE_VERSION = "stationary-feature-lease-v1"


@dataclass(frozen=True, slots=True)
class StationaryFeatureLease:
    """One reusable density-feature value and its compiler-derived lifetime."""

    feature: str
    value_name: str
    consumers: tuple[str, ...]
    retain: bool

    def semantic_payload(self) -> typing.Any:
        return {
            "feature": self.feature,
            "value_name": self.value_name,
            "consumers": self.consumers,
            "retain": self.retain,
        }


@dataclass(frozen=True, slots=True)
class StationaryFeatureLeasePlan:
    """Runtime-facing feature inventory with no named-functional dispatch."""

    prepared_identity: str
    leases: tuple[StationaryFeatureLease, ...]
    version: str = STATIONARY_FEATURE_LEASE_VERSION

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(lease.feature for lease in self.leases)

    @property
    def retained_features(self) -> tuple[str, ...]:
        return tuple(lease.feature for lease in self.leases if lease.retain)

    def semantic_payload(self) -> typing.Any:
        return {
            "version": self.version,
            "prepared_identity": self.prepared_identity,
            "leases": [lease.semantic_payload() for lease in self.leases],
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.semantic_payload())


def plan_stationary_feature_leases(
    prepared: StationaryPreparedPlan,
) -> StationaryFeatureLeasePlan:
    """Project graph/lifetime decisions onto the CUDA-grid feature boundary."""

    if not isinstance(prepared, StationaryPreparedPlan):
        raise TypeError("stationary feature leases require StationaryPreparedPlan")

    leases = []
    for feature in prepared.graph.grid_features:
        value_name = f"grid_feature:{feature}"
        value = prepared.graph.value(value_name)
        lifetime = prepared.lifetimes.value(value_name)
        leases.append(
            StationaryFeatureLease(
                feature=feature,
                value_name=value_name,
                consumers=value.consumers,
                retain=lifetime.placement == DEVICE_RETAINED,
            )
        )
    return StationaryFeatureLeasePlan(
        prepared_identity=prepared.identity,
        leases=tuple(leases),
    )
