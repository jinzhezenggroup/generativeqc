"""Provider-neutral runtime-indexed tensor domains with no owned sparse storage."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from .ir import (
    Node,
    input_tensor,
    runtime_cartesian_scatter_add,
    runtime_cartesian_select,
)

if TYPE_CHECKING:
    from .types import Index, TensorSpec


@dataclass(frozen=True)
class IndexedTensorLayout:
    """Independent local axes sharing immutable runtime control tensors.

    The maps remain ProgramIR inputs, not static coordinates baked into source
    identity. Selecting two AO axes can share one vector without constructing
    a Cartesian incidence table or an intermediate local-by-global panel.
    Allocation, admission thresholds, and device leases belong to execution.
    """

    global_spec: TensorSpec
    selections: tuple[tuple[int, Node, Index], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selections", tuple(self.selections))
        self.select(input_tensor("layout_source", self.global_spec))

    @cached_property
    def local_spec(self) -> TensorSpec:
        return self.select(input_tensor("layout_source", self.global_spec)).spec

    @property
    def map_bytes(self) -> int:
        """Count shared maps once; these are logical inputs, not allocations."""
        return sum(
            node.spec.size * node.spec.itemsize
            for node in dict.fromkeys(mapping for _, mapping, _ in self.selections)
        )

    def select(self, source: Node) -> Node:
        """Gather directly into the declared local Cartesian tensor domain."""
        if source.spec.signature != self.global_spec.signature:
            raise ValueError(
                "indexed layout source belongs to a different global domain"
            )
        return runtime_cartesian_select(source, self.selections)

    def scatter_add(self, local: Node) -> Node:
        """Transpose this layout, preserving duplicate-coordinate multiplicity."""
        if local.spec.signature != self.local_spec.signature:
            raise ValueError("indexed layout value belongs to a different local domain")
        return runtime_cartesian_scatter_add(
            local,
            tuple((axis, mapping) for axis, mapping, _ in self.selections),
            self.global_spec.indices,
        )
