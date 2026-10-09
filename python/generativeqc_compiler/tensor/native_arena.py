"""Compiler-owned symbolic storage for materialized native TensorIR lowering.

This is the existing CPU/CUDA arena schedule, not an allocator or a solver cache.
Runtime owners still admit the complete endpoint, allocate storage, establish an
immutable execution epoch, and publish only after successful preparation/replay.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

from .program import Program

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .ir import Index, Node


@dataclass(frozen=True)
class SymbolicArenaPlan:
    """Exclusive retained slots plus last-reader-colored dynamic FP64 storage.

    Slot capacities are products of runtime extent symbols, not representative
    numeric shapes. Outputs stay live through return because native callers
    borrow their pointers. Assignments use positions in the supplied node order.
    """

    slots: tuple[tuple[str, ...], ...]
    node_slots: Mapping[int, int]

    @property
    def identity(self) -> str:
        """Identify storage only; this is not scientific or numeric cache validity."""
        return canonical_hash(
            {
                "schema": "generativeqc.tensor.symbolic-arena.v1",
                "slots": self.slots,
                "assignments": tuple(self.node_slots.items()),
            }
        )

    @property
    def sizes(self) -> tuple[str, ...]:
        """Emit checked element capacities for the existing CPU/CUDA native ABI."""
        return tuple(
            "checked_product({" + ",".join(shape) + "})" if shape else "1"
            for shape in self.slots
        )


def plan_symbolic_arena(
    program: Program,
    *,
    dimension_symbol: Callable[[Index], str],
    execution_nodes: Sequence[Node] | None = None,
    retained_nodes: Sequence[Node] = (),
) -> SymbolicArenaPlan:
    """Plan the established materialized native schedule without backend policy.

    The caller deterministically binds each index to a runtime extent symbol and may
    supply an already-selected dependency order. Every live node must occur once;
    no operation is reordered, removed, fused, or rematerialized here. Lowering
    must write each noninput node into distinct dense FP64 storage: views or
    opaque physical aliases require the separate common storage analysis.

    ``retained_nodes`` declares exclusive storage, not a proof of invariance.
    Use ``analyze_iteration_reuse`` to prove reuse and let the runtime establish
    its lifetime. Retained slots are reserved before any dynamic slot so a replay
    cannot overwrite a later invariant before its original graph position.
    """
    if not isinstance(program, Program):
        raise TypeError("symbolic arena planning requires a TensorIR Program")
    if not callable(dimension_symbol):
        raise TypeError("arena dimension symbols require an index binding")
    for label, values in (
        ("execution nodes", execution_nodes),
        ("retained nodes", retained_nodes),
    ):
        if values is not None and (
            not isinstance(values, Sequence) or isinstance(values, (str, bytes))
        ):
            raise TypeError(f"arena {label} must be a sequence")
    nodes = tuple(program.live_nodes if execution_nodes is None else execution_nodes)
    expected_nodes = nodes if execution_nodes is None else program.dependency_order
    numbers = {node: number for number, node in enumerate(nodes)}
    if (
        len(numbers) != len(nodes)
        or len(nodes) != len(expected_nodes)
        or any(node not in numbers for node in expected_nodes)
    ):
        raise ValueError("arena execution order must contain every live node once")
    last_use = list(range(len(nodes)))
    shapes: dict[int, tuple[str, ...]] = {}
    dimensions: dict[Index, str] = {}
    for number, node in enumerate(nodes):
        for source in node.inputs:
            if numbers[source] >= number:
                raise ValueError("arena execution order must be dependency ordered")
            last_use[numbers[source]] = number
        if node.op == "input":
            continue
        if node.spec.dtype != "float64":
            raise ValueError("native tensor arena requires FP64 intermediates")
        symbols = []
        for index in node.spec.indices:
            symbol = dimensions.get(index)
            if symbol is None:
                symbol = dimension_symbol(index)
                if (
                    not isinstance(symbol, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) is None
                ):
                    raise ValueError(
                        "arena dimensions must bind to runtime extent identifiers"
                    )
                dimensions[index] = symbol
            symbols.append(symbol)
        shapes[number] = tuple(sorted(symbols))
    for node in program.outputs.values():
        last_use[numbers[node]] = len(nodes)

    releases: dict[int, list[int]] = defaultdict(list)
    available: dict[tuple[str, ...], list[int]] = defaultdict(list)
    slots: list[tuple[str, ...]] = []
    node_slots: dict[int, int] = {}
    for node in retained_nodes:
        number = numbers.get(node)
        if number is None or node.op == "input" or number in node_slots:
            raise ValueError("invalid retained native arena node")
        node_slots[number] = len(slots)
        slots.append(shapes[number])
    for number, node in enumerate(nodes):
        for slot in releases[number]:
            available[slots[slot]].append(slot)
        if node.op == "input" or number in node_slots:
            continue
        shape = shapes[number]
        if available[shape]:
            slot = available[shape].pop()
        else:
            slot = len(slots)
            slots.append(shape)
        node_slots[number] = slot
        releases[last_use[number] + 1].append(slot)

    return SymbolicArenaPlan(tuple(slots), MappingProxyType(node_slots))
