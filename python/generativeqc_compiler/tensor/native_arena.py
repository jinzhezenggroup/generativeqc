"""Compiler-owned symbolic storage for materialized native TensorIR lowering.

This is the existing CPU/CUDA arena schedule, not an allocator or a solver cache.
Runtime owners still admit the complete endpoint, allocate storage, establish an
immutable execution epoch, and publish only after successful preparation/replay.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class NativeCopyRoundTrips:
    """Consecutive native FP64 copies returning to already populated storage.

    The scientific Program and complete materialized arena remain unchanged.
    Every triple names an audited producer, its sole-reader unit copy, and the
    final unit copy whose original arena slot equals the producer's slot.
    Skipping both copies therefore leaves the final pointer and value intact;
    this does not introduce a view, donation, or a new lifetime assumption.
    """

    program_identity: str
    arena_identity: str
    triples: tuple[tuple[int, int, int], ...]

    @property
    def elided_nodes(self) -> tuple[int, ...]:
        return tuple(number for triple in self.triples for number in triple[1:])

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": "generativeqc.tensor.native-copy-roundtrips.v1",
                "program": self.program_identity,
                "arena": self.arena_identity,
                "triples": self.triples,
            }
        )


def analyze_native_copy_roundtrips(
    program: Program,
    *,
    dimension_symbol: Callable[[Index], str],
    execution_nodes: Sequence[Node] | None = None,
) -> NativeCopyRoundTrips:
    """Prove bounded elisions for the existing audited native FP64 emitter.

    This is not an algebraic TensorIR rewrite: native singleton addition emits
    exactly ``1.0 * source``, without a leading zero addition. Finite values,
    signed zero and subnormals survive both copies bit-for-bit. The producer
    must itself audit every output, preserving its earlier sticky error even
    for nonfinite values. Matrix callbacks and borrowed inputs are excluded.

    Adjacency, single-reader edges and the existing exact symbolic arena slot
    equality prove that no intervening write or hidden reader can observe the
    omitted storage. A future unsupported graph simply retains its copies.
    """
    arena = plan_symbolic_arena(
        program, dimension_symbol=dimension_symbol, execution_nodes=execution_nodes
    )
    nodes = tuple(program.live_nodes if execution_nodes is None else execution_nodes)
    readers: dict[Node, set[Node]] = {node: set() for node in nodes}
    for node in nodes:
        for source in node.inputs:
            readers[source].add(node)
    outputs = set(program.outputs.values())

    def unit_copy(node: Node, source: Node) -> bool:
        return (
            node.op == "add"
            and node.inputs == (source,)
            and node.attrs["coefficients"] == ((1, 1),)
            and node.spec.dtype == "float64"
            and replace(source.spec, role=node.spec.role) == node.spec
        )

    def audited_scalar(node: Node) -> bool:
        if node.spec.dtype != "float64":
            return False
        if node.op in ("add", "multiply", "transpose"):
            return True
        if node.op != "einsum" or len(node.inputs) != 2:
            return False
        labels = {label for operand in node.attrs["labels"] for label in operand}
        return labels == set(node.attrs["output"])

    triples = []
    used_positions: set[int] = set()
    for number in range(len(nodes) - 2):
        positions = (number, number + 1, number + 2)
        if any(position in used_positions for position in positions):
            continue
        producer, first, final = nodes[number : number + 3]
        if (
            audited_scalar(producer)
            and unit_copy(first, producer)
            and unit_copy(final, first)
            and readers[producer] == {first}
            and readers[first] == {final}
            and producer not in outputs
            and first not in outputs
            and arena.node_slots[number] == arena.node_slots[number + 2]
        ):
            triples.append(positions)
            used_positions.update(positions)
    return NativeCopyRoundTrips(program.logical_hash, arena.identity, tuple(triples))


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
