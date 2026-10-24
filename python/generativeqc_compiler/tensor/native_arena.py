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
from generativeqc_compiler.common.storage import (
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_lifetimes,
)

from .program import Program

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .ir import Index, Node


@dataclass(frozen=True)
class SymbolicArenaPlan:
    """Exclusive retained slots plus uniformly typed last-reader-colored storage.

    Capacities are maxima of runtime extent products, not representative numeric
    shapes. Outputs stay live through return because native callers
    borrow their pointers. Assignments use positions in the supplied node order.
    """

    slots: tuple[tuple[tuple[str, ...], ...], ...]
    node_slots: Mapping[int, int]
    storage_dtype: str = "float64"

    @property
    def identity(self) -> str:
        """Identify storage only; this is not scientific or numeric cache validity."""
        return canonical_hash(
            {
                "schema": "generativeqc.tensor.symbolic-arena.v2",
                "slots": self.slots,
                "assignments": tuple(self.node_slots.items()),
                **(
                    {"storage_dtype": self.storage_dtype}
                    if self.storage_dtype != "float64"
                    else {}
                ),
            }
        )

    @property
    def sizes(self) -> tuple[str, ...]:
        """Emit checked element capacities for the existing CPU/CUDA native ABI."""
        sizes = []
        for capacity in self.slots:
            terms = tuple(
                "checked_product({" + ",".join(shape) + "})"
                if shape
                else ("1" if len(capacity) == 1 else "std::size_t{1}")
                for shape in capacity
            )
            sizes.append(
                terms[0] if len(terms) == 1 else "std::max({" + ",".join(terms) + "})"
            )
        return tuple(sizes)


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
    storage_dtype: str = "float64",
    cross_size_reuse: bool = False,
) -> SymbolicArenaPlan:
    """Plan the established materialized native schedule without backend policy.

    The caller deterministically binds each index to a runtime extent symbol and may
    supply an already-selected dependency order. Every live node must occur once;
    no operation is reordered, removed, fused, or rematerialized here. Lowering
    must write each noninput node into distinct dense uniform floating storage: views or
    opaque physical aliases require the separate common storage analysis.

    ``retained_nodes`` declares exclusive storage, not a proof of invariance.
    Use ``analyze_iteration_reuse`` to prove reuse and let the runtime establish
    its lifetime. Retained slots are reserved before any dynamic slot so a replay
    cannot overwrite a later invariant before its original graph position.

    The opt-in cross-size candidate merges whole equal-capacity slots only when
    all their owner intervals are disjoint. Their runtime maximum cannot exceed
    the original sum for any admitted shape. Retained plans keep the established
    coloring so optional-cache admission is unchanged.
    """
    if not isinstance(program, Program):
        raise TypeError("symbolic arena planning requires a TensorIR Program")
    if storage_dtype not in ("float32", "float64"):
        raise ValueError("native tensor arena requires a floating storage dtype")
    if not callable(dimension_symbol):
        raise TypeError("arena dimension symbols require an index binding")
    if type(cross_size_reuse) is not bool:
        raise TypeError("cross-size arena reuse must be bool")
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
    shapes: dict[int, tuple[str, ...]] = {}
    dimensions: dict[Index, str] = {}
    for number, node in enumerate(nodes):
        for source in node.inputs:
            if numbers[source] >= number:
                raise ValueError("arena execution order must be dependency ordered")
        if node.op == "input":
            continue
        if node.spec.dtype != storage_dtype:
            label = "FP64" if storage_dtype == "float64" else "FP32"
            raise ValueError(
                f"native tensor arena requires uniform {label} intermediates"
            )
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
    # Unit capacities resolve materialized-owner lifetimes only; concrete sizes
    # never participate in the symbolic capacity or slot assignment.
    lifetimes = analyze_lifetimes(
        [
            BufferValue(number, 8, "native", compiler_owned=node.op != "input")
            for number, node in enumerate(nodes)
        ],
        [
            BufferOp(
                number,
                tuple(numbers[source] for source in node.inputs),
                () if node.op == "input" else (number,),
                MemoryEffect.EXPLICIT,
            )
            for number, node in enumerate(nodes)
        ],
        inputs=tuple(number for number, node in enumerate(nodes) if node.op == "input"),
        outputs=tuple(
            dict.fromkeys(numbers[node] for node in program.outputs.values())
        ),
    )
    last_use = {item.owner: item.last_phase - 1 for item in lifetimes.ranges}

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

    capacities = tuple((shape,) for shape in slots)
    if cross_size_reuse and not retained_nodes:
        intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for number, slot in node_slots.items():
            intervals[slot].append((number, last_use[number]))
        groups: list[list[int]] = []
        group_intervals: list[list[tuple[int, int]]] = []
        remap: dict[int, int] = {}
        for slot in range(len(slots)):
            for group, occupied in enumerate(group_intervals):
                if all(
                    last < first_other or last_other < first
                    for first, last in intervals[slot]
                    for first_other, last_other in occupied
                ):
                    groups[group].append(slot)
                    occupied.extend(intervals[slot])
                    remap[slot] = group
                    break
            else:
                remap[slot] = len(groups)
                groups.append([slot])
                group_intervals.append(list(intervals[slot]))
        capacities = tuple(
            tuple(sorted({slots[slot] for slot in group})) for group in groups
        )
        node_slots = {number: remap[slot] for number, slot in node_slots.items()}
    return SymbolicArenaPlan(capacities, MappingProxyType(node_slots), storage_dtype)
