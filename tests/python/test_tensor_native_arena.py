"""Shared symbolic storage legality, replay, and native ownership boundaries."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from generativeqc_compiler.common.storage import (
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_storage,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    multiply,
)
from generativeqc_compiler.tensor.iteration_reuse import analyze_iteration_reuse
from generativeqc_compiler.tensor.native_arena import plan_symbolic_arena

from tools import generate_rccsd_native as codegen


def _input(name: str, extent: str = "points", dtype: str = "float64") -> Node:
    spec = TensorSpec(
        (Index("i", IndexSpace(extent, "batch", 3)),), dtype=dtype, role="input"
    )
    return input_tensor(name, spec)


def _symbol(index: Index) -> str:
    return index.space.name


def _chain() -> Program:
    source = _input("source")
    first = multiply(source, source)
    second = add(first, source)
    third = multiply(second, source)
    return Program({"result": add(third, source)})


def test_reuse_agrees_with_common_storage_lifetimes() -> None:
    """Independent shared lifetime analysis checks the native coloring contract."""
    program = _chain()
    nodes = program.live_nodes
    plan = plan_symbolic_arena(program, dimension_symbol=_symbol)
    storage = analyze_storage(
        [
            BufferValue(
                node, node.spec.size * 8, "host", compiler_owned=node.op != "input"
            )
            for node in nodes
        ],
        [
            BufferOp(node, node.inputs, (node,), MemoryEffect.EXPLICIT)
            for node in nodes
            if node.op != "input"
        ],
        inputs=tuple(node for node in nodes if node.op == "input"),
        outputs=tuple(program.outputs.values()),
    )
    lifetimes = {item.owner: item for item in storage.ranges}
    assert len(plan.slots) == 2
    assert plan.sizes == ("checked_product({points})",) * 2
    for left, right in combinations(plan.node_slots, 2):
        if plan.node_slots[left] == plan.node_slots[right]:
            assert (
                lifetimes[nodes[left]].last_phase < lifetimes[nodes[right]].first_phase
            )
    for number, node in enumerate(nodes):
        for source in node.inputs:
            if source.op != "input":
                assert plan.node_slots[number] != plan.node_slots[nodes.index(source)]
    with pytest.raises(TypeError):
        plan.node_slots[1] = 100  # type: ignore[index]
    fresh = plan_symbolic_arena(_chain(), dimension_symbol=_symbol)
    assert fresh == plan and fresh.identity == plan.identity


def test_equal_representative_shapes_do_not_merge_distinct_symbols() -> None:
    rows, columns = _input("rows", "row_extent"), _input("columns", "column_extent")
    row_value, column_value = multiply(rows, rows), multiply(columns, columns)
    program = Program(
        {"rows": einsum("i->", row_value), "columns": einsum("i->", column_value)}
    )
    nodes = program.dependency_order
    plan = plan_symbolic_arena(program, dimension_symbol=_symbol, execution_nodes=nodes)
    assert row_value.spec.size == column_value.spec.size
    assert (
        plan.node_slots[nodes.index(row_value)]
        != plan.node_slots[nodes.index(column_value)]
    )
    assert ("row_extent",) in plan.slots and ("column_extent",) in plan.slots
    assert "1" in plan.sizes


def test_permuted_equal_symbolic_products_can_share_capacity() -> None:
    rows = Index("rows", IndexSpace("rows", "batch", 2))
    columns = Index("columns", IndexSpace("columns", "batch", 3))
    source = input_tensor("source", TensorSpec((rows, columns), role="input"))
    first = multiply(source, source)
    transposed = einsum("ij->ji", first)
    last = multiply(transposed, transposed)
    program = Program({"result": last})
    nodes = program.live_nodes
    plan = plan_symbolic_arena(program, dimension_symbol=_symbol)
    assert plan.node_slots[nodes.index(first)] == plan.node_slots[nodes.index(last)]
    assert plan.sizes == ("checked_product({columns,rows})",) * 2


def test_borrowed_outputs_cannot_be_overwritten_by_later_branches() -> None:
    source = _input("source")
    first = multiply(source, source)
    second = add(first, source)
    program = Program({"borrowed": first, "later": multiply(second, source)})
    nodes = program.dependency_order
    plan = plan_symbolic_arena(program, dimension_symbol=_symbol, execution_nodes=nodes)
    output_slot = plan.node_slots[nodes.index(first)]
    assert all(
        slot != output_slot
        for number, slot in plan.node_slots.items()
        if number != nodes.index(first)
    )


@pytest.mark.parametrize("changed_reference", [False, True])
def test_late_invariants_are_exclusive_across_density_replay(
    changed_reference: bool,
) -> None:
    """Execute ordinary TensorIR primitives in the actual planned numeric slots."""
    geometry, density = _input("geometry"), _input("density")
    dynamic = multiply(multiply(density, density), density)
    invariant = multiply(multiply(geometry, geometry), geometry)
    program = Program({"result": add(dynamic, invariant)})
    reuse = analyze_iteration_reuse(program, invariant_inputs=("geometry",))
    nodes = program.dependency_order
    numbers = {node: number for number, node in enumerate(nodes)}
    assert numbers[dynamic] < numbers[invariant]
    plan = plan_symbolic_arena(
        program,
        dimension_symbol=_symbol,
        execution_nodes=nodes,
        retained_nodes=reuse.invariant_nodes,
    )
    pinned = {plan.node_slots[numbers[node]] for node in reuse.invariant_nodes}
    assert len(pinned) == len(reuse.invariant_nodes)
    assert not pinned.intersection(
        plan.node_slots[numbers[node]] for node in reuse.dynamic_nodes
    )
    arenas = [np.full(3, np.nan) for _ in plan.slots]
    values = {}
    feeds = {"geometry": np.array([1.0, 2.0, 4.0]) + 3 * changed_reference}
    executed = 0

    def evaluate(node: Node) -> None:
        nonlocal executed
        arguments = tuple(
            input_tensor(f"arg_{number}", replace(source.spec, role="input"))
            for number, source in enumerate(node.inputs)
        )
        lowered = Node(node.op, arguments, node.spec, node.attributes)
        frame = {
            argument.attrs["name"]: (
                feeds[source.attrs["name"]] if source.op == "input" else values[source]
            )
            for argument, source in zip(arguments, node.inputs, strict=True)
        }
        output = execute(Program({"value": lowered}), frame).outputs["value"]
        slot = arenas[plan.node_slots[numbers[node]]]
        slot[:] = output
        values[node] = slot
        executed += 1

    for node in reuse.invariant_nodes:
        evaluate(node)
    for iteration in range(4):
        feeds["density"] = np.array([0.5, -1.0, 2.0]) + iteration
        for node in nodes:
            if node in reuse.dynamic_nodes:
                evaluate(node)
        expected = execute(program, feeds).outputs["result"]
        np.testing.assert_array_equal(values[program.outputs["result"]], expected)
    assert executed == len(reuse.invariant_nodes) + 4 * len(reuse.dynamic_nodes)


@pytest.mark.parametrize("order", ["missing", "duplicate", "reversed", "foreign"])
def test_invalid_execution_orders_fail_closed(order: str) -> None:
    program = _chain()
    nodes = program.live_nodes
    invalid = {
        "missing": nodes[:-1],
        "duplicate": (*nodes[:-1], nodes[0]),
        "reversed": tuple(reversed(nodes)),
        "foreign": (*nodes[:-1], _input("foreign")),
    }[order]
    with pytest.raises(ValueError, match="execution order"):
        plan_symbolic_arena(program, dimension_symbol=_symbol, execution_nodes=invalid)


@pytest.mark.parametrize("retained", ["input", "duplicate", "foreign"])
def test_invalid_retained_storage_fails_closed(retained: str) -> None:
    program = _chain()
    nodes = program.live_nodes
    invalid = {
        "input": (nodes[0],),
        "duplicate": (nodes[1], nodes[1]),
        "foreign": (multiply(_input("foreign"), _input("other")),),
    }[retained]
    with pytest.raises(ValueError, match="retained"):
        plan_symbolic_arena(program, dimension_symbol=_symbol, retained_nodes=invalid)


@pytest.mark.parametrize("symbol", ["", "extent+1", "a,b", 3, None])
def test_invalid_dimension_bindings_fail_closed(symbol: object) -> None:
    with pytest.raises(ValueError, match="extent identifiers"):
        plan_symbolic_arena(_chain(), dimension_symbol=lambda index: symbol)


def test_unqualified_precision_is_not_admitted() -> None:
    source = _input("source", dtype="float32")
    with pytest.raises(ValueError, match="FP64"):
        plan_symbolic_arena(
            Program({"result": multiply(source, source)}), dimension_symbol=_symbol
        )


def test_selected_order_does_not_repeat_liveness_or_dimension_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplied schedules must not pay for another full middle-end liveness pass."""
    program = _chain()
    nodes = program.dependency_order
    bound = []

    def unexpected_analysis(program: Program) -> None:
        raise AssertionError("repeated live-node analysis")

    def bind(index: Index) -> str:
        bound.append(index)
        return _symbol(index)

    monkeypatch.setattr(Program, "live_nodes", property(unexpected_analysis))
    plan = plan_symbolic_arena(program, dimension_symbol=bind, execution_nodes=nodes)
    assert len(plan.slots) == 2
    assert len(bound) == len(set(bound)) == 1


def test_native_generator_is_only_a_domain_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual native consumers must call the canonical compiler planner."""
    program = codegen.iteration_program(2, 3)
    reuse = codegen._iteration_reuse_plan(program)
    expected = plan_symbolic_arena(
        program,
        dimension_symbol=codegen._dim,
        execution_nodes=codegen._execution_nodes(program),
        retained_nodes=reuse.invariant_nodes,
    )
    calls = []

    def planner(actual: Program, **bindings: object) -> object:
        calls.append((actual, bindings))
        return expected

    monkeypatch.setattr(codegen, "plan_symbolic_arena", planner)
    assert (
        codegen._arena_plan(program, retained_nodes=reuse.invariant_nodes) is expected
    )
    assert calls == [
        (
            program,
            {
                "dimension_symbol": codegen._dim,
                "execution_nodes": codegen._execution_nodes(program),
                "retained_nodes": reuse.invariant_nodes,
            },
        )
    ]
