"""Dependency proofs and numerical replay before CPU/CUDA lowering."""

from dataclasses import replace as replace_spec
from typing import Literal

import numpy as np
import pytest
from generativeqc_compiler.common.liveness import EffectKind
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    execute,
    input_tensor,
    multiply,
    prepare_for_backend,
)
from generativeqc_compiler.tensor.ir import Node
from generativeqc_compiler.tensor.iteration_reuse import (
    IterationReusePlan,
    analyze_iteration_reuse,
)
from generativeqc_compiler.tensor.scf import fock_composition_program


def graph() -> tuple[Program, Node, Node]:
    spec = TensorSpec((Index("i", IndexSpace("points", "batch", 3)),), role="input")
    g, b, p, d = (
        input_tensor(n, spec) for n in ("geometry", "basis", "parameters", "density")
    )
    ao = multiply(g, b)
    immutable = add(ao, p)
    return Program({"result": multiply(immutable, d)}), ao, immutable


def split(program: Program, plan: IterationReusePlan) -> tuple[Program, Program]:
    # Test adapter uses existing interpreter algebra, not another numerical implementation.
    names = {n: f"retained_{i}" for i, n in enumerate(plan.invariant_nodes)}
    prepare = Program({names[n]: n for n in plan.invariant_nodes})
    replace = {}
    for n in program.live_nodes:
        replace[n] = (
            input_tensor(names[n], replace_spec(n.spec, role="input"))
            if n in names
            else Node(n.op, tuple(replace[s] for s in n.inputs), n.spec, n.attributes)
        )
    return prepare, Program({k: replace[n] for k, n in program.outputs.items()})


def test_transitive_dependencies_and_dynamic_density() -> None:
    program, ao, immutable = graph()
    plan = analyze_iteration_reuse(
        program, invariant_inputs=("geometry", "basis", "parameters")
    )
    assert plan.invariant_nodes == (ao, immutable)
    assert plan.dynamic_nodes == (program.outputs["result"],)
    assert dict(plan.input_dependencies)[immutable] == (
        "basis",
        "geometry",
        "parameters",
    )
    assert plan.retained_bytes == 48
    prepare, replay = split(program, plan)
    feeds = {n: np.arange(1.0, 4.0) for n in plan.invariant_inputs}
    saved = execute(prepare, feeds).outputs
    for density in (np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 4.0])):
        expected = execute(program, {**feeds, "density": density}).outputs["result"]
        actual = execute(replay, {**feeds, **saved, "density": density}).outputs[
            "result"
        ]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("changed", ["geometry", "basis", "parameters"])
def test_changed_invariants_need_fresh_epoch(changed: str) -> None:
    program, _, _ = graph()
    plan = analyze_iteration_reuse(
        program, invariant_inputs=("geometry", "basis", "parameters")
    )
    prepare, replay = split(program, plan)
    feeds = {
        n: np.arange(1.0, 4.0) for n in ("geometry", "basis", "parameters", "density")
    }
    old = execute(prepare, feeds).outputs
    feeds[changed] = feeds[changed] + 2
    fresh = execute(prepare, feeds).outputs
    reference = execute(program, feeds).outputs["result"]
    np.testing.assert_array_equal(
        execute(replay, {**feeds, **fresh}).outputs["result"], reference
    )
    # Same-shaped data and same plan identity are NOT numeric validity proofs.
    assert not np.array_equal(
        execute(replay, {**feeds, **old}).outputs["result"], reference
    )


@pytest.mark.parametrize("effect", [EffectKind.OPAQUE, EffectKind.EFFECTFUL])
def test_effectful_ancestry_is_not_reusable(effect: EffectKind) -> None:
    program, ao, immutable = graph()
    plan = analyze_iteration_reuse(
        program,
        invariant_inputs=("geometry", "basis", "parameters"),
        effects={"multiply": effect},
    )
    assert not plan.invariant_nodes
    assert ao in plan.dynamic_nodes and immutable in plan.dynamic_nodes
    assert (
        plan.identity
        != analyze_iteration_reuse(
            program, invariant_inputs=("geometry", "basis", "parameters")
        ).identity
    )


def test_strict_input_contract_and_deterministic_identity() -> None:
    program, _, _ = graph()
    assert not analyze_iteration_reuse(program, invariant_inputs=()).invariant_nodes
    for names in (("missing",), ("basis", "basis"), ("",)):
        with pytest.raises(ValueError):
            analyze_iteration_reuse(program, invariant_inputs=names)
    with pytest.raises(TypeError):
        analyze_iteration_reuse(program, invariant_inputs="basis")
    with pytest.raises(TypeError):
        analyze_iteration_reuse(program, invariant_inputs=(), effects={"add": "pure"})
    a = analyze_iteration_reuse(program, invariant_inputs=("basis", "geometry"))
    b = analyze_iteration_reuse(program, invariant_inputs=("geometry", "basis"))
    assert a.identity == b.identity
    assert (
        a.identity
        != analyze_iteration_reuse(program, invariant_inputs=("basis",)).identity
    )


@pytest.mark.parametrize("reference", ["restricted", "unrestricted"])
def test_real_hf_ks_keeps_density_dependent_j_k_xc_dynamic(
    reference: Literal["restricted", "unrestricted"],
) -> None:
    program = fock_composition_program(
        1, 2, reference=reference, exact_exchange="1/4", include_local_potential=True
    )
    plan = analyze_iteration_reuse(program, invariant_inputs=("hcore",))
    assert len(plan.invariant_nodes) == 1
    assert plan.invariant_nodes[0].op == "broadcast"
    assert dict(plan.input_dependencies)[plan.invariant_nodes[0]] == ("hcore",)
    assert program.outputs["fock"] in plan.dynamic_nodes


def test_cpu_cuda_share_schedule_before_emission() -> None:
    program, _, _ = graph()
    plans = [
        analyze_iteration_reuse(
            prepare_for_backend(program, backend, preserve_reduction_order=True),
            invariant_inputs=("geometry", "basis", "parameters"),
        )
        for backend in ("cpu", "cuda")
    ]
    assert plans[0].identity == plans[1].identity
    assert plans[0].retained_bytes == plans[1].retained_bytes


def test_real_ccsd_never_reuses_amplitude_dependents() -> None:
    from tools.generate_rccsd_native import iteration_program

    program = iteration_program(2, 3)
    immutable = tuple(
        n.attrs["name"]
        for n in program.live_nodes
        if n.op == "input" and n.attrs["name"] not in {"t1", "t2"}
    )
    plan = analyze_iteration_reuse(program, invariant_inputs=immutable)
    assert plan.invariant_nodes and plan.dynamic_nodes
    assert all(
        not {"t1", "t2"}.intersection(deps)
        for n, deps in plan.input_dependencies
        if n in plan.invariant_nodes
    )
    assert all(n in plan.dynamic_nodes for n in program.outputs.values())


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (3, 3)])
def test_real_ccsd_five_replays_match_bitwise(shape: tuple[int, int]) -> None:
    from generativeqc_compiler.tensor import PackedLayout

    from tools.generate_rccsd_native import iteration_program

    program = prepare_for_backend(iteration_program(*shape), "cpu")
    immutable = tuple(
        n.attrs["name"]
        for n in program.live_nodes
        if n.op == "input" and n.attrs["name"] not in {"t1", "t2"}
    )
    plan = analyze_iteration_reuse(program, invariant_inputs=immutable)
    prepare, replay = split(program, plan)
    rng = np.random.default_rng(4811)
    feeds = {}
    for node in program.live_nodes:
        if node.op != "input":
            continue
        if node.spec.symmetries:
            layout = PackedLayout.from_spec(node.spec)
            value = layout.unpack(rng.normal(scale=0.05, size=layout.size))
        else:
            value = rng.normal(scale=0.05, size=node.spec.shape)
        feeds[node.attrs["name"]] = value
    for name in ("d1", "d2"):
        feeds[name] = np.full_like(feeds[name], -1.3)
    saved = execute(prepare, feeds).outputs
    for _ in range(5):
        feeds["t1"] = rng.normal(scale=0.02, size=feeds["t1"].shape)
        value = rng.normal(scale=0.02, size=feeds["t2"].shape)
        feeds["t2"] = (value + value.transpose(1, 0, 3, 2)) / 2
        expected = execute(program, feeds).outputs
        actual = execute(replay, {**feeds, **saved}).outputs
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_portable_reference_transform_traffic_is_not_contraction_flops() -> None:
    """Qualify logical traffic, without claiming DRAM bytes or device timing."""
    from tools.generate_rccsd_native import iteration_program

    program = prepare_for_backend(iteration_program(8, 16), "cuda")
    immutable = tuple(
        n.attrs["name"]
        for n in program.live_nodes
        if n.op == "input" and n.attrs["name"] not in {"t1", "t2"}
    )
    plan = analyze_iteration_reuse(program, invariant_inputs=immutable)
    assert len(plan.invariant_nodes) == 9
    assert plan.retained_bytes == 1_084_928
    reads = 0
    for node in plan.invariant_nodes:
        assert len(node.inputs) == 1
        if node.op == "einsum":
            assert node.attrs["coefficient"] == (1, 1)
            assert set(node.attrs["labels"][0]) == set(node.attrs["output"])
        else:
            assert node.op == "add"
            assert node.attrs["coefficients"] == ((1, 1),)
        reads += node.inputs[0].spec.size * node.inputs[0].spec.itemsize
    assert reads == plan.retained_bytes
    assert 8 * (reads + plan.retained_bytes) == 17_358_848
