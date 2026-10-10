"""Native copy elision preserves the original graph, slots and audited values."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _cc_owner_test_support import write_df_cpu_headers
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
from generativeqc_compiler.tensor.native_arena import (
    NativeCopyRoundTrips,
    analyze_native_copy_roundtrips,
    plan_symbolic_arena,
)

from tools import generate_df_ccsd_spectator_pairs as paired
from tools import generate_rccsd_native as native


def _input(name: str, size: int = 3) -> Node:
    return input_tensor(
        name,
        TensorSpec((Index("a", IndexSpace("virtual", "virtual", size)),), role="input"),
    )


def _chain(copies: int = 2, *, size: int = 3) -> Program:
    source, ones = _input("source", size), _input("ones", size)
    result = multiply(source, ones)
    for _ in range(copies):
        result = add(result)
    return Program({"result": result})


def _plan(program: Program) -> NativeCopyRoundTrips:
    return analyze_native_copy_roundtrips(
        program,
        dimension_symbol=native._dim,
        execution_nodes=program.dependency_order,
    )


def test_copy_roundtrip_has_original_storage_and_graph_identity() -> None:
    program = _chain()
    original = program.dumps()
    nodes = program.dependency_order
    plan = _plan(program)
    arena = plan_symbolic_arena(
        program, dimension_symbol=native._dim, execution_nodes=nodes
    )
    assert plan.triples == ((2, 3, 4),)
    assert plan.elided_nodes == (3, 4)
    assert arena.node_slots[2] == arena.node_slots[4] != arena.node_slots[3]
    assert plan.program_identity == program.logical_hash
    assert plan.arena_identity == arena.identity
    assert program.dumps() == original
    assert plan.identity == _plan(_chain()).identity


@pytest.mark.parametrize("copies", range(2, 9))
def test_long_chains_have_disjoint_roundtrips(copies: int) -> None:
    plan = _plan(_chain(copies))
    positions = [position for triple in plan.triples for position in triple]
    assert len(set(positions)) == len(positions)
    assert len(plan.elided_nodes) % 2 == 0


@pytest.mark.parametrize("coefficient", (-1, 0, 2))
def test_nonunit_copies_remain_materialized(coefficient: int) -> None:
    source = multiply(_input("left"), _input("right"))
    program = Program({"result": add(add(source, coefficients=(coefficient,)))})
    assert not _plan(program).elided_nodes


def test_borrowed_inputs_and_matrix_reductions_keep_their_checks() -> None:
    source = _input("source")
    borrowed = Program({"result": add(add(source))})
    reduction = einsum("a,a->", source, source)
    reduced = Program({"result": add(add(reduction))})
    assert not _plan(borrowed).elided_nodes
    assert not _plan(reduced).elided_nodes


def test_extra_readers_or_outputs_refuse_elision() -> None:
    source = multiply(_input("left"), _input("right"))
    first = add(source)
    final = add(first)
    for retained in (source, first):
        assert not _plan(Program({"result": final, "retained": retained})).elided_nodes
    extra_reader = multiply(first, _input("extra"))
    assert not _plan(Program({"result": final, "branch": extra_reader})).elided_nodes


def test_value_metadata_cannot_be_forged_into_a_unit_copy() -> None:
    source = multiply(_input("left"), _input("right"))
    first = add(source)
    with pytest.raises(ValueError, match="declared add result"):
        Node(
            "add",
            (first,),
            replace(first.spec, differentiable=not first.spec.differentiable),
            first.attributes,
        )


def test_interleaved_execution_is_not_a_roundtrip() -> None:
    left, right = _input("left"), _input("right")
    source = multiply(left, right)
    first, independent = add(source), multiply(left, left)
    final = add(first)
    program = Program({"result": final, "other": independent})
    order = (left, right, source, independent, first, final)
    plan = analyze_native_copy_roundtrips(
        program, dimension_symbol=native._dim, execution_nodes=order
    )
    assert not plan.elided_nodes


def _native_replay(
    program: Program, values: np.ndarray, *, elide: bool
) -> tuple[np.ndarray, int]:
    """Independently execute native term/slot semantics, not TensorIR's zero seed."""
    nodes = program.dependency_order
    arena = plan_symbolic_arena(
        program, dimension_symbol=native._dim, execution_nodes=nodes
    )
    removed = set(_plan(program).elided_nodes) if elide else set()
    storage = [np.full(values.shape, 17.0) for _ in arena.slots]
    buffers = {}
    first_error = 0
    for number, node in enumerate(nodes):
        if node.op == "input":
            buffers[node] = (
                values if node.attrs["name"] == "source" else np.ones_like(values)
            )
            continue
        buffers[node] = storage[arena.node_slots[number]]
        if number in removed:
            continue
        operands = [buffers[source] for source in node.inputs]
        with np.errstate(invalid="ignore", over="ignore"):
            value = (
                operands[0] * operands[1]
                if node.op == "multiply"
                else 1.0 * operands[0]
            )
        if not np.isfinite(value).all() and not first_error:
            first_error = number + 1
        np.copyto(buffers[node], value)
    return buffers[program.outputs["result"]].view(np.uint64).copy(), first_error


@pytest.mark.parametrize("copies", range(2, 9))
def test_native_replay_is_bitwise_and_retains_first_error(copies: int) -> None:
    bits = np.array(
        [
            0,
            1 << 63,
            1,
            (1 << 63) | 1,
            0x3FF0000000000000,
            0x7FEFFFFFFFFFFFFF,
            0x7FF0000000000000,
            0x7FF8000000000017,
        ],
        dtype=np.uint64,
    )
    values = bits.view(np.float64)
    program = _chain(copies, size=len(values))
    original, original_error = _native_replay(program, values, elide=False)
    selected, selected_error = _native_replay(program, values, elide=True)
    assert np.array_equal(original, selected)
    assert original_error == selected_error != 0


def test_native_plan_does_not_change_tensor_signed_zero_semantics() -> None:
    program = _chain()
    before = program.dumps()
    _plan(program)
    actual = execute(program, {"source": np.full(3, -0.0), "ones": np.ones(3)})
    assert not np.signbit(actual.outputs["result"]).any()
    assert program.dumps() == before


def test_df_copy_plans_elide_only_the_proven_six_operations() -> None:
    for name in ("auxiliary_packed", "auxiliary_batched"):
        program = paired.programs()[name]
        plan = analyze_native_copy_roundtrips(
            program,
            dimension_symbol=native._dim,
            execution_nodes=native._execution_nodes(program),
        )
        assert len(plan.triples) == 3
        assert len(plan.elided_nodes) == 6
        arguments = {
            "input_overrides": {name: "s." + name for name in paired.INPUTS},
            "output_fields": paired.FIELDS,
            "batch_dim": name.endswith("batched"),
        }
        legacy = native._cuda_program(program, "proof", "Outputs", **arguments)
        selected = native._cuda_program(
            program,
            "proof",
            "Outputs",
            elide_native_copy_roundtrips=True,
            **arguments,
        )
        for number in plan.elided_nodes:
            assert f"proof_node_{number}<<<" in legacy
            assert f"proof_node_{number}<<<" not in selected
        assert selected.count("<<<") == legacy.count("<<<") - 6


@pytest.mark.parametrize(
    "arguments", ({"emit_kernels": False}, {"kernel_prefix": "external"})
)
def test_partial_kernel_schedules_refuse_copy_elision(arguments: dict) -> None:
    with pytest.raises(ValueError, match="complete original kernel schedule"):
        native._cuda_program(
            _chain(),
            "proof",
            "Outputs",
            elide_native_copy_roundtrips=True,
            **arguments,
        )


def test_cuda_copy_roundtrips_preserve_bits_canaries_and_first_error(
    tmp_path: Path,
) -> None:
    """Run both native schedules on exact IEEE input bits in a scheduled job."""
    if os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("explicit scheduled real-GPU qualification is required")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires CUDA and the existing compiler cache")
    root = Path(__file__).resolve().parents[2]
    write_df_cpu_headers(tmp_path)
    programs = []
    for copies in (2, 5):
        program = _chain(copies, size=16)
        assert len(_plan(program).elided_nodes) >= 2
        for selected in (False, True):
            prefix = f"{'selected' if selected else 'original'}_{copies}"
            programs.append(
                native._cuda_program(
                    program,
                    prefix,
                    "Outputs",
                    input_overrides={"source": "s.source", "ones": "s.ones"},
                    state_type="State",
                    arena_field="arena",
                    output_fields=("result",),
                    reset_error=False,
                    elide_native_copy_roundtrips=selected,
                )
            )
    (tmp_path / "generated_native_copy_roundtrips.cuh").write_text("\n".join(programs))
    executable = tmp_path / "probe"
    command = [
        cache,
        compiler,
        "-std=c++20",
        "-O2",
        "-arch=sm_120",
        "-I" + str(tmp_path),
        "-I" + str(root / "src"),
        "-I" + str(root / "include"),
        str(root / "tests/native/native_copy_roundtrip_probe.cu"),
        "-lcublas",
        "-o",
        str(executable),
    ]
    build = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=180
    )
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "bitwise, canaries, and first-error gates passed"
