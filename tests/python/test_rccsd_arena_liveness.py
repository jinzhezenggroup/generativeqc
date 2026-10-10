"""Protect runtime-shape arena lifetimes and shared CPU/CUDA admission."""

from collections import defaultdict

import pytest
from generativeqc_compiler.tensor.native_arena import SymbolicArenaPlan
from generativeqc_compiler.tensor.program import Program

from tools import generate_rccsd_native as codegen

Programs = tuple[dict[str, Program], dict[str, Program]]


@pytest.fixture(scope="module")
def programs() -> Programs:
    """Collect every production graph at the actual emission boundary."""
    cpu, cuda = {}, {}
    with pytest.MonkeyPatch.context() as patch:

        def collect_cpu(
            program: Program, name: str, *args: object, **kwargs: object
        ) -> str:
            cpu[name.removeprefix("run_").removesuffix("_cpu")] = program
            return ""

        def collect_cuda(
            program: Program, name: str, *args: object, **kwargs: object
        ) -> str:
            cuda[name] = program
            return ""

        patch.setattr(codegen, "_cpu_function", collect_cpu)
        patch.setattr(codegen, "_cuda_program", collect_cuda)
        codegen.cpu_header()
        codegen.cuda_source()
    return cpu, cuda


def test_cuda_uses_the_same_plan_as_host_admission(programs: Programs) -> None:
    """CUDA owners reserve through the CPU header: both plans must agree."""
    cpu, cuda = programs
    assert {
        "iteration",
        "iteration_prepared",
        "replay_strict",
        "replay_reassociated",
        "lambda_transpose",
        "hamiltonian_weights",
    } <= cuda.keys()
    assert any(
        codegen._packed_matrix_gemm(node) is not None
        for node in cuda["iteration_prepared"].live_nodes
    )
    for name, program in cuda.items():
        cpu_name = "iteration" if name == "iteration_prepared" else name
        assert program.logical_hash == cpu[cpu_name].logical_hash, name
        assert codegen._arena_plan(program) == codegen._arena_plan(cpu[cpu_name]), name


def test_every_slot_excludes_live_inputs_and_retains_all_outputs(
    programs: Programs,
) -> None:
    """Check pairwise intervals, independently of the allocator's free lists."""
    cpu, _ = programs
    for name, program in cpu.items():
        nodes = codegen._execution_nodes(program)
        numbers = {id(node): index for index, node in enumerate(nodes)}
        outputs = {id(node) for node in program.outputs.values()}
        consumers = defaultdict(list)
        for index, node in enumerate(nodes):
            for source in node.inputs:
                consumers[id(source)].append(index)
        plan = codegen._arena_plan(program)
        assert isinstance(plan, SymbolicArenaPlan)
        owners = defaultdict(list)
        for number, slot in plan.node_slots.items():
            node = nodes[number]
            assert node.op != "input"
            assert plan.slots[slot] == tuple(
                sorted(codegen._dim(index) for index in node.spec.indices)
            )
            end = len(nodes) if id(node) in outputs else max(consumers[id(node)])
            for previous_start, previous_end in owners[slot]:
                assert previous_end < number or end < previous_start, (name, slot)
            owners[slot].append((number, end))
        assert set(plan.node_slots) == {
            numbers[id(node)] for node in nodes if node.op != "input"
        }
        assert len(plan.slots) < len(plan.node_slots), name


@pytest.mark.parametrize("o,v,q", [(1, 1, 1), (2, 7, 3), (7, 2, 5), (20, 80, 16)])
def test_arena_capacity_reduces_without_shape_specialization(
    programs: Programs, o: int, v: int, q: int
) -> None:
    cpu, _ = programs
    extents = {"o": o, "v": v, "n": o + v, "q": q}

    def elements(shape: tuple[str, ...]) -> int:
        result = 1
        for axis in shape:
            result *= extents[axis]
        return result

    for name in (
        "iteration",
        "replay_strict",
        "replay_reassociated",
        "lambda_transpose",
        "triples_response",
    ):
        program = cpu[name]
        plan = codegen._arena_plan(program)
        old = sum(
            elements(tuple(codegen._dim(index) for index in node.spec.indices))
            for node in program.live_nodes
            if node.op != "input"
        )
        new = sum(elements(shape) for shape in plan.slots)
        assert 0 < new < old, (name, o, v, q, old, new)
