"""Semantic admission and allocated-device parity for shared streamed work."""

import os
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    divide,
    einsum,
    input_tensor,
    linearize,
    multiply,
    reduce_sum,
    transpose,
    transpose_program,
)
from generativeqc_compiler.tensor.cuda_plan import (
    TensorSchedule,
    estimated_cuda_launches,
    plan_cuda,
)
from generativeqc_compiler.tensor.cuda_reduction import streamed_reduction_fusion_groups
from generativeqc_compiler.tensor.cuda_search import estimate_schedule, execution_key
from generativeqc_compiler.tensor.interpreter import execute


def shared_program(
    q: int = 7, k: int = 129, dtype: str = "float64", lanes: int = 2
) -> Program:
    axes = (
        Index("q", IndexSpace("q", "batch", q)),
        Index("k", IndexSpace("k", "batch", k)),
    )
    a = input_tensor(
        "a", TensorSpec(axes, dtype=dtype, role="parameter", differentiable=True)
    )
    b = input_tensor(
        "b", TensorSpec(axes, dtype=dtype, role="parameter", differentiable=True)
    )
    shared = divide(multiply(a, b), add(a, b))
    outputs = {
        f"result_{i}": reduce_sum(
            einsum("qk,qk->q", shared, add(a, b, coefficients=(i + 1, 1))), (0,)
        )
        for i in range(lanes)
    }
    return Program(outputs)


def shared_plan(
    program: Program, enabled: bool = True, architecture: str = "sm_80"
) -> typing.Any:
    return plan_cuda(
        program,
        cuda_target_info(architecture),
        schedule=TensorSchedule(
            stream_reductions=True,
            streamed_gemm_reduction=True,
            shared_streamed_reductions=enabled,
        ),
        reassociate_contractions=False,
    )


@pytest.mark.parametrize("q,k", [(0, 129), (7, 0), (7, 31), (7, 32), (7, 129)])
def test_bounded_launch_partition_and_tail_domains(q: int, k: int) -> None:
    program = shared_program(q, k)
    baseline, candidate = shared_plan(program, False), shared_plan(program)
    admitted = bool(q and k >= 32)
    assert bool(candidate.executable_reduction_groups) == admitted
    assert baseline.executable_reduction_groups == ()
    assert candidate.arena_bytes == baseline.arena_bytes
    assert candidate.program.logical_hash == baseline.program.logical_hash
    assert candidate.identity != baseline.identity
    assert (execution_key(candidate) != execution_key(baseline)) == admitted
    assert estimated_cuda_launches(candidate) == estimated_cuda_launches(
        baseline
    ) - int(admitted)
    if admitted:
        before, after = estimate_schedule(baseline), estimate_schedule(candidate)
        assert after["estimated_effective_flops"] < before["estimated_effective_flops"]
        assert after["estimated_shared_bytes"] >= before["estimated_shared_bytes"]


def test_overlapping_candidates_form_bounded_disjoint_partition() -> None:
    plan = shared_plan(shared_program(lanes=9))
    claimed = []
    for group in plan.executable_reduction_groups:
        assert 2 <= len(group.steps) <= 4
        claimed.extend(group.steps)
    assert len(claimed) == len(set(claimed))
    assert len(claimed) >= 8


def test_existing_storage_and_late_input_preconditions_fail_closed() -> None:
    plan = shared_plan(shared_program())
    group = plan.executable_reduction_groups[0]
    steps = list(plan.steps)
    steps[group.steps[1]] = replace(
        steps[group.steps[1]], offset=steps[group.steps[0]].offset
    )
    assert replace(plan, steps=tuple(steps)).executable_reduction_groups == ()
    limited = replace(plan.target, shared_memory_per_block=1)
    assert replace(plan, target=limited).executable_reduction_groups == ()


def test_unproved_coordinates_retain_unfused_execution() -> None:
    axes = (
        Index("q", IndexSpace("q", "batch", 64)),
        Index("k", IndexSpace("k", "batch", 64)),
    )
    a = input_tensor("a", TensorSpec(axes, role="input"))
    b = input_tensor("b", TensorSpec(axes, role="input"))
    shared = multiply(a, b)
    plan = shared_plan(
        Program(
            {
                "rows": reduce_sum(shared, (1,)),
                "columns": reduce_sum(transpose(shared, (1, 0)), (1,)),
            }
        )
    )
    assert plan.executable_reduction_groups == ()


def test_recursive_scalar_expansion_has_finite_fallback() -> None:
    axes = (
        Index("q", IndexSpace("q", "batch", 7)),
        Index("k", IndexSpace("k", "batch", 64)),
    )
    a = input_tensor("a", TensorSpec(axes, role="input"))
    b = input_tensor("b", TensorSpec(axes, role="input"))
    shared = multiply(a, b)
    left, right = shared, shared
    for _ in range(12):
        left = multiply(left, left)
        right = add(right, a)
    plan = shared_plan(
        Program(
            {
                "left": reduce_sum(einsum("qk,qk->q", left, a), (0,)),
                "right": reduce_sum(einsum("qk,qk->q", right, b), (0,)),
            }
        )
    )
    assert streamed_reduction_fusion_groups(plan)
    assert plan.executable_reduction_groups == ()


@pytest.mark.parametrize("diamond", [False, True])
def test_deep_unshared_branch_preserves_shared_emission(diamond: bool) -> None:
    from generativeqc_compiler.tensor.cuda_emit import emit_cuda
    from generativeqc_compiler.tensor.cuda_resident_emit import resident_source

    axes = (
        Index("q", IndexSpace("q", "batch", 7)),
        Index("k", IndexSpace("k", "batch", 64)),
    )
    a, b, c = (
        input_tensor(name, TensorSpec(axes, role="input")) for name in ("a", "b", "c")
    )
    shared = multiply(a, b)
    unrelated = c
    for _ in range(1200):
        unrelated = multiply(unrelated, unrelated if diamond else c)
    program = Program(
        {
            "left": reduce_sum(add(unrelated, shared), (1,)),
            "right": reduce_sum(multiply(shared, a), (1,)),
        }
    )
    baseline, candidate = shared_plan(program, False), shared_plan(program)
    assert candidate.executable_reduction_groups
    assert candidate.program.logical_hash == baseline.program.logical_hash
    assert candidate.arena_bytes == baseline.arena_bytes
    for plan in (baseline, candidate):
        assert emit_cuda(plan)
        assert resident_source(plan)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU",
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("mode", ["ordinary", "resident"])
def test_source_matched_shared_execution_against_numpy_and_unfused(
    tmp_path: Path, dtype: str, mode: str
) -> None:
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_resident import (
        PreparedResident,
        compile_resident,
    )

    nvcc = find_nvcc()
    assert nvcc is not None
    arch = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(arch))
    program = shared_program(dtype=dtype)
    rng = np.random.default_rng(783)
    feeds = {
        "a": rng.uniform(0.3, 0.9, (7, 129)).astype(dtype),
        "b": rng.uniform(0.2, 0.8, (7, 129)).astype(dtype),
    }
    oracle = execute(program, feeds).outputs
    tolerance = 3e-6 if dtype == "float32" else 3e-13
    results = []
    for enabled in (False, True):
        plan = shared_plan(program, enabled, arch)
        assert bool(plan.executable_reduction_groups) == enabled
        compile_fn = compile_cuda if mode == "ordinary" else compile_resident
        owner = PreparedCuda if mode == "ordinary" else PreparedResident
        artifact = compile_fn(plan, compiler, tmp_path / str(enabled))
        with owner(plan, artifact) as prepared:

            def run(values: dict) -> dict:
                if mode == "ordinary":
                    return prepared.execute(values).outputs
                prepared.upload(values)
                leases, _ = prepared.run()
                return {
                    name: prepared.download(leases[name]) for name in program.outputs
                }

            for scale in (1, 2, 1):
                values = {name: value * scale for name, value in feeds.items()}
                result = run(values)
                for name in oracle:
                    np.testing.assert_allclose(
                        result[name],
                        oracle[name] * scale**2,
                        rtol=tolerance,
                        atol=tolerance,
                    )
            bad = {"a": feeds["a"], "b": -feeds["a"]}
            with pytest.raises(
                (RuntimeError, ValueError), match="arithmetic|division|TensorIR|tensor"
            ):
                run(bad)
            recovered = run(feeds)
            for name in oracle:
                np.testing.assert_allclose(
                    recovered[name], oracle[name], rtol=tolerance, atol=tolerance
                )
            results.append(recovered)
    for name in oracle:
        np.testing.assert_array_equal(results[0][name], results[1][name])


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_generated_response_graph_retains_independent_analytic_oracle(
    mode: str,
) -> None:
    program, feeds, expected = response_case(mode)
    actual = execute(program, feeds).outputs
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=3e-13, atol=3e-13)
    candidate = shared_plan(program)
    baseline = shared_plan(program, False)
    assert candidate.program.logical_hash == baseline.program.logical_hash


def response_case(mode: str) -> tuple[Program, dict, dict]:
    primal = shared_program()
    rng = np.random.default_rng(7831)
    a, b = rng.uniform(0.3, 0.9, (7, 129)), rng.uniform(0.2, 0.8, (7, 129))
    feeds = {"a": a, "b": b}
    derivatives = []
    for coefficient in (1, 2):
        ga = b**2 / (a + b) ** 2 * (coefficient * a + b) + coefficient * a * b / (a + b)
        gb = a**2 / (a + b) ** 2 * (coefficient * a + b) + a * b / (a + b)
        derivatives.append((ga, gb))
    if mode == "jvp":
        generated = linearize(primal, ["a", "b"])
        da, db = rng.normal(size=a.shape), rng.normal(size=b.shape)
        feeds.update(d_a=da, d_b=db)
        expected = {
            f"d_result_{i}": np.asarray(np.sum(ga * da + gb * db))
            for i, (ga, gb) in enumerate(derivatives)
        }
    else:
        generated = transpose_program(primal, list(primal.outputs), inputs=["a", "b"])
        feeds.update(bar_result_0=np.asarray(1.0), bar_result_1=np.asarray(2.0))
        expected = {
            f"bar_{name}": derivatives[0][axis] + 2 * derivatives[1][axis]
            for axis, name in enumerate(("a", "b"))
        }
    return generated.program, feeds, expected


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU",
)
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_generated_responses_execute_on_cuda(tmp_path: Path, mode: str) -> None:
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda

    nvcc = find_nvcc()
    assert nvcc is not None
    architecture = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
    program, feeds, expected = response_case(mode)
    for enabled in (False, True):
        plan = shared_plan(program, enabled, architecture)
        artifact = compile_cuda(plan, compiler, tmp_path / str(enabled))
        with PreparedCuda(plan, artifact) as owner:
            result = owner.execute(feeds)
            for name, value in expected.items():
                np.testing.assert_allclose(
                    result.outputs[name], value, rtol=3e-12, atol=3e-12
                )


def test_prepared_provider_recipes_preserve_unfused_fallback() -> None:
    from tools.tensor_reduction_qualification import qualification_reduction_plan

    plan = shared_plan(shared_program())
    assert plan.executable_reduction_groups
    for implementation in ("generated", "cub"):
        bound = qualification_reduction_plan(plan, implementation=implementation)
        assert bound.executable_reduction_groups == ()
        assert bound.program.logical_hash == plan.program.logical_hash


def alias_layout_plan(architecture: str = "sm_80") -> typing.Any:
    from generativeqc_compiler.common.layout import DenseLayout

    axes = tuple(
        Index(name, IndexSpace(name, "batch", size))
        for name, size in (("p", 3), ("q", 7), ("k", 129))
    )
    a = input_tensor("a", TensorSpec(axes, role="input"))
    b = input_tensor("b", TensorSpec(axes, role="input"))
    shared = transpose(multiply(a, b), (1, 0, 2))
    program = Program(
        {
            "a_total": reduce_sum(
                einsum("pqk,pqk->pq", shared, transpose(a, (1, 0, 2))), (0, 1)
            ),
            "b_total": reduce_sum(
                einsum("pqk,pqk->pq", shared, transpose(b, (1, 0, 2))), (0, 1)
            ),
        }
    )
    plan = shared_plan(program, True, architecture)
    assert plan.executable_reduction_groups
    members = {
        index for group in plan.executable_reduction_groups for index in group.steps
    }
    steps = tuple(
        replace(step, layout=DenseLayout(step.node.spec.shape, (1, 0), alignment=256))
        if index in members
        else step
        for index, step in enumerate(plan.steps)
    )
    return replace(plan, steps=steps)


def test_alias_frontier_layout_is_part_of_execution_identity() -> None:
    physical = alias_layout_plan()
    baseline = shared_plan(physical.program)
    assert physical.program.logical_hash == baseline.program.logical_hash
    assert physical.executable_reduction_groups
    assert physical.arena_bytes == baseline.arena_bytes
    assert execution_key(physical) != execution_key(baseline)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU",
)
def test_shared_alias_layout_and_cuda_graph_replay(tmp_path: Path) -> None:
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda

    nvcc = find_nvcc()
    assert nvcc is not None
    architecture = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
    plan = alias_layout_plan(architecture)
    artifact = compile_cuda(plan, compiler, tmp_path)
    rng = np.random.default_rng(7833)
    a, b = rng.normal(size=(3, 7, 129)), rng.normal(size=(3, 7, 129))
    expected = {"a_total": np.sum(a * b * a), "b_total": np.sum(a * b * b)}
    with PreparedCuda(plan, artifact, execution_mode="cuda-graph") as owner:
        for _ in range(3):
            result = owner.execute({"a": a, "b": b})
            for name, value in expected.items():
                np.testing.assert_allclose(
                    result.outputs[name], value, rtol=3e-12, atol=3e-12
                )
        assert result.metrics["graph_mode"] == "replay"
        assert result.metrics["graph_captures"] == 1
        with pytest.raises(ValueError, match="non-finite"):
            owner.execute({"a": np.full_like(a, np.nan), "b": b})
        recovered = owner.execute({"a": a, "b": b})
        for name, value in expected.items():
            np.testing.assert_allclose(
                recovered.outputs[name], value, rtol=3e-12, atol=3e-12
            )


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU",
)
def test_widened_accumulation_and_cub_oracle(tmp_path: Path) -> None:
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.precision import (
        PrecisionDirective,
        lower_precision,
    )

    from tools.tensor_reduction_qualification import qualification_reduction_plan

    nvcc = find_nvcc()
    assert nvcc is not None
    architecture = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_90")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
    program = shared_program()
    program = lower_precision(
        program,
        {
            program.debug_names[output.inputs[0]]: PrecisionDirective(
                "float32",
                "float32",
                "float64",
                qualification="fixture/shared-widened-accumulation",
            )
            for output in program.outputs.values()
        },
    )
    rng = np.random.default_rng(7832)
    a, b = rng.uniform(0.3, 0.9, (7, 129)), rng.uniform(0.2, 0.8, (7, 129))
    shared = (a * b / (a + b)).astype(np.float32)
    expected = {}
    for i in (0, 1):
        operand = ((i + 1) * a + b).astype(np.float32)
        rows = (shared * operand).astype(np.float64).sum(axis=1).astype(np.float32)
        expected[f"result_{i}"] = rows.astype(np.float64).sum()
    fused = shared_plan(program, True, architecture)
    assert fused.executable_reduction_groups
    assert all(
        group.accumulation_dtype == "float64"
        for group in fused.executable_reduction_groups
    )
    baseline = shared_plan(program, False, architecture)
    cub = qualification_reduction_plan(fused, compiler=compiler, implementation="cub")
    assert cub.executable_reduction_groups == ()
    for label, plan in (("unfused", baseline), ("shared", fused), ("cub", cub)):
        artifact = compile_cuda(plan, compiler, tmp_path / label)
        with PreparedCuda(plan, artifact) as owner:
            result = owner.execute({"a": a, "b": b})
            for name, value in expected.items():
                np.testing.assert_allclose(
                    result.outputs[name], value, rtol=2e-7, atol=2e-7
                )
