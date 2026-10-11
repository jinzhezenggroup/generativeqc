"""Reduction implementations are prepared bindings, not scientific schedules."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.lowering_selection import select_lowering_binding
from generativeqc_compiler.common.provenance import find_nvcc
from generativeqc_compiler.common.specialization import TargetCapabilities
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PrecisionDirective,
    Program,
    TensorSpec,
    cuda_execute,
    input_tensor,
    lower_precision,
    reduce_sum,
)
from generativeqc_compiler.tensor.cuda_emit import emit_cuda
from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from generativeqc_compiler.tensor.cuda_plan import TensorPlan, TensorSchedule, plan_cuda
from generativeqc_compiler.tensor.cuda_providers import (
    bind_reduction_lowering,
    reduction_provider_candidates,
    tensor_lowering_diagnostics,
)
from generativeqc_compiler.tensor.cuda_search import execution_key

from tools.tensor_reduction_qualification import qualification_reduction_plan


def _plan(mode: str = "float64", architecture: str = "sm_80") -> TensorPlan:
    rows = Index("row", IndexSpace("rows", "batch", 3))
    inner = Index("inner", IndexSpace("reduction", "batch", 129))
    values = input_tensor(
        "values",
        TensorSpec(
            (rows, inner), dtype="float64" if mode == "mixed" else mode, role="input"
        ),
    )
    reduced = reduce_sum(values, (1,))
    program = Program({"result": reduced})
    if mode == "mixed":
        program = lower_precision(
            program,
            {
                program.debug_names[reduced]: PrecisionDirective(
                    "float32",
                    "float32",
                    "float64",
                    qualification="fixture/mixed-reduction",
                )
            },
        )
    return plan_cuda(
        program,
        cuda_target_info(architecture),
        schedule=TensorSchedule(stream_reductions=True),
    )


@pytest.mark.parametrize("mode", ["float32", "float64", "mixed"])
def test_binding_changes_execution_not_science_or_schedule(mode: str) -> None:
    plan = _plan(mode)
    bound = qualification_reduction_plan(plan)
    index, binding = bound.lowering_bindings[0]
    assert plan.program is bound.program
    assert plan.schedule == bound.schedule
    assert plan.precision_schedule.identity == bound.precision_schedule.identity
    assert execution_key(plan) != execution_key(bound)
    assert plan.identity != bound.identity
    assert binding.selected.request == binding.fallbacks[0].request
    assert (
        binding.selected.execution.precision == binding.fallbacks[0].execution.precision
    )
    source = emit_cuda(bound)
    assert "cub::BlockReduce<" in source
    if mode == "mixed":
        assert "cub::BlockReduce<\n        double, 128" in source
    report = tensor_lowering_diagnostics(bound)
    assert report["bindings"][0]["step"] == index
    assert report["bindings"][0]["fallbacks"] == [binding.fallbacks[0].identity]


def test_selection_counts_complete_cost_and_preserves_strict_fallback() -> None:
    plan = _plan()
    index, fixture = qualification_reduction_plan(plan).lowering_bindings[0]
    candidates = tuple(
        replace(candidate, cost=replace(candidate.cost, prepare_ns=100))
        if candidate.execution.algorithm == "cub-block-reduce"
        else candidate
        for candidate in fixture.candidates
    )
    cold = select_lowering_binding(
        fixture.selected.request,
        fixture.target,
        fixture.compilation,
        candidates,
    )
    warm = select_lowering_binding(
        fixture.selected.request,
        fixture.target,
        fixture.compilation,
        candidates,
        expected_replays=200,
    )
    assert cold.selected.execution.algorithm == "generated-cooperative"
    assert warm.selected.execution.algorithm == "cub-block-reduce"
    assert warm.fallbacks[0].execution.algorithm == "generated-cooperative"
    assert "cub::BlockReduce" not in emit_cuda(
        bind_reduction_lowering(plan, index, cold)
    )
    assert "cub::BlockReduce" in emit_cuda(bind_reduction_lowering(plan, index, warm))


def test_missing_header_selects_generated_and_retains_rejection() -> None:
    plan = _plan()
    index, fixture = qualification_reduction_plan(plan).lowering_bindings[0]
    target = TargetCapabilities(plan.target.target_info)
    candidates = tuple(
        replace(
            candidate,
            providers=tuple(
                replace(provider, version="fixture") for provider in candidate.providers
            ),
            cost=fixture.selected.cost,
        )
        for candidate in reduction_provider_candidates(
            plan, index, target_capabilities=target
        )
    )
    binding = select_lowering_binding(
        candidates[0].request,
        target,
        fixture.compilation,
        candidates,
    )
    bound = bind_reduction_lowering(plan, index, binding)
    assert binding.selected.execution.algorithm == "generated-cooperative"
    assert any("header" in reason for _, reason in binding.rejections)
    assert "cub::BlockReduce" not in emit_cuda(bound)


@pytest.mark.parametrize("debt", ["versions", "cost"])
def test_missing_evidence_cannot_select_an_alternative(debt: str) -> None:
    plan = _plan()
    _, fixture = qualification_reduction_plan(plan).lowering_bindings[0]
    candidates = tuple(
        replace(candidate, cost=None)
        if debt == "cost"
        else replace(
            candidate,
            providers=tuple(
                replace(provider, version=None) for provider in candidate.providers
            ),
        )
        for candidate in fixture.candidates
    )
    with pytest.raises(ValueError, match="no admitted"):
        select_lowering_binding(
            fixture.selected.request, fixture.target, fixture.compilation, candidates
        )


def test_only_generated_can_be_retained_as_unmeasured_incumbent() -> None:
    plan = _plan()
    index, fixture = qualification_reduction_plan(plan).lowering_bindings[0]
    candidates = tuple(
        replace(candidate, cost=None) for candidate in fixture.candidates
    )
    for candidate in candidates:
        retained = select_lowering_binding(
            candidate.request,
            fixture.target,
            fixture.compilation,
            candidates,
            qualified_incumbent=candidate.identity,
        )
        if candidate.execution.algorithm == "cub-block-reduce":
            with pytest.raises(ValueError, match="qualified unmeasured incumbent"):
                bind_reduction_lowering(plan, index, retained)
        else:
            assert (
                bind_reduction_lowering(plan, index, retained)
                .lowering_bindings[0][1]
                .retained_incumbent
            )


@pytest.mark.parametrize(
    "change", ["schedule", "threads", "unroll", "target", "duplicate", "index"]
)
def test_stale_or_malformed_plan_replacements_fail_closed(change: str) -> None:
    bound = qualification_reduction_plan(_plan())
    index, binding = bound.lowering_bindings[0]
    if change == "schedule":
        changes = {"schedule": TensorSchedule()}
    elif change == "threads":
        changes = {"schedule": replace(bound.schedule, threads=64)}
    elif change == "unroll":
        changes = {"schedule": replace(bound.schedule, reduction_unroll=2)}
    elif change == "target":
        changes = {"target": cuda_target_info("sm_120")}
    elif change == "duplicate":
        changes = {"lowering_bindings": ((index, binding), (index, binding))}
    else:
        changes = {"lowering_bindings": ((len(bound.steps), binding),)}
    with pytest.raises(ValueError):
        replace(bound, **changes)


@pytest.mark.parametrize("change", ["algorithm", "workspace", "determinism"])
def test_forged_adapter_contract_is_not_executable(change: str) -> None:
    plan = _plan()
    index, fixture = qualification_reduction_plan(plan).lowering_bindings[0]
    selected = fixture.selected
    if change == "workspace":
        selected = replace(selected, workspace_bytes=256)
    else:
        execution = replace(
            selected.execution,
            **(
                {"algorithm": "unimplemented"}
                if change == "algorithm"
                else {"determinism": "unspecified"}
            ),
        )
        selected = replace(selected, execution=execution)
    candidates = tuple(
        selected if candidate == fixture.selected else candidate
        for candidate in fixture.candidates
    )
    binding = select_lowering_binding(
        selected.request, fixture.target, fixture.compilation, candidates
    )
    with pytest.raises(ValueError, match="canonical provider offers"):
        bind_reduction_lowering(plan, index, binding)


def test_cost_only_updates_do_not_invalidate_executable_identity() -> None:
    plan = _plan()
    bound = qualification_reduction_plan(plan)
    index, fixture = bound.lowering_bindings[0]
    candidates = tuple(
        replace(candidate, cost=replace(candidate.cost, source="fixture/remeasured"))
        for candidate in fixture.candidates
    )
    binding = select_lowering_binding(
        fixture.selected.request, fixture.target, fixture.compilation, candidates
    )
    rebound = bind_reduction_lowering(plan, index, binding)
    assert bound.identity == rebound.identity
    assert emit_cuda(bound) == emit_cuda(rebound)


def test_binding_cannot_change_admitted_precision_context() -> None:
    index, binding = qualification_reduction_plan(_plan()).lowering_bindings[0]
    with pytest.raises(ValueError, match="prepared lowering context changed"):
        bind_reduction_lowering(_plan("float32"), index, binding)


def test_cached_binding_index_cannot_bypass_validation() -> None:
    bound = qualification_reduction_plan(_plan())
    index, binding = bound.lowering_bindings[0]
    with pytest.raises(TypeError):
        bound.lowering_by_step[index] = binding


def test_binding_pairs_cannot_mutate_plan_after_validation() -> None:
    plan = _plan()
    index, cub = qualification_reduction_plan(plan).lowering_bindings[0]
    _, generated = qualification_reduction_plan(
        plan, implementation="generated"
    ).lowering_bindings[0]
    pair = [index, cub]
    pairs = [pair]
    bound = replace(plan, lowering_bindings=pairs)
    source, identity = emit_cuda(bound), bound.identity
    pair[1] = generated
    pairs.clear()
    assert bound.lowering_bindings == ((index, cub),)
    assert bound.lowering_by_step[index] == cub
    assert bound.identity == identity
    assert emit_cuda(bound) == source


def test_qualification_compile_cli_keeps_checkout_pythonpath_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "tools/validate_tensor_cuda_compile.py", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "python")},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "--nvcc" in result.stdout


def test_compilation_rejects_stale_recipe_context_before_nvcc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = qualification_reduction_plan(_plan())
    compiler = CudaCompilerAdapter(Path("fixture/nvcc"), bound.target)
    monkeypatch.setattr(
        cuda_execute, "_tensor_compiler_facts", lambda *_: {"context": "changed"}
    )
    with pytest.raises(ValueError, match="compiler context changed"):
        compile_cuda(bound, compiler, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires an explicitly allocated GPU",
)
@pytest.mark.parametrize("mode", ["float32", "float64", "mixed"])
@pytest.mark.parametrize("implementation", ["generated", "cub"])
@pytest.mark.parametrize("execution_mode", ["ordinary", "cuda-graph"])
def test_bound_reduction_executes_and_replays_against_numpy(
    tmp_path: Path, mode: str, implementation: str, execution_mode: str
) -> None:
    nvcc = find_nvcc()
    assert nvcc is not None, "allocated GPU test requires NVCC"
    architecture = os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(architecture))
    plan = qualification_reduction_plan(
        _plan(mode, architecture), implementation=implementation, compiler=compiler
    )
    artifact = compile_cuda(plan, compiler, tmp_path)
    dtype = np.float32 if mode == "float32" else np.float64
    values = np.random.default_rng(1889).normal(size=(3, 129)).astype(dtype)
    if mode == "mixed":
        values.fill(0)
        values[:, :4] = [1e8, 1, -1e8, 1]
    expected = values.astype(np.float32 if mode == "mixed" else dtype).sum(
        axis=1, dtype=np.float64
    )
    with PreparedCuda(plan, artifact, execution_mode=execution_mode) as prepared:
        for scale in (1, 2, 1):
            result = prepared.execute({"values": values * scale})
            tolerance = 2e-6 if mode == "float32" else 2e-13
            np.testing.assert_allclose(
                result.outputs["result"],
                expected * scale,
                rtol=tolerance,
                atol=tolerance,
            )
        if execution_mode == "cuda-graph":
            if mode == "float64":
                assert result.metrics["graph_mode"] == "replay"
                assert result.metrics["graph_captures"] == 1
            else:
                assert result.metrics["graph_mode"] == "fallback"
                assert "precision" in result.metrics["graph_reason"]
                assert result.metrics["graph_captures"] == 0
        with pytest.raises(ValueError, match="non-finite"):
            prepared.execute({"values": np.full(values.shape, np.nan, dtype=dtype)})
        recovered = prepared.execute({"values": values})
        np.testing.assert_allclose(
            recovered.outputs["result"], expected, rtol=tolerance, atol=tolerance
        )
