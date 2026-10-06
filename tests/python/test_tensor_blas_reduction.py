from __future__ import annotations

from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    input_tensor,
    reduce_sum,
)
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract
from generativeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from generativeqc_compiler.tensor.cuda_providers import tensor_lowering_diagnostics
from generativeqc_compiler.tensor.optimize import prepare_for_backend

TARGET = cuda_target_info("sm_80")


def _partial_force_program(pages: int = 64, coordinates: int = 21) -> Program:
    page = Index("page", IndexSpace("force_pages", "batch", pages))
    coordinate = Index(
        "coordinate", IndexSpace("force_coordinates", "matrix", coordinates)
    )
    partials = input_tensor(
        "partials",
        TensorSpec((page, coordinate), dtype="float64", role="input"),
    )
    return Program({"force": reduce_sum(partials, (0,))})


def test_dense_force_reduction_can_lower_through_existing_cublas_path() -> None:
    program = _partial_force_program()
    baseline = plan_cuda(program, TARGET)
    candidate = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(blas_reductions=True),
    )

    assert any(step.node.op == "reduce" for step in baseline.steps)
    assert not any(
        step.node.op == "einsum" and step.gemm != "none" for step in baseline.steps
    )

    gemms = [
        step
        for step in candidate.steps
        if step.node.op == "einsum" and step.gemm != "none"
    ]
    assert len(gemms) == 1
    contract = gemm_contract(gemms[0].node)
    assert contract is not None
    assert (contract.m, contract.n, contract.k) == (21, 1, 64)

    report = tensor_lowering_diagnostics(candidate)
    selected = next(
        row
        for row in report["candidates"]
        if row["implementation"].startswith("tensor-gemm-")
    )
    assert [provider["name"] for provider in selected["providers"]] == [
        "nvidia.cublas",
        "generativeqc.generated_cuda",
    ]


def test_blas_reduction_retains_source_equation_identity() -> None:
    program = _partial_force_program()
    prepared = prepare_for_backend(program, "cuda")
    candidate = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(blas_reductions=True),
    )

    marker = candidate.program.provenance["cuda_blas_reduction"]
    assert marker["source_equation"] == prepared.logical_hash
    assert marker["rewritten_nodes"] == 1
    assert candidate.precision_schedule.source_equation == prepared.logical_hash
    assert candidate.program.logical_hash != prepared.logical_hash


def test_small_dense_reduction_keeps_generated_path() -> None:
    program = _partial_force_program(pages=16)
    candidate = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(blas_reductions=True),
    )

    assert any(step.node.op == "reduce" for step in candidate.steps)
    assert not any(step.node.op == "einsum" for step in candidate.steps)
    assert "cuda_blas_reduction" not in candidate.program.provenance
