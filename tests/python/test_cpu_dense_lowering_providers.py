"""CPU and CUDA dense providers share one canonical TensorIR operation boundary."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.specialization import TargetCapabilities
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.cpu_providers import (
    CPU_SCALAR_PROVIDER,
    OPENBLAS_PROVIDER,
    cpu_dense_provider_candidates,
)
from generativeqc_compiler.tensor.cuda_cublaslt import CublasLtMatmulProvider
from generativeqc_compiler.tensor.cuda_cutlass import CutlassAotProvider
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.matrix_view import matrix_contraction

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

CPU_TARGET = TargetCapabilities(
    TargetInfo("cpu", "x86_64-generic", None, 1, None),
    features=(
        ("cpu-linalg-provider-threads", 1),
        ("cpu-linalg-thread-ownership", "task-parallel"),
        ("openblas", True),
        ("openblas-global-thread-control", False),
        ("openblas-local-thread-control", True),
        ("openblas-version", "0.3.34"),
    ),
)


def _requests(
    equation: str = "mk,kn->mn", dtype: str = "float64"
) -> tuple[LoweringRequest, LoweringRequest]:
    sizes = {"b": 2, "m": 3, "k": 5, "n": 7}
    axes = {
        label: Index(label, IndexSpace(label, "batch", size))
        for label, size in sizes.items()
    }
    labels = equation.split("->")[0].split(",")
    inputs = [
        input_tensor(
            f"input{i}",
            TensorSpec(tuple(axes[x] for x in operand), dtype=dtype, role="input"),
        )
        for i, operand in enumerate(labels)
    ]
    node = einsum(equation, *inputs)
    program = Program({"result": node})
    adapter = TensorLoweringAdapter(program)
    return (
        adapter.request(node, backend="cpu"),
        adapter.request(node, backend="cuda"),
    )


def test_cpu_and_cuda_share_semantic_identity_and_matrix_layout_proof() -> None:
    cpu, cuda = _requests()
    assert cpu.identity != cuda.identity
    assert cpu.semantic_identity == cuda.semantic_identity
    assert cpu.scientific_identity == cuda.scientific_identity

    cpu_recipe = matrix_contraction(cpu)
    cuda_recipe = matrix_contraction(cuda)
    assert cpu_recipe.layouts == cuda_recipe.layouts
    assert cpu_recipe.batches == cuda_recipe.batches == 1


def test_cpu_scalar_and_openblas_are_candidates_for_the_same_request() -> None:
    cpu, _ = _requests()
    offers = cpu_dense_provider_candidates(cpu, CPU_TARGET)
    assert len(offers) == 2
    assert all(candidate.request == cpu for candidate in offers)
    assert {candidate.providers[0].name for candidate in offers} == {
        CPU_SCALAR_PROVIDER.name,
        OPENBLAS_PROVIDER.name,
    }
    assert {candidate.status for candidate in offers} == {"ready"}
    by_provider = {candidate.providers[0].name: candidate for candidate in offers}
    assert by_provider[CPU_SCALAR_PROVIDER.name].execution is not None
    assert by_provider[CPU_SCALAR_PROVIDER.name].execution.determinism == "exact-order"
    assert by_provider[OPENBLAS_PROVIDER.name].execution is not None
    assert (
        dict(by_provider[OPENBLAS_PROVIDER.name].provenance)["thread_ownership"]
        == "task-parallel"
    )


def test_missing_openblas_thread_control_keeps_negative_evidence() -> None:
    cpu, _ = _requests()
    target = replace(
        CPU_TARGET,
        features=tuple(
            (name, False if name == "openblas-local-thread-control" else value)
            for name, value in CPU_TARGET.features
        ),
    )
    offers = cpu_dense_provider_candidates(cpu, target)
    scalar = next(row for row in offers if row.providers[0] == CPU_SCALAR_PROVIDER)
    openblas = next(
        row for row in offers if row.providers[0].name == OPENBLAS_PROVIDER.name
    )
    assert scalar.status == "ready"
    assert openblas.status == "unsupported"
    assert openblas.reason and "thread-local" in openblas.reason


def test_current_cpu_runtime_limits_are_explicit_provider_rejections() -> None:
    batched, _ = _requests("bmk,bkn->bmn")
    assert {
        row.status for row in cpu_dense_provider_candidates(batched, CPU_TARGET)
    } == {"unsupported"}

    fp32, _ = _requests(dtype="float32")
    offers = cpu_dense_provider_candidates(fp32, CPU_TARGET)
    assert {row.status for row in offers} == {"unsupported"}
    assert all(row.reason and "FP64" in row.reason for row in offers)

    padded, _ = _requests()
    a, b, output = padded.operands
    padded = replace(padded, operands=(replace(a, strides=(6, 1)), b, output))
    offers = cpu_dense_provider_candidates(padded, CPU_TARGET)
    assert {row.status for row in offers} == {"unsupported"}
    assert all(row.reason and "compact matrix views" in row.reason for row in offers)


def test_cuda_matrix_providers_do_not_accept_cpu_requests_after_shared_proof() -> None:
    cpu, _ = _requests()
    cublaslt = CublasLtMatmulProvider(version="12.9.1").candidates(cpu, CPU_TARGET)[0]
    cutlass = CutlassAotProvider(version="3.9.2").candidates(cpu, CPU_TARGET)[0]
    assert cublaslt.status == "unsupported"
    assert cublaslt.reason and "CUDA" in cublaslt.reason
    assert cutlass.status == "unsupported"
    assert cutlass.reason and "CUDA" in cutlass.reason
