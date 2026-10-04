"""cuTENSOR offers share canonical TensorIR identity and retain negative evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.lowering_contract import (
    LoweringConstraints,
    LoweringPrecision,
)
from generativeqc_compiler.common.lowering_provider import collect_lowering_candidates
from generativeqc_compiler.common.precision import (
    CastBoundary,
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)
from generativeqc_compiler.common.specialization import TargetCapabilities
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.cuda_cutensor import (
    CutensorContractionProvider,
    cutensor_contract,
    cutensor_opportunities,
    cutensor_provider_candidates,
)
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.tensor.cuda_providers import resolved_lowering_candidates

if TYPE_CHECKING:
    from generativeqc_compiler.tensor.cuda_plan import TensorPlan

TARGET = cuda_target_info("sm_80")
CAPABILITIES = TargetCapabilities(
    TARGET.target_info,
    features=(
        ("cutensor", True),
        ("cutensor-version", "2.3.0"),
        ("cutensor-workspace-bytes", 512),
        ("cutensor-provider-bytes", 1024),
        ("cutensor-host-bytes", 128),
        ("cutensor-cache-bytes", 256),
    ),
)


def packed_plan(dtype: str = "float64") -> tuple[TensorPlan, int]:
    """A transposed result forces material packing in the incumbent lowering."""
    i, j, k = (
        Index(name, IndexSpace(name, "batch", size))
        for name, size in zip("ijk", (7, 11, 13), strict=True)
    )
    a = input_tensor("a", TensorSpec((i, k), dtype=dtype, role="input"))
    b = input_tensor("b", TensorSpec((k, j), dtype=dtype, role="input"))
    plan = plan_cuda(Program({"result": einsum("ik,kj->ji", a, b)}), TARGET)
    index = next(i for i, step in enumerate(plan.steps) if step.gemm == "packed")
    return plan, index


@pytest.mark.parametrize("dtype,itemsize", [("float32", 4), ("float64", 8)])
def test_opportunity_and_incumbent_use_the_exact_same_request(
    dtype: str, itemsize: int
) -> None:
    plan, index = packed_plan(dtype)
    before = plan.identity
    contract = cutensor_contract(plan, index)
    assert contract is not None
    assert cutensor_opportunities(plan) == (contract,)
    assert (contract.a_extents, contract.b_extents, contract.c_extents) == (
        (7, 13),
        (13, 11),
        (11, 7),
    )
    assert (contract.a_strides, contract.b_strides, contract.c_strides) == (
        (13, 1),
        (11, 1),
        (7, 1),
    )
    assert contract.flops == 2 * 7 * 11 * 13
    assert contract.packing_bytes_avoided == 2 * itemsize * (7 * 13 + 13 * 11 + 7 * 11)
    incumbent = next(
        row
        for row in resolved_lowering_candidates(plan)
        if dict(row.provenance)["step_index"] == index
    )
    candidate = cutensor_provider_candidates(
        plan, index, target_capabilities=CAPABILITIES
    )[0]
    assert candidate.status == "ready"
    assert candidate.request == incumbent.request
    assert candidate.request.operation == "einsum"
    assert candidate.request.semantic_identity == contract.semantic_identity
    assert "packing_bytes_avoided" not in dict(candidate.request.semantics)
    assert candidate.execution is not None
    assert candidate.execution.precision == incumbent.execution.precision
    assert candidate.execution.layouts == candidate.request.operands
    assert (
        candidate.workspace_bytes,
        candidate.provider_bytes,
        candidate.execution.host_bytes,
        candidate.execution.cache_bytes,
    ) == (512, 1024, 128, 256)
    assert candidate.providers[0].version == "2.3.0"
    assert candidate.cost is None
    assert plan.identity == before


def test_every_admitted_precision_survives_provider_rejection() -> None:
    plan, index = packed_plan()
    request = cutensor_provider_candidates(plan, index)[0].request
    mixed = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (
                (
                    "operation",
                    PrecisionDirective(
                        "float64", "float32", "float32", "test-mixed-gate"
                    ),
                ),
            )
        ),
        "operation",
        ("float64", "float64"),
        "float64",
    )
    casts = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (
                (
                    "operation",
                    PrecisionDirective(
                        "float32", "float32", "float32", "test-cast-gate"
                    ),
                ),
            )
        ),
        "operation",
        ("float32", "float32"),
        "float64",
        casts=(
            CastBoundary("input:0", "float64", "float32", 91, 728, 364),
            CastBoundary("input:1", "float64", "float32", 143, 1144, 572),
            CastBoundary("output", "float32", "float64", 77, 308, 616),
        ),
    )
    joint = replace(request, precisions=(*request.precisions, mixed, casts))
    assert joint.semantic_identity == request.semantic_identity
    offers = collect_lowering_candidates(
        joint, CAPABILITIES, (CutensorContractionProvider(version="2.3.0"),)
    )
    assert len(offers) == 3
    by_precision = {row.execution.precision.identity: row for row in offers}
    assert by_precision[request.precisions[0].identity].status == "ready"
    assert "mixed arithmetic" in by_precision[mixed.identity].reason
    assert "cast/refinement/audit" in by_precision[casts.identity].reason
    assert all(row.request == joint for row in offers)


@pytest.mark.parametrize(
    "missing",
    [
        "cutensor",
        "cutensor-version",
        "cutensor-workspace-bytes",
        "cutensor-provider-bytes",
        "cutensor-host-bytes",
        "cutensor-cache-bytes",
    ],
)
def test_each_capability_and_resource_bound_is_required(missing: str) -> None:
    plan, index = packed_plan()
    target = replace(
        CAPABILITIES,
        features=tuple(row for row in CAPABILITIES.features if row[0] != missing),
    )
    offer = cutensor_provider_candidates(plan, index, target_capabilities=target)[0]
    assert offer.status == "unsupported" and offer.reason


@pytest.mark.parametrize("value", [True, -1, "1024", 1.5])
def test_resource_bound_requires_a_nonnegative_integer(value: object) -> None:
    plan, index = packed_plan()
    features = dict(CAPABILITIES.features)
    features["cutensor-provider-bytes"] = value
    target = replace(CAPABILITIES, features=tuple(features.items()))
    row = cutensor_provider_candidates(plan, index, target_capabilities=target)[0]
    assert row.status == "unsupported" and "provider byte bound" in row.reason


@pytest.mark.parametrize(
    "limits,reason",
    [
        (
            LoweringConstraints(additional_device_bytes=512 + 1024 + 256 - 1),
            "additional_device_bytes",
        ),
        (LoweringConstraints(host_bytes=127), "host_bytes"),
        (LoweringConstraints(capture_required=True), "capture/replay"),
        (LoweringConstraints(determinism="exact-order"), "determinism"),
    ],
)
def test_shared_resource_capture_and_order_gates_apply(
    limits: LoweringConstraints, reason: str
) -> None:
    plan, index = packed_plan()
    request = replace(
        cutensor_provider_candidates(plan, index)[0].request, constraints=limits
    )
    row = collect_lowering_candidates(
        request, CAPABILITIES, (CutensorContractionProvider(version="2.3.0"),)
    )[0]
    assert row.status == "unsupported" and reason in row.reason


def test_virtual_overlapping_aliased_and_diagonal_views_are_rejected() -> None:
    plan, index = packed_plan()
    request = cutensor_provider_candidates(plan, index)[0].request
    a, b, c = request.operands
    provider = CutensorContractionProvider(version="2.3.0")
    for operands, reason in (
        ((replace(a, strides=None), b, c), "materialized"),
        ((replace(a, strides=(1, 1)), b, c), "nonoverlapping"),
        (
            (replace(a, alias_group="borrowed"), b, replace(c, alias_group="borrowed")),
            "overwrite",
        ),
        (
            (
                replace(a, modes=(0, 0), shape=(7, 7)),
                replace(b, modes=(2, 3)),
                replace(c, modes=(3, 2), shape=(11, 13)),
            ),
            "diagonals",
        ),
    ):
        row = provider.candidates(replace(request, operands=operands), CAPABILITIES)[0]
        assert row.status == "unsupported" and reason in row.reason


def test_foreign_target_and_version_cannot_reuse_capabilities() -> None:
    plan, index = packed_plan()
    with pytest.raises(ValueError, match="planned target"):
        cutensor_provider_candidates(
            plan,
            index,
            target_capabilities=replace(
                CAPABILITIES, target=cuda_target_info("sm_120").target_info
            ),
        )
    request = cutensor_provider_candidates(plan, index)[0].request
    row = CutensorContractionProvider(version="2.4.0").candidates(
        request, CAPABILITIES
    )[0]
    assert row.status == "unsupported" and "version does not match" in row.reason
