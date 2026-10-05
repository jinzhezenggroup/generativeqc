"""cuBLASLt layout recipes preserve canonical science and negative evidence."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.lowering_contract import LoweringConstraints
from generativeqc_compiler.common.lowering_provider import collect_lowering_candidates
from generativeqc_compiler.common.precision import (
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
from generativeqc_compiler.tensor.cuda_cublaslt import (
    CublasLtMatmul,
    CublasLtMatmulProvider,
    cublaslt_matmul,
    cublaslt_provider_candidates,
)
from generativeqc_compiler.tensor.cuda_cutensor import CutensorContractionProvider
from generativeqc_compiler.tensor.cuda_plan import TensorPlan, plan_cuda
from generativeqc_compiler.tensor.cuda_providers import resolved_lowering_candidates

TARGET = cuda_target_info("sm_120")
CAPABILITIES = TargetCapabilities(
    TARGET.target_info,
    features=(
        ("cublaslt", True),
        ("cublaslt-version", "12.9.1"),
        ("cublaslt-workspace-bytes", 512),
        ("cublaslt-provider-bytes", 1024),
        ("cublaslt-host-bytes", 128),
        ("cublaslt-cache-bytes", 256),
    ),
)


def _plan(
    equation: str = "mk,kn->nm",
    dtype: str = "float64",
    sizes: tuple[int, int, int] = (3, 5, 7),
) -> tuple[TensorPlan, int]:
    """Build real TensorIR so tests cannot manufacture a provider-only request."""
    axes = {
        name: Index(name, IndexSpace(name, "batch", size))
        for name, size in zip("mknb", (*sizes, 2), strict=True)
    }
    labels = equation.split("->")[0].split(",")
    inputs = [
        input_tensor(
            f"input{i}",
            TensorSpec(tuple(axes[x] for x in labels[i]), dtype=dtype, role="input"),
        )
        for i in range(2)
    ]
    node = einsum(equation, *inputs)
    plan = plan_cuda(Program({"result": node}), TARGET)
    return plan, next(i for i, step in enumerate(plan.steps) if step.node == node)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sizes", [(3, 5, 7), (1, 3, 1), (1, 1, 1)])
@pytest.mark.parametrize("batched", [False, True])
def test_layouts_against_independent_einsum(
    dtype: str, sizes: tuple[int, int, int], batched: bool
) -> None:
    """Check all input/result transposes, unequal sizes, batches and unit axes."""
    prefix = "b" if batched else ""
    for al, bl, cl in product(("mk", "km"), ("kn", "nk"), ("mn", "nm")):
        equation = f"{prefix}{al},{prefix}{bl}->{prefix}{cl}"
        plan, index = _plan(equation, dtype, sizes)
        candidate = cublaslt_provider_candidates(
            plan, index, target_capabilities=CAPABILITIES
        )[0]
        assert candidate.status == "ready", candidate.reason
        incumbent = next(
            row
            for row in resolved_lowering_candidates(plan)
            if dict(row.provenance)["step_index"] == index
        )
        assert candidate.request == incumbent.request
        recipe = cublaslt_matmul(candidate.request)
        assert recipe.identity == cublaslt_matmul(candidate.request).identity
        arrays = [
            np.arange(np.prod(v.shape), dtype=dtype).reshape(v.shape) / 16
            for v in candidate.request.operands[:2]
        ]
        reference = np.einsum(equation, *arrays)
        flat = np.zeros(reference.size, dtype=dtype)

        def offset(
            which: int,
            batch: int,
            row: int,
            column: int,
            bound_recipe: CublasLtMatmul = recipe,
        ) -> int:
            layout = bound_recipe.layouts[which]
            matrix = (
                row * layout.leading_dimension + column
                if layout.order == "row"
                else column * layout.leading_dimension + row
            )
            return batch * layout.batch_stride + matrix

        m, k, n = sizes
        for batch, row, column in product(range(recipe.batches), range(m), range(n)):
            flat[offset(2, batch, row, column)] = sum(
                arrays[0].flat[offset(0, batch, row, reduction)]
                * arrays[1].flat[offset(1, batch, reduction, column)]
                for reduction in range(k)
            )
        np.testing.assert_array_equal(flat.reshape(reference.shape), reference)


def test_padded_layout_and_common_resource_gates() -> None:
    plan, index = _plan()
    original = cublaslt_provider_candidates(plan, index)[0].request
    a, b, c = original.operands
    request = replace(original, operands=(replace(a, strides=(11, 1)), b, c))
    recipe = cublaslt_matmul(request)
    assert recipe.layouts[0].leading_dimension == 11
    provider = CublasLtMatmulProvider(version="12.9.1")
    candidate = collect_lowering_candidates(request, CAPABILITIES, (provider,))[0]
    assert candidate.request == request
    assert candidate.cost is None
    assert candidate.execution is not None
    assert (
        candidate.workspace_bytes,
        candidate.provider_bytes,
        candidate.execution.host_bytes,
        candidate.execution.cache_bytes,
    ) == (512, 1024, 128, 256)
    for constraint in (
        LoweringConstraints(additional_device_bytes=1791),
        LoweringConstraints(host_bytes=127),
        LoweringConstraints(capture_required=True),
        LoweringConstraints(determinism="exact-order"),
    ):
        rejected = collect_lowering_candidates(
            replace(request, constraints=constraint), CAPABILITIES, (provider,)
        )[0]
        assert rejected.status == "unsupported"
    # Same operation can retain both providers; missing cuTENSOR facts remain a
    # visible rejection, not a second request or a forced cuBLASLt selection.
    offers = collect_lowering_candidates(
        request, CAPABILITIES, (provider, CutensorContractionProvider())
    )
    assert len(offers) == 2 and all(row.request == request for row in offers)


@pytest.mark.parametrize(
    "feature",
    [
        "cublaslt",
        "cublaslt-version",
        "cublaslt-workspace-bytes",
        "cublaslt-provider-bytes",
        "cublaslt-host-bytes",
        "cublaslt-cache-bytes",
    ],
)
def test_missing_capabilities_are_rejections(feature: str) -> None:
    plan, index = _plan()
    target = replace(
        CAPABILITIES,
        features=tuple((k, v) for k, v in CAPABILITIES.features if k != feature),
    )
    assert (
        cublaslt_provider_candidates(plan, index, target_capabilities=target)[0].status
        == "unsupported"
    )


def test_precision_obligations_remain_visible() -> None:
    plan, index = _plan()
    request = cublaslt_provider_candidates(plan, index)[0].request
    strict = request.precisions[0]
    mixed = replace(
        strict,
        schedule=ExecutionPrecisionSchedule(
            (
                (
                    "operation",
                    PrecisionDirective(
                        "float64", "float32", "float32", "test-qualified"
                    ),
                ),
            )
        ),
    )
    audit = replace(strict, audit="independent-reference-required")
    request = replace(request, precisions=(strict, mixed, audit))
    offers = CublasLtMatmulProvider(version="12.9.1").candidates(request, CAPABILITIES)
    statuses = {}
    for row in offers:
        assert row.execution is not None
        statuses[row.execution.precision.identity] = row.status
    assert statuses == {
        strict.identity: "ready",
        mixed.identity: "unsupported",
        audit.identity: "unsupported",
    }
    assert all(row.request == request for row in offers)


@pytest.mark.parametrize(
    "failure", ["alias", "virtual", "triangle", "overlap", "diagonal"]
)
def test_unsupported_views_retain_negative_evidence(failure: str) -> None:
    plan, index = _plan()
    request = cublaslt_provider_candidates(plan, index)[0].request
    a, b, c = request.operands
    if failure == "alias":
        a, c = replace(a, alias_group="shared"), replace(c, alias_group="shared")
    elif failure == "virtual":
        a = replace(a, strides=None)
    elif failure == "triangle":
        c = replace(c, triangle="upper")
    elif failure == "overlap":
        a = replace(a, strides=(1, 1))
    else:
        assert a.strides is not None
        a = replace(
            a,
            modes=(*a.modes, a.modes[0]),
            shape=(*a.shape, a.shape[0]),
            strides=(*a.strides, 1),
        )
    request = replace(request, operands=(a, b, c))
    candidate = CublasLtMatmulProvider(version="12.9.1").candidates(
        request, CAPABILITIES
    )[0]
    assert candidate.status == "unsupported"
    assert candidate.reason and candidate.request == request


def test_batch_overlap_and_one_sided_reduction_reject() -> None:
    plan, index = _plan("bmk,bkn->bmn")
    request = cublaslt_provider_candidates(plan, index)[0].request
    a, b, c = request.operands
    with pytest.raises(ValueError, match="nonoverlapping matrix batches"):
        cublaslt_matmul(
            replace(request, operands=(replace(a, strides=(1, 5, 1)), b, c))
        )
    plan, index = _plan("mk,kn->m")
    assert (
        cublaslt_provider_candidates(plan, index, target_capabilities=CAPABILITIES)[
            0
        ].status
        == "unsupported"
    )


def test_target_version_and_noninteger_resource_evidence_reject() -> None:
    plan, index = _plan()
    request = cublaslt_provider_candidates(plan, index)[0].request
    assert (
        CublasLtMatmulProvider(version="12.8.0")
        .candidates(request, CAPABILITIES)[0]
        .status
        == "unsupported"
    )
    features = dict(CAPABILITIES.features)
    features["cublaslt-host-bytes"] = True
    target = replace(CAPABILITIES, features=tuple(features.items()))
    assert (
        cublaslt_provider_candidates(plan, index, target_capabilities=target)[0].status
        == "unsupported"
    )
    with pytest.raises(ValueError, match="planned target"):
        cublaslt_provider_candidates(
            plan,
            index,
            target_capabilities=TargetCapabilities(
                cuda_target_info("sm_80").target_info
            ),
        )
