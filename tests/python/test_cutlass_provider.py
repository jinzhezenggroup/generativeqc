"""AOT offers share canonical science while retaining artifact/resource gates."""

from __future__ import annotations

from dataclasses import replace

import pytest
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
from generativeqc_compiler.tensor.cuda_cublaslt import CublasLtMatmulProvider
from generativeqc_compiler.tensor.cuda_cutensor import CutensorContractionProvider
from generativeqc_compiler.tensor.cuda_cutlass import (
    CUTLASS_FAMILY,
    CutlassAotProvider,
    cutlass_provider_candidates,
)
from generativeqc_compiler.tensor.cuda_providers import resolved_lowering_candidates
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from test_cublaslt_provider import CAPABILITIES, _plan

FEATURES = (
    ("cutlass-aot", True),
    ("cutlass-version", "3.9.2"),
    ("cutlass-aot-family", CUTLASS_FAMILY),
    ("cutlass-aot-artifact", "a" * 64),
    ("cutlass-aot-host-bytes", 512),
    ("cutlass-aot-module-bytes", 4096),
)
TARGET = replace(CAPABILITIES, features=(*CAPABILITIES.features, *FEATURES))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("equation", ["mk,kn->mn", "km,nk->nm", "bmk,bkn->bmn"])
def test_same_request_and_retained_module_resources(dtype: str, equation: str) -> None:
    plan, index = _plan(equation, dtype)
    candidate = cutlass_provider_candidates(plan, index, target_capabilities=TARGET)[0]
    assert candidate.status == "ready", candidate.reason
    incumbent = next(
        row
        for row in resolved_lowering_candidates(plan)
        if dict(row.provenance)["step_index"] == index
    )
    assert candidate.request == incumbent.request
    assert candidate.execution is not None
    assert candidate.execution.precision in candidate.request.precisions
    assert (
        candidate.execution.host_bytes == 512
        and candidate.execution.cache_bytes == 4096
    )
    assert candidate.execution.topology.tiles == (32, 64, 8)
    assert candidate.execution.topology.workgroup_threads == 128
    assert candidate.cost is None
    assert (candidate.workspace_bytes, candidate.provider_bytes) == (0, 0)
    offers = collect_lowering_candidates(
        candidate.request,
        TARGET,
        (
            CutlassAotProvider(version="3.9.2"),
            CublasLtMatmulProvider(version="12.9.1"),
            CutensorContractionProvider(),
        ),
    )
    assert [row.status for row in offers] == ["ready", "ready", "unsupported"]
    assert all(row.request == candidate.request for row in offers)


@pytest.mark.parametrize("feature", [key for key, _ in FEATURES])
def test_missing_artifact_and_qualification_are_negative_evidence(feature: str) -> None:
    plan, index = _plan()
    target = replace(
        TARGET, features=tuple((k, v) for k, v in TARGET.features if k != feature)
    )
    offer = cutlass_provider_candidates(plan, index, target_capabilities=target)[0]
    assert offer.status == "unsupported" and offer.reason


@pytest.mark.parametrize("value", [True, 0, -1, 1 << 63])
@pytest.mark.parametrize("kind", ["host", "module"])
def test_bad_resource_facts_fail_closed(kind: str, value: int) -> None:
    plan, index = _plan()
    features = dict(TARGET.features)
    features[f"cutlass-aot-{kind}-bytes"] = value
    offer = cutlass_provider_candidates(
        plan,
        index,
        target_capabilities=replace(TARGET, features=tuple(features.items())),
    )[0]
    assert offer.status == "unsupported"


@pytest.mark.parametrize("artifact", [None, True, "", "A" * 64, "a" * 63])
def test_actual_artifact_digest_is_required(artifact: str | bool | None) -> None:
    plan, index = _plan()
    features = dict(TARGET.features)
    if artifact is None:
        del features["cutlass-aot-artifact"]
    else:
        features["cutlass-aot-artifact"] = artifact
    offer = cutlass_provider_candidates(
        plan,
        index,
        target_capabilities=replace(TARGET, features=tuple(features.items())),
    )[0]
    assert offer.status == "unsupported"


def test_artifact_changes_execution_identity_without_changing_science() -> None:
    plan, index = _plan()
    before = cutlass_provider_candidates(plan, index, target_capabilities=TARGET)[0]
    features = dict(TARGET.features)
    features["cutlass-aot-artifact"] = "b" * 64
    after = cutlass_provider_candidates(
        plan,
        index,
        target_capabilities=replace(TARGET, features=tuple(features.items())),
    )[0]
    assert after.status == "ready"
    assert before.request == after.request and before.identity != after.identity
    assert dict(after.provenance)["artifact"] == "b" * 64


def test_precision_obligations_and_common_resource_admission() -> None:
    plan, index = _plan()
    request = cutlass_provider_candidates(plan, index)[0].request
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
    request = replace(
        request, precisions=(strict, mixed, replace(strict, audit="independent"))
    )
    provider = CutlassAotProvider(version="3.9.2")
    offers = collect_lowering_candidates(request, TARGET, (provider,))
    for row in offers:
        assert row.execution is not None
        assert (row.status == "ready") == (row.execution.precision == strict)
    for constraint in (
        LoweringConstraints(host_bytes=511),
        LoweringConstraints(additional_device_bytes=4095),
        LoweringConstraints(capture_required=True),
        LoweringConstraints(determinism="exact-order"),
    ):
        assert all(
            row.status == "unsupported"
            for row in collect_lowering_candidates(
                replace(request, constraints=constraint), TARGET, (provider,)
            )
        )
    exact = replace(
        request,
        precisions=(strict,),
        constraints=LoweringConstraints(
            host_bytes=512, additional_device_bytes=4096, determinism="reproducible"
        ),
    )
    assert collect_lowering_candidates(exact, TARGET, (provider,))[0].status == "ready"


@pytest.mark.parametrize("columns", [65535 * 64, 65535 * 64 + 1])
@pytest.mark.parametrize("output", ["mn", "nm"])
def test_native_grid_bound_accounts_for_transposed_output(
    columns: int, output: str
) -> None:
    # Both output axes must be nonunit: unit axes have no observable stride,
    # so the shared proof can canonically choose row order for either equation.
    # This tests a provider's grid bound, independently of the arena planner's
    # default 256 MiB caller-storage budget. Construct the real scientific IR
    # directly; no tensor data is materialized for these large symbolic shapes.
    m, k, n = (
        Index(name, IndexSpace(name, "batch", size))
        for name, size in (("m", 2), ("k", 1), ("n", columns))
    )
    a = input_tensor("a", TensorSpec((m, k), role="input"))
    b = input_tensor("b", TensorSpec((k, n), role="input"))
    node = einsum("mk,kn->" + output, a, b)
    request = TensorLoweringAdapter(Program({"out": node})).request(
        node, backend="cuda"
    )
    offer = collect_lowering_candidates(
        request, TARGET, (CutlassAotProvider(version="3.9.2"),)
    )[0]
    assert (offer.status == "ready") == (output == "nm" or columns == 65535 * 64)


def test_missing_family_version_and_target_geometry_reject() -> None:
    plan, index = _plan()
    request = cutlass_provider_candidates(plan, index)[0].request
    assert (
        CutlassAotProvider(version="4.0.0").candidates(request, TARGET)[0].status
        == "unsupported"
    )
    for target in (
        replace(TARGET, target=replace(TARGET.target, subgroup_size=64)),
        replace(TARGET, target=replace(TARGET.target, maximum_workgroup_threads=64)),
    ):
        assert (
            CutlassAotProvider(version="3.9.2").candidates(request, target)[0].status
            == "unsupported"
        )
    with pytest.raises(ValueError, match="planned target"):
        cutlass_provider_candidates(
            plan,
            index,
            target_capabilities=TargetCapabilities(
                replace(TARGET.target, architecture="other")
            ),
        )
