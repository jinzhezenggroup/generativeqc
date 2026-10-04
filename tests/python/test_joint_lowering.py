"""Joint precision/provider binding gates, independent of any installed runtime."""

import typing
from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
    LoweringCost,
    LoweringPrecision,
    OperandLayout,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.lowering_selection import select_lowering_binding
from generativeqc_compiler.common.precision import (
    CastBoundary,
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    einsum,
    execute,
    input_tensor,
)
from generativeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from generativeqc_compiler.tensor.cuda_providers import resolved_lowering_candidates
from generativeqc_compiler.tensor.lowering import tensor_lowering_request

SCIENCE = "a" * 64
COMPILATION = CompilationIdentity(SCIENCE, "b" * 64)
TARGET = TargetCapabilities(
    cuda_target_info("sm_80").target_info,
    features=(("test-library", True), ("toolkit-version", "12.9")),
)
GENERATED = ProviderDescriptor(
    "test.generated", "generated", "scalar", version="source-v1"
)
LIBRARY = ProviderDescriptor(
    "test.library",
    "library",
    "matmul",
    version="1.0",
    required_features=("test-library",),
)


def _precision(
    dtype: str = "float64",
    accumulation: str | None = None,
    qualification: str | None = None,
) -> LoweringPrecision:
    accumulation = accumulation or dtype
    directive = PrecisionDirective(dtype, dtype, accumulation, qualification)
    casts = (
        ()
        if dtype == "float64"
        else (
            CastBoundary("input:0", "float64", "float32", 6, 48, 24),
            CastBoundary("input:1", "float64", "float32", 6, 48, 24),
            CastBoundary("output", "float32", "float64", 4, 16, 32),
        )
    )
    return LoweringPrecision(
        ExecutionPrecisionSchedule((("operation", directive),)),
        "operation",
        (dtype, dtype),
        "float64",
        casts=casts,
        audit=None if dtype == "float64" else "independent-gate-v1",
    )


def _request(**changes: typing.Any) -> LoweringRequest:
    request = LoweringRequest(
        "tensor",
        "einsum",
        "cuda",
        "float64",
        "float64",
        (2, 2),
        scientific_identity=SCIENCE,
        operands=(
            OperandLayout("a", (0, 2), (2, 3), (3, 1)),
            OperandLayout("b", (2, 1), (3, 2), (2, 1)),
            OperandLayout("c", (0, 1), (2, 2), (2, 1), access="write"),
        ),
        precisions=(
            _precision(),
            _precision("float32", qualification="independent-gate-v1"),
        ),
        constraints=LoweringConstraints(),
        semantics=(("alpha", 1.0), ("beta", 0.0)),
    )
    return replace(request, **changes)


def _cost(
    precision: LoweringPrecision,
    *,
    prepare: int = 0,
    kernel: int = 100,
    cast: int = 0,
    pack: int = 0,
    audit: int = 0,
) -> LoweringCost:
    return LoweringCost(
        "test-fixture-complete-phase-cost",
        "estimated",
        prepare_ns=prepare,
        kernel_ns=kernel,
        cast_ns=cast,
        pack_ns=pack,
        refinement_ns=0,
        audit_ns=audit,
        fallback_ns=0,
        cast_bytes=sum(
            boundary.read_bytes + boundary.write_bytes for boundary in precision.casts
        ),
        launches=1,
    )


def _offer(
    request: LoweringRequest,
    provider: ProviderDescriptor = GENERATED,
    *,
    dtype: str = "float64",
    cost: LoweringCost | None = None,
    **changes: typing.Any,
) -> LoweringCandidate:
    precision = next(
        p for p in request.precisions if p.directive.compute_dtype == dtype
    )
    return replace(
        LoweringCandidate(
            request,
            "test-operation",
            (provider,),
            "ready",
            f"{dtype}->{precision.directive.accumulation_dtype}",
            execution=CandidateExecution(precision, "test-algorithm", request.operands),
            cost=cost or _cost(precision),
            target=TARGET
            if request.backend == "cuda"
            else TargetCapabilities(TargetInfo("cpu", "scalar", None, 1, None)),
        ),
        **changes,
    )


class _Provider:
    """Controlled provider double; it makes no real-device execution claim."""

    def __init__(
        self, descriptor: ProviderDescriptor, offers: tuple[LoweringCandidate, ...]
    ) -> None:
        self.descriptor = descriptor
        self.offers = offers

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        return self.offers


def test_registry_compares_joint_variants_and_retains_negative_evidence() -> None:
    request = _request()
    strict = _offer(request)
    mixed = _offer(request, LIBRARY, dtype="float32")
    rejected = replace(
        mixed,
        implementation="unsupported-arithmetic",
        status="unsupported",
        reason="FP32 multiply with FP64 accumulation unavailable",
    )
    offered = collect_lowering_candidates(
        request,
        TARGET,
        (
            _Provider(GENERATED, (strict,)),
            _Provider(LIBRARY, (mixed, rejected)),
        ),
    )
    assert {candidate.request_hash for candidate in offered} == {request.identity}
    assert offered[-1].reason == rejected.reason
    assert len({candidate.execution.precision.identity for candidate in offered}) == 2
    with pytest.raises(ValueError, match="another request"):
        collect_lowering_candidates(
            request,
            TARGET,
            (
                _Provider(
                    GENERATED,
                    (replace(strict, request=replace(request, operation="matmul")),),
                ),
            ),
        )


def test_conversion_and_audit_cost_can_reverse_the_kernel_winner() -> None:
    request = _request()
    strict = _offer(request)
    low = next(p for p in request.precisions if p.directive.compute_dtype == "float32")
    mixed = _offer(
        request, LIBRARY, dtype="float32", cost=_cost(low, kernel=10, cast=70, audit=30)
    )
    binding = select_lowering_binding(request, TARGET, COMPILATION, (mixed, strict))
    assert binding.selected == strict
    assert binding.fallbacks == ()  # a strict selection never falls back to narrowing
    fast_mixed = replace(mixed, cost=_cost(low, kernel=10, cast=5, audit=5))
    binding = select_lowering_binding(
        request, TARGET, COMPILATION, (strict, fast_mixed)
    )
    assert binding.selected == fast_mixed
    assert binding.fallbacks == (strict,)


def test_prepare_cost_is_amortized_explicitly_and_registration_order_is_irrelevant() -> (
    None
):
    request = _request()
    ordinary = _offer(request)
    planned = _offer(
        request, LIBRARY, cost=_cost(_precision(), prepare=1000, kernel=10)
    )
    cold = select_lowering_binding(request, TARGET, COMPILATION, (planned, ordinary))
    warm = select_lowering_binding(
        request, TARGET, COMPILATION, (planned, ordinary), expected_replays=100
    )
    assert cold.selected == ordinary
    assert warm.selected == planned
    assert (
        warm.to_payload()
        == select_lowering_binding(
            request, TARGET, COMPILATION, (ordinary, planned), expected_replays=100
        ).to_payload()
    )


@pytest.mark.parametrize(
    ("constraints", "execution", "workspace", "provider", "reason"),
    [
        (LoweringConstraints(workspace_bytes=0), {}, 1, 0, "workspace_bytes"),
        (LoweringConstraints(provider_bytes=0), {}, 0, 1, "provider_bytes"),
        (
            LoweringConstraints(additional_device_bytes=10),
            {"temporary_bytes": 3, "cache_bytes": 3},
            3,
            3,
            "additional_device_bytes",
        ),
        (LoweringConstraints(host_bytes=0), {"host_bytes": 1}, 0, 0, "host_bytes"),
        (LoweringConstraints(capture_required=True), {}, 0, 0, "capture/replay"),
        (
            LoweringConstraints(determinism="exact-order"),
            {"determinism": "reproducible"},
            0,
            0,
            "determinism",
        ),
    ],
)
def test_shared_resource_capture_and_order_gates_keep_negative_offers(
    constraints: LoweringConstraints,
    execution: dict[str, typing.Any],
    workspace: int,
    provider: int,
    reason: str,
) -> None:
    request = _request(constraints=constraints)
    candidate = _offer(request, workspace_bytes=workspace, provider_bytes=provider)
    candidate = replace(candidate, execution=replace(candidate.execution, **execution))
    (rejected,) = collect_lowering_candidates(
        request, TARGET, (_Provider(GENERATED, (candidate,)),)
    )
    assert rejected.status == "unsupported"
    assert reason in rejected.reason
    # Direct selection cannot bypass registry admission by reusing the ready row.
    with pytest.raises(ValueError, match=reason):
        select_lowering_binding(request, TARGET, COMPILATION, (candidate,))


def test_selection_fails_closed_on_absence_incomplete_cost_and_missing_strict_candidate() -> (
    None
):
    request = _request()
    strict = _offer(request)
    missing = replace(strict, cost=replace(strict.cost, cast_ns=None))
    with pytest.raises(ValueError, match="complete prepare"):
        select_lowering_binding(request, TARGET, COMPILATION, (missing,))
    mixed = _offer(request, LIBRARY, dtype="float32")
    with pytest.raises(ValueError, match="strict requested-precision"):
        select_lowering_binding(request, TARGET, COMPILATION, (mixed,))
    unknown = replace(strict, providers=(replace(GENERATED, version=None),))
    with pytest.raises(ValueError, match="version/source identity"):
        select_lowering_binding(request, TARGET, COMPILATION, (unknown,))
    absent = replace(TARGET, features=(("test-library", 1),))
    with pytest.raises(ValueError, match="required provider capability"):
        select_lowering_binding(
            request,
            absent,
            COMPILATION,
            (replace(_offer(request, LIBRARY), target=absent),),
        )


def test_binding_identity_covers_context_but_not_measured_latency() -> None:
    request = _request()
    strict = _offer(request)
    bind = lambda req, target, compilation, offer: select_lowering_binding(
        req, target, compilation, (offer,)
    )
    binding = bind(request, TARGET, COMPILATION, strict)
    binding.validate_context(request, TARGET, COMPILATION)
    changed = replace(strict, cost=replace(strict.cost, kernel_ns=101))
    assert (
        binding.cache_identity
        == bind(request, TARGET, COMPILATION, changed).cache_identity
    )
    toolkit = replace(
        TARGET, features=(("test-library", True), ("toolkit-version", "13.0"))
    )
    assert (
        binding.cache_identity
        != bind(
            request, toolkit, COMPILATION, replace(strict, target=toolkit)
        ).cache_identity
    )
    for stale in (
        toolkit,
        replace(TARGET, target=cuda_target_info("sm_120").target_info),
    ):
        with pytest.raises(ValueError, match="context changed"):
            binding.validate_context(request, stale, COMPILATION)
    algorithm = replace(
        strict,
        execution=replace(
            strict.execution,
            algorithm="algorithm-2",
            topology=ScheduleTopology(fusion="cast-matmul-publish"),
        ),
    )
    assert (
        binding.cache_identity
        != bind(request, TARGET, COMPILATION, algorithm).cache_identity
    )
    version = replace(strict, providers=(replace(GENERATED, version="source-v2"),))
    assert (
        binding.cache_identity
        != bind(request, TARGET, COMPILATION, version).cache_identity
    )
    with pytest.raises(ValueError, match="scientific identity"):
        bind(request, TARGET, replace(COMPILATION, scientific_hash="c" * 64), strict)


def test_precision_layout_contract_rejects_implicit_changes() -> None:
    request = _request()
    with pytest.raises(ValueError, match="scientific qualification"):
        _request(precisions=(_precision(), _precision("float32")))
    with pytest.raises(ValueError, match="arithmetic mode"):
        PrecisionDirective("float32", "float32", "float32", math_mode="tf32")
    with pytest.raises(ValueError, match="outside the admitted"):
        replace(
            _offer(request),
            execution=CandidateExecution(
                _precision("float32", "float64", "another-gate"),
                "test",
                request.operands,
            ),
        )
    with pytest.raises(ValueError, match="logical axes"):
        replace(
            _offer(request),
            execution=replace(
                _offer(request).execution,
                layouts=(
                    replace(request.operands[0], modes=(2, 0)),
                    *request.operands[1:],
                ),
            ),
        )
    with pytest.raises(ValueError, match="omits explicit cast traffic"):
        mixed = _offer(request, dtype="float32")
        replace(mixed, cost=replace(mixed.cost, cast_bytes=0))
    with pytest.raises(ValueError, match="cast traffic"):
        CastBoundary("bad", "float64", "float32", 2, 8, 8)


def test_candidate_bound_is_checked_before_selection_and_does_not_truncate() -> None:
    request = _request(constraints=LoweringConstraints(maximum_candidates=1))
    candidates = (_offer(request), _offer(request, LIBRARY))
    with pytest.raises(ValueError, match="bound exceeded"):
        collect_lowering_candidates(
            request, TARGET, (_Provider(GENERATED, candidates),)
        )
    with pytest.raises(ValueError, match="bound exceeded"):
        select_lowering_binding(request, TARGET, COMPILATION, iter(candidates))


def _tensor() -> tuple[Program, Node]:
    i, j, k = (
        Index(name, IndexSpace(name, "batch", size))
        for name, size in (("i", 2), ("j", 2), ("k", 3))
    )
    a = input_tensor("a", TensorSpec((i, k), role="input"))
    b = input_tensor("b", TensorSpec((k, j), role="input"))
    node = einsum("ik,kj->ij", a, b)
    return Program({"result": node}), node


def test_tensor_cpu_cuda_and_gemm_schedules_share_semantic_identity_and_equation() -> (
    None
):
    program, node = _tensor()
    original = program.dumps()
    cpu = tensor_lowering_request(program, node, backend="cpu")
    cuda = tensor_lowering_request(program, node, backend="cuda")
    assert cpu.identity != cuda.identity
    assert cpu.semantic_identity == cuda.semantic_identity
    assert cpu.scientific_identity == cuda.scientific_identity == program.logical_hash
    requests = []
    for schedule in (TensorSchedule(), TensorSchedule(direct_gemm=False)):
        plan = plan_cuda(
            program,
            cuda_target_info("sm_80"),
            schedule=schedule,
            reassociate_contractions=False,
        )
        candidate = next(
            c
            for c in resolved_lowering_candidates(plan)
            if c.request.operation == "einsum"
        )
        requests.append(candidate.request)
    assert requests[0] == requests[1]
    assert requests[0].semantic_identity == cuda.semantic_identity
    assert program.dumps() == original
    a = np.arange(6, dtype=np.float64).reshape(2, 3) - 2
    b = np.arange(6, dtype=np.float64).reshape(3, 2) / 3
    np.testing.assert_allclose(
        execute(program, {"a": a, "b": b}).outputs["result"],
        a @ b,
        rtol=1e-15,
        atol=1e-15,
    )


def test_cpu_selection_requires_no_cuda_provider_and_preserves_science() -> None:
    request = _request(backend="cpu")
    target = TargetCapabilities(TargetInfo("cpu", "scalar", None, 1, None))
    binding = select_lowering_binding(request, target, COMPILATION, (_offer(request),))
    assert binding.selected.request.semantic_identity == _request().semantic_identity
    with pytest.raises(ValueError, match="requested backend"):
        select_lowering_binding(request, TARGET, COMPILATION, (_offer(request),))


def test_unmeasured_incumbent_retention_still_requires_legal_strict_fallback() -> None:
    request = _request()
    strict = replace(_offer(request), cost=None)
    mixed = replace(_offer(request, LIBRARY, dtype="float32"), cost=None)
    for candidates, incumbent in (
        ((mixed,), mixed.identity),
        (
            (strict, replace(mixed, status="unsupported", reason="missing capability")),
            mixed.identity,
        ),
        ((strict, mixed), "0" * 64),
    ):
        with pytest.raises(ValueError, match="incumbent|strict"):
            select_lowering_binding(
                request, TARGET, COMPILATION, candidates, qualified_incumbent=incumbent
            )
    limited = replace(request, constraints=LoweringConstraints(provider_bytes=0))
    over_budget = replace(mixed, request=limited, provider_bytes=1)
    with pytest.raises(ValueError, match="incumbent"):
        select_lowering_binding(
            limited,
            TARGET,
            COMPILATION,
            (replace(strict, request=limited), over_budget),
            qualified_incumbent=over_budget.identity,
        )
