"""Resolved provider provenance for the existing TensorIR CUDA lowering.

This module does not choose a new implementation.  It projects the already
planned TensorIR execution into the shared lowering-provider contract so future
cuBLASLt/CUTLASS/CUB candidates can be compared without adding provider-specific
branches to scientific IR.
"""

from __future__ import annotations

from generativeqc_compiler.common.lowering_contract import CandidateExecution
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
    lowering_diagnostics,
)
from generativeqc_compiler.common.specialization import TargetCapabilities

from .cuda_gemm import gemm_contract
from .cuda_plan import TensorPlan
from .cuda_reduction import cooperative_reduction_provider
from .lowering import TensorLoweringAdapter, plan_lowering_request

GENERATED_CUDA_PROVIDER = ProviderDescriptor(
    name="generativeqc.generated_cuda",
    kind="generated",
    implementation="tensor-cuda-emitter",
    provenance=(("version_source", "compiler-source-identity"),),
)

CUBLAS_PROVIDER = ProviderDescriptor(
    name="nvidia.cublas",
    kind="library",
    implementation="cublas-gemm",
    required_features=("cublas",),
    provenance=(("version_source", "runtime-probe"),),
)


CUDA_RUNTIME_PROVIDER = ProviderDescriptor(
    name="nvidia.cuda_runtime",
    kind="runtime",
    implementation="cuda-runtime",
    required_features=("cuda-runtime",),
    provenance=(("version_source", "runtime-probe"),),
)


CUB_REDUCTION_PROVIDER = ProviderDescriptor(
    name="nvidia.cccl.cub",
    kind="library",
    implementation="cub-block-reduce",
    required_features=("cuda", "cub-block-reduce-header"),
    provenance=(("version_source", "cuda-toolkit-cccl-header"),),
)


class GeneratedReductionProvider:
    """Advertise the existing generated cooperative reduction."""

    descriptor = GENERATED_CUDA_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        return _reduction_candidates(request, target, self.descriptor)


class CubReductionProvider:
    """Advertise the opt-in CUB BlockReduce implementation."""

    descriptor = CUB_REDUCTION_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        return _reduction_candidates(request, target, self.descriptor)


def _reduction_candidates(
    request: LoweringRequest, target: TargetCapabilities, descriptor: ProviderDescriptor
) -> tuple[LoweringCandidate, ...]:
    """Offer each admitted precision under exactly the same operation identity."""
    candidates = []
    for precision in request.precisions or (None,):
        reason = _cooperative_reduction_rejection(request, target)
        if precision is not None:
            directive = precision.directive
            if precision.casts or precision.refinement or precision.audit:
                reason = "cooperative adapter does not implement standalone cast/refinement/audit obligations"
            elif any(
                dtype != directive.compute_dtype for dtype in precision.input_dtypes
            ):
                reason = (
                    "cooperative adapter requires inputs in the requested compute dtype"
                )
            numerical_mode = (
                f"{directive.compute_dtype}->{directive.accumulation_dtype}"
            )
            execution = CandidateExecution(
                precision=precision,
                algorithm="cub-block-reduce"
                if descriptor == CUB_REDUCTION_PROVIDER
                else "generated-cooperative",
                layouts=request.operands,
                determinism="reproducible",
                capture_safe=True,
            )
        else:
            numerical_mode = f"{request.dtype}->{request.accumulation_dtype}"
            execution = None
        cub = descriptor == CUB_REDUCTION_PROVIDER
        if (
            reason is None
            and cub
            and dict(target.features).get("cub-block-reduce-header") is not True
        ):
            reason = "CUB requires explicit cub-block-reduce-header capability"
        candidates.append(
            LoweringCandidate(
                request=request,
                implementation="tensor-reduce-cub-block-reduce"
                if cub
                else "tensor-reduce-generated-cooperative",
                providers=(descriptor, GENERATED_CUDA_PROVIDER)
                if cub
                else (descriptor,),
                status="unsupported" if reason else "ready",
                numerical_mode=numerical_mode,
                reason=reason,
                execution=execution,
            )
        )
    return tuple(candidates)


def _cooperative_reduction_rejection(
    request: LoweringRequest, target: TargetCapabilities
) -> str | None:
    semantics = dict(request.semantics)
    if request.operation not in ("reduce", "einsum"):
        return "provider requires a TensorIR reduction or streamed einsum request"
    extent = semantics.get("reduction_extent")
    if extent is None:
        if request.operation != "reduce" or len(request.shape) != 2:
            return "provider requires explicit reduction extent"
        extent = request.shape[1]  # retained v1 flattened diagnostic request
    if request.dtype not in (
        "float32",
        "float64",
    ) or request.accumulation_dtype not in ("float32", "float64"):
        return "provider supports float32/float64 reduction arithmetic only"
    if target.target.subgroup_size != 32:
        return "cooperative reduction pilot requires CUDA subgroup size 32"
    if type(extent) is not int or extent < 0:
        return "reduction extent must be an explicit non-negative integer"
    if extent < 32:
        return "reduction extent is below the cooperative subgroup threshold"
    return None


def _numerical_mode(plan: TensorPlan, index: int) -> str:
    node = plan.steps[index].node
    value = plan.precision_by_node.get(node)
    if value is None:
        return f"{node.spec.dtype}->{node.spec.dtype}"
    return f"{value.compute_dtype}->{value.accumulation_dtype}"


def resolved_lowering_candidates(plan: TensorPlan) -> tuple[LoweringCandidate, ...]:
    """Describe the providers already selected by one TensorPlan.

    GEMM is a composite lowering: cuBLAS owns the contraction, while generated
    CUDA owns packing/scatter and/or the checked coefficient epilogue.  Empty-K
    contractions use the CUDA runtime zero-fill path and therefore do not claim
    a generated kernel or cuBLAS provider.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("Tensor CUDA lowering diagnostics require a TensorPlan")
    candidates: list[LoweringCandidate] = []
    adapter = TensorLoweringAdapter(plan.program)
    for index, step in enumerate(plan.steps):
        node = step.node
        if step.virtual or node.op in ("input", "constant") or not node.spec.size:
            continue
        contract = gemm_contract(node)
        reduction_provider = cooperative_reduction_provider(plan, index)
        is_gemm = step.gemm != "none" and contract is not None
        uses_cublas = is_gemm and contract.k > 0
        request = plan_lowering_request(plan, index, adapter)
        if uses_cublas:
            providers = (CUBLAS_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = f"tensor-gemm-{step.gemm}"
        elif is_gemm:
            providers = (CUDA_RUNTIME_PROVIDER,)
            implementation = "tensor-gemm-zero-fill"
        elif reduction_provider == "cub":
            providers = (CUB_REDUCTION_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = "tensor-reduce-cub-block-reduce"
        elif reduction_provider == "generated":
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = "tensor-reduce-generated-cooperative"
        else:
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = f"tensor-generated-{node.op}"
        candidates.append(
            LoweringCandidate(
                request=request,
                implementation=implementation,
                providers=providers,
                status="ready",
                execution=(
                    CandidateExecution(
                        precision=request.precisions[0],
                        algorithm=implementation,
                        layouts=request.operands,
                        # These are resolved diagnostics, not a new determinism
                        # or capture qualification for the complete endpoint.
                    )
                    if request.precisions
                    else None
                ),
                numerical_mode=_numerical_mode(plan, index),
                workspace_bytes=plan.library_bytes if uses_cublas else 0,
                provider_bytes=plan.provider_bytes if uses_cublas else 0,
                provenance=(
                    ("plan_identity", plan.identity),
                    ("step_index", index),
                ),
            )
        )
    return tuple(candidates)


def reduction_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Advertise reductions using explicit, plan-bound toolkit capability facts.

    GPU architecture alone does not establish that CUB headers are installed.
    Without caller-supplied header evidence the CUB offer is unsupported; the
    generated offer remains available. This routine does not probe a toolkit.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("reduction provider candidates require a TensorPlan")
    if cooperative_reduction_provider(plan, index) is None:
        raise ValueError("step is not an eligible cooperative reduction")
    request = plan_lowering_request(plan, index, TensorLoweringAdapter(plan.program))
    target = TargetCapabilities(
        plan.target.target_info,
        features=tuple(
            (feature, True) for feature in plan.target.required_cuda_features
        ),
    )
    if target_capabilities is not None:
        if not isinstance(target_capabilities, TargetCapabilities):
            raise TypeError("reduction capabilities require TargetCapabilities")
        if target_capabilities.target != plan.target.target_info:
            raise ValueError("reduction capabilities do not match the planned target")
        target = target_capabilities
    return collect_lowering_candidates(
        request,
        target,
        (GeneratedReductionProvider(), CubReductionProvider()),
    )


def tensor_lowering_diagnostics(plan: TensorPlan) -> dict[str, object]:
    """Return deterministic provider provenance for a resolved Tensor CUDA plan."""

    return lowering_diagnostics(resolved_lowering_candidates(plan))
