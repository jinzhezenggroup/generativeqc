"""Pure cuBLASLt offers for the canonical TensorIR contraction request.

Physical matrix recognition is provider implementation metadata. It does not
rewrite equations, invent precision variants, probe a device or choose an
algorithm. Native preparation must resolve a bounded heuristic once and retain
its algorithm/workspace identity before any replay.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringPrecision,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.precision import STRICT_MATH_MODE
from generativeqc_compiler.common.specialization import TargetCapabilities

from .lowering import TensorLoweringAdapter, plan_lowering_request
from .matrix_view import MatrixContraction, matrix_contraction

if TYPE_CHECKING:
    from .cuda_plan import TensorPlan

CUBLASLT_PROVIDER = ProviderDescriptor(
    name="nvidia.cublaslt",
    kind="library",
    implementation="cublaslt-matmul",
    required_features=("cublaslt",),
    provenance=(("algorithm_discovery", "bounded-prepare-only"),),
)


def cublaslt_matmul(request: LoweringRequest) -> MatrixContraction:
    """Project the shared affine proof; preserve provider-specific diagnostics."""
    try:
        return matrix_contraction(request)
    except ValueError as error:
        raise ValueError(str(error).replace("matrix view", "cuBLASLt")) from error


def _precision_rejection(precision: LoweringPrecision) -> str | None:
    """Do not invent conversions or silently enable TF32/reduced accumulation."""
    d = precision.directive
    if precision.casts or precision.refinement or precision.audit:
        return "cuBLASLt adapter does not implement cast/refinement/audit obligations"
    if (
        d.math_mode != STRICT_MATH_MODE
        or d.storage_dtype not in ("float32", "float64")
        or d.compute_dtype != d.storage_dtype
        or d.accumulation_dtype != d.compute_dtype
        or precision.publication_dtype != d.storage_dtype
        or precision.input_dtypes != (d.storage_dtype, d.storage_dtype)
    ):
        return "cuBLASLt adapter requires homogeneous pedantic FP32 or FP64 arithmetic"
    return None


class CublasLtMatmulProvider:
    """Advertise bounded preparation eligibility, never a resolved algorithm.

    Resource facts are externally qualified simultaneous ceilings. Native
    preparation still has to return an algorithm with workspace within the cap,
    reject unavailable shapes, and charge opaque/lazy provider state. Unknown
    costs remain unknown, so these offers alone cannot promote a method default.
    """

    descriptor = CUBLASLT_PROVIDER

    def __init__(self, *, version: str | None = None) -> None:
        self.descriptor = replace(CUBLASLT_PROVIDER, version=version)

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        features = dict(target.features)
        reason = None
        recipe = None
        try:
            recipe = cublaslt_matmul(request)
        except ValueError as error:
            reason = str(error)
        bounds = {
            key: features.get(f"cublaslt-{key}-bytes")
            for key in ("workspace", "provider", "host", "cache")
        }

        def bound(key: str) -> int:
            value = bounds[key]
            return value if type(value) is int and value >= 0 else 0

        version = self.descriptor.version
        if reason is None:
            if features.get("cublaslt") is not True:
                reason = "cuBLASLt requires explicit cublaslt=True capability"
            elif not version or features.get("cublaslt-version") != version:
                reason = "cuBLASLt requires a matching explicit provider version"
            else:
                for key, value in bounds.items():
                    if type(value) is not int or value < 0:
                        reason = f"cuBLASLt requires an explicit non-negative {key} byte bound"
                        break
        candidates = []
        for precision in request.precisions or (None,):
            rejection = reason
            execution = None
            if precision is None:
                rejection = (
                    rejection or "cuBLASLt requires an admitted precision schedule"
                )
            else:
                rejection = rejection or _precision_rejection(precision)
                execution = CandidateExecution(
                    precision,
                    "cublaslt-prepare-heuristic",
                    request.operands,
                    host_bytes=bound("host"),
                    cache_bytes=bound("cache"),
                )
            candidates.append(
                LoweringCandidate(
                    request=request,
                    implementation="tensor-einsum-cublaslt",
                    providers=(self.descriptor,),
                    status="unsupported" if rejection else "ready",
                    reason=rejection,
                    execution=execution,
                    numerical_mode=(
                        f"{precision.directive.compute_dtype}->{precision.directive.accumulation_dtype}"
                        if precision
                        else f"{request.dtype}->{request.accumulation_dtype}"
                    ),
                    workspace_bytes=bound("workspace"),
                    provider_bytes=bound("provider"),
                    target=target,
                    provenance=(
                        ("matrix_recipe", recipe.identity if recipe else "unsupported"),
                        ("algorithm_discovery", "prepare-only"),
                        ("epilogue", "default"),
                        ("selection", "candidate-only-not-promoted"),
                    ),
                )
            )
        return tuple(candidates)


def cublaslt_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Project the exact request already used by cuBLAS/generated diagnostics."""
    request = plan_lowering_request(plan, index, TensorLoweringAdapter(plan.program))
    target = target_capabilities or TargetCapabilities(plan.target.target_info)
    if target.target != plan.target.target_info:
        raise ValueError("cuBLASLt capabilities do not match the planned target")
    version = dict(target.features).get("cublaslt-version")
    return collect_lowering_candidates(
        request,
        target,
        (CublasLtMatmulProvider(version=version if type(version) is str else None),),
    )
