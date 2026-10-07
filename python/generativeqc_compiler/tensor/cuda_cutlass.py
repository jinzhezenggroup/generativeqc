"""Pure CUTLASS AOT offers for the canonical TensorIR contraction request.

Availability describes an actual compiled family and its qualified resources.
It does not invoke a compiler, probe a device or promote an execution default.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import CandidateExecution, digest
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.precision import STRICT_MATH_MODE
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import TargetCapabilities

from .lowering import TensorLoweringAdapter, plan_lowering_request
from .matrix_view import matrix_contraction

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_contract import LoweringPrecision

    from .cuda_plan import TensorPlan

CUTLASS_FAMILY = "simt-32x64x8-v1"
CUTLASS_PROVIDER = ProviderDescriptor(
    name="nvidia.cutlass-aot",
    kind="generated",
    implementation="cutlass-simt-matrix",
    required_features=("cutlass-aot",),
    provenance=(("preparation", "fixed-aot-parameters"),),
)


def _precision_rejection(precision: LoweringPrecision) -> str | None:
    """The initial native family executes only admitted homogeneous arithmetic."""
    d = precision.directive
    if precision.casts or precision.audit or precision.refinement:
        return "CUTLASS AOT family does not implement cast/audit/refinement obligations"
    if (
        d.math_mode != STRICT_MATH_MODE
        or d.storage_dtype not in ("float32", "float64")
        or d.compute_dtype != d.storage_dtype
        or d.accumulation_dtype != d.storage_dtype
        or precision.publication_dtype != d.storage_dtype
        or precision.input_dtypes != (d.storage_dtype, d.storage_dtype)
    ):
        return "CUTLASS AOT family requires homogeneous strict FP32 or FP64"
    return None


class CutlassAotProvider:
    """Offer the exact fixed SIMT family implemented by the native AOT owner.

    Artifact identity includes real sources, external headers, toolchain and
    flags. Host bytes bound the native descriptor objects; module bytes reserve
    context-retained device code and remain charged after local plan release.
    They must come from qualification, never be inferred from a GPU name.
    """

    descriptor = CUTLASS_PROVIDER

    def __init__(self, *, version: str | None = None) -> None:
        self.descriptor = replace(CUTLASS_PROVIDER, version=version)

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        features = dict(target.features)
        reason, recipe = None, None
        try:
            if request.backend != "cuda":
                raise ValueError("CUTLASS requires the canonical CUDA einsum request")
            recipe = matrix_contraction(request)
            output = recipe.layouts[2]
            columns = output.columns if output.order == "row" else output.rows
            if recipe.batches > 65535 or (columns + 63) // 64 > 65535:
                raise ValueError("CUTLASS AOT family exceeds its CUDA grid bound")
            # Match the native family's signed-int intermediate bounds. CUTLASS
            # rounds kernel M/K by its fixed tiles; column output swaps M/N.
            rows = output.rows if output.order == "row" else output.columns
            int_max = (1 << 31) - 1
            if rows > int_max - 32 or recipe.layouts[0].columns > int_max - 8:
                raise ValueError(
                    "CUTLASS AOT family exceeds its signed kernel tile bound"
                )
        except ValueError as error:
            reason = str(error)
        bounds = {
            key: features.get(f"cutlass-aot-{key}-bytes") for key in ("host", "module")
        }

        def bound(key: str) -> int:
            value = bounds[key]
            return value if type(value) is int and 0 < value < (1 << 63) else 0

        artifact = features.get("cutlass-aot-artifact")
        if reason is None:
            if features.get("cutlass-aot") is not True:
                reason = "CUTLASS requires explicit cutlass-aot=True capability"
            elif (
                not self.descriptor.version
                or features.get("cutlass-version") != self.descriptor.version
            ):
                reason = "CUTLASS requires a matching explicit header version"
            elif features.get("cutlass-aot-family") != CUTLASS_FAMILY:
                reason = "CUTLASS requires the compiled SIMT family identity"
            elif (
                target.target.subgroup_size != 32
                or target.target.maximum_workgroup_threads < 128
            ):
                reason = "CUTLASS SIMT family requires four 32-thread warps"
            else:
                try:
                    if type(artifact) is not str:
                        raise ValueError(
                            "CUTLASS AOT artifact must be a SHA-256 digest"
                        )
                    digest(artifact, "CUTLASS AOT artifact")
                except ValueError as error:
                    reason = str(error)
                for key in bounds:
                    if not bound(key):
                        reason = (
                            reason
                            or f"CUTLASS requires a positive qualified {key} byte bound"
                        )
        candidates = []
        for precision in request.precisions or (None,):
            rejection = reason
            execution = None
            if precision is None:
                rejection = (
                    rejection or "CUTLASS requires an admitted precision schedule"
                )
            else:
                rejection = rejection or _precision_rejection(precision)
                execution = CandidateExecution(
                    precision,
                    CUTLASS_FAMILY,
                    request.operands,
                    topology=ScheduleTopology(
                        tiles=(32, 64, 8),
                        workgroup_threads=128,
                        subgroup_size=32,
                        fusion="canonical-alpha-beta-epilogue",
                        materialization="direct-affine-views",
                    ),
                    determinism="reproducible",
                    host_bytes=bound("host"),
                    cache_bytes=bound("module"),
                )
            candidates.append(
                LoweringCandidate(
                    request=request,
                    implementation="tensor-einsum-cutlass-aot",
                    providers=(self.descriptor,),
                    status="unsupported" if rejection else "ready",
                    reason=rejection,
                    execution=execution,
                    numerical_mode=(
                        f"{precision.directive.compute_dtype}->{precision.directive.accumulation_dtype}"
                        if precision
                        else f"{request.dtype}->{request.accumulation_dtype}"
                    ),
                    target=target,
                    provenance=(
                        ("matrix_recipe", recipe.identity if recipe else "unsupported"),
                        (
                            "artifact",
                            artifact if type(artifact) is str else "unavailable",
                        ),
                        ("module_lifetime", "cuda-context"),
                        ("selection", "candidate-only-not-promoted"),
                    ),
                )
            )
        return tuple(candidates)


def cutlass_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Use the exact planned request consumed by the other matrix providers."""
    request = plan_lowering_request(plan, index, TensorLoweringAdapter(plan.program))
    target = target_capabilities or TargetCapabilities(plan.target.target_info)
    if target.target != plan.target.target_info:
        raise ValueError("CUTLASS capabilities do not match the planned target")
    version = dict(target.features).get("cutlass-version")
    return collect_lowering_candidates(
        request,
        target,
        (
            CutlassAotProvider(
                version=version if type(version) is str and version.strip() else None
            ),
        ),
    )
