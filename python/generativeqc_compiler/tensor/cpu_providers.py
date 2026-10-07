"""CPU dense-linear-algebra offers for canonical TensorIR contractions.

The existing native `tensor::cpu_linalg` owner already hides scalar/OpenBLAS
execution from scientific callers.  This module projects that owner into the
same provider-neutral lowering contract used by CUDA tensor providers.  It does
not call BLAS, probe a host, change CPU provider defaults, or invent another
scientific operation.
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

from .matrix_view import matrix_contraction

if TYPE_CHECKING:
    from generativeqc_compiler.common.specialization import TargetCapabilities

    from .matrix_view import MatrixContraction

CPU_SCALAR_PROVIDER = ProviderDescriptor(
    name="generativeqc.cpu_linalg.scalar",
    kind="runtime",
    implementation="cpu-linalg-scalar-gemm",
    version="1",
    provenance=(
        ("execution_owner", "src/tensor/cpu_linalg.cpp"),
        ("thread_ownership", "task-or-provider-explicit"),
    ),
)

OPENBLAS_PROVIDER = ProviderDescriptor(
    name="openblas.cblas",
    kind="library",
    implementation="openblas-cblas-dgemm",
    required_features=("openblas",),
    provenance=(
        ("execution_owner", "src/tensor/cpu_linalg.cpp"),
        ("version_source", "runtime-probe"),
        ("thread_ownership", "explicit"),
    ),
)


def cpu_matrix_contraction(request: LoweringRequest) -> MatrixContraction:
    """Return the shared affine matrix proof for a CPU TensorIR request."""
    if request.backend != "cpu":
        raise ValueError("CPU dense provider requires the canonical CPU einsum request")
    recipe = matrix_contraction(request)
    if recipe.batches != 1:
        raise ValueError("current CPU dense runtime does not expose batched GEMM")
    for layout in recipe.layouts:
        compact_leading_dimension = (
            layout.columns if layout.order == "row" else layout.rows
        )
        if layout.leading_dimension != compact_leading_dimension:
            raise ValueError("current CPU dense runtime requires compact matrix views")
    if recipe.layouts[2].order != "row":
        raise ValueError("current CPU dense runtime requires row-major output storage")
    return recipe


def _precision_rejection(precision: LoweringPrecision) -> str | None:
    """Match the currently executable native `cpu_gemm(double)` contract."""
    directive = precision.directive
    if precision.casts or precision.refinement or precision.audit:
        return "CPU dense adapter does not implement cast/refinement/audit obligations"
    if (
        directive.math_mode != STRICT_MATH_MODE
        or directive.storage_dtype != "float64"
        or directive.compute_dtype != "float64"
        or directive.accumulation_dtype != "float64"
        or precision.publication_dtype != "float64"
        or precision.input_dtypes != ("float64", "float64")
    ):
        return "current CPU dense runtime requires homogeneous strict FP64 arithmetic"
    return None


def _target_rejection(
    request: LoweringRequest, target: TargetCapabilities
) -> tuple[str | None, MatrixContraction | None]:
    if target.target.backend != "cpu":
        return "CPU dense provider requires a CPU target", None
    try:
        return None, cpu_matrix_contraction(request)
    except ValueError as error:
        return str(error), None


class CpuScalarMatmulProvider:
    """Expose the existing deterministic scalar GEMM as a lowering candidate."""

    descriptor = CPU_SCALAR_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        reason, recipe = _target_rejection(request, target)
        candidates = []
        for precision in request.precisions or (None,):
            rejection = reason
            execution = None
            if precision is None:
                rejection = (
                    rejection or "CPU dense provider requires admitted precision"
                )
            else:
                rejection = rejection or _precision_rejection(precision)
                execution = CandidateExecution(
                    precision,
                    "cpu-linalg-scalar-gemm",
                    request.operands,
                    determinism="exact-order",
                )
            candidates.append(
                LoweringCandidate(
                    request=request,
                    implementation="tensor-einsum-cpu-scalar",
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
                        ("selection", "candidate-only-not-promoted"),
                    ),
                )
            )
        return tuple(candidates)


class OpenBlasMatmulProvider:
    """Expose the native OpenBLAS adapter with explicit thread-ownership facts."""

    descriptor = OPENBLAS_PROVIDER

    def __init__(self, *, version: str | None = None) -> None:
        self.descriptor = replace(OPENBLAS_PROVIDER, version=version)

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        reason, recipe = _target_rejection(request, target)
        features = dict(target.features)
        ownership = features.get("cpu-linalg-thread-ownership")
        threads = features.get("cpu-linalg-provider-threads")
        version = self.descriptor.version
        if reason is None:
            if features.get("openblas") is not True:
                reason = "OpenBLAS requires explicit openblas=True capability"
            elif not version or features.get("openblas-version") != version:
                reason = "OpenBLAS requires a matching explicit provider version"
            elif ownership not in ("task-parallel", "provider-parallel"):
                reason = (
                    "OpenBLAS requires explicit CPU linear-algebra thread ownership"
                )
            elif type(threads) is not int or threads < 1:
                reason = "OpenBLAS requires a positive explicit provider thread count"
            elif ownership == "task-parallel" and threads != 1:
                reason = "task-parallel CPU linear algebra requires one provider thread"
            elif (
                ownership == "task-parallel"
                and features.get("openblas-local-thread-control") is not True
            ):
                reason = "task-parallel OpenBLAS requires thread-local runtime control"
            elif (
                ownership == "provider-parallel"
                and features.get("openblas-local-thread-control") is not True
                and features.get("openblas-global-thread-control") is not True
            ):
                reason = "provider-parallel OpenBLAS requires local or global runtime thread control"

        candidates = []
        for precision in request.precisions or (None,):
            rejection = reason
            execution = None
            if precision is None:
                rejection = rejection or "OpenBLAS requires admitted precision"
            else:
                rejection = rejection or _precision_rejection(precision)
                execution = CandidateExecution(
                    precision,
                    "openblas-cblas-dgemm",
                    request.operands,
                    determinism="unspecified",
                )
            candidates.append(
                LoweringCandidate(
                    request=request,
                    implementation="tensor-einsum-openblas",
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
                            "thread_ownership",
                            ownership if type(ownership) is str else "unknown",
                        ),
                        ("provider_threads", threads if type(threads) is int else -1),
                        ("selection", "candidate-only-not-promoted"),
                    ),
                )
            )
        return tuple(candidates)


def cpu_dense_provider_candidates(
    request: LoweringRequest, target: TargetCapabilities
) -> tuple[LoweringCandidate, ...]:
    """Collect scalar/OpenBLAS offers for one canonical CPU TensorIR request.

    Runtime/build discovery supplies OpenBLAS availability, version and thread
    control facts through `TargetCapabilities`.  Missing facts retain the
    OpenBLAS offer as negative evidence while the scalar candidate stays visible.
    """
    features = dict(target.features)
    version = features.get("openblas-version")
    return collect_lowering_candidates(
        request,
        target,
        (
            CpuScalarMatmulProvider(),
            OpenBlasMatmulProvider(version=version if type(version) is str else None),
        ),
    )
