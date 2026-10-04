"""Pure cuBLASLt offers for the canonical TensorIR contraction request.

Physical matrix recognition is provider implementation metadata. It does not
rewrite equations, invent precision variants, probe a device or choose an
algorithm. Native preparation must resolve a bounded heuristic once and retain
its algorithm/workspace identity before any replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringPrecision,
    OperandLayout,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.precision import STRICT_MATH_MODE
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.specialization import TargetCapabilities

from .lowering import TensorLoweringAdapter, plan_lowering_request

if TYPE_CHECKING:
    from .cuda_plan import TensorPlan

CUBLASLT_PROVIDER = ProviderDescriptor(
    name="nvidia.cublaslt",
    kind="library",
    implementation="cublaslt-matmul",
    required_features=("cublaslt",),
    provenance=(("algorithm_discovery", "bounded-prepare-only"),),
)


@dataclass(frozen=True)
class CublasLtMatrixLayout:
    """A logical matrix with native row/column storage; strides are elements."""

    rows: int
    columns: int
    order: str
    leading_dimension: int
    batch_stride: int


@dataclass(frozen=True)
class CublasLtMatmul:
    """Provider recipe retaining the canonical request and physical layouts.

    A and B have logical shapes M,K and K,N; D has M,N. Row/column orders
    describe the existing memory directly, including a transposed result. There
    is no generated packing, output scatter, broadcast or epilogue in this offer.
    """

    request_identity: str
    batches: int
    layouts: tuple[CublasLtMatrixLayout, ...]

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))


def _collapsed_axis(view: OperandLayout, modes: tuple[int, ...]) -> tuple[int, int]:
    """Prove lexicographic flattening preserves every nonunit axis address.

    Groups use the same semantic mode order in every operand. Equal dimension
    products alone would incorrectly admit swapped reduction axes. Unit axes
    have no observable stride and do not constrain the physical matrix cut.
    """
    assert view.strides is not None
    extent, stride = 1, None
    for mode in reversed(modes):
        axis = view.modes.index(mode)
        size, physical = view.shape[axis], view.strides[axis]
        if size != 1:
            if stride is None:
                stride = physical
            if physical != extent * stride:
                raise ValueError("cuBLASLt mode group is not physically contiguous")
        extent *= size
    return extent, 1 if stride is None else stride


def _matrix_layout(
    view: OperandLayout,
    row: tuple[int, ...],
    column: tuple[int, ...],
    batch: tuple[int, ...],
    batches: int,
) -> CublasLtMatrixLayout:
    """Prove the view is addressable without copying or overlapping batches."""
    if view.strides is None or any(s <= 0 or s > (1 << 63) - 1 for s in view.strides):
        raise ValueError("cuBLASLt requires materialized positive matrix strides")
    rows, rs = _collapsed_axis(view, row)
    columns, cs = _collapsed_axis(view, column)
    # A unit axis has no observable stride. Resolve it deterministically without
    # rejecting equivalent dense or padded views emitted by the storage owner.
    if (columns == 1 or cs == 1) and (rows == 1 or rs >= columns):
        order, ld = "row", rs if rows > 1 else columns
    elif (rows == 1 or rs == 1) and (columns == 1 or cs >= rows):
        order, ld = "column", cs if columns > 1 else rows
    else:
        raise ValueError("cuBLASLt requires a native row/column matrix layout")
    stride = _collapsed_axis(view, batch)[1] if batch else 0
    span = (rows - 1) * rs + (columns - 1) * cs + 1
    if batches > 1 and stride < span:
        raise ValueError("cuBLASLt adapter requires nonoverlapping matrix batches")
    if max(rows, columns, ld, batches) > (1 << 31) - 1:
        raise ValueError("cuBLASLt adapter matrix dimensions exceed its native bound")
    return CublasLtMatrixLayout(rows, columns, order, ld, stride)


def cublaslt_matmul(request: LoweringRequest) -> CublasLtMatmul:
    """Recognize affine matmul while retaining the original einsum identity.

    M/N/K and batch may each group existing modes only when their ordered
    strides prove a matrix view without packing. One-sided reductions,
    diagonals, broadcast and scientific symmetry remain explicit rejections.
    """
    if request.backend != "cuda" or request.operation != "einsum":
        raise ValueError("cuBLASLt requires the canonical CUDA einsum request")
    if request.scientific_identity is None or len(request.operands) != 3:
        raise ValueError(
            "cuBLASLt requires two inputs, one output and scientific identity"
        )
    a, b, c = request.operands
    if (a.access, b.access, c.access) != ("read", "read", "write"):
        raise ValueError("cuBLASLt adapter requires two reads and a fresh output")
    if c.alias_group is not None and c.alias_group in (a.alias_group, b.alias_group):
        raise ValueError("cuBLASLt adapter cannot overwrite a borrowed input")
    sizes: dict[int, int] = {}
    for view in request.operands:
        if len(view.modes) > 8:
            raise ValueError("cuBLASLt operand exceeds the native rank bound of eight")
        if view.triangle != "full" or not view.shape or not all(view.shape):
            raise ValueError("cuBLASLt requires positive full matrix operands")
        if len(set(view.modes)) != len(view.modes):
            raise ValueError(
                "cuBLASLt adapter does not implement repeated-mode diagonals"
            )
        for mode, size in zip(view.modes, view.shape, strict=True):
            if sizes.setdefault(mode, size) != size:
                raise ValueError("cuBLASLt semantic mode extents disagree")
    am, bm, cm = (set(view.modes) for view in request.operands)
    batch, m, n, k = am & bm & cm, (am & cm) - bm, (bm & cm) - am, (am & bm) - cm
    if (
        not m
        or not n
        or not k
        or am != batch | m | k
        or bm != batch | k | n
        or cm != batch | m | n
    ):
        raise ValueError(
            "cuBLASLt adapter requires M/N/K groups and shared batch modes"
        )
    mi = tuple(mode for mode in c.modes if mode in m)
    ni = tuple(mode for mode in c.modes if mode in n)
    ki = tuple(mode for mode in a.modes if mode in k)
    bi = tuple(mode for mode in c.modes if mode in batch)
    batches = prod(sizes[mode] for mode in bi)
    return CublasLtMatmul(
        request.identity,
        batches,
        tuple(
            _matrix_layout(view, row, column, bi, batches)
            for view, row, column in ((a, mi, ki), (b, ki, ni), (c, mi, ni))
        ),
    )


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
