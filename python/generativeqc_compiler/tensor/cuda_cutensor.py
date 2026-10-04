"""Pure cuTENSOR eligibility for packed TensorIR contractions.

This module does not link or execute cuTENSOR.  It identifies a conservative
subset of existing packed binary einsums whose materialized dense views can
be described directly by cuTENSOR modes/strides.  Runtime promotion remains
separate and requires explicit provider/workspace bounds.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, replace

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
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.specialization import TargetCapabilities

from .cuda_gemm import gemm_contract
from .cuda_layout import conversion_bytes
from .cuda_plan import TensorPlan
from .lowering import TensorLoweringAdapter, plan_lowering_request

CUTENSOR_PROVIDER = ProviderDescriptor(
    name="nvidia.cutensor",
    kind="library",
    implementation="cutensor-contraction",
    required_features=("cutensor",),
    provenance=(
        ("version_source", "runtime-probe"),
        ("resource_contract", "explicit-workspace-and-provider-bounds"),
    ),
)


@dataclass(frozen=True)
class CutensorContract:
    """One stride-addressable replacement for an existing packed GEMM site."""

    step_index: int
    semantic_identity: str
    a_modes: tuple[int, ...]
    b_modes: tuple[int, ...]
    c_modes: tuple[int, ...]
    a_extents: tuple[int, ...]
    b_extents: tuple[int, ...]
    c_extents: tuple[int, ...]
    a_strides: tuple[int, ...]
    b_strides: tuple[int, ...]
    c_strides: tuple[int, ...]
    alpha: float
    batch: int
    m: int
    n: int
    k: int
    flops: int
    packing_bytes_avoided: int

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": "generativeqc.tensor.cutensor-contract.v2",
            **asdict(self),
        }


def cutensor_contract(
    plan: TensorPlan, index: int, *, adapter: TensorLoweringAdapter | None = None
) -> CutensorContract | None:
    """Return a conservative cuTENSOR descriptor for one planned contraction.

    The first slice deliberately targets only sites that already have an audited
    binary-GEMM interpretation but still require generated packing/scatter.  It
    therefore changes no equation inventory or contraction tree.  Virtual
    operands are rejected because cuTENSOR needs addressable tensor storage.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("cuTENSOR eligibility requires a TensorPlan")
    if type(index) is not int or not 0 <= index < len(plan.steps):
        raise ValueError("cuTENSOR step index is outside the plan")
    step = plan.steps[index]
    node = step.node
    if (
        step.virtual
        or step.gemm != "packed"
        or node.op != "einsum"
        or len(step.inputs) != 2
        or not node.spec.size
    ):
        return None
    contract = gemm_contract(node)
    if contract is None or min(contract.batch, contract.m, contract.n, contract.k) == 0:
        return None
    a_step, b_step = (plan.steps[source] for source in step.inputs)
    layouts = (a_step.layout, b_step.layout, step.layout)
    if any(layout is None for layout in layouts):
        return None
    a_layout, b_layout, c_layout = typing.cast(
        "tuple[typing.Any, typing.Any, typing.Any]", layouts
    )
    packing = conversion_bytes(contract, plan.schedule, node.spec.itemsize)
    if not packing:
        return None
    labels = node.attrs["labels"]
    output = tuple(node.attrs["output"])
    return CutensorContract(
        step_index=index,
        semantic_identity=plan_lowering_request(
            plan, index, adapter or TensorLoweringAdapter(plan.program)
        ).semantic_identity,
        a_modes=tuple(labels[0]),
        b_modes=tuple(labels[1]),
        c_modes=output,
        a_extents=tuple(node.inputs[0].spec.shape),
        b_extents=tuple(node.inputs[1].spec.shape),
        c_extents=tuple(node.spec.shape),
        a_strides=tuple(a_layout.element_strides),
        b_strides=tuple(b_layout.element_strides),
        c_strides=tuple(c_layout.element_strides),
        alpha=contract.coefficient,
        batch=contract.batch,
        m=contract.m,
        n=contract.n,
        k=contract.k,
        flops=contract.flops,
        packing_bytes_avoided=packing,
    )


def cutensor_opportunities(plan: TensorPlan) -> tuple[CutensorContract, ...]:
    """Return physical opportunities without filtering scientifically admitted dtypes."""

    if not isinstance(plan, TensorPlan):
        raise TypeError("cuTENSOR opportunities require a TensorPlan")
    adapter = TensorLoweringAdapter(plan.program)
    return tuple(
        contract
        for index in range(len(plan.steps))
        if (contract := cutensor_contract(plan, index, adapter=adapter)) is not None
    )


def _layout_rejection(request: LoweringRequest) -> str | None:
    """Check only physical contraction legality, without admitting arithmetic."""
    if request.backend != "cuda" or request.operation != "einsum":
        return "cuTENSOR requires the canonical CUDA einsum request"
    if request.scientific_identity is None or len(request.operands) != 3:
        return "cuTENSOR requires two inputs and one output with scientific identity"
    a, b, c = request.operands
    if (a.access, b.access, c.access) != ("read", "read", "write"):
        return "cuTENSOR adapter requires a fresh output and two read operands"
    for operand in request.operands:
        if operand.triangle != "full":
            return "cuTENSOR adapter does not infer triangular or scientific symmetry"
        if not operand.shape or not all(operand.shape):
            return "cuTENSOR adapter requires positive nonscalar extents"
        if len(set(operand.modes)) != len(operand.modes):
            return "cuTENSOR adapter does not implement repeated-mode diagonals"
        if operand.strides is None:
            return "cuTENSOR requires materialized affine operands"
        # Sufficient nonoverlap proof for dense/permuted/padded layouts. General
        # interleaved or broadcast views retain generated execution for now.
        span = 1
        for stride, extent in sorted(zip(operand.strides, operand.shape, strict=True)):
            if extent == 1:
                continue
            if stride < span:
                return "cuTENSOR adapter requires nonoverlapping positive strides"
            span += (extent - 1) * stride
    if c.alias_group is not None and c.alias_group in (a.alias_group, b.alias_group):
        return "cuTENSOR adapter cannot overwrite a borrowed input"
    if not set(c.modes) <= set(a.modes) | set(b.modes):
        return "cuTENSOR output modes must originate in the inputs"
    if (set(a.modes) ^ set(b.modes)) - set(c.modes):
        return "cuTENSOR adapter does not implement one-sided reductions"
    return None


def _precision_rejection(precision: LoweringPrecision) -> str | None:
    """Keep every admitted variant visible, including unsupported obligations."""
    directive = precision.directive
    if precision.casts or precision.refinement or precision.audit:
        return "standalone cuTENSOR adapter does not implement cast/refinement/audit obligations"
    if directive.math_mode != STRICT_MATH_MODE:
        return "cuTENSOR adapter requires strict arithmetic"
    if (
        directive.storage_dtype not in ("float32", "float64")
        or directive.compute_dtype != directive.storage_dtype
        or directive.accumulation_dtype != directive.compute_dtype
        or precision.publication_dtype != directive.storage_dtype
        or any(dtype != directive.storage_dtype for dtype in precision.input_dtypes)
    ):
        return "cuTENSOR adapter supports homogeneous FP32 or FP64; mixed arithmetic needs a qualified composite"
    return None


class CutensorContractionProvider:
    """Describe each admitted variant with explicit capability/resource evidence.

    This is pure compiler metadata. Ready means the supplied capability facts
    satisfy this descriptor; native plan construction and endpoint qualification
    remain necessary before executing or caching a binding. Timing stays unknown.
    """

    descriptor = CUTENSOR_PROVIDER

    def __init__(self, *, version: str | None = None) -> None:
        self.descriptor = replace(CUTENSOR_PROVIDER, version=version)

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        features = dict(target.features)
        reason = _layout_rejection(request)
        bounds = {
            name: features.get(f"cutensor-{name}-bytes")
            for name in ("workspace", "provider", "host", "cache")
        }

        def bound(name: str) -> int:
            value = bounds[name]
            return value if type(value) is int and value >= 0 else 0

        version = self.descriptor.version
        if reason is None:
            if features.get("cutensor") is not True:
                reason = "cuTENSOR requires explicit cutensor=True capability"
            elif type(version) is not str or not version.startswith("2."):
                reason = "cuTENSOR requires an explicit 2.x provider version"
            elif features.get("cutensor-version") != version:
                reason = "cuTENSOR provider version does not match target capability evidence"
            else:
                for name, value in bounds.items():
                    if type(value) is not int or value < 0:
                        reason = f"cuTENSOR requires an explicit non-negative {name} byte bound"
                        break
        candidates = []
        for precision in request.precisions or (None,):
            rejection = reason
            execution = None
            numerical = f"{request.dtype}->{request.accumulation_dtype}"
            if precision is None:
                rejection = (
                    rejection
                    or "cuTENSOR requires an explicit admitted precision schedule"
                )
            else:
                rejection = rejection or _precision_rejection(precision)
                numerical = f"{precision.directive.compute_dtype}->{precision.directive.accumulation_dtype}"
                execution = CandidateExecution(
                    precision,
                    "cutensor-affine-contraction",
                    request.operands,
                    host_bytes=bound("host"),
                    cache_bytes=bound("cache"),
                )
            candidates.append(
                LoweringCandidate(
                    request=request,
                    implementation="tensor-einsum-cutensor",
                    providers=(self.descriptor,),
                    status="unsupported" if rejection else "ready",
                    numerical_mode=numerical,
                    workspace_bytes=bound("workspace"),
                    provider_bytes=bound("provider"),
                    reason=rejection,
                    execution=execution,
                    target=target,
                    provenance=(("selection", "candidate-only-not-promoted"),),
                )
            )
        return tuple(candidates)


def cutensor_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Describe the optional cuTENSOR provider for one eligible packed site."""

    adapter = TensorLoweringAdapter(plan.program)
    contract = cutensor_contract(plan, index, adapter=adapter)
    if contract is None:
        raise ValueError("step is not an eligible packed cuTENSOR contraction")
    request = plan_lowering_request(plan, index, adapter)
    target = TargetCapabilities(
        plan.target.target_info,
        features=tuple(
            (feature, True) for feature in plan.target.required_cuda_features
        ),
    )
    if target_capabilities is not None:
        if not isinstance(target_capabilities, TargetCapabilities):
            raise TypeError("cuTENSOR capabilities require TargetCapabilities")
        if target_capabilities.target != plan.target.target_info:
            raise ValueError("cuTENSOR capabilities do not match the planned target")
        target = target_capabilities
    version = dict(target.features).get("cutensor-version")
    return collect_lowering_candidates(
        request,
        target,
        (
            CutensorContractionProvider(
                version=version if type(version) is str else None
            ),
        ),
    )
