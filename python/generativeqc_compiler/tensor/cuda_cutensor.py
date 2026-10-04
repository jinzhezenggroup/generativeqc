"""Pure cuTENSOR eligibility for packed TensorIR contractions.

This module does not link or execute cuTENSOR.  It identifies a conservative
subset of existing packed FP64 binary einsums whose materialized dense views can
be described directly by cuTENSOR modes/strides.  Runtime promotion remains
separate and requires explicit provider/workspace bounds.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.specialization import TargetCapabilities

from .cuda_gemm import fp64_coefficient, gemm_contract
from .cuda_layout import conversion_bytes
from .cuda_plan import TensorPlan

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
            "schema": "generativeqc.tensor.cutensor-contract.v1",
            **asdict(self),
        }


def cutensor_contract(plan: TensorPlan, index: int) -> CutensorContract | None:
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
    if (
        contract is None
        or contract.dtype != "float64"
        or min(contract.batch, contract.m, contract.n, contract.k) == 0
        or any(source.spec.dtype != "float64" for source in node.inputs)
    ):
        return None
    precision = plan.precision_by_node.get(node)
    if precision is not None and (
        precision.compute_dtype != "float64"
        or precision.accumulation_dtype != "float64"
    ):
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
        a_modes=tuple(labels[0]),
        b_modes=tuple(labels[1]),
        c_modes=output,
        a_extents=tuple(node.inputs[0].spec.shape),
        b_extents=tuple(node.inputs[1].spec.shape),
        c_extents=tuple(node.spec.shape),
        a_strides=tuple(a_layout.element_strides),
        b_strides=tuple(b_layout.element_strides),
        c_strides=tuple(c_layout.element_strides),
        alpha=fp64_coefficient(node.attrs["coefficient"]),
        batch=contract.batch,
        m=contract.m,
        n=contract.n,
        k=contract.k,
        flops=contract.flops,
        packing_bytes_avoided=packing,
    )


def cutensor_opportunities(plan: TensorPlan) -> tuple[CutensorContract, ...]:
    """Return every packed FP64 site eligible for direct stride contraction."""

    if not isinstance(plan, TensorPlan):
        raise TypeError("cuTENSOR opportunities require a TensorPlan")
    return tuple(
        contract
        for index in range(len(plan.steps))
        if (contract := cutensor_contract(plan, index)) is not None
    )


class CutensorContractionProvider:
    """Advertise cuTENSOR only with explicit availability and resource bounds."""

    descriptor = CUTENSOR_PROVIDER

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        features = dict(target.features)
        reason = None
        workspace = features.get("cutensor-workspace-bytes")
        provider = features.get("cutensor-provider-bytes")
        if request.operation != "tensor-contraction":
            reason = "cuTENSOR pilot requires a binary tensor-contraction request"
        elif request.dtype != "float64" or request.accumulation_dtype != "float64":
            reason = "cuTENSOR pilot is qualified for FP64 storage/accumulation only"
        elif features.get("cutensor") is not True:
            reason = "cuTENSOR requires explicit cutensor=True capability"
        elif type(workspace) is not int or workspace < 0:
            reason = "cuTENSOR requires an explicit non-negative workspace byte bound"
        elif type(provider) is not int or provider < 0:
            reason = "cuTENSOR requires an explicit non-negative provider byte bound"
        return (
            LoweringCandidate(
                request=request,
                implementation="tensor-contraction-cutensor",
                providers=(self.descriptor,),
                status="unsupported" if reason else "ready",
                numerical_mode=f"{request.dtype}->{request.accumulation_dtype}",
                workspace_bytes=workspace if reason is None else 0,
                provider_bytes=provider if reason is None else 0,
                reason=reason,
                provenance=(("selection", "candidate-only-not-promoted"),),
            ),
        )


def cutensor_provider_candidates(
    plan: TensorPlan,
    index: int,
    *,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Describe the optional cuTENSOR provider for one eligible packed site."""

    contract = cutensor_contract(plan, index)
    if contract is None:
        raise ValueError("step is not an eligible packed FP64 cuTENSOR contraction")
    request = LoweringRequest(
        consumer="tensor.cuda",
        operation="tensor-contraction",
        backend="cuda",
        dtype="float64",
        accumulation_dtype="float64",
        shape=(contract.batch, contract.m, contract.n, contract.k),
        semantics=(
            ("descriptor_hash", contract.identity),
            ("packing_bytes_avoided", contract.packing_bytes_avoided),
            ("program_hash", plan.program.logical_hash),
            ("step_index", index),
        ),
    )
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
    return collect_lowering_candidates(
        request,
        target,
        (CutensorContractionProvider(),),
    )
