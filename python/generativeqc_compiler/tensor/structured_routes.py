"""Discover reusable tensor-provider routes from existing compiler plans.

The TensorIR equations and source hashes are authoritative.  Discovery does not
rewrite contractions, call a vendor, or change production defaults.  Every
implementation is offered against the SAME resolved semantic request, retaining
explicit negative eligibility evidence.  Existing plan selection remains the
qualified incumbent until full costs and numerical/endpoint gates authorize a
different prepared binding.

In particular A^T B A remains two ordered binary contraction sites.  This
module does not silently reassociate or fuse them, which could change FP64
rounding, intermediate storage or publication/failure semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    collect_lowering_candidates,
)
from generativeqc_compiler.common.lowering_selection import (
    LoweringBinding,
    select_lowering_binding,
)
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)

from .cuda_cublaslt import CublasLtMatmulProvider
from .cuda_cutensor import CutensorContractionProvider
from .cuda_cutlass import CutlassAotProvider
from .cuda_gemm import GemmContract, gemm_contract
from .lowering import TensorLoweringAdapter, plan_lowering_request
from .symmetric_rank_k import (
    symmetric_rank_k_candidates,
    symmetric_rank_k_request,
)

if TYPE_CHECKING:
    from .cuda_plan import TensorPlan
    from .ir import Node


@dataclass(frozen=True)
class StructuredRoute:
    """A candidate-only lowering for one unmodified scientific operation."""

    family: str
    step_index: int | None
    output_name: str | None
    request: LoweringRequest
    candidates: tuple[LoweringCandidate, ...]

    @property
    def scientific_identity(self) -> str | None:
        return self.request.scientific_identity


def _family(node: Node, contract: GemmContract) -> str:
    """Describe structure without changing the einsum or promising fusion."""
    if any(
        child.op == "einsum" and gemm_contract(child) is not None
        for child in node.inputs
    ):
        return "matrix-chain"
    if (
        node.inputs[0] is node.inputs[1]
        and contract.m_labels
        and contract.n_labels
        and contract.k_labels
        and contract.m == contract.n
    ):
        return "gram"
    if contract.m_labels and contract.n_labels:
        return "gemm"
    if contract.m_labels or contract.n_labels:
        return "gemv"
    return "dot"


def _version(features: dict[str, object], name: str) -> str | None:
    value = features.get(name)
    return value if type(value) is str and value.strip() else None


def structured_cuda_routes(
    plan: TensorPlan,
    *,
    source_identity: str,
    target_capabilities: TargetCapabilities | None = None,
) -> tuple[StructuredRoute, ...]:
    """Discover GEMM/GEMV/Gram, matrix-chain and symmetric weighted-Gram routes.

    Binary einsums offer cuBLASLt, cuTENSOR and CUTLASS AOT through the same
    canonical plan request.  Provider-specific constraints (including absent
    toolkit, actual strides, workspace, arithmetic and capture) remain explicit
    unsupported candidates.  The qualified TensorPlan generated/cuBLAS route
    remains the incumbent and is NOT replaced by these candidate-only offers.

    Three-factor C diag(w) C^T uses the existing strictly checked, signed rank-k
    TensorIR request and its already implemented generated/cuBLAS candidates.
    Other ternary, aliased, indexed, non-FP64 and nonfinite-policy-sensitive
    shapes are deliberately not transformed into GEMM or silently advertised.
    """
    from .cuda_plan import TensorPlan

    if not isinstance(plan, TensorPlan):
        raise TypeError("structured routes require a resolved TensorPlan")
    if not isinstance(source_identity, str) or not source_identity.strip():
        raise ValueError("structured routes require a nonempty AOT source identity")
    target = target_capabilities or TargetCapabilities(
        plan.target.target_info,
        features=tuple((name, True) for name in plan.target.required_cuda_features),
    )
    if not isinstance(target, TargetCapabilities):
        raise TypeError("structured routes require TargetCapabilities")
    if target.target != plan.target.target_info:
        raise ValueError("structured route capabilities do not match the plan target")

    versions = dict(target.features)
    providers = (
        CublasLtMatmulProvider(version=_version(versions, "cublaslt-version")),
        CutensorContractionProvider(version=_version(versions, "cutensor-version")),
        CutlassAotProvider(version=_version(versions, "cutlass-version")),
    )
    adapter = TensorLoweringAdapter(plan.program)
    routes: list[StructuredRoute] = []

    for index, step in enumerate(plan.steps):
        if step.virtual or not step.node.spec.size:
            continue
        contract = gemm_contract(step.node)
        if contract is None or contract.k == 0:
            continue
        request = plan_lowering_request(plan, index, adapter)
        candidates = collect_lowering_candidates(request, target, providers)
        routes.append(
            StructuredRoute(
                _family(step.node, contract),
                index,
                None,
                request,
                candidates,
            )
        )

    for name, node in sorted(plan.program.outputs.items()):
        if node.op != "einsum" or len(node.inputs) != 3:
            continue
        if node.inputs[0] is not node.inputs[2]:
            continue
        try:
            request = symmetric_rank_k_request(
                plan.program, name, update="overwrite"
            )
        except ValueError:
            # The rank-k provider must prove exact domains, aliasing, FP64,
            # arithmetic and transactional publication before admission.
            continue
        candidates = symmetric_rank_k_candidates(request, source_identity, target)
        routes.append(StructuredRoute("weighted-gram", None, name, request, candidates))
    return tuple(routes)


def bind_structured_route(
    route: StructuredRoute,
    target: TargetCapabilities,
    compilation: CompilationIdentity,
    *,
    expected_replays: int = 1,
    qualified_incumbent: str | None = None,
) -> LoweringBinding:
    """Use the existing evidence-gated selector, not a new shape heuristic.

    A portfolio without complete phase costs cannot displace an incumbent.
    Calling this API does NOT install the selected binding into an SCF/DFT/GFN
    execution path; that requires native resource/capture and endpoint gates.
    """
    return select_lowering_binding(
        route.request,
        target,
        compilation,
        route.candidates,
        expected_replays=expected_replays,
        qualified_incumbent=qualified_incumbent,
    )
