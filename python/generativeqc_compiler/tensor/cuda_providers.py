"""Resolved provider provenance for the existing TensorIR CUDA lowering.

It projects planned execution into the shared lowering-provider contract and
binds eligible reductions through that contract. Scientific IR and schedule
search never accept a provider name. Unbound reductions retain generated CUDA;
alternative bindings require explicit legality, version and complete-cost facts.
"""

from __future__ import annotations

from dataclasses import replace

from generativeqc_compiler.common.lowering_contract import CandidateExecution
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
    collect_lowering_candidates,
    lowering_diagnostics,
)
from generativeqc_compiler.common.lowering_selection import (
    LoweringBinding,
    select_lowering_binding,
)
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import TargetCapabilities

from .cuda_gemm import gemm_contract
from .cuda_plan import TensorPlan
from .cuda_reduction import (
    cooperative_reduction_eligible,
    cooperative_reduction_provider,
)
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
        cooperative_provider = cooperative_reduction_provider(plan, index)
        is_gemm = step.gemm != "none" and contract is not None
        uses_cublas = is_gemm and contract.k > 0
        request = plan_lowering_request(plan, index, adapter)
        if uses_cublas:
            providers = (CUBLAS_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = f"tensor-gemm-{step.gemm}"
        elif is_gemm:
            providers = (CUDA_RUNTIME_PROVIDER,)
            implementation = "tensor-gemm-zero-fill"
        elif cooperative_provider == "cub":
            providers = (CUB_REDUCTION_PROVIDER, GENERATED_CUDA_PROVIDER)
            implementation = "tensor-reduce-cub-block-reduce"
        elif cooperative_provider == "generated":
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = "tensor-reduce-generated-cooperative"
        else:
            providers = (GENERATED_CUDA_PROVIDER,)
            implementation = f"tensor-generated-{node.op}"
        binding = plan.lowering_by_step.get(index)
        if binding is not None:
            candidates.append(binding.selected)
            continue
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
    adapter: TensorLoweringAdapter | None = None,
) -> tuple[LoweringCandidate, ...]:
    """Advertise reductions using explicit, plan-bound toolkit capability facts.

    GPU architecture alone does not establish that CUB headers are installed.
    Without caller-supplied header evidence the CUB offer is unsupported; the
    generated offer remains available. This routine does not probe a toolkit.
    """

    if not isinstance(plan, TensorPlan):
        raise TypeError("reduction provider candidates require a TensorPlan")
    if not cooperative_reduction_eligible(plan, index):
        raise ValueError("step is not an eligible cooperative reduction")
    if adapter is None:
        adapter = TensorLoweringAdapter(plan.program)
    elif adapter.program is not plan.program:
        raise ValueError("reduction adapter does not match the planned program")
    request = plan_lowering_request(plan, index, adapter)
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
    candidates = collect_lowering_candidates(
        request,
        target,
        (GeneratedReductionProvider(), CubReductionProvider()),
    )
    topology = ScheduleTopology(
        workgroup_threads=plan.schedule.threads,
        subgroup_size=plan.target.warp_size,
        fusion="fused" if plan.schedule.fuse else "unfused",
        materialization="recompute" if plan.schedule.recompute else "materialize",
        residency="planned-device",
        reduction=f"unroll-{plan.schedule.reduction_unroll}",
        cooperative=True,
    )
    return tuple(
        replace(candidate, execution=replace(candidate.execution, topology=topology))
        if candidate.execution is not None
        else candidate
        for candidate in candidates
    )


def _reduction_offer(candidate: LoweringCandidate) -> LoweringCandidate:
    """Separate adapter legality from externally supplied cost/version evidence."""
    return replace(
        candidate,
        cost=None,
        provenance=(),
        providers=tuple(
            replace(provider, version=None) for provider in candidate.providers
        ),
    )


def validate_reduction_bindings(plan: TensorPlan) -> None:
    """Fail closed on stale/forged recipes, including dataclass plan replacements.

    Reconstruct the bounded shared selection and provider offers before codegen.
    A binding cannot smuggle in a different arithmetic/layout/resource contract
    or an unsupported algorithm merely by claiming a known provider name.
    """
    indices = tuple(index for index, _ in plan.lowering_bindings)
    if any(
        type(index) is not int or not 0 <= index < len(plan.steps) for index in indices
    ):
        raise ValueError("reduction binding step is outside the tensor plan")
    if indices != tuple(sorted(set(indices))):
        raise ValueError("reduction bindings require unique sorted steps")
    adapter = TensorLoweringAdapter(plan.program)
    for index, binding in plan.lowering_bindings:
        if not isinstance(binding, LoweringBinding):
            raise TypeError("reduction execution requires a shared LoweringBinding")
        offers = reduction_provider_candidates(
            plan, index, target_capabilities=binding.target, adapter=adapter
        )
        request = offers[0].request
        binding.validate_context(request, binding.target, binding.compilation)
        canonical = tuple(_reduction_offer(candidate) for candidate in offers)
        if any(
            _reduction_offer(candidate) not in canonical
            for candidate in binding.candidates
        ):
            raise ValueError(
                "reduction binding does not match canonical provider offers"
            )
        reconstructed = select_lowering_binding(
            request,
            binding.target,
            binding.compilation,
            binding.candidates,
            expected_replays=binding.expected_replays,
            qualified_incumbent=(
                binding.selected.identity if binding.retained_incumbent else None
            ),
        )
        if reconstructed != binding:
            raise ValueError(
                "reduction binding differs from shared candidate selection"
            )
        if binding.retained_incumbent and binding.selected.implementation != (
            "tensor-reduce-generated-cooperative"
        ):
            raise ValueError(
                "only generated reduction is a qualified unmeasured incumbent"
            )


def bind_reduction_lowering(
    plan: TensorPlan, index: int, binding: LoweringBinding
) -> TensorPlan:
    """Install a prepared recipe without changing equations, precision or schedule.

    The shared selector owns complete preparation/replay costs, strict fallback,
    target capabilities and version provenance. This adapter owns exact supported
    algorithms and their code emission. It does not tune or probe at execution.
    """
    bindings = dict(plan.lowering_bindings)
    bindings[index] = binding
    return replace(plan, lowering_bindings=tuple(sorted(bindings.items())))


def tensor_lowering_diagnostics(plan: TensorPlan) -> dict[str, object]:
    """Return deterministic provider provenance for a resolved Tensor CUDA plan."""

    report = lowering_diagnostics(resolved_lowering_candidates(plan))
    report["bindings"] = [
        {"step": index, **binding.to_payload()}
        for index, binding in plan.lowering_bindings
    ]
    return report
