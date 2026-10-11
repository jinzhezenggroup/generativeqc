"""Explicit provider pinning for compile/numerical fixtures, never production policy.

Synthetic costs exercise the binding machinery, not measured profitability or
promotion. Device-free fixtures use synthetic versions; compiled fixtures bind
actual source/toolchain facts that compilation must independently revalidate.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from generativeqc_compiler.common.lowering_contract import LoweringCost
from generativeqc_compiler.common.lowering_selection import select_lowering_binding
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)
from generativeqc_compiler.tensor.cuda_providers import (
    bind_reduction_lowering,
    reduction_provider_candidates,
)
from generativeqc_compiler.tensor.cuda_reduction import cooperative_reduction_eligible

if TYPE_CHECKING:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_plan import TensorPlan


def qualification_reduction_plan(
    plan: TensorPlan,
    *,
    implementation: Literal["generated", "cub"] = "cub",
    compiler: CudaCompilerAdapter | None = None,
) -> TensorPlan:
    """Pin eligible steps in a fixture while retaining the other strict fallback.

    Compiled fixtures obtain actual source/toolchain identities before binding.
    Device-free emission fixtures use synthetic identities. Ranking costs remain
    synthetic in both cases and may never select production policy.
    """
    if implementation not in ("generated", "cub"):
        raise ValueError("unknown qualification reduction implementation")
    compiler_hash = canonical_hash("qualification-fixture-only")
    version = "qualification-fixture-only"
    if compiler is not None:
        from generativeqc_compiler.tensor.cuda_execute import tensor_compiler_identity

        compiler_hash = tensor_compiler_identity(plan, compiler)
        version = f"prepared-context:{compiler_hash}"
    target = TargetCapabilities(
        plan.target.target_info,
        features=(("cuda", True), ("cub-block-reduce-header", True)),
    )
    for index in range(len(plan.steps)):
        if not cooperative_reduction_eligible(plan, index):
            continue
        candidates = tuple(
            replace(
                candidate,
                providers=tuple(
                    replace(provider, version=version)
                    for provider in candidate.providers
                ),
                cost=LoweringCost(
                    source="qualification/synthetic-ranking-not-performance",
                    kind="estimated",
                    prepare_ns=0,
                    kernel_ns=1
                    if implementation in candidate.execution.algorithm
                    else 2,
                    cast_ns=0,
                    pack_ns=0,
                    refinement_ns=0,
                    audit_ns=0,
                    fallback_ns=0,
                ),
            )
            for candidate in reduction_provider_candidates(
                plan, index, target_capabilities=target
            )
        )
        request = candidates[0].request
        binding = select_lowering_binding(
            request,
            target,
            CompilationIdentity(
                request.scientific_identity,
                compiler_hash,
            ),
            candidates,
        )
        plan = bind_reduction_lowering(plan, index, binding)
    return plan
