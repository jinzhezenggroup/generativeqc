"""Bounded structured matrix regions using the existing scientific TensorIR.

The AOT hashes identify templates; native bindings additionally retain resolved
dimensions, context and scratch bounds. No method name selects a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
)
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)

from .basis_transform import basis_transform_program
from .ir import add, einsum, input_tensor, power
from .lowering import TensorLoweringAdapter
from .program import Program
from .symmetric_rank_k import symmetric_rank_k_request
from .types import Index, IndexSpace, TensorSpec


def structured_program(
    dimension: int, *, inverse_root: bool = False, pair: bool = False
) -> Program:
    """Describe column-major weighted Gram, inverse-root and symmetric pairs."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    space = IndexSpace("matrix", "ao", dimension)
    orbital, row = Index("orbital", space), Index("row", space)
    vectors = input_tensor("vectors", TensorSpec((row, orbital), role="input"))
    if pair:
        other = input_tensor("other", TensorSpec((row, orbital), role="input"))
        product = add(
            einsum("ik,jk->ij", vectors, other), einsum("ik,jk->ij", other, vectors)
        )
    else:
        weights = input_tensor("weights", TensorSpec((orbital,), role="input"))
        if inverse_root:
            weights = power(weights, "-1/2")
        product = einsum("ik,k,jk->ij", vectors, weights, vectors)
    return Program({"product": product})


@dataclass(frozen=True)
class StructuredPortfolio:
    """Canonical template offers; native preparation resolves resource extents."""

    request: LoweringRequest
    candidates: tuple[LoweringCandidate, ...]
    target: TargetCapabilities
    compilation: CompilationIdentity


def structured_scf_portfolios() -> tuple[StructuredPortfolio, ...]:
    """Return seven bounded regions, including negative provider evidence.

    Templates do not establish a measured crossover. The final basis region
    includes hcore in its scientific identity, even though publication fuses it.
    """
    programs = [
        structured_program(5),
        structured_program(5),
        structured_program(5, inverse_root=True),
    ]
    basis = basis_transform_program(5, 7)
    regions = [(program, "product") for program in programs]
    regions.extend(
        (basis, name)
        for name in ("density_right", "density_direct", "fock_left", "fock_public")
    )
    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    portfolios = []
    for slot, (program, name) in enumerate(regions):
        adapter = TensorLoweringAdapter(program)
        request = (
            symmetric_rank_k_request(
                program, name, order="column-major", update="overwrite"
            )
            if slot < 2
            else adapter.request(program.outputs[name], backend="cuda")
        )
        if slot < 2:
            semantics = dict(request.semantics)
            semantics["publication"] = "lower-triangle-mirrored"
            request = replace(
                request,
                semantics=tuple(semantics.items()),
                operands=tuple(
                    replace(operand, triangle="lower")
                    if operand.access == "write"
                    else operand
                    for operand in request.operands
                ),
            )
        request = replace(
            request,
            effects=(
                *request.effects,
                ("activity", "device-per-system"),
                ("inactive_publication", "preserve-bytes"),
            ),
            semantics=(
                *request.semantics,
                ("structured_region", slot),
                ("storage", "column-major-system-spin"),
                (
                    "weight_domain",
                    "nonnegative-occupied-counts"
                    if slot == 0
                    else "signed"
                    if slot == 1
                    else "positive-finite-eigenvalues"
                    if slot == 2
                    else "not-applicable",
                ),
            ),
        )
        candidates = []
        for provider, algorithm in (
            ("generated.cuda", "native"),
            ("cublas", "gemm"),
            ("cublas", "syrk"),
            ("cublas", "syr2k"),
            ("cublaslt", "matmul"),
            ("cutensor", "contraction"),
            ("cutlass-aot", "fused"),
        ):
            reason = None
            if algorithm == "syrk" and slot != 0:
                reason = "SYRK requires an unsigned Gram domain"
            elif algorithm == "syr2k":
                reason = "SYR2K requires a symmetric pair domain"
            elif provider not in ("generated.cuda", "cublas"):
                reason = "exact-shape plans do not implement this bounded domain"
            precision = request.precisions[0]
            candidates.append(
                LoweringCandidate(
                    request,
                    "structured-" + algorithm,
                    (
                        ProviderDescriptor(
                            provider,
                            "generated"
                            if provider in ("generated.cuda", "cutlass-aot")
                            else "library",
                            algorithm,
                            version="runtime-bound-fp64",
                        ),
                    ),
                    "unsupported" if reason else "ready",
                    precision.directive.math_mode,
                    reason=reason,
                    execution=CandidateExecution(
                        precision,
                        "scaled-panel-and-masked-publication",
                        request.operands,
                        ScheduleTopology(
                            materialization="borrowed-charged-scratch",
                            reduction="provider-reproducible",
                        ),
                        determinism="reproducible",
                        capture_safe=True,
                    ),
                    target=target,
                )
            )
        if request.scientific_identity is None:
            raise ValueError(
                "structured region requires a scientific equation identity"
            )
        portfolios.append(
            StructuredPortfolio(
                request,
                tuple(candidates),
                target,
                CompilationIdentity(
                    request.scientific_identity,
                    canonical_hash({"structured_schema": 1, "slot": slot}),
                ),
            )
        )
    return tuple(portfolios)


def emit_structured_scf_cuda() -> str:
    """Emit native/GEMM/SYRK/SYR2K offers, retaining unsupported evidence."""
    portfolios = structured_scf_portfolios()
    pieces = [
        "// Generated canonical structured matrix regions; do not edit.",
        "#pragma once",
        '#include "runtime/lowering_binding.hpp"',
        "namespace generativeqc::tensor::structured_lowering {",
        "struct Portfolio { const runtime::NativeLoweringRequest& request;",
        "const std::array<runtime::NativeLoweringCandidate, 7>& candidates;",
        "std::string_view target, compilation; };",
    ]
    for slot, portfolio in enumerate(portfolios):
        pieces.append(
            native_lowering_portfolio(
                portfolio.request,
                portfolio.candidates,
                portfolio.target,
                portfolio.compilation,
                name=f"structured_{slot}",
            )
        )
    pieces.append("inline Portfolio portfolio(std::size_t slot) { switch (slot) {")
    for slot in range(len(portfolios)):
        pieces.append(
            f"case {slot}: return {{structured_{slot}_request, structured_{slot}_candidates, structured_{slot}_target, structured_{slot}_compilation}};"
        )
    pieces.extend(
        [
            'default: throw std::invalid_argument("invalid structured region");',
            "} }",
            "}",
        ]
    )
    return "\n".join(pieces) + "\n"
