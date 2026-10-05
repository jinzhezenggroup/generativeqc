"""Project the existing LibraryRequest vocabulary into shared solver lowering.

The ordinary native owner binds runtime dimensions separately from this AOT
template. No generalized eigenproblem, method policy or new algebra lives here.
Only the existing independently qualified FP64 symmetric solver is admitted;
runtime residual-producing requests need a different executable implementation.
"""

from __future__ import annotations

from .backend import TargetInfo
from .library import LibraryRequest
from .lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
    LoweringPrecision,
    OperandLayout,
)
from .lowering_provider import LoweringCandidate, LoweringRequest, ProviderDescriptor
from .native_lowering import native_lowering_portfolio
from .precision import ExecutionPrecisionSchedule, PrecisionDirective
from .provenance import canonical_hash
from .specialization import CompilationIdentity, TargetCapabilities


def symmetric_eigh_request(
    operation: LibraryRequest, *, backend: str
) -> LoweringRequest:
    """Adapt an in-place, full-symmetric input with column-major eigenvectors.

    Both triangles must describe the same matrix: the small native algorithm
    and library algorithm need not read the same triangle. Eigenvalues are sorted
    ascending. Status and active masks are control effects, not floating operands.
    """
    if (
        operation.operation != "symmetric_eigh"
        or operation.layout != "column_major"
        or operation.residual_policy != "qualification"
    ):
        raise ValueError(
            "ordinary solver requires qualified column-major symmetric_eigh"
        )
    n = operation.shape[0]
    science = canonical_hash(
        {
            "operation": operation.operation,
            "shape": operation.shape,
            "dtype": operation.dtype,
            "residual_policy": operation.residual_policy,
            "residual_tolerance": operation.residual_tolerance,
            "vectors": "columns",
            "spectrum": "ascending",
        }
    )
    precision = LoweringPrecision(
        ExecutionPrecisionSchedule(
            (("operation", PrecisionDirective("float64", "float64", "float64")),)
        ),
        "operation",
        ("float64",),
        "float64",
    )
    return LoweringRequest(
        "solver.symmetric_eigh",
        operation.operation,
        backend,
        "float64",
        "float64",
        (n,),
        scientific_identity=science,
        operands=(
            OperandLayout("matrix_vectors", (0, 1), (n, n), (1, n), "read-write"),
            OperandLayout("eigenvalues", (1,), (n,), (1,), "write"),
        ),
        precisions=(precision,),
        constraints=LoweringConstraints(
            workspace_bytes=operation.workspace_limit_bytes, maximum_candidates=2
        ),
        semantics=(("symmetric-input", "full"), ("eigenvalues", "ascending")),
        effects=(
            ("matrix", "overwritten-by-column-eigenvectors"),
            ("active", "inactive-results-unpublished"),
            ("completion", "stream-ordered-with-device-info"),
            ("residual", "independent-qualification-only"),
            ("batch", "runtime-serialized-workspace-reuse"),
            ("caller-scratch", "disjoint-batch*n*n-fp64"),
            ("buffers", "borrowed-disjoint-until-stream-completion"),
        ),
    )


def ordinary_eigh_portfolio(
    source_identity: str,
) -> tuple[
    LoweringRequest,
    tuple[LoweringCandidate, ...],
    TargetCapabilities,
    CompilationIdentity,
]:
    """Emit two qualified algorithm templates with unknown performance costs.

    Preparation resolves the disjoint size domains and queried resources; an
    unknown cost retains the incumbent and cannot promote a new implementation.
    Provider version placeholders must be resolved before native selection.
    """
    request = symmetric_eigh_request(
        LibraryRequest(
            "symmetric_eigh",
            (17,),
            layout="column_major",
            residual_tolerance=2e-11,
            residual_policy="qualification",
        ),
        backend="cuda",
    )
    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    assert request.scientific_identity is not None
    compilation = CompilationIdentity(request.scientific_identity, source_identity)
    candidates = tuple(
        LoweringCandidate(
            request,
            algorithm,
            (ProviderDescriptor(provider, kind, algorithm, version=version),),
            "ready",
            "strict",
            execution=CandidateExecution(
                request.precisions[0],
                algorithm,
                request.operands,
                capture_safe=capture,
            ),
            target=target,
        )
        for provider, kind, algorithm, version, capture in (
            (
                "native.cuda",
                "runtime",
                "small-symmetric-eigh",
                source_identity,
                True,
            ),
            ("cusolver", "library", "xsyevd", "runtime-query-required", False),
        )
    )
    return request, candidates, target, compilation


def ordinary_eigh_header(source_identity: str) -> str:
    """Generate pure metadata; native preparation owns queries and allocation."""
    return (
        '#pragma once\n#include "runtime/lowering_binding.hpp"\n'
        "namespace generativeqc::scf::cuda_execution {\n"
        + native_lowering_portfolio(
            *ordinary_eigh_portfolio(source_identity), name="ordinary_eigh"
        )
        + "}\n"
    )
