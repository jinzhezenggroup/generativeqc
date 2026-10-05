"""Recognize a symmetric cross product in the existing TensorIR algebra.

This is a fused lowering view of ``X + transpose(X)``, not a new scientific
operation or a method-specific BLAS API. Input panels remain compiler-owned.
"""

from dataclasses import replace

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
)
from generativeqc_compiler.common.lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    ProviderDescriptor,
)
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.common.schedule import ScheduleTopology
from generativeqc_compiler.common.specialization import (
    CompilationIdentity,
    TargetCapabilities,
)

from .lowering import TensorLoweringAdapter
from .program import Program


def symmetric_product_request(
    program: Program, output: str, *, backend: str = "cuda"
) -> LoweringRequest:
    """Project packed ``...m,...n->mn`` plus its transpose, strictly in FP64.

    Triangle ownership belongs to the fused implementation. The complete output
    is still published, and both read operands are the original materialized
    panels. The parent add-node hash preserves the complete scientific identity.
    """
    node = program.outputs[output]
    if (
        node.op != "add"
        or len(node.inputs) != 2
        or tuple(node.attrs["coefficients"]) != ((1, 1), (1, 1))
    ):
        raise ValueError("symmetric product requires an unscaled binary sum")
    cross, reflected = node.inputs
    if (
        cross.op != "einsum"
        or reflected.op != "transpose"
        or reflected.inputs != (cross,)
        or reflected.attrs["axes"] != (1, 0)
    ):
        raise ValueError("symmetric product requires a contraction and its transpose")
    labels, result = cross.attrs["labels"], cross.attrs["output"]
    if (
        len(labels) != 2
        or len(result) != 2
        or labels[0][:-1] != labels[1][:-1]
        or labels[0][-1:] != result[:1]
        or labels[1][-1:] != result[1:]
        or not labels[0][:-1]
        or cross.attrs["coefficient"] != (1, 1)
        or any(len(set(row)) != len(row) for row in labels)
        or set(result) & set(labels[0][:-1])
        or cross.spec.shape[0] != cross.spec.shape[1]
    ):
        raise ValueError("symmetric product requires packed square cross-product axes")
    adapter = TensorLoweringAdapter(program)
    request = adapter.request(cross, backend=backend)
    if any(value.spec.dtype != "float64" for value in (*cross.inputs, cross, node)):
        raise ValueError("symmetric product currently requires strict FP64")
    if any(
        p.directive.compute_dtype != "float64"
        or p.directive.accumulation_dtype != "float64"
        for p in request.precisions
    ):
        raise ValueError("symmetric product cannot silently change admitted arithmetic")
    semantics = dict(request.semantics)
    semantics.update(
        parent_node_hash=adapter.hashes[node],
        symmetric_cross_product=True,
        publication="upper-triangle-mirrored",
        update="overwrite-or-accumulate",
    )
    return replace(
        request,
        semantics=tuple(semantics.items()),
        operands=(
            *request.operands[:-1],
            replace(request.operands[-1], access="read-write", triangle="upper"),
        ),
        # Accumulating tiles also read the old output. Name that storage type
        # explicitly instead of hiding beta/update reads from the contract.
        input_dtypes=(*request.input_dtypes, "float64"),
        precisions=tuple(
            replace(p, input_dtypes=(*p.input_dtypes, "float64"))
            for p in request.precisions
        ),
        effects=(("output", "overwrite-or-accumulate-symmetric"),),
        constraints=LoweringConstraints(
            determinism="reproducible", capture_required=True
        ),
    )


def emit_symmetric_product_portfolio(
    program: Program, output: str, source: str, *, name: str
) -> str:
    """Offer generated and library execution of the same fused request.

    Generated execution is the qualified incumbent. A library offer needs both
    an explicit resource allowance and endpoint evidence before promotion.
    Native qualification can exercise it without changing scientific APIs.
    """
    request = symmetric_product_request(program, output)
    target = TargetCapabilities(
        TargetInfo("cuda", "current-aot-module", 32, 1024, None)
    )
    precision = request.precisions[0]
    candidates = tuple(
        LoweringCandidate(
            request,
            algorithm,
            (ProviderDescriptor(provider, kind, algorithm, version=version),),
            "ready",
            precision.directive.math_mode,
            execution=CandidateExecution(
                precision,
                algorithm,
                request.operands,
                ScheduleTopology(
                    materialization="caller-owned-panels",
                    reduction="provider-reproducible",
                ),
                determinism="reproducible",
                capture_safe=True,
            ),
            target=target,
        )
        for provider, kind, algorithm, version in (
            ("generated.cuda", "generated", "symmetric-cross-generated", source),
            ("cublas", "library", "symmetric-cross-rank2k", "runtime-bound-pedantic"),
        )
    )
    return native_lowering_portfolio(
        request,
        candidates,
        target,
        CompilationIdentity(request.scientific_identity, source),
        name=name,
    )
