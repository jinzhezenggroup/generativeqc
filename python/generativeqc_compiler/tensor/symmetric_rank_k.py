"""Project a symmetric weighted rank-k TensorIR contraction to CUDA providers.

The existing ternary einsum is the scientific owner.  This module only records
one physical batch slice and offers implementations of that same operation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from generativeqc_compiler.common.backend import TargetInfo
from generativeqc_compiler.common.lowering_contract import (
    CandidateExecution,
    LoweringConstraints,
    OperandLayout,
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

from .ir import Node, add, input_tensor, multiply
from .lowering import TensorLoweringAdapter
from .program import Program
from .types import TensorSpec, checked_shape

MatrixOrder = Literal["row-major", "column-major"]


def _domains(indices: tuple) -> tuple:
    """Compare scientific index domains without treating notation as identity."""
    return tuple(index.domain for index in indices)


def symmetric_rank_k_update_program() -> Program:
    """Own the runtime ``alpha * product + beta * old_output`` scalar update."""

    def scalar(name: str) -> Node:
        return input_tensor(name, TensorSpec((), dtype="float64", role="input"))

    alpha = scalar("alpha")
    product = scalar("product")
    beta = scalar("beta")
    old_output = scalar("old_output")
    return Program(
        {"updated": add(multiply(alpha, product), multiply(beta, old_output))},
        provenance={"source": "tensor symmetric rank-k scalar update"},
    )


def symmetric_rank_k_request(
    program: Program,
    output: str,
    *,
    order: MatrixOrder = "row-major",
) -> LoweringRequest:
    """Recognize ``C[...,p,i] * w[...,i] * C[...,q,i]`` with signed weights.

    The two coefficient legs must be the *same* original node, and the AO axes
    must have the same index domain.  Physical strides describe one batch slice;
    the surrounding owner traverses all batch indices without changing science.
    """
    if order not in ("row-major", "column-major"):
        raise ValueError("rank-k matrix order must be row-major or column-major")
    node = program.outputs[output]
    if node.op != "einsum" or node.attrs.get("coefficient") != (1, 1):
        raise ValueError("rank-k requires an unscaled ternary einsum")
    if len(node.inputs) != 3:
        raise ValueError("rank-k requires coefficient, weight, coefficient")
    left, weights, right = node.inputs
    if left is not right or left.op != "input":
        raise ValueError("rank-k coefficient legs must share one input node")
    labels = node.attrs["labels"]
    result = node.attrs["output"]
    if (
        len(labels) != 3
        or len(labels[0]) < 2
        or labels[0][:-2] != labels[1][:-1]
        or labels[0][:-2] != labels[2][:-2]
        or labels[0][:-2] != result[:-2]
        or labels[0][-1] != labels[1][-1]
        or labels[0][-1] != labels[2][-1]
        or labels[0][-2] != result[-2]
        or labels[2][-2] != result[-1]
        or labels[0][-2] == labels[2][-2]
        or len(set(labels[0])) != len(labels[0])
        or len(set(labels[1])) != len(labels[1])
        or len(set(labels[2])) != len(labels[2])
        or len(set(result)) != len(result)
    ):
        raise ValueError("rank-k einsum labels do not describe a symmetric Gram")
    if (
        _domains(left.spec.indices[:-2]) != _domains(weights.spec.indices[:-1])
        or _domains(left.spec.indices[:-2]) != _domains(node.spec.indices[:-2])
        or left.spec.indices[-1].domain != weights.spec.indices[-1].domain
        or left.spec.indices[-2].domain != node.spec.indices[-2].domain
        or left.spec.indices[-2].domain != node.spec.indices[-1].domain
        or left.spec.shape[-2] != node.spec.shape[-1]
    ):
        raise ValueError("rank-k input and output scientific domains differ")
    if any(
        index.selection is not None
        for value in (left, weights, node)
        for index in value.spec.indices
    ):
        raise ValueError("rank-k gathered axes require an explicit packing contract")
    if weights.op != "input" and not (
        weights.op == "multiply"
        and len(weights.inputs) == 2
        and all(
            term.op == "input"
            and _domains(term.spec.indices) == _domains(weights.spec.indices)
            and term.spec.dtype == weights.spec.dtype
            for term in weights.inputs
        )
    ):
        raise ValueError("rank-k weight must be an input or original input product")
    if any(
        value.spec.dtype != "float64"
        for value in (left, weights, node, *weights.inputs)
    ):
        raise ValueError("rank-k currently requires strict FP64 storage")
    adapter = TensorLoweringAdapter(program)
    base = adapter.request(node, backend="cuda")
    arithmetic = adapter.directives[node]
    if arithmetic.storage_dtype != "float64" or any(
        adapter.directives[value] != arithmetic
        for value in (left, weights, *weights.inputs)
    ):
        raise ValueError(
            "rank-k weight and contraction require one strict FP64 schedule"
        )
    if any(
        precision.directive.compute_dtype != "float64"
        or precision.directive.accumulation_dtype != "float64"
        or precision.casts
        or precision.refinement
        or precision.audit
        for precision in base.precisions
    ):
        raise ValueError("rank-k cannot silently change arithmetic or audit")
    update = symmetric_rank_k_update_program()
    update_inputs = {
        value.attrs["name"]: value for value in update.live_nodes if value.op == "input"
    }
    update_root = update.outputs["updated"]
    update_products = [value for value in update_root.inputs if value.op == "multiply"]
    if (
        set(update_inputs) != {"alpha", "product", "beta", "old_output"}
        or any(
            value.spec.shape or value.spec.dtype != "float64"
            for value in update.live_nodes
        )
        or update_root.op != "add"
        or update_root.attrs["coefficients"] != ((1, 1), (1, 1))
        or len(update_products) != 2
        or {frozenset(value.inputs) for value in update_products}
        != {
            frozenset((update_inputs["alpha"], update_inputs["product"])),
            frozenset((update_inputs["beta"], update_inputs["old_output"])),
        }
        or len(update.live_nodes) != 7
    ):
        raise ValueError("rank-k scalar update TensorIR changed")
    n, k = left.spec.shape[-2:]
    checked_shape((n, k, n), 8)
    panel_strides = (k, 1) if order == "row-major" else (1, n)
    matrix_strides = (n, 1) if order == "row-major" else (1, n)
    physical = (
        OperandLayout(
            "input:0", (0, 2), (n, k), panel_strides, alias_group="coefficient"
        ),
        OperandLayout("input:1", (2,), (k,), (1,)),
        OperandLayout(
            "input:2", (1, 2), (n, k), panel_strides, alias_group="coefficient"
        ),
        OperandLayout(
            "output",
            (0, 1),
            (n, n),
            matrix_strides,
            access="read-write",
            triangle="upper",
        ),
    )
    semantics = dict(base.semantics)
    semantics.update(
        parent_node_hash=adapter.hashes[node],
        symmetric_rank_k=True,
        transpose="coefficient-times-weighted-coefficient-transpose",
        signed_weights=True,
        weights_materialization="preceding-multiply"
        if weights.op == "multiply"
        else "borrowed",
        publication="upper-triangle-mirrored",
        scalar_input_roles="alpha,product,beta,old_output",
        scalar_update_hash=update.logical_hash,
        update="alpha-product-plus-beta-output",
    )
    return replace(
        base,
        semantics=tuple(semantics.items()),
        operands=physical,
        input_dtypes=(*base.input_dtypes, "float64"),
        precisions=tuple(
            replace(precision, input_dtypes=(*precision.input_dtypes, "float64"))
            for precision in base.precisions
        ),
        effects=(("output", "transactional-symmetric-overwrite-or-accumulate"),),
        constraints=LoweringConstraints(
            determinism="reproducible", capture_required=True
        ),
    )


def emit_symmetric_rank_k_portfolio(
    program: Program,
    output: str,
    source: str,
    *,
    name: str,
    order: MatrixOrder = "row-major",
) -> str:
    """Emit generated and signed-GEMM candidates for one rank-k request.

    Native preparation resolves exact resource bytes and endpoint eligibility.
    Unknown timing cannot promote the optional library provider by itself.
    """
    request = symmetric_rank_k_request(program, output, order=order)
    if request.scientific_identity is None:
        raise ValueError("rank-k requires an original scientific identity")
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
                    materialization=materialization,
                    reduction="provider-reproducible",
                ),
                determinism="reproducible",
                capture_safe=True,
            ),
            target=target,
        )
        for provider, kind, algorithm, version, materialization in (
            (
                "generated.cuda",
                "generated",
                "symmetric-rank-k-generated",
                source,
                "borrowed-panels/two-pass-transactional-publication",
            ),
            (
                "cublas",
                "library",
                "symmetric-rank-k-signed-gemm",
                "runtime-bound-pedantic",
                "signed-column-scale/gram-scratch/mirror",
            ),
        )
    )
    return native_lowering_portfolio(
        request,
        candidates,
        target,
        CompilationIdentity(request.scientific_identity, source),
        name=name,
    )
