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

from .ir import Node, add, broadcast, input_tensor, multiply
from .lowering import TensorLoweringAdapter
from .program import Program
from .types import TensorSpec, checked_shape

MatrixOrder = Literal["row-major", "column-major"]
UpdateMode = Literal["overwrite", "update"]


def _domains(indices: tuple) -> tuple:
    """Compare scientific index domains without treating notation as identity."""
    return tuple(index.domain for index in indices)


def _rank_k_overwrite(alpha: Node, product: Node) -> Node:
    """The beta-zero TensorIR branch, which has no old-output input."""
    return multiply(alpha, product)


def _rank_k_update(alpha: Node, product: Node, beta: Node, old_output: Node) -> Node:
    """The nonzero-beta formula shared by the dense root and scalar helper."""
    return add(_rank_k_overwrite(alpha, product), multiply(beta, old_output))


def symmetric_rank_k_scalar_overwrite_program() -> Program:
    """Scalarize the checked beta-zero overwrite branch."""

    def scalar(name: str) -> Node:
        return input_tensor(name, TensorSpec((), dtype="float64", role="input"))

    return Program(
        {"updated": _rank_k_overwrite(scalar("alpha"), scalar("product"))},
        provenance={"source": "tensor symmetric rank-k scalar overwrite"},
    )


def symmetric_rank_k_scalar_update_program() -> Program:
    """Scalarize the checked runtime update without inventing another formula."""

    def scalar(name: str) -> Node:
        return input_tensor(name, TensorSpec((), dtype="float64", role="input"))

    alpha = scalar("alpha")
    product = scalar("product")
    beta = scalar("beta")
    old_output = scalar("old_output")
    return Program(
        {"updated": _rank_k_update(alpha, product, beta, old_output)},
        provenance={"source": "tensor symmetric rank-k scalar update"},
    )


def symmetric_rank_k_overwrite_program(program: Program, output: str) -> Program:
    """Compose the original Gram with its explicit alpha input."""
    product = program.outputs[output]
    alpha = input_tensor("rank_k_alpha", TensorSpec((), dtype="float64", role="input"))
    return Program(
        {
            "updated": _rank_k_overwrite(
                broadcast(alpha, product.spec.indices, ()), product
            )
        },
        provenance={
            "source": "tensor symmetric rank-k dense overwrite composition",
            "rank_k_product_program": program.logical_hash,
            "rank_k_product_output": output,
        },
    )


def symmetric_rank_k_update_program(program: Program, output: str) -> Program:
    """Compose the original Gram with explicit scalar and old-output inputs."""
    product = program.outputs[output]
    alpha = input_tensor("rank_k_alpha", TensorSpec((), dtype="float64", role="input"))
    beta = input_tensor("rank_k_beta", TensorSpec((), dtype="float64", role="input"))
    old_output = input_tensor("rank_k_old_output", replace(product.spec, role="input"))
    updated = _rank_k_update(
        broadcast(alpha, product.spec.indices, ()),
        product,
        broadcast(beta, product.spec.indices, ()),
        old_output,
    )
    return Program(
        {"updated": updated},
        provenance={
            "source": "tensor symmetric rank-k dense update composition",
            "rank_k_product_program": program.logical_hash,
            "rank_k_product_output": output,
        },
    )


def symmetric_rank_k_request(
    program: Program,
    output: str,
    *,
    order: MatrixOrder = "row-major",
    update: UpdateMode = "update",
) -> LoweringRequest:
    """Recognize ``C[...,p,i] * w[...,i] * C[...,q,i]`` with signed weights.

    The two coefficient legs must be the *same* original node, and the AO axes
    must have the same index domain.  Physical strides describe one batch slice;
    the surrounding owner traverses all batch indices without changing science.
    """
    if order not in ("row-major", "column-major"):
        raise ValueError("rank-k matrix order must be row-major or column-major")
    if update not in ("overwrite", "update"):
        raise ValueError("rank-k update must be overwrite or update")
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
    source_adapter = TensorLoweringAdapter(program)
    source_request = source_adapter.request(node, backend="cuda")
    source_arithmetic = source_adapter.directives[node]
    if source_arithmetic.storage_dtype != "float64" or any(
        source_adapter.directives[value] != source_arithmetic
        for value in (left, weights, *weights.inputs)
    ):
        raise ValueError(
            "rank-k weight and contraction require one strict FP64 schedule"
        )
    if (
        len(source_request.precisions) != 1
        or source_adapter.precision.strict_audit_dtype != "float64"
        or any(
            precision.directive.compute_dtype != "float64"
            or precision.directive.accumulation_dtype != "float64"
            or precision.casts
            or precision.refinement
            or precision.audit
            for precision in source_request.precisions
        )
    ):
        raise ValueError("rank-k cannot silently change arithmetic or audit")
    composition = (
        symmetric_rank_k_overwrite_program(program, output)
        if update == "overwrite"
        else symmetric_rank_k_update_program(program, output)
    )
    update_root = composition.outputs["updated"]
    adapter = TensorLoweringAdapter(composition)
    base = adapter.request(update_root, backend="cuda")
    arithmetic = adapter.directives[update_root]
    if arithmetic.storage_dtype != "float64" or any(
        adapter.directives[value] != arithmetic for value in composition.live_nodes
    ):
        raise ValueError(
            "rank-k weight and contraction require one strict FP64 schedule"
        )
    if len(base.precisions) != 1 or any(
        precision.directive.compute_dtype != "float64"
        or precision.directive.accumulation_dtype != "float64"
        or precision.casts
        or precision.refinement
        or precision.audit
        for precision in base.precisions
    ):
        raise ValueError("rank-k cannot silently change arithmetic or audit")
    named_inputs = {
        value.attrs["name"]: value
        for value in composition.live_nodes
        if value.op == "input"
    }
    alpha = named_inputs.get("rank_k_alpha")
    beta = named_inputs.get("rank_k_beta")
    old_output = named_inputs.get("rank_k_old_output")
    update_products = [value for value in update_root.inputs if value.op == "multiply"]
    alpha_views = [
        value
        for value in composition.live_nodes
        if value.op == "broadcast" and value.inputs == (alpha,)
    ]
    beta_views = [
        value
        for value in composition.live_nodes
        if value.op == "broadcast" and value.inputs == (beta,)
    ]
    common_invalid = (
        alpha is None
        or any(value.spec.dtype != "float64" for value in composition.live_nodes)
        or len(alpha_views) != 1
        or alpha.spec.shape
        or alpha_views[0].attrs["axes"]
        or base.scientific_identity != composition.logical_hash
    )
    if update == "overwrite":
        invalid_composition = (
            common_invalid
            or beta is not None
            or old_output is not None
            or update_root.op != "multiply"
            or frozenset(update_root.inputs) != frozenset((alpha_views[0], node))
        )
        scalar_update = symmetric_rank_k_scalar_overwrite_program()
    else:
        invalid_composition = (
            common_invalid
            or beta is None
            or old_output is None
            or update_root.op != "add"
            or update_root.attrs["coefficients"] != ((1, 1), (1, 1))
            or len(update_products) != 2
            or len(beta_views) != 1
            or {frozenset(value.inputs) for value in update_products}
            != {
                frozenset((alpha_views[0], node)),
                frozenset((beta_views[0], old_output)),
            }
            or beta.spec.shape
            or _domains(old_output.spec.indices) != _domains(node.spec.indices)
            or beta_views[0].attrs["axes"]
        )
        scalar_update = symmetric_rank_k_scalar_update_program()
    if invalid_composition:
        raise ValueError("rank-k dense update TensorIR changed")
    scalar_inputs = {
        value.attrs["name"]: value
        for value in scalar_update.live_nodes
        if value.op == "input"
    }
    scalar_root = scalar_update.outputs["updated"]
    scalar_products = [value for value in scalar_root.inputs if value.op == "multiply"]
    expected_scalar_inputs = (
        {"alpha", "product"}
        if update == "overwrite"
        else {"alpha", "product", "beta", "old_output"}
    )
    actual_scalar_products = (
        {frozenset(scalar_root.inputs)}
        if update == "overwrite"
        else {frozenset(value.inputs) for value in scalar_products}
    )
    expected_scalar_products = (
        {frozenset((scalar_inputs["alpha"], scalar_inputs["product"]))}
        if update == "overwrite"
        else {
            frozenset((scalar_inputs["alpha"], scalar_inputs["product"])),
            frozenset((scalar_inputs["beta"], scalar_inputs["old_output"])),
        }
    )
    if (
        set(scalar_inputs) != expected_scalar_inputs
        or any(
            value.spec.shape or value.spec.dtype != "float64"
            for value in scalar_update.live_nodes
        )
        or (update == "overwrite" and scalar_root.op != "multiply")
        or (update == "update" and scalar_root.op != "add")
        or (
            update == "update" and scalar_root.attrs["coefficients"] != ((1, 1), (1, 1))
        )
        or actual_scalar_products != expected_scalar_products
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
            access="write" if update == "overwrite" else "read-write",
            triangle="upper",
        ),
    )
    semantics = dict(base.semantics)
    semantics.update(
        parent_node_hash=adapter.hashes[node],
        source_program_hash=program.logical_hash,
        source_request_identity=source_request.identity,
        source_scientific_identity=source_request.scientific_identity or "",
        source_precision_identity=source_request.precisions[0].identity,
        alpha_input_hash=adapter.hashes[alpha],
        beta_input_hash="" if beta is None else adapter.hashes[beta],
        old_output_hash="" if old_output is None else adapter.hashes[old_output],
        update_root_hash=adapter.hashes[update_root],
        update_program_hash=composition.logical_hash,
        update_mode=update,
        symmetric_rank_k=True,
        transpose="coefficient-times-weighted-coefficient-transpose",
        signed_weights=True,
        weights_materialization="preceding-multiply"
        if weights.op == "multiply"
        else "borrowed",
        publication="upper-triangle-mirrored",
        scalar_input_roles=(
            "alpha,product"
            if update == "overwrite"
            else "alpha,product,beta,old_output"
        ),
        scalar_update_hash=scalar_update.logical_hash,
        update="alpha-product-plus-beta-output",
    )
    return replace(
        base,
        semantics=tuple(semantics.items()),
        operands=physical,
        input_dtypes=("float64",) * (3 if update == "overwrite" else 4),
        precisions=tuple(
            replace(
                precision,
                input_dtypes=("float64",) * (3 if update == "overwrite" else 4),
            )
            for precision in base.precisions
        ),
        effects=(
            (
                "output",
                "transactional-symmetric-overwrite"
                if update == "overwrite"
                else "transactional-symmetric-overwrite-or-accumulate",
            ),
        ),
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
    update: UpdateMode = "update",
) -> str:
    """Emit generated and signed-GEMM candidates for one rank-k request.

    Native preparation resolves exact resource bytes and endpoint eligibility.
    Unknown timing cannot promote the optional library provider by itself.
    """
    request = symmetric_rank_k_request(program, output, order=order, update=update)
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
