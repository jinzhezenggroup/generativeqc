"""Opt-in backend rewrite of dense one-sided reductions into BLAS contractions.

The scientific TensorIR keeps reduce as the source equation. This module only
rewrites an already prepared CUDA program when the caller explicitly asks for
the BLAS experiment. A rank-two reduction

    y[j] = sum_i x[i, j]

is represented as the exactly equivalent contraction

    y[j] = sum_i x[i, j] * 1[i]

so the existing binary-einsum/cuBLAS lowering can own execution. The inserted
unit vector is an exact compiler constant. This is an execution-order change,
not a new mathematical operation, and therefore remains opt-in until complete
numerical and endpoint performance qualification exists.
"""

from __future__ import annotations

from .ir import Node, constant
from .program import Program
from .types import TensorSpec

SCHEMA = "generativeqc.tensor.cuda.blas-reduction.v1"
MIN_REDUCTION_EXTENT = 32


def _eligible(node: Node) -> bool:
    if node.op != "reduce" or len(node.inputs) != 1:
        return False
    source = node.inputs[0]
    axes = tuple(node.attrs["axes"])
    if (
        source.spec.dtype not in ("float32", "float64")
        or len(source.spec.indices) != 2
        or len(axes) != 1
        or node.spec.symmetries
        or node.spec.size == 0
    ):
        return False
    axis = axes[0]
    return source.spec.indices[axis].extent >= MIN_REDUCTION_EXTENT


def _as_blas_einsum(node: Node, source: Node) -> Node:
    axis = tuple(node.attrs["axes"])[0]
    reduced = source.spec.indices[axis]
    unit = constant(
        (1,) * reduced.extent,
        TensorSpec(
            (reduced,),
            dtype=source.spec.dtype,
            representation=source.spec.representation,
            role="constant",
            differentiable=False,
        ),
    )
    labels = tuple(range(len(source.spec.indices)))
    output = tuple(label for label in labels if label != axis)
    return Node(
        "einsum",
        (source, unit),
        node.spec,
        (
            ("coefficient", (1, 1)),
            ("labels", (labels, (axis,))),
            ("output", output),
        ),
    )


def rewrite_dense_reductions_for_blas(program: Program) -> Program:
    """Rewrite eligible CUDA reductions through the existing cuBLAS GEMM path.

    The rewrite is deliberately narrow for the first experiment: one rank-two
    floating input, one reduced axis, no output symmetry, and at least one warp
    of reduction work. Programs with explicit precision-execution/request
    bindings are rejected until those node-name bindings are remapped through
    this backend-only rewrite.

    precision_source_equation retains the pre-rewrite scientific identity; the
    rewritten logical hash remains the concrete execution equation.
    """

    if not isinstance(program, Program):
        raise TypeError("BLAS reduction rewrite requires a TensorIR Program")
    provenance = program.provenance
    if any(
        key in provenance
        for key in (
            "precision_execution",
            "precision_request",
            "precision_request_identity",
        )
    ):
        raise ValueError(
            "BLAS reduction rewrite does not yet remap explicit precision bindings"
        )

    replacements: dict[Node, Node] = {}
    rewritten = 0
    for node in program.nodes:
        inputs = tuple(replacements[child] for child in node.inputs)
        updated = (
            node
            if inputs == node.inputs
            else Node(node.op, inputs, node.spec, node.attributes)
        )
        if _eligible(updated):
            updated = _as_blas_einsum(updated, inputs[0])
            rewritten += 1
        replacements[node] = updated

    if not rewritten:
        return program

    outputs = {name: replacements[node] for name, node in program.outputs.items()}
    definitions = tuple(replacements[node] for node in program.definitions)
    provenance.setdefault("precision_source_equation", program.logical_hash)
    provenance["cuda_blas_reduction"] = {
        "schema": SCHEMA,
        "source_equation": program.logical_hash,
        "rewritten_nodes": rewritten,
        "minimum_reduction_extent": MIN_REDUCTION_EXTENT,
        "execution": "binary-einsum-with-exact-unit-vector",
    }
    return Program(outputs, definitions, provenance)
