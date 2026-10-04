"""Project canonical TensorIR lowering requests into native AOT descriptors.

Runtime extents are supplied by the existing native emitter; this adapter owns
no orbital/method policy and does not construct an alternative scientific IR.
Representative request hashes identify the AOT template, while native binding
compatibility additionally includes every resolved operand extent and stride.
"""

from __future__ import annotations

import json
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from .ir import Node
    from .lowering import TensorLoweringAdapter
    from .types import Index


def affine_contraction_initializer(
    adapter: TensorLoweringAdapter,
    node: Node,
    dimension: Callable[[Index], str],
    *,
    coefficient: str,
    beta: str = "0.0",
) -> str:
    """Project the original binary axes without requiring a GEMM factorization.

    Zero matrix dimensions deliberately prevent accidental matrix execution.
    General providers validate the same descriptor's affine semantic fields;
    the matrix provider additionally requires its existing physical recipe.
    """
    return contraction_initializer(
        adapter,
        node,
        dimension,
        transpose=("N", "N"),
        extents=("1", "0", "0", "0"),
        coefficient=coefficient,
        beta=beta,
    )


def contraction_initializer(
    adapter: TensorLoweringAdapter,
    node: Node,
    dimension: Callable[[Index], str],
    *,
    transpose: tuple[str, str],
    extents: tuple[str, str, str, str],
    coefficient: str,
    row_axes: tuple[int, int, int] | None = None,
    leading_dimensions: tuple[str, str, str] | None = None,
    beta: str = "0.0",
) -> str:
    """Emit a typed descriptor for an already recognized binary matrix einsum.

    Optional row cuts/strides describe unbatched affine views; beta can also
    update a dense output. Native validation checks the physical recipe against
    the original semantic modes before execution.

    Matrix recognition belongs to the existing physical lowerer. The descriptor
    retains original mode labels and operand shapes, so provider execution can
    validate its matrix recipe against that same semantic request. No provider
    or execution precision choice is made by a scientific/method owner here.
    """
    if node.op != "einsum" or len(node.inputs) != 2:
        raise ValueError("native contraction projection requires binary einsum")
    request = adapter.request(node, backend="cuda")
    precision = request.precisions[0]
    directive = precision.directive
    if (
        len(request.precisions) != 1
        or precision.casts
        or precision.refinement
        or precision.audit
        or len({*precision.input_dtypes, precision.publication_dtype}) != 1
        or directive.storage_dtype != directive.compute_dtype
        or directive.compute_dtype != directive.accumulation_dtype
    ):
        raise ValueError(
            "native dense candidate does not implement requested precision"
        )

    def dtype(name: str) -> str:
        return (
            "generativeqc::runtime::PrecisionDtype::"
            + {
                "float64": "Fp64",
                "float32": "Fp32",
            }[name]
        )

    operands = []
    if (row_axes is None) != (leading_dimensions is None):
        raise ValueError("matrix view cuts and strides must be supplied together")
    for i, (value, layout) in enumerate(
        zip((*node.inputs, node), request.operands, strict=True)
    ):
        modes = ",".join(map(str, layout.modes))
        shape = ",".join(dimension(index) for index in value.spec.indices)
        view = "matrix_view" if row_axes is not None else "dense"
        extra = ""
        if row_axes is not None:
            assert leading_dimensions is not None
            extra = f",{row_axes[i]},{leading_dimensions[i]}"
        operands.append(
            f"generativeqc::tensor::ContractionOperand::{view}("
            f"{{{modes}}},{{{shape}}},{dtype(value.spec.dtype)}{extra})"
        )
    arithmetic = ",".join(
        (
            dtype(directive.storage_dtype),
            dtype(directive.compute_dtype),
            dtype(directive.accumulation_dtype),
            json.dumps(directive.qualification or ""),
            json.dumps(directive.math_mode),
        )
    )
    return (
        "generativeqc::tensor::ContractionRequest{"
        f'"{request.scientific_identity}","{request.semantic_identity}",'
        f'"{precision.identity}",'
        "{" + ",".join(operands) + "},"
        "{"
        + arithmetic
        + "},"
        + dtype(precision.publication_dtype)
        + f",'{transpose[0]}','{transpose[1]}',"
        + ",".join((*extents, coefficient))
        + (
            ",{" + ",".join(leading_dimensions or ()) + "}," + beta
            if leading_dimensions is not None or beta != "0.0"
            else ""
        )
        + "}"
    )
