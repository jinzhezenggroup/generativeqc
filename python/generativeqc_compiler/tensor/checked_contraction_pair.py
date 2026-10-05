"""Bind a coupled forward/transpose pair to its original scalar program."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .checked_contraction import _bind_scalar_checks

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node
    from .lowering import TensorLoweringAdapter
    from .program import Program


@dataclass(frozen=True)
class CheckedTransposePair:
    """Recognition metadata referencing existing graphs, never new equations.

    Scalar roles are the two accumulators, forward/transpose matrix elements,
    and two vector elements. Output roles follow the same accumulator order.
    """

    first: Node
    second: Node
    scalar_inputs: tuple[str, str, str, str, str, str]
    scalar_outputs: tuple[str, str]

    def role(self, node: Node) -> int:
        if node is self.first:
            return 1
        if node is self.second:
            return 2
        raise ValueError("checked pair requires its original contraction node")


def checked_transpose_pair_request(
    adapter: TensorLoweringAdapter,
    pair: CheckedTransposePair,
    update: Program,
    *,
    role: int,
    backend: str,
) -> LoweringRequest:
    """Recognize two square batched matvecs with joint checked publication.

    The original matrix is read in both directions and never copied. Recognition
    binds both products and both scalar updates, so separate execution cannot
    expose a successful output after the other update fails.
    """
    first, second = pair.first, pair.second
    if (
        role not in (1, 2)
        or len(adapter.program.outputs) != 2
        or set(adapter.program.outputs.values()) != {first, second}
    ):
        raise ValueError("checked pair requires its complete two-output region")
    for node in (first, second):
        if (
            node.op != "einsum"
            or len(node.inputs) != 2
            or node.attrs["coefficient"] != (1, 1)
            or any(value.op != "input" for value in node.inputs)
        ):
            raise ValueError("checked pair requires two unit external contractions")
    matrix, vector = first.inputs
    matrix_other, vector_other = second.inputs
    if (
        matrix is not matrix_other
        or len(matrix.spec.shape) != 3
        or matrix.spec.shape[1] != matrix.spec.shape[2]
        or vector.spec.shape != matrix.spec.shape[:2]
        or vector_other.spec.shape != vector.spec.shape
        or first.attrs["labels"] != ((0, 1, 2), (0, 2))
        or first.attrs["output"] != (0, 1)
        or second.attrs["labels"] != ((0, 1, 2), (0, 1))
        or second.attrs["output"] != (0, 2)
        or adapter.directives[first] != adapter.directives[second]
        or any(
            value.spec.dtype != "float64"
            for value in (matrix, vector, vector_other, first, second)
        )
    ):
        raise ValueError("checked pair requires its square FP64 transpose views")
    inputs = {
        node.attrs["name"]: node for node in update.live_nodes if node.op == "input"
    }
    if (
        len(set(pair.scalar_inputs)) != 6
        or set(inputs) != set(pair.scalar_inputs)
        or len(set(pair.scalar_outputs)) != 2
        or set(update.outputs) != set(pair.scalar_outputs)
        or len(update.live_nodes) != 10
        or any(
            node.spec.shape or node.spec.dtype != "float64"
            for node in update.live_nodes
        )
    ):
        raise ValueError("checked pair requires its complete scalar FP64 program")
    a, b, matrix_rc, matrix_cr, vector_rc, vector_cr = (
        inputs[name] for name in pair.scalar_inputs
    )
    for name, accumulator, factors in (
        (pair.scalar_outputs[0], a, {matrix_rc, vector_rc}),
        (pair.scalar_outputs[1], b, {matrix_cr, vector_cr}),
    ):
        root = update.outputs[name]
        products = [node for node in root.inputs if node.op == "multiply"]
        if (
            root.op != "add"
            or root.attrs["coefficients"] != ((1, 1), (1, 1))
            or accumulator not in root.inputs
            or len(products) != 1
            or set(products[0].inputs) != factors
        ):
            raise ValueError("checked pair cannot change either scalar update graph")
    node = first if role == 1 else second
    request = adapter.request(node, backend=backend)
    if (
        len(request.precisions) != 1
        or not request.precisions[0].schedule.is_strict_fp64
        or request.input_dtypes != ("float64", "float64")
        or request.accumulation_dtype != "float64"
    ):
        raise ValueError("checked pair requires its complete strict FP64 region")
    semantics = dict(request.semantics)
    semantics.update(
        checked_transpose_pair_role=role,
        checked_pair_first_hash=adapter.hashes[first],
        checked_pair_second_hash=adapter.hashes[second],
        checked_pair_scalar_inputs=",".join(pair.scalar_inputs),
        checked_pair_scalar_outputs=",".join(pair.scalar_outputs),
    )
    request = _bind_scalar_checks(
        replace(request, semantics=tuple(semantics.items())), update, ""
    )
    return replace(
        request,
        effects=(*request.effects, ("paired_publication", "zero-both-on-failure")),
    )
