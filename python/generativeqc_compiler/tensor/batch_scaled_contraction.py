"""Recognize a batch-weighted contraction region in the existing TensorIR."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node
    from .lowering import TensorLoweringAdapter


def batch_scaled_contraction_request(
    adapter: TensorLoweringAdapter, root: Node, *, backend: str
) -> LoweringRequest:
    """Bind `weights[batch] * einsum(left,right)` as one complete region.

    The original two-node SSA graph remains the scientific authority. Recognition
    requires a leading shared batch axis, homogeneous arithmetic and an exclusive
    intermediate. Every candidate publishes the weighted output, including its
    finite-value policy; a split implementation cannot expose an unweighted result.
    """
    if root.op != "einsum" or len(root.inputs) != 2:
        raise ValueError("batch scaling requires a binary outer einsum")
    products = [value for value in root.inputs if value.op == "einsum"]
    weights = [value for value in root.inputs if value.op == "input"]
    if len(products) != 1 or len(weights) != 1 or len(products[0].inputs) != 2:
        raise ValueError("batch scaling requires one contraction and one input weight")
    product, weight = products[0], weights[0]
    if any(value.op != "input" for value in product.inputs):
        raise ValueError("batch contraction factors must be external input views")
    if product in adapter.program.outputs.values() or any(
        product in value.inputs for value in adapter.live_nodes if value is not root
    ):
        raise ValueError("batch scaling requires an exclusive contraction intermediate")
    weight_position = root.inputs.index(weight)
    product_position = root.inputs.index(product)
    outer_weight_modes = root.attrs["labels"][weight_position]
    outer_product_modes = root.attrs["labels"][product_position]
    if (
        len(weight.spec.shape) != 1
        or len(outer_weight_modes) != 1
        or not outer_product_modes
        or outer_weight_modes != outer_product_modes[:1]
        or root.attrs["output"] != outer_product_modes
        or root.spec != product.spec
        or weight.spec.dtype != product.spec.dtype
        or root.attrs["coefficient"] != (1, 1)
        or product.attrs["coefficient"] != (1, 1)
    ):
        raise ValueError(
            "batch scaling cannot reduce, reorder, broadcast or cast the product"
        )
    batch_mode = product.attrs["output"][0]
    if any(labels[:1] != (batch_mode,) for labels in product.attrs["labels"]):
        raise ValueError("weight axis must be the leading shared contraction batch")
    if adapter.directives[root] != adapter.directives[product]:
        raise ValueError("batch scaling cannot change intermediate arithmetic")
    request = adapter.request(root, backend=backend)
    contraction = adapter.request(product, backend=backend)
    dtypes = (*contraction.input_dtypes, weight.spec.dtype)
    semantics = dict(request.semantics)
    semantics.update(
        {
            "contraction_node_hash": adapter.hashes[product],
            "batch_scale_mode": batch_mode,
            "reduction_extent": dict(contraction.semantics)["reduction_extent"],
        }
    )
    return replace(
        request,
        input_dtypes=dtypes,
        precisions=tuple(
            replace(value, input_dtypes=dtypes) for value in request.precisions
        ),
        operands=(
            *contraction.operands[:-1],
            replace(
                request.operands[weight_position],
                operand="input:2",
                modes=(batch_mode,),
            ),
            contraction.operands[-1],
        ),
        semantics=tuple(semantics.items()),
        effects=(*request.effects, ("nonfinite_publication", "sticky-error-and-zero")),
    )
