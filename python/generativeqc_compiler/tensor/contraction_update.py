"""Canonical boundary for an existing TensorIR einsum-plus-seed region.

The mathematical graph stays immutable SSA. A prepared implementation may donate
its explicit seed allocation to the result after the seed's last use; this
execution precondition is shared by every candidate, not chosen by a provider.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node
    from .lowering import TensorLoweringAdapter


def contraction_update_request(
    adapter: TensorLoweringAdapter, root: Node, *, backend: str
) -> LoweringRequest:
    """Project `seed + einsum(A,B)` with an explicit donated seed/result pair.

    Recognition is intentionally bounded: one binary contraction, one external
    input seed, identical shapes/dtypes and one arithmetic directive throughout.
    No reassociation, implicit casts, or scientific approximation is introduced.
    The root Node hash and program identity remain the scientific authority.
    """
    if root.op != "add" or len(root.inputs) != 2:
        raise ValueError("contraction update requires an existing binary add")
    products = [node for node in root.inputs if node.op == "einsum"]
    seeds = [node for node in root.inputs if node.op == "input"]
    if len(products) != 1 or len(seeds) != 1 or len(products[0].inputs) != 2:
        raise ValueError("contraction update requires one binary einsum and input seed")
    product, seed = products[0], seeds[0]
    if tuple(root.attrs["coefficients"]) != ((1, 1), (1, 1)):
        raise ValueError("contraction update requires unit add coefficients")
    if seed in adapter.program.outputs.values() or any(
        seed in node.inputs for node in adapter.live_nodes if node is not root
    ):
        raise ValueError("contraction seed donation requires its exclusive last use")
    if any(
        node.spec.shape != root.spec.shape or node.spec.dtype != root.spec.dtype
        for node in (product, seed)
    ):
        raise ValueError("contraction update cannot broadcast or convert the seed")
    request = adapter.request(root, backend=backend)
    contraction = adapter.request(product, backend=backend)
    if adapter.directives[root] != adapter.directives[product]:
        raise ValueError("contraction update cannot change intermediate arithmetic")
    dtypes = (*contraction.input_dtypes, seed.spec.dtype)
    output = contraction.operands[-1]
    return replace(
        request,
        input_dtypes=dtypes,
        precisions=tuple(
            replace(precision, input_dtypes=dtypes) for precision in request.precisions
        ),
        operands=(
            *contraction.operands[:-1],
            replace(
                output, operand="input:2", access="read", alias_group="donated-seed"
            ),
            replace(output, alias_group="donated-seed"),
        ),
        semantics=(
            *request.semantics,
            ("contraction_node_hash", adapter.hashes[product]),
        ),
        effects=(("output", "donated"), ("donated_input", "input:2")),
    )
