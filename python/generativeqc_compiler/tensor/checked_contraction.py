"""Bind existing scalar update programs to an ordered contraction request."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import LoweringConstraints

from .program import Program, node_hashes

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node


def checked_contraction_request(
    request: LoweringRequest,
    update: Program,
    publication: Program | None = None,
    *,
    contraction: Node,
) -> LoweringRequest:
    """Retain the compiler's checked scalar arithmetic at every reduction step.

    Recognition is deliberately bounded to ``accumulator + left * right`` and
    optional ``factor * value`` publication. The original scalar programs remain
    authoritative and their hashes bind the generated execution. A final-output
    audit cannot replace an intermediate update check or change reduction order.
    """
    if (
        contraction.op != "einsum"
        or len(contraction.inputs) != 2
        or contraction.attrs["coefficient"] != (1, 1)
    ):
        raise ValueError(
            "checked scalar execution requires a unit original contraction"
        )
    # The request may describe a leading-axis slice or a weighted region. Bind
    # the proof to its original product, never an unrelated unit-coefficient node.
    proof = node_hashes(Program({"result": contraction}).nodes)[contraction]
    semantics = dict(request.semantics)
    expected = semantics.get("contraction_node_hash" if publication else "node_hash")
    if proof != expected:
        raise ValueError("checked contraction proof differs from its original node")
    inputs = {n.attrs["name"]: n for n in update.live_nodes if n.op == "input"}
    if (
        set(inputs) != {"accumulator", "left", "right"}
        or set(update.outputs) != {"updated"}
        or any(n.spec.shape or n.spec.dtype != "float64" for n in update.live_nodes)
    ):
        raise ValueError("checked contraction requires the scalar FP64 update program")
    root = update.outputs["updated"]
    products = [n for n in root.inputs if n.op == "multiply"]
    if (
        root.op != "add"
        or root.attrs.get("coefficients") != ((1, 1), (1, 1))
        or inputs["accumulator"] not in root.inputs
        or len(products) != 1
        or set(products[0].inputs) != {inputs["left"], inputs["right"]}
        or len(update.live_nodes) != 5
    ):
        raise ValueError("checked contraction cannot change the scalar update graph")
    if (
        request.operation != "einsum"
        or any(dtype != "float64" for dtype in request.input_dtypes)
        or request.dtype != "float64"
        or request.accumulation_dtype != "float64"
        or len(request.input_dtypes) != (3 if publication else 2)
        or bool("batch_scale_mode" in dict(request.semantics)) != bool(publication)
    ):
        raise ValueError("checked scalar execution requires its complete FP64 region")
    publication_hash = ""
    if publication is not None:
        pinputs = {
            n.attrs["name"]: n for n in publication.live_nodes if n.op == "input"
        }
        if (
            set(pinputs) != {"factor", "value"}
            or set(publication.outputs) != {"scaled"}
            or any(
                n.spec.shape or n.spec.dtype != "float64"
                for n in publication.live_nodes
            )
            or publication.outputs["scaled"].op != "multiply"
            or set(publication.outputs["scaled"].inputs) != set(pinputs.values())
            or len(publication.live_nodes) != 3
        ):
            raise ValueError("checked publication must retain the scalar scaling graph")
        publication_hash = publication.logical_hash
    semantics = dict(request.semantics)
    semantics.update(
        scalar_update_hash=update.logical_hash,
        scalar_publication_hash=publication_hash,
    )
    effects = dict(request.effects)
    effects.update(
        reduction_step="finite-inputs-and-output-or-zero",
        nonfinite_publication="sticky-error-and-zero",
    )
    return replace(
        request,
        semantics=tuple(semantics.items()),
        effects=tuple(effects.items()),
        constraints=replace(
            request.constraints or LoweringConstraints(), determinism="exact-order"
        ),
    )
