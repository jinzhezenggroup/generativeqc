"""Bind existing scalar update programs to an ordered contraction request."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from generativeqc_compiler.common.lowering_contract import LoweringConstraints

from .program import Program, node_hashes

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringRequest

    from .ir import Node
    from .lowering import TensorLoweringAdapter


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
    return _bind_scalar_checks(request, update, publication_hash)


def checked_symmetric_right_request(
    adapter: TensorLoweringAdapter,
    contraction: Node,
    update: Program,
    *,
    scalar_inputs: tuple[str, str, str],
    backend: str,
) -> LoweringRequest:
    """Fuse the original half-scaled contraction with a borrowed RHS transpose.

    Recognize only ``1/2 * einsum(A, B + transpose(B))`` and its existing scalar
    update ``accumulator + (a * 1/2) * (b + bt)``. The scalar graph fixes rounding
    and finite checks; the dense graph fixes the complete scientific region.
    Neither virtual intermediate is published or materialized. Both RHS reads
    borrow the same square, readonly external input.
    """
    if contraction.op != "einsum" or len(contraction.inputs) != 2:
        raise ValueError("checked symmetric RHS requires a binary contraction")
    left, summed = contraction.inputs
    if (
        left.op != "input"
        or summed.op != "add"
        or len(summed.inputs) != 2
        or summed.attrs["coefficients"] != ((1, 1), (1, 1))
        or contraction.attrs["coefficient"] != (1, 2)
    ):
        raise ValueError(
            "checked symmetric RHS requires its original half-scaled graph"
        )
    right, transposed = summed.inputs
    if (
        right.op != "input"
        or transposed.op != "transpose"
        or transposed.inputs != (right,)
        or transposed.attrs["axes"] != (1, 0)
        or len(right.spec.shape) != 2
        or right.spec.shape[0] != right.spec.shape[1]
        or len(left.spec.shape) != 2
        or contraction.attrs["labels"] != ((0, 1), (1, 2))
        or contraction.attrs["output"] != (0, 2)
        or any(
            n.spec.dtype != "float64"
            for n in (left, right, summed, transposed, contraction)
        )
        or any(
            adapter.directives[n] != adapter.directives[contraction]
            for n in (summed, transposed)
        )
    ):
        raise ValueError("checked symmetric RHS requires its square FP64 borrowed view")
    for intermediate, consumer in ((summed, contraction), (transposed, summed)):
        if intermediate in adapter.program.outputs.values() or any(
            intermediate in node.inputs
            for node in adapter.live_nodes
            if node is not consumer
        ):
            raise ValueError(
                "checked symmetric RHS requires exclusive virtual intermediates"
            )
    inputs = {n.attrs["name"]: n for n in update.live_nodes if n.op == "input"}
    if (
        len(set(scalar_inputs)) != 3
        or set(inputs) != {"accumulator", *scalar_inputs}
        or set(update.outputs) != {"updated"}
        or any(n.spec.shape or n.spec.dtype != "float64" for n in update.live_nodes)
    ):
        raise ValueError("checked symmetric RHS requires its scalar FP64 inputs")
    a, b, bt = (inputs[name] for name in scalar_inputs)
    root = update.outputs["updated"]
    products = [n for n in root.inputs if n.op == "multiply"]
    if (
        root.op != "add"
        or root.attrs["coefficients"] != ((1, 1), (1, 1))
        or inputs["accumulator"] not in root.inputs
        or len(products) != 1
    ):
        raise ValueError("checked symmetric RHS cannot change the scalar update graph")
    scaled = [n for n in products[0].inputs if n.op == "multiply"]
    sums = [n for n in products[0].inputs if n.op == "add"]
    if (
        len(scaled) != 1
        or len(sums) != 1
        or set(sums[0].inputs) != {b, bt}
        or sums[0].attrs["coefficients"] != ((1, 1), (1, 1))
        or a not in scaled[0].inputs
        or len(update.live_nodes) != 9
    ):
        raise ValueError(
            "checked symmetric RHS cannot change scalar scaling or pairing"
        )
    literals = [n for n in scaled[0].inputs if n.op == "constant"]
    if len(literals) != 1 or literals[0].attrs["values"] != ((1, 2),):
        raise ValueError(
            "checked symmetric RHS requires the original scalar half factor"
        )
    request = adapter.request(contraction, backend=backend)
    if (
        request.input_dtypes != ("float64", "float64")
        or request.dtype != "float64"
        or request.accumulation_dtype != "float64"
        or len(request.precisions) != 1
        or not request.precisions[0].schedule.is_strict_fp64
    ):
        raise ValueError("checked symmetric RHS requires its complete FP64 region")
    semantics = dict(request.semantics)
    semantics.update(
        right_transform="symmetric-part",
        scalar_input_roles=",".join(scalar_inputs),
        right_source_hash=adapter.hashes[right],
        right_sum_hash=adapter.hashes[summed],
        right_transpose_hash=adapter.hashes[transposed],
    )
    # Rebind the virtual sum to its one immutable external matrix. The second
    # view is compiler-proven transpose aliasing, not a third external input.
    rhs = replace(
        request.operands[1], strides=(right.spec.shape[1], 1), alias_group="input:1"
    )
    request = replace(
        request,
        operands=(request.operands[0], rhs, request.operands[2]),
        semantics=tuple(semantics.items()),
    )
    return _bind_scalar_checks(request, update, "")


def _bind_scalar_checks(
    request: LoweringRequest, update: Program, publication_hash: str
) -> LoweringRequest:
    """Publish the common exact-order effects after structural recognition."""
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
