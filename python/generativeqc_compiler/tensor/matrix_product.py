"""Semantic square products of borrowed column-major system/spin matrices.

Physical spin broadcast and transpose are mathematical views, not provider
switches. Activity is a per-system publication predicate: both implementations
must preserve every inactive output byte, including during graph replay.
"""

from __future__ import annotations

from dataclasses import replace

from generativeqc_compiler.common.provenance import canonical_hash

from .ir import broadcast, einsum, input_tensor
from .lowering import TensorLoweringAdapter
from .native_lowering import emit_contraction_region_portfolio
from .program import Program
from .types import Index, IndexSpace, TensorSpec


def matrix_product_program(
    dimension: int,
    systems: int,
    spins: int = 1,
    *,
    spin_product: bool = False,
    left_is_spin: bool = False,
    right_is_spin: bool = False,
    transpose_left: bool = False,
) -> Program:
    """Describe the transposed storage views without packing or symmetry.

    All buffers use column-major matrices within system-major/spin-major
    storage. Consequently the row-major TensorIR output is R.T @ L.T (or
    R.T @ L for a transposed left operand). A physical operand lacks the spin
    index and broadcasts without reading nonexistent spin storage.
    """
    if any(
        type(value) is not int or value < 1 for value in (dimension, systems, spins)
    ):
        raise ValueError("matrix product dimensions must be positive integers")
    flags = (spin_product, left_is_spin, right_is_spin, transpose_left)
    if any(type(value) is not bool for value in flags):
        raise TypeError("matrix product view flags must be boolean")
    if not spin_product and (spins != 1 or left_is_spin or right_is_spin):
        raise ValueError("plain products cannot carry spin views")
    batch = Index("b", IndexSpace("system", "batch", systems))
    spin = Index("s", IndexSpace("spin", "batch", spins))
    space = IndexSpace("matrix", "ao", dimension)
    row, column = Index("i", space), Index("j", space)
    left_axes = (batch, spin, row, column) if left_is_spin else (batch, row, column)
    right_axes = (batch, spin, row, column) if right_is_spin else (batch, row, column)
    left = input_tensor("left", TensorSpec(left_axes, role="input"))
    right = input_tensor("right", TensorSpec(right_axes, role="input"))
    left_labels = ("bs" if left_is_spin else "b") + ("jk" if transpose_left else "kj")
    right_labels = ("bs" if right_is_spin else "b") + "ik"
    output_labels = ("bs" if spin_product else "b") + "ij"
    if spin_product and not left_is_spin and not right_is_spin:
        # The result is identical for each spin. The original executor publishes
        # each spin separately; no extra numeric input or materialization exists.
        product = einsum(f"{right_labels},{left_labels}->bij", right, left)
        product = broadcast(product, (batch, spin, row, column), (0, 2, 3))
    else:
        product = einsum(f"{right_labels},{left_labels}->{output_labels}", right, left)
    scale = input_tensor("scale", TensorSpec((), role="input"))
    result = einsum(f",{output_labels}->{output_labels}", scale, product)
    return Program({"product": result})


def emit_matrix_product_cuda() -> str:
    """Emit bounded native portfolios; numerical admission stays strict FP64."""
    variants = [(False, False, False, transpose) for transpose in (False, True)]
    variants.extend(
        (True, left, right, transpose)
        for left in (False, True)
        for right in (False, True)
        for transpose in (False, True)
    )
    pieces = [
        "// Generated from column-major matrix-product TensorIR; do not edit.",
        "#pragma once",
        '#include "runtime/lowering_binding.hpp"',
        "namespace generativeqc::tensor::matrix_product_lowering {",
        "struct Portfolio {",
        "  const runtime::NativeLoweringRequest& request;",
        "  const std::array<runtime::NativeLoweringCandidate, 5>& candidates;",
        "  std::string_view target, compilation;",
        "};",
    ]
    for slot, (spin, left, right, transpose) in enumerate(variants):
        program = matrix_product_program(
            5,
            3,
            2 if spin else 1,
            spin_product=spin,
            left_is_spin=left,
            right_is_spin=right,
            transpose_left=transpose,
        )
        adapter = TensorLoweringAdapter(program)
        node = program.outputs["product"]
        contraction = next(
            value
            for value in adapter.live_nodes
            if value.op == "einsum"
            and all(
                child.op == "input" and child.attrs["name"] in ("left", "right")
                for child in value.inputs
            )
        )
        request = adapter.request(node, backend="cuda")
        product_request = adapter.request(contraction, backend="cuda")
        dtypes = (*product_request.input_dtypes, "float64")
        # Node-local mode ordinals must be composed through logical indices;
        # scalar scaling and spin broadcast renumber their own local domains.
        modes = dict(
            zip(
                contraction.spec.indices,
                product_request.operands[-1].modes,
                strict=True,
            )
        )
        next_mode = (
            max(mode for operand in product_request.operands for mode in operand.modes)
            + 1
        )
        for label in node.spec.indices:
            if label not in modes:
                modes[label] = next_mode
                next_mode += 1
        request = replace(
            request,
            input_dtypes=dtypes,
            precisions=tuple(
                replace(precision, input_dtypes=dtypes)
                for precision in request.precisions
            ),
            operands=(
                *product_request.operands[:-1],
                replace(
                    request.operands[0],
                    operand="input:2",
                    alias_group="host-scalar-scale",
                ),
                replace(
                    request.operands[-1],
                    modes=tuple(modes[label] for label in node.spec.indices),
                ),
            ),
            effects=(
                *request.effects,
                ("activity", "per-system-device-predicate"),
                ("inactive_publication", "preserve-output-bytes"),
            ),
            semantics=tuple(
                (name, dict(product_request.semantics)["reduction_extent"])
                if name == "reduction_extent"
                else (name, value)
                for name, value in request.semantics
            )
            + (
                ("storage", "column-major-system-spin"),
                ("publication", "preserve-inactive-system-bytes"),
                ("scale", "uniform-finite-scalar"),
                (
                    "output_spins",
                    "broadcast" if spin and not left and not right else "physical",
                ),
                ("contraction_node_hash", adapter.hashes[contraction]),
            ),
        )
        pieces.append(
            emit_contraction_region_portfolio(
                request,
                canonical_hash({"matrix_product_schema": 1, "slot": slot}),
                name=f"matrix_{slot}",
                bounded_dense=True,
                blas_context_contract="runtime-bound-fp64",
            )
        )
    pieces.append("inline Portfolio portfolio(std::size_t slot) { switch (slot) {")
    for slot in range(len(variants)):
        pieces.append(
            f"case {slot}: return {{matrix_{slot}_request, matrix_{slot}_candidates, "
            f"matrix_{slot}_target, matrix_{slot}_compilation}};"
        )
    pieces.extend(
        [
            'default: throw std::invalid_argument("invalid matrix product site");',
            "} }",
            "} // namespace generativeqc::tensor::matrix_product_lowering",
        ]
    )
    return "\n".join(pieces) + "\n"
