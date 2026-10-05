"""Independent mathematics and structural gates for coupled checked matvecs."""

from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.dft.cosx_contraction import (
    cosx_bidirectional_pair,
    cosx_bidirectional_program,
)
from generativeqc_compiler.method.cosx_derivative_runtime import (
    build_cosx_bidirectional_update_program,
)
from generativeqc_compiler.tensor import Program, einsum, execute
from generativeqc_compiler.tensor.checked_contraction_pair import (
    checked_transpose_pair_request,
)
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import contraction_initializer


def test_pair_binds_both_products_and_original_scalar_program() -> None:
    program = cosx_bidirectional_program(5, 4)
    pair = cosx_bidirectional_pair(program)
    update = build_cosx_bidirectional_update_program()
    adapter = TensorLoweringAdapter(program)
    for role, node in ((1, pair.first), (2, pair.second)):
        requests = [
            checked_transpose_pair_request(adapter, pair, update, role=role, backend=b)
            for b in ("cpu", "cuda")
        ]
        assert requests[0].semantic_identity == requests[1].semantic_identity
        for request in requests:
            assert request.scientific_identity == program.logical_hash
            assert request.constraints.determinism == "exact-order"
            assert dict(request.semantics)["scalar_update_hash"] == update.logical_hash
            assert dict(request.semantics)["checked_transpose_pair_role"] == role
            assert dict(request.effects)["paired_publication"] == "zero-both-on-failure"
        emitted = contraction_initializer(
            adapter,
            node,
            lambda index: index.space.name,
            transpose=("N" if role == 1 else "T", "N"),
            extents=("points", "columns", "1", "columns"),
            coefficient="1.0",
            checked_update=update,
            checked_pair=pair,
        )
        assert emitted.endswith(f",false,{role}" + "}")
        assert update.logical_hash in emitted


def test_both_asymmetric_directions_match_independent_long_double() -> None:
    p, n = 5, 4
    matrix = (np.arange(p * n * n).reshape(p, n, n) - 23) / 41
    vector = (np.arange(p * n).reshape(p, n) - 7) / 29
    opposite = (np.arange(p * n).reshape(p, n) - 11) / 17
    right, left = (np.zeros((p, n), dtype=np.longdouble) for _ in range(2))
    for point in range(p):
        for row in range(n):
            for k in range(n):
                right[point, row] += (
                    np.longdouble(matrix[point, row, k]) * vector[point, k]
                )
                left[point, row] += (
                    np.longdouble(matrix[point, k, row]) * opposite[point, k]
                )
    result = execute(
        cosx_bidirectional_program(p, n),
        {"esp": matrix, "projected": vector, "symmetric_projection": opposite},
    )
    np.testing.assert_allclose(result.outputs["right"], right, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(result.outputs["left"], left, atol=3e-14, rtol=3e-14)


def test_pair_rejects_scalar_role_swaps_and_partial_science() -> None:
    program = cosx_bidirectional_program(5, 4)
    pair = cosx_bidirectional_pair(program)
    update = build_cosx_bidirectional_update_program()
    adapter = TensorLoweringAdapter(program)
    swapped = replace(pair, scalar_outputs=tuple(reversed(pair.scalar_outputs)))
    with pytest.raises(ValueError, match="scalar update graph"):
        checked_transpose_pair_request(adapter, swapped, update, role=1, backend="cuda")
    partial = TensorLoweringAdapter(Program({"right": pair.first}))
    with pytest.raises(ValueError, match="complete two-output"):
        checked_transpose_pair_request(partial, pair, update, role=1, backend="cuda")
    alias_output = TensorLoweringAdapter(
        Program({"right": pair.first, "left": pair.second, "extra": pair.first})
    )
    with pytest.raises(ValueError, match="complete two-output"):
        checked_transpose_pair_request(
            alias_output, pair, update, role=1, backend="cuda"
        )
    changed = einsum("pmn,pn->pm", *pair.first.inputs, coefficient=2)
    changed_pair = replace(pair, first=changed)
    changed_adapter = TensorLoweringAdapter(
        Program({"right": changed, "left": pair.second})
    )
    with pytest.raises(ValueError, match="unit external"):
        checked_transpose_pair_request(
            changed_adapter, changed_pair, update, role=1, backend="cuda"
        )
    with pytest.raises(ValueError, match="fresh transpose"):
        contraction_initializer(
            adapter,
            pair.second,
            lambda index: index.space.name,
            transpose=("N", "N"),
            extents=("points", "columns", "1", "columns"),
            coefficient="1.0",
            checked_update=update,
            checked_pair=pair,
        )
