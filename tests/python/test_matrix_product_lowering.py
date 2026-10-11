"""Canonical square-product views independently match column-major algebra."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.matrix_product import (
    emit_matrix_product_cuda,
    matrix_product_program,
)


@pytest.mark.parametrize("systems,spins", [(1, 1), (3, 2)])
@pytest.mark.parametrize(
    "left_spin,right_spin,transpose", itertools.product((False, True), repeat=3)
)
def test_asymmetric_spin_views_match_numpy(
    systems: int, spins: int, left_spin: bool, right_spin: bool, transpose: bool
) -> None:
    rng = np.random.default_rng(1873)
    left = rng.normal(size=(systems, spins if left_spin else 1, 5, 5))
    right = rng.normal(size=(systems, spins if right_spin else 1, 5, 5))
    program = matrix_product_program(
        5,
        systems,
        spins,
        spin_product=True,
        left_is_spin=left_spin,
        right_is_spin=right_spin,
        transpose_left=transpose,
    )
    left_view = left.swapaxes(-1, -2).copy()
    right_view = right.swapaxes(-1, -2).copy()
    actual = execute(
        program,
        {
            "left": left_view if left_spin else left_view[:, 0],
            "right": right_view if right_spin else right_view[:, 0],
            "scale": np.asarray(-0.375),
        },
    ).outputs["product"]
    expected = np.matmul(left.swapaxes(-1, -2) if transpose else left, right)
    expected = np.broadcast_to(expected * -0.375, (systems, spins, 5, 5))
    np.testing.assert_allclose(
        actual.swapaxes(-1, -2), expected, rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_plain_request_has_one_backend_neutral_semantic_identity(
    transpose: bool,
) -> None:
    program = matrix_product_program(7, 3, transpose_left=transpose)
    adapter = TensorLoweringAdapter(program)
    node = program.outputs["product"]
    cpu, cuda = (adapter.request(node, backend=backend) for backend in ("cpu", "cuda"))
    assert cpu.semantic_identity == cuda.semantic_identity
    assert len(cuda.precisions) == 1
    assert cuda.precisions[0].directive.compute_dtype == "float64"
    assert cuda.precisions[0].directive.accumulation_dtype == "float64"


@pytest.mark.parametrize(
    "dimension,systems,spins", [(0, 1, 1), (1, 0, 1), (1, 1, -1), (True, 1, 1)]
)
def test_dimensions_fail_closed(dimension: int, systems: int, spins: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        matrix_product_program(dimension, systems, spins)


def test_emission_is_deterministic_and_preserves_negative_offers() -> None:
    source = emit_matrix_product_cuda()
    assert source == emit_matrix_product_cuda()
    assert (
        source.count("exact-shape plans do not implement a bounded runtime domain")
        == 30
    )
    assert "matrix_9_request" in source
