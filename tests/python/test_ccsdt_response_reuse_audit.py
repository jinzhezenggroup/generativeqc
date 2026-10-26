"""Binding identity and runtime work units for the opt-in response audit."""

from __future__ import annotations

import pytest
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)

from benchmarks.ccsdt_response_reuse_audit import compare


def contraction(binding: str, dtype: str = "float64") -> Program:
    """A tiny template whose physical binding is explicit in the input name."""
    occupied = IndexSpace("audit_occupied", "occupied", 2)
    virtual = IndexSpace("audit_virtual", "virtual", 3)
    i, a = Index("i", occupied), Index("a", virtual)
    matrix = input_tensor(binding, TensorSpec((i, a), dtype=dtype, role="input"))
    vector = input_tensor("shared_vector", TensorSpec((a,), dtype=dtype, role="input"))
    return Program({"result": einsum("ia,a->i", matrix, vector)})


def test_identical_requests_need_identical_physical_bindings() -> None:
    dimensions = {"occupied": 5, "virtual": 7}
    same = compare(
        {"left": contraction("matrix"), "right": contraction("matrix")}, dimensions
    )
    different = compare(
        {"left": contraction("matrix"), "right": contraction("other_matrix")},
        dimensions,
    )
    # The concrete oracle is one 5-by-7 matrix-vector product, not the 2-by-3
    # representative used for code generation and not twice its FLOP count.
    assert same["eliminated_contraction_summands"] == 35
    assert different["eliminated_contraction_summands"] == 0
    assert not different["exact_common_nodes"]


def test_audit_does_not_silently_compose_different_precision() -> None:
    with pytest.raises(ValueError, match="strict-FP64"):
        compare({"mixed": contraction("matrix", "float32")}, {})
