"""Independent checks for the physical-to-logical rank-k output binding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.ir import einsum, input_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import density_program
from generativeqc_compiler.tensor.symmetric_rank_k import (
    emit_symmetric_rank_k_old_output_binding,
    symmetric_rank_k_bind_old_output,
    symmetric_rank_k_overwrite_program,
    symmetric_rank_k_request,
    symmetric_rank_k_update_program,
)
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("beta", [1.0, -2.0, 0.0, -0.0])
@pytest.mark.parametrize("poison_lower", [False, True])
def test_dense_bound_equation_matches_upper_authoritative_oracle(
    order: str, beta: float, poison_lower: bool
) -> None:
    program = density_program(2, 2, spin_count=2, orbital_count=1)
    coefficients = np.arange(1.0, 9.0).reshape(2, 2, 2, 1)
    weights = np.array([1.0, -1.0, 2.0, -2.0]).reshape(2, 2, 1)
    old = np.array(np.arange(10.0, 170.0, 10.0).reshape(2, 2, 2, 2), order=order)
    if poison_lower:
        old[..., 1, 0] = np.nan
    if beta == 0.0:
        old[...] = np.nan
    feeds = {
        "coefficients": coefficients,
        "occupations": weights,
        "rank_k_alpha": np.array(1.0),
    }
    if beta == 0.0:
        composition = symmetric_rank_k_overwrite_program(program, "density")
    else:
        composition = symmetric_rank_k_update_program(program, "density")
        feeds.update(
            rank_k_beta=np.array(beta),
            rank_k_bound_old_output=symmetric_rank_k_bind_old_output(
                program, "density", beta, old
            ),
        )
    actual = execute(composition, feeds).outputs["updated"]
    expected = np.empty_like(old)
    for batch in np.ndindex(2, 2):
        for row in range(2):
            for column in range(row, 2):
                gram = (
                    coefficients[batch + (row, 0)]
                    * weights[batch + (0,)]
                    * coefficients[batch + (column, 0)]
                )
                prior = 0.0 if beta == 0.0 else old[batch + (row, column)]
                value = gram + beta * prior
                expected[batch + (row, column)] = value
                expected[batch + (column, row)] = value
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("beta", [0.0, -0.0])
def test_zero_beta_binding_does_not_inspect_old_output(beta: float) -> None:
    class Unreadable:
        def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
            pytest.fail("zero beta inspected the old output")

    program = density_program(1, 2, orbital_count=1)
    bound = symmetric_rank_k_bind_old_output(program, "density", beta, Unreadable())
    np.testing.assert_array_equal(bound, np.zeros((1, 1, 2, 2)))
    assert not np.signbit(bound).any()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_binding_rejects_nonfinite_beta_and_read_upper_cells(bad: float) -> None:
    program = density_program(1, 2, orbital_count=1)
    old = np.ones((1, 1, 2, 2))
    with pytest.raises(ValueError, match="beta must be finite"):
        symmetric_rank_k_bind_old_output(program, "density", bad, old)
    old[..., 0, 1] = bad
    with pytest.raises(ValueError, match="upper triangle must be finite"):
        symmetric_rank_k_bind_old_output(program, "density", 1.0, old)


def test_raw_nonsymmetric_output_is_not_a_logical_bound_input() -> None:
    program = density_program(1, 2, orbital_count=1)
    with pytest.raises(ValueError, match="declared symmetry"):
        execute(
            symmetric_rank_k_update_program(program, "density"),
            {
                "coefficients": np.array([[[[1.0], [2.0]]]]),
                "occupations": np.ones((1, 1, 1)),
                "rank_k_alpha": np.array(1.0),
                "rank_k_beta": np.array(1.0),
                "rank_k_bound_old_output": np.array([[[[10.0, 20.0], [30.0, 40.0]]]]),
            },
        )


@pytest.mark.parametrize(
    "name",
    ["rank_k_alpha", "rank_k_beta", "rank_k_old_output", "rank_k_bound_old_output"],
)
def test_reserved_same_domain_input_names_fail_closed(name: str) -> None:
    ao = IndexSpace("ao", "ao", 2)
    p, i = Index("p", ao), Index("i", ao)
    coefficients = input_tensor(name, TensorSpec((p, i), role="input"))
    weights = input_tensor("weights", TensorSpec((i,), role="input"))
    program = Program(
        {"gram": einsum("pi,i,qi->pq", coefficients, weights, coefficients)}
    )
    for builder in (
        symmetric_rank_k_update_program,
        symmetric_rank_k_overwrite_program,
    ):
        with pytest.raises(ValueError, match="reserved update binding names"):
            builder(program, "gram")
    for mode in ("update", "overwrite"):
        with pytest.raises(ValueError, match="reserved update binding names"):
            symmetric_rank_k_request(program, "gram", update=mode)


def test_binding_source_participates_only_in_update_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import generativeqc_compiler.tensor.symmetric_rank_k as rank_k

    program = density_program(1, 2, orbital_count=1)
    update_before = symmetric_rank_k_request(program, "density")
    overwrite_before = symmetric_rank_k_request(program, "density", update="overwrite")
    source = emit_symmetric_rank_k_old_output_binding()
    monkeypatch.setattr(
        rank_k, "emit_symmetric_rank_k_old_output_binding", lambda: source + "\n"
    )
    update_after = symmetric_rank_k_request(program, "density")
    overwrite_after = symmetric_rank_k_request(program, "density", update="overwrite")
    assert update_before.scientific_identity == update_after.scientific_identity
    assert update_before.semantic_identity != update_after.semantic_identity
    assert update_before.identity != update_after.identity
    assert overwrite_before.identity == overwrite_after.identity
    assert dict(overwrite_before.semantics)["old_output_binding_identity"] == ""


def test_all_native_old_output_reads_use_the_generated_binding() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/tensor/cuda_symmetric_rank_k.cuh"
    ).read_text()
    assert source.count("rank_k_generated::rank_k_bound_old_output(") == 3
    assert source.count("call.beta == 0.0") == 1
    assert source.index("const auto panel = contraction_product") < source.index(
        "const bool small_row_request"
    )
