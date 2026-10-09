"""The rank-k provider view must remain tied to the original signed-weight IR."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Literal

import pytest
from generativeqc_compiler.tensor.ir import einsum, input_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import density_program, weighted_density_program
from generativeqc_compiler.tensor.symmetric_rank_k import (
    emit_symmetric_rank_k_portfolio,
    symmetric_rank_k_request,
)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("order", ["row-major", "column-major"])
def test_rank_k_reuses_original_equation_and_has_one_signed_contract(
    weighted: bool, order: Literal["row-major", "column-major"]
) -> None:
    name = "weighted_density" if weighted else "density"
    program = (
        weighted_density_program(2, 3, spin_count=2, orbital_count=5)
        if weighted
        else density_program(2, 3, spin_count=2, orbital_count=5)
    )
    request = symmetric_rank_k_request(program, name, order=order)
    assert request.scientific_identity
    assert request.operation == "einsum"
    assert request.input_dtypes == ("float64",) * 4
    assert dict(request.semantics)["signed_weights"] is True
    assert dict(request.semantics)["weights_materialization"] == (
        "preceding-multiply" if weighted else "borrowed"
    )
    assert request.operands[0].alias_group == request.operands[2].alias_group
    assert request.operands[-1].triangle == "upper"
    assert request.operands[-1].access == "read-write"
    assert request.constraints.capture_required
    assert request.effects == (
        ("output", "transactional-symmetric-overwrite-or-accumulate"),
    )
    emitted = emit_symmetric_rank_k_portfolio(
        program, name, sha256(b"source").hexdigest(), name="rank_k", order=order
    )
    assert "symmetric-rank-k-generated" in emitted
    assert "symmetric-rank-k-signed-gemm" in emitted


def test_rank_k_rejects_changed_coefficient_or_weight_topology() -> None:
    program = density_program(1, 3, spin_count=2, orbital_count=5)
    root = program.outputs["density"]
    other = input_tensor("other", root.inputs[0].spec)
    changed = Program(
        {
            "density": einsum(
                "bspi,bsi,bsqi->bspq", root.inputs[0], root.inputs[1], other
            )
        }
    )
    with pytest.raises(ValueError, match="share one input node"):
        symmetric_rank_k_request(changed, "density")

    with pytest.raises(ValueError, match="matrix order"):
        symmetric_rank_k_request(program, "density", order="not-an-order")
    with pytest.raises(ValueError, match="unscaled"):
        symmetric_rank_k_request(
            Program(
                {"density": einsum("bspi,bsi,bsqi->bspq", *root.inputs, coefficient=2)}
            ),
            "density",
        )


def test_rank_k_layout_changes_physical_identity_but_not_science() -> None:
    program = density_program(1, 3, spin_count=2, orbital_count=5)
    row = symmetric_rank_k_request(program, "density", order="row-major")
    column = symmetric_rank_k_request(program, "density", order="column-major")
    assert row.scientific_identity == column.scientific_identity
    assert row.identity != column.identity
    assert row.operands[0].strides == (5, 1)
    assert column.operands[0].strides == (1, 3)


def test_rank_k_rejects_mixed_precision() -> None:
    program = density_program(1, 3, spin_count=2, orbital_count=5)
    root = program.outputs["density"]
    changed_spec = replace(root.inputs[0].spec, dtype="float32")
    coefficients = input_tensor("coefficients", changed_spec)
    weights = input_tensor("occupations", replace(root.inputs[1].spec, dtype="float32"))
    with pytest.raises(ValueError, match="strict FP64"):
        symmetric_rank_k_request(
            Program(
                {
                    "density": einsum(
                        "bspi,bsi,bsqi->bspq", coefficients, weights, coefficients
                    )
                }
            ),
            "density",
        )
