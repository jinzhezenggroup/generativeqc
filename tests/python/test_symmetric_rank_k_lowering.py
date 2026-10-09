"""The rank-k provider view must remain tied to the original signed-weight IR."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from generativeqc_compiler.tensor.ir import einsum, input_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import density_program, weighted_density_program
from generativeqc_compiler.tensor.symmetric_rank_k import (
    emit_symmetric_rank_k_portfolio,
    symmetric_rank_k_request,
)
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

ROOT = Path(__file__).resolve().parents[2]


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
    assert row.semantic_identity == column.semantic_identity
    assert row.identity != column.identity
    assert row.operands[0].strides == (5, 1)
    assert column.operands[0].strides == (1, 3)


def test_rank_k_batch_dummy_names_do_not_change_domains() -> None:
    batch = IndexSpace("batch", "batch", 1)
    spin = IndexSpace("spin", "spin", 2)
    ao = IndexSpace("ao", "ao", 3)
    orbital = IndexSpace("orbital", "orbital", 5)
    coefficients = input_tensor(
        "coefficients",
        TensorSpec(
            (
                Index("coefficient_batch", batch),
                Index("coefficient_spin", spin),
                Index("coefficient_ao", ao),
                Index("coefficient_orbital", orbital),
            ),
            role="input",
        ),
    )
    weights = input_tensor(
        "weights",
        TensorSpec(
            (
                Index("weight_batch", batch),
                Index("weight_spin", spin),
                Index("weight_orbital", orbital),
            ),
            role="input",
        ),
    )
    program = Program(
        {"density": einsum("bspi,bsi,bsqi->bspq", coefficients, weights, coefficients)}
    )
    assert symmetric_rank_k_request(program, "density").scientific_identity


def test_rank_k_generator_bootstraps_checkout_and_binds_toolchain(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    toolkit = tmp_path / "cuda"
    for relative in (
        "bin/nvcc",
        "bin/ptxas",
        "lib64/libcublas.so.12",
        "lib64/libcublasLt.so.12",
        "lib64/libcudart.so.12",
    ):
        path = toolkit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    output = tmp_path / "generated.cuh"

    def generate() -> bytes:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/generate_symmetric_rank_k_cuda.py"),
                "--output",
                str(output),
                "--toolkit-root",
                str(toolkit),
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return output.read_bytes()

    before = generate()
    (toolkit / "bin/nvcc").write_text("changed compiler bytes")
    assert generate() != before


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
