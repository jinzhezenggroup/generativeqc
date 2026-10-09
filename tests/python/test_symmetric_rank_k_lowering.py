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
from generativeqc_compiler.common.precision import PrecisionDirective
from generativeqc_compiler.tensor.ir import add, einsum, input_tensor
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.precision import lower_precision
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.scf import density_program, weighted_density_program
from generativeqc_compiler.tensor.symmetric_rank_k import (
    emit_symmetric_rank_k_portfolio,
    symmetric_rank_k_overwrite_program,
    symmetric_rank_k_request,
    symmetric_rank_k_scalar_overwrite_program,
    symmetric_rank_k_scalar_update_program,
    symmetric_rank_k_update_program,
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
    assert request.operation == "add"
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
    update = symmetric_rank_k_update_program(program, name)
    source_request = TensorLoweringAdapter(program).request(
        program.outputs[name], backend="cuda"
    )
    assert request.scientific_identity == update.logical_hash
    assert request.scientific_identity != program.logical_hash
    assert dict(request.semantics)["update_program_hash"] == update.logical_hash
    assert dict(request.semantics)["source_program_hash"] == program.logical_hash
    assert dict(request.semantics)["source_request_identity"] == source_request.identity
    assert dict(request.semantics)["source_scientific_identity"] == (
        source_request.scientific_identity
    )
    assert dict(request.semantics)["source_precision_identity"] == (
        source_request.precisions[0].identity
    )
    assert dict(request.semantics)["scalar_update_hash"] == (
        symmetric_rank_k_scalar_update_program().logical_hash
    )
    assert dict(request.semantics)["scalar_input_roles"] == (
        "alpha,product,beta,old_output"
    )
    changed = Program(
        {"updated": add(*update.outputs["updated"].inputs, coefficients=(2, 1))}
    )
    assert changed.logical_hash != request.scientific_identity
    emitted = emit_symmetric_rank_k_portfolio(
        program, name, sha256(b"source").hexdigest(), name="rank_k", order=order
    )
    assert "symmetric-rank-k-generated" in emitted
    assert "symmetric-rank-k-signed-gemm" in emitted
    overwrite = symmetric_rank_k_request(program, name, order=order, update="overwrite")
    overwrite_program = symmetric_rank_k_overwrite_program(program, name)
    assert overwrite.operation == "multiply"
    assert overwrite.scientific_identity == overwrite_program.logical_hash
    assert overwrite.input_dtypes == ("float64",) * 3
    assert overwrite.operands[-1].access == "write"
    assert dict(overwrite.semantics)["scalar_update_hash"] == (
        symmetric_rank_k_scalar_overwrite_program().logical_hash
    )
    assert overwrite.scientific_identity != request.scientific_identity


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
    with pytest.raises(ValueError, match="update must be overwrite or update"):
        symmetric_rank_k_request(program, "density", update="unknown")


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
        "bin/nvcc.profile",
        "bin/cudafe++",
        "bin/fatbinary",
        "bin/nvlink",
        "bin/ptxas",
        "bin/crt/link.stub",
        "nvvm/bin/cicc",
        "nvvm/libdevice/libdevice.10.bc",
        "include/cuda_runtime.h",
        "lib64/libcublas.so.12",
        "lib64/libcublasLt.so.12",
        "lib64/libcudadevrt.a",
        "lib64/libcudart.so.12",
    ):
        path = toolkit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    host_compiler = tmp_path / "host-compiler"
    host_compiler.write_text("qualified host compiler bytes")
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
                "--host-compiler",
                str(host_compiler),
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
    assert b"rank_k_alpha_beta_update" in before
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


def test_rank_k_rejects_changed_or_forged_source_precision_provenance() -> None:
    program = density_program(1, 3, spin_count=2, orbital_count=5)
    with pytest.raises(ValueError, match="cannot silently change arithmetic or audit"):
        symmetric_rank_k_request(
            lower_precision(program, {}, strict_audit_dtype="float32"), "density"
        )
    root_name = program.debug_names[program.outputs["density"]]
    with pytest.raises(
        ValueError, match="unscaled ternary einsum|strict FP64|cannot silently change"
    ):
        symmetric_rank_k_request(
            lower_precision(
                program,
                {
                    root_name: PrecisionDirective(
                        "float32",
                        "float32",
                        "float32",
                        qualification="test/rank-k-rejected",
                    )
                },
            ),
            "density",
        )

    lowered = lower_precision(program, {})
    provenance = lowered.provenance
    provenance["precision_request_identity"] = "0" * 64
    altered = Program(lowered.outputs, lowered.definitions, provenance=provenance)
    with pytest.raises(
        ValueError, match="precision request scope or identity mismatch"
    ):
        symmetric_rank_k_request(altered, "density")
