"""Public/direct stage requests preserve the column-major SCF algebra."""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor.basis_transform import (
    basis_transform_program,
    emit_basis_transform_cuda,
)
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter


def _product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Independent increasing-k, platform-independent high-precision oracle."""
    result = np.empty((left.shape[0], right.shape[1]), dtype=object)
    with localcontext() as context:
        context.prec = 80
        for row in range(left.shape[0]):
            for column in range(right.shape[1]):
                value = Decimal(0)
                for reduction in range(left.shape[1]):
                    lhs, rhs = left[row, reduction], right[reduction, column]
                    value += (
                        lhs
                        if isinstance(lhs, Decimal)
                        else Decimal.from_float(float(lhs))
                    ) * (
                        rhs
                        if isinstance(rhs, Decimal)
                        else Decimal.from_float(float(rhs))
                    )
                result[row, column] = value
    return result


@pytest.mark.parametrize("public,direct", [(1, 1), (5, 7), (7, 5), (12, 17)])
def test_rectangular_asymmetric_column_major_oracle(public: int, direct: int) -> None:
    rng = np.random.default_rng(1878 + public * 31 + direct)
    transform = rng.normal(size=(public, direct))
    density = rng.normal(size=(public, public))
    direct_fock = rng.normal(size=(direct, direct))
    hcore = rng.normal(size=(public, public))
    result = execute(
        basis_transform_program(public, direct),
        {
            "transform": transform.T.copy(),
            "density": density.T.copy(),
            "direct_fock": direct_fock.T.copy(),
            "hcore": hcore.T.copy(),
        },
    ).outputs
    density_right = _product(density, transform)
    density_direct = _product(transform.T, density_right)
    fock_left = _product(transform, direct_fock)
    fock_right = _product(fock_left, transform.T)
    expected = {
        "density_right": density_right.T,
        "density_direct": density_direct.T,
        "fock_left": fock_left.T,
        "fock_right": fock_right.T,
        "fock_public": (
            fock_right
            + np.vectorize(lambda value: Decimal.from_float(float(value)))(hcore)
        ).T,
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(
            result[name], np.asarray(reference, dtype=float), rtol=2e-13, atol=2e-13
        )


@pytest.mark.parametrize("public,direct", [(0, 2), (2, 0), (-1, 2), (2, 1.5)])
def test_invalid_dimensions_fail_closed(public: int, direct: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        basis_transform_program(public, direct)


def test_four_stage_identities_are_backend_neutral() -> None:
    program = basis_transform_program(5, 7)
    adapter = TensorLoweringAdapter(program)
    for name in ("density_right", "density_direct", "fock_left", "fock_right"):
        node = program.outputs[name]
        cpu = adapter.request(node, backend="cpu")
        cuda = adapter.request(node, backend="cuda")
        assert cpu.scientific_identity == cuda.scientific_identity
        assert cpu.semantic_identity == cuda.semantic_identity
        assert len(node.inputs) == 2
    assert program.outputs["fock_public"].op == "add"


def test_generated_portfolios_and_domain_guard_are_deterministic() -> None:
    source = emit_basis_transform_cuda()
    assert source == emit_basis_transform_cuda()
    for name in ("density_right", "density_direct", "fock_left", "fock_right"):
        assert f"basis_{name}_request" in source
        assert f"basis_{name}_candidates" in source
    assert source.count("inline tensor::ContractionRequest ") == 4
    assert '#include "runtime/lowering_binding.hpp"' in source
    assert "require_packed_stage" in source
    assert "batch_size != 1 || spin_count != 1 || active || shell_spans" in source
    assert "public_nbf <= 0 || direct_nbf <= 0" in source
    assert "cublasDgemm" not in source
    assert "cublasCreate" not in source


def test_cli_generates_the_qualification_header(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_basis_transform_cuda.cuh"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_basis_transform_cuda.py"),
            "--output",
            str(output),
        ],
        check=True,
        timeout=30,
    )
    assert output.read_text(encoding="utf-8") == emit_basis_transform_cuda()
