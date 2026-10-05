"""Canonical potential algebra and real prepared native XC qualification."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.dft.xc_potential_lowering import potential_panel_program
from generativeqc_compiler.tensor import Program, add, execute
from generativeqc_compiler.tensor.symmetric_product import symmetric_product_request


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 31, 19), (4, 33, 17)])
def test_panel_identity_matches_independent_symmetric_cross_product(
    shape: tuple[int, int, int],
) -> None:
    rng = np.random.default_rng(193)
    a = rng.normal(size=shape)
    w = rng.normal(size=shape)
    program = potential_panel_program(*shape)
    result = execute(program, {"ao": a, "weighted": w}).outputs["potential"]
    cross = sum(x.T @ y for x, y in zip(a, w, strict=True))
    np.testing.assert_allclose(result, cross + cross.T, atol=2e-12, rtol=2e-13)
    cuda = symmetric_product_request(program, "potential")
    cpu = symmetric_product_request(program, "potential", backend="cpu")
    assert cuda.semantic_identity == cpu.semantic_identity
    assert cuda.scientific_identity == cpu.scientific_identity
    assert cuda.constraints.capture_required
    assert cuda.input_dtypes == ("float64",) * 3
    assert cuda.operands[-1].access == "read-write"
    assert cuda.operands[-1].triangle == "upper"
    assert dict(cuda.semantics)["publication"] == "upper-triangle-mirrored"


def test_asymmetric_or_scaled_algebra_cannot_become_rank2k() -> None:
    valid = potential_panel_program(1, 3, 2)
    cross, reflected = valid.outputs["potential"].inputs
    for node in (cross, add(cross, cross), add(cross, reflected, coefficients=(1, 2))):
        with pytest.raises(ValueError, match="symmetric product"):
            symmetric_product_request(Program({"potential": node}), "potential")
