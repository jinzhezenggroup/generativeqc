"""The polynomial lowering must track its actual energy and fail closed."""

import numpy as np
import pytest
from generativeqc_compiler.method.scalar_graph_ad import polynomial_partial
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ir import (
    constant,
    divide,
    input_tensor,
    multiply,
    sqrt,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec


@pytest.mark.parametrize("coefficient", ("1/3", "2/3", "-7/3"))
def test_exact_polynomial_response_changes_with_energy(coefficient: str) -> None:
    charge = input_tensor("charge", TensorSpec((), role="input", differentiable=True))
    energy = multiply(constant(coefficient), multiply(multiply(charge, charge), charge))
    primal = Program({"energy": energy})
    response = polynomial_partial(primal, "energy", "charge", ("charge", "charge"))
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 60
    numerator, denominator = coefficient.split("/")
    for point in (-2.3, 0, 0.8):
        expected = mp.diff(
            lambda value: mp.mpf(numerator) * value**3 / mp.mpf(denominator),
            mp.mpf(point),
        )
        actual = execute(
            Program({"response": response}), {"charge": np.asarray(point, dtype=float)}
        ).outputs["response"]
        np.testing.assert_allclose(actual, float(expected), atol=2e-14, rtol=3e-15)
    with pytest.raises(ValueError, match="factors"):
        polynomial_partial(primal, "energy", "charge", ("charge",))


def test_nonpolynomial_responses_are_rejected() -> None:
    coordinate = input_tensor(
        "coordinate", TensorSpec((), role="input", differentiable=True)
    )
    with pytest.raises(ValueError, match="nonpolynomial"):
        polynomial_partial(
            Program({"value": sqrt(coordinate)}), "value", "coordinate", ("coordinate",)
        )
    with pytest.raises(ValueError, match="nonpolynomial"):
        polynomial_partial(
            Program({"value": divide(constant(1), coordinate)}),
            "value",
            "coordinate",
            ("coordinate",),
        )
