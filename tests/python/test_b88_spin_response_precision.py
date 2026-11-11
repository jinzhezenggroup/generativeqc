"""Independent high-precision B88 response at minority-spin orbital nodes."""

from fractions import Fraction

import numpy as np
import pytest
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)
from generativeqc_compiler.xc.spec import FunctionalSpec


@pytest.mark.parametrize(
    "beta_density,beta_gradient",
    (
        (
            7.4225911645623816e-9,
            (-7.077433096138145e-6, 2.903863230182963e-7, 2.44166821128845e-7),
        ),
        (
            7.957986101616474e-8,
            (2.3052433743646647e-5, 1.6726573348585462e-6, -7.182692982576625e-7),
        ),
        (0.2, (0.0, 0.0, 0.0)),
        (0.2, (1e-7, -2e-7, 3e-7)),
    ),
)
def test_b88_cartesian_response_matches_independent_primal_differentiation(
    beta_density: float, beta_gradient: tuple[float, float, float]
) -> None:
    """Keep absolute AO-action accuracy without clipping a minority tangent."""
    precision = pytest.importorskip("mpmath").mp.clone()
    precision.dps = 80
    values = np.array(
        [
            0.11122928188611153,
            -0.18367220223785358,
            0.07765706640944295,
            0.1253058523729783,
            beta_density,
            *beta_gradient,
        ]
    )
    moving = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -0.02076973226014628,
            0.013436420725166147,
            -0.010249968910499467,
            -0.020557649119123632,
        ]
    )
    fields, direction = np.zeros((14, 1)), np.zeros((14, 1))
    indices = [0, 1, 2, 3, 5, 6, 7, 8]
    fields[indices, 0], direction[indices, 0] = values, moving
    fields[10, 0] = 0.02659000611220951
    spec = FunctionalSpec("B88 precision", (("GGA_X_B88", Fraction(1)),))
    actual = StationaryGridSecondOrder(spec).evaluate(fields, direction)

    def independent_energy(arguments: list) -> object:
        """Original B88 spin scaling, not the compiler's rearranged AD roots."""
        exchange_constant = (
            precision.mpf(3) / 4 * (6 / precision.pi) ** (precision.mpf(1) / 3)
        )
        beta = precision.mpf("0.0042")
        energy = precision.mpf(0)
        for offset in (0, 4):
            density = arguments[offset]
            sigma = sum(value**2 for value in arguments[offset + 1 : offset + 4])
            reduced_gradient = precision.sqrt(sigma) / density ** (precision.mpf(4) / 3)
            enhancement = exchange_constant + beta * reduced_gradient**2 / (
                1 + 6 * beta * reduced_gradient * precision.asinh(reduced_gradient)
            )
            energy -= density ** (precision.mpf(4) / 3) * enhancement
        return energy

    expected = []
    for coordinate in range(8):

        def perturbed(
            left: object, right: object, coordinate: int = coordinate
        ) -> object:
            return independent_energy(
                [
                    precision.mpf(float(value))
                    + right * precision.mpf(float(delta))
                    + (left if index == coordinate else 0)
                    for index, (value, delta) in enumerate(
                        zip(values, moving, strict=True)
                    )
                ]
            )

        expected.append(float(precision.diff(perturbed, (0, 0), (1, 1))))
    expected = np.asarray(expected) * fields[10, 0]
    np.testing.assert_allclose(
        actual.directional_coefficients[indices, 0], expected, atol=2e-12, rtol=8e-14
    )
    orbital, gradient = 0.33, np.array([-0.12, 0.18, 0.09])
    actual_action = actual.directional_coefficients[
        5, 0
    ] * orbital**2 + 2 * orbital * np.dot(
        actual.directional_coefficients[6:9, 0], gradient
    )
    expected_action = expected[4] * orbital**2 + 2 * orbital * np.dot(
        expected[5:], gradient
    )
    assert abs(actual_action - expected_action) < 2e-9
