"""High-precision primal-only oracles for complete AO contraction responses."""

import numpy as np
import pytest
from generativeqc_compiler.xc.contractions import (
    _geometry_feature_direction,
    _geometry_feature_directions,
)
from generativeqc_compiler.xc.potential import (
    assemble_coefficients,
    assemble_coefficients_directional,
)


@pytest.mark.parametrize("family", ("lda", "gga"))
def test_feature_responses_against_indefinite_density_primal(family: str) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 70
    rng = np.random.default_rng(7431)
    panels = rng.normal(size=(4, 4, 1, 3))
    density = rng.normal(size=(2, 3, 3))
    density = 0.5 * (density + density.swapaxes(1, 2))
    delta = rng.normal(size=density.shape)
    delta = 0.5 * (delta + delta.swapaxes(1, 2))
    reference = {"gradient": rng.normal(size=(2, 1, 3))}
    results = _geometry_feature_directions(reference, *panels, density, delta, family)
    single = _geometry_feature_direction(
        reference, panels[0], panels[2], density, delta, family
    )
    for key in single:
        np.testing.assert_allclose(single[key], results[1][key], atol=1e-13, rtol=1e-13)

    def primal(
        left: object, right: object, point: object, spin: int, axis: int
    ) -> object:
        values = []
        for orbital in range(3):
            values.append(
                sum(
                    mp.mpf(float(panels[motion, row, 0, orbital]))
                    * (1, left, right, left * right)[motion]
                    * (1 if row == 0 else point)
                    for motion in range(4)
                    for row in (0, axis + 1)
                )
            )
        return sum(
            values[first]
            * values[second]
            * (
                mp.mpf(float(density[spin, first, second]))
                + right * mp.mpf(float(delta[spin, first, second]))
            )
            for first in range(3)
            for second in range(3)
        )

    for result, orders in zip(results, ((1, 0), (0, 1), (1, 1)), strict=True):
        for spin in range(2):
            for row in range(1 if family == "lda" else 4):
                expected = mp.diff(
                    lambda left, right, point, spin=spin, row=row: primal(
                        left, right, point, spin, max(0, row - 1)
                    ),
                    (0, 0, 0),
                    (*orders, int(row != 0)),
                )
                actual = (
                    result["rho"][spin, 0]
                    if row == 0
                    else result["gradient"][spin, 0, row - 1]
                )
                np.testing.assert_allclose(
                    actual, float(expected), atol=2e-14, rtol=3e-14
                )


@pytest.mark.parametrize(
    "ingredients", (("rho",), ("rho", "gradient"), ("rho", "gradient", "tau"))
)
def test_matrix_jvp_against_high_precision_primal(ingredients: tuple[str, ...]) -> None:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 70
    rng = np.random.default_rng(8572)
    jets, moving_jets = rng.normal(size=(2, 4, 3, 2))
    weights, moving_weights = rng.normal(size=(2, 3))
    base = {
        "rho": rng.normal(size=(2, 3)),
        "gradient": rng.normal(size=(2, 3, 3)),
        "tau": rng.normal(size=(2, 3)),
    }
    moving = {key: rng.normal(size=value.shape) for key, value in base.items()}
    base = {key: base[key] for key in ingredients}
    moving = {key: moving[key] for key in ingredients}
    actual = assemble_coefficients_directional(
        jets, moving_jets, base, moving, weights, moving_weights
    )

    def interpolate(value: float, tangent: float, step: object) -> object:
        return mp.mpf(float(value)) + step * mp.mpf(float(tangent))

    for spin in range(2):
        for first in range(2):
            for second in range(2):

                def primal(
                    step: object,
                    spin: int = spin,
                    first: int = first,
                    second: int = second,
                ) -> object:
                    total = mp.mpf(0)
                    for point in range(3):
                        orbitals = [
                            [
                                interpolate(
                                    jets[row, point, column],
                                    moving_jets[row, point, column],
                                    step,
                                )
                                for column in (first, second)
                            ]
                            for row in range(4)
                        ]
                        coefficient = {
                            key: [
                                interpolate(
                                    base[key][spin, point, axis],
                                    moving[key][spin, point, axis],
                                    step,
                                )
                                for axis in range(3)
                            ]
                            if key == "gradient"
                            else interpolate(
                                base[key][spin, point], moving[key][spin, point], step
                            )
                            for key in ingredients
                        }
                        value = coefficient["rho"] * orbitals[0][0] * orbitals[0][1]
                        if "gradient" in ingredients:
                            value += sum(
                                coefficient["gradient"][axis]
                                * (
                                    orbitals[0][0] * orbitals[axis + 1][1]
                                    + orbitals[axis + 1][0] * orbitals[0][1]
                                )
                                for axis in range(3)
                            )
                        if "tau" in ingredients:
                            value += coefficient["tau"] * sum(
                                orbitals[row][0] * orbitals[row][1]
                                for row in range(1, 4)
                            )
                        total += (
                            interpolate(weights[point], moving_weights[point], step)
                            * value
                        )
                    return total

                np.testing.assert_allclose(
                    actual[spin, first, second],
                    float(mp.diff(primal, 0)),
                    atol=3e-14,
                    rtol=3e-14,
                )
    np.testing.assert_allclose(actual, actual.swapaxes(1, 2), atol=0, rtol=0)
    assert np.isfinite(assemble_coefficients(jets, base, weights)).all()
