"""Primal-only Gaussian/density oracles for sparse spatial/nuclear AD."""

import itertools
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.features import contract_density_response

DIRECTIONS = tuple(itertools.product(("point", "center"), range(3)))


@pytest.fixture(scope="module")
def gaussian_density() -> Any:
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 70
    centers = (("0.1", "-0.2", "0.3"), ("1.2", "0.7", "-0.4"), ("-0.6", "0.9", "0.2"))
    exponents = tuple(map(mp.mpf, (".7", ".23", "1.1")))
    point = tuple(map(mp.mpf, (".31", "-.17", ".43")))
    owned = np.array((True, False, True))
    density = np.array(((-1.2, 0.3, -0.15), (0.3, 0.7, 0.09), (-0.15, 0.09, 0.2)))

    def orbital(index: int, coordinates: Any, center_motion: Any) -> Any:
        return mp.exp(
            -exponents[index]
            * sum(
                (coordinate - mp.mpf(center) - (motion if owned[index] else 0)) ** 2
                for coordinate, center, motion in zip(
                    coordinates, centers[index], center_motion, strict=True
                )
            )
        )

    jets = {
        index: np.array(
            [
                [
                    float(
                        mp.diff(
                            lambda *coordinates, orbital_index=orbital_index: orbital(
                                orbital_index, coordinates, (0, 0, 0)
                            ),
                            point,
                            index,
                        )
                    )
                    for orbital_index in range(3)
                ]
            ]
        )
        for index in jet_indices(2)
    }
    products = {
        (index, selected): (values * owned if selected else values) @ density
        for index, values in jets.items()
        for selected in (False, True)
    }
    return mp, point, owned, density, orbital, jets, products


@pytest.mark.parametrize(
    "directions",
    [
        (),
        *[(direction,) for direction in DIRECTIONS],
        *itertools.product(DIRECTIONS, repeat=2),
    ],
)
def test_density_spatial_and_center_terms_against_primal_only_gaussians(
    gaussian_density: Any, directions: Any
) -> None:
    mp, point, owned, density, orbital, jets, products = gaussian_density

    def primal(*steps: Any) -> Any:
        coordinates = list(point)
        center_motion = [mp.mpf(0)] * 3
        for step, (kind, axis) in zip(steps, directions, strict=True):
            if kind == "point":
                coordinates[axis] += step
            else:
                center_motion[axis] += step
        values = [orbital(index, coordinates, center_motion) for index in range(3)]
        return sum(
            mp.mpf(float(density[first, second])) * values[first] * values[second]
            for first in range(3)
            for second in range(3)
        )

    expected = (
        mp.diff(primal, (0,) * len(directions), (1,) * len(directions))
        if directions
        else primal()
    )
    actual = contract_density_response(jets, products, directions, owned=owned)
    np.testing.assert_allclose(actual, float(expected), atol=3e-15, rtol=2e-13)


def test_density_hessian_factors_preserve_finite_cancellation() -> None:
    jets = {
        (0, 0, 0): np.array([[0.0, 1.0]]),
        (1, 0, 0): np.array([[1e154, 0.0]]),
        (2, 0, 0): np.array([[0.0, 1e308]]),
    }
    density = np.diag((-1.0, 1.0))
    products = {
        (index, False): values @ density
        for index, values in jets.items()
        if sum(index) < 2
    }
    with np.errstate(all="raise"):
        actual = contract_density_response(jets, products, (("point", 0), ("point", 0)))
    np.testing.assert_array_equal(actual, np.zeros(1))
