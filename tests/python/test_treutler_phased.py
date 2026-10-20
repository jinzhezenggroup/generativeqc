"""Independent moving-point oracle and bounded/phased Treutler contracts."""

from __future__ import annotations

import ctypes as ct
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_becke_phased import helper as helper  # noqa: PLC0414
from test_grid_response import CENTERS, DC, POINTS, decimal_partition


def adjusted_run(
    helper: ct.CDLL,
    points: np.ndarray,
    centers: np.ndarray,
    owners: np.ndarray,
    seeds: np.ndarray,
    radii: np.ndarray,
    *,
    cached: bool,
    phased: bool,
) -> tuple[int, np.ndarray, tuple[int, ...]]:
    """Run the same emitted graph with or without retained pair panels."""
    pointer = ct.POINTER(ct.c_double)
    output = np.full_like(centers, 31415.0)
    counts = (ct.c_size_t * 3)()
    status = helper.run_adjusted(
        points.ctypes.data_as(pointer),
        len(points),
        centers.ctypes.data_as(pointer),
        len(centers),
        owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
        seeds.ctypes.data_as(pointer),
        cached,
        phased,
        output.ctypes.data_as(pointer),
        counts,
        radii.ctypes.data_as(pointer),
    )
    return status, output, tuple(counts)


@pytest.mark.parametrize("atoms", [2, 5, 33, 48, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_adjusted_phases_preserve_derivatives_and_pair_work(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    rng = np.random.default_rng(262600 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    owners = rng.integers(atoms, size=17, dtype=np.int64)
    offsets = rng.normal(size=(len(owners), 3)) * 0.4
    seeds = rng.normal(size=len(owners))
    seeds[::3] = 0  # Signed measures and zero cotangents share the same schedule.
    radii = np.resize(np.array([0.35, 0.66, 1.4, 2.6]), atoms)
    for motion in (0, 0.03, 0):
        geometry = centers + motion * np.sin(np.arange(atoms))[:, None]
        points = np.ascontiguousarray(geometry[owners] + offsets)
        arguments = (helper, points, geometry, owners, seeds, radii)
        expected = adjusted_run(*arguments, cached=cached, phased=False)
        actual = adjusted_run(*arguments, cached=cached, phased=True)
        assert actual[0] == expected[0] == 0
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)
        np.testing.assert_allclose(actual[1].sum(axis=0), 0, atol=3e-13)
        assert actual[2][0] == len(points) * atoms * (atoms - 1) // 2
        assert expected[2][0] == 2 * actual[2][0]


@pytest.mark.parametrize("radii", [(0.66, 1.14, 2.55), (0.01, 1.0, 100.0)])
def test_adjusted_phases_match_independent_decimal_motion(
    helper: ct.CDLL, radii: tuple[float, ...]
) -> None:
    """Check chain rule, radius caps, moving owners and orientation reversal."""
    owners = np.array([0, 1, 2], dtype=np.int64)
    seeds = np.array([0.3, -0.2, 0.7])
    radii = np.array(radii)
    status, gradient, _ = adjusted_run(
        helper, POINTS, CENTERS, owners, seeds, radii, cached=True, phased=True
    )
    assert status == 0
    with localcontext() as context:
        context.prec = 65
        step = Decimal("1e-18")

        def displaced(
            values: np.ndarray, motion: np.ndarray, sign: int
        ) -> list[list[Decimal]]:
            return [
                [
                    Decimal(str(v)) + sign * step * Decimal(str(d))
                    for v, d in zip(row, delta, strict=True)
                ]
                for row, delta in zip(values, motion, strict=True)
            ]

        upper, lower = [
            decimal_partition(
                displaced(POINTS, DC[owners], sign),
                displaced(CENTERS, DC, sign),
                helper.iterations,
                [Decimal(str(r)) for r in radii],
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (a[owner] - b[owner]) / (2 * step)
            for seed, a, b, owner in zip(seeds, upper, lower, owners, strict=True)
        )
    assert np.sum(gradient * DC) == pytest.approx(float(expected), abs=3e-14)
    permutation = np.array([2, 0, 1])
    status, permuted, _ = adjusted_run(
        helper,
        POINTS,
        CENTERS[permutation],
        np.argsort(permutation)[owners],
        seeds,
        radii[permutation],
        cached=True,
        phased=True,
    )
    assert status == 0
    np.testing.assert_allclose(permuted, gradient[permutation], rtol=5e-13, atol=2e-14)


@pytest.mark.parametrize("case", ["saturated", "rounded_zero", "collision", "nan_seed"])
def test_adjusted_phase_edge_publication(helper: ct.CDLL, case: str) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    owners = np.array([0, 1], dtype=np.int64)
    seeds = np.array([0.3, -0.7])
    if case == "rounded_zero":
        points[0, 1], owners[0], seeds[0] = 2e-5, 1, 1e12
    elif case == "collision":
        points[1] = centers[0]
    elif case == "nan_seed":
        seeds[0] = np.nan
    args = (helper, points, centers, owners, seeds, np.array([0.35, 0.66, 2.6]))
    expected = adjusted_run(*args, cached=True, phased=False)
    actual = adjusted_run(*args, cached=True, phased=True)
    if case == "nan_seed":
        # The raw serial core assumes its caller validated the seed. The
        # phased normalization boundary validates before any publication.
        assert actual[0] != 0
    else:
        assert actual[0] == expected[0]
    if actual[0]:
        np.testing.assert_array_equal(actual[1], 31415.0)
    else:
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)
