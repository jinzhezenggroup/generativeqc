"""Opt-in CUDA qualification of the emitted phase probe, always under Slurm."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import test_becke_cooperative as retained
from test_becke_phased import helper
from test_grid_response import CENTERS, DC, POINTS, decimal_partition

__all__ = ["helper"]
Probe = Callable[..., tuple[int, np.ndarray]]


@pytest.fixture(scope="module")
def probe() -> Probe:
    path = os.environ.get("GENERATIVEQC_BECKE_PHASED_PROBE")
    if not path:
        pytest.skip("isolated emitted CUDA probe is not configured")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("real-GPU qualification requires a finite Slurm allocation")
    library = ct.CDLL(str(Path(path).resolve()))
    pointer = ct.POINTER(ct.c_double)
    library.probe.argtypes = [
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        pointer,
        pointer,
        ct.POINTER(ct.c_int64),
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    library.probe.restype = ct.c_int

    def run(
        points: np.ndarray,
        centers: np.ndarray,
        owners: np.ndarray,
        seeds: np.ndarray,
        tile_points: int = 256,
    ) -> tuple[int, np.ndarray]:
        old = np.full((len(points), len(centers), 3), 12345.0)
        new = np.full_like(old, 12345.0)
        times = np.empty(8)
        as_pointer = lambda array: array.ctypes.data_as(pointer)
        status = library.probe(
            len(centers),
            len(points),
            tile_points,
            as_pointer(centers),
            as_pointer(points),
            owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
            as_pointer(seeds),
            as_pointer(old),
            as_pointer(new),
            as_pointer(times),
        )
        if status == 0:
            np.testing.assert_allclose(new, old, rtol=5e-12, atol=2e-12)
        else:
            np.testing.assert_array_equal(old, 12345.0)
            np.testing.assert_array_equal(new, 12345.0)
        return status, new

    return run


@pytest.mark.parametrize("atoms", [1, 2, 24, 33, 48, 96, 128])
@pytest.mark.parametrize("tile_points", [17, 256])
def test_cuda_moved_geometry_tail_and_host_gate(
    probe: Probe, helper: ct.CDLL, atoms: int, tile_points: int
) -> None:
    if helper.iterations != 3:
        pytest.skip("CUDA probe is specialized to three partition iterations")
    rng = np.random.default_rng(1010400 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(257, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    moved = centers + rng.normal(size=centers.shape) * 0.01
    for geometry in (centers, moved, centers):
        expected = retained.run(helper, points, geometry, owners, seeds, True, 0)
        status, actual = probe(points, geometry, owners, seeds, tile_points)
        assert status == expected[0] == 0
        np.testing.assert_allclose(
            actual.sum(axis=0), expected[1], rtol=5e-12, atol=2e-11
        )


@pytest.mark.parametrize(
    "case",
    ["saturated", "rounded_zero", "coincident", "collision", "nonfinite", "nan_seed"],
)
def test_cuda_failure_publication_and_zero_semantics(probe: Probe, case: str) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    owners = np.array([0, 1], dtype=np.int64)
    seeds = np.array([0.3, -0.7])
    if case == "rounded_zero":
        points[0, 1], owners[0], seeds[0] = 2e-5, 1, 1e12
    elif case == "coincident":
        centers[1] = centers[0]
    elif case == "collision":
        points[-1] = centers[0]
    elif case == "nonfinite":
        points[-1, 1] = np.inf
    elif case == "nan_seed":
        seeds[0] = np.nan
    status, _ = probe(points, centers, owners, seeds)
    assert (status == 0) == (case in {"saturated", "rounded_zero"})


def test_cuda_independent_decimal_directional_derivative(probe: Probe) -> None:
    owners = np.array([0, 1, 2], dtype=np.int64)
    seeds = np.array([0.3, -0.2, 0.7])
    status, gradient = probe(POINTS, CENTERS, owners, seeds)
    assert status == 0
    with localcontext() as context:
        context.prec = 60
        step = Decimal("1e-16")

        def moved(
            values: np.ndarray, motion: np.ndarray, sign: int
        ) -> list[list[Decimal]]:
            return [
                [
                    Decimal(str(value)) + sign * step * Decimal(str(delta))
                    for value, delta in zip(row, direction)
                ]
                for row, direction in zip(values, motion)
            ]

        plus, minus = [
            decimal_partition(
                moved(POINTS, DC[owners], sign), moved(CENTERS, DC, sign), 3
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (upper[owner] - lower[owner]) / (2 * step)
            for seed, upper, lower, owner in zip(seeds, plus, minus, owners)
        )
    assert abs(float(expected) - np.sum(gradient.sum(axis=0) * DC)) < 2e-14
