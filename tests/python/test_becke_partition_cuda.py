"""Real-device primitive gates; opt-in library and finite Slurm step required."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import test_becke_cooperative as retained
from test_becke_partition_primitive import helper
from test_grid_response import CENTERS, DC, POINTS, decimal_partition

__all__ = ["helper"]
Probe = Callable[..., tuple[int, np.ndarray, np.ndarray]]


@pytest.fixture(scope="module")
def probe() -> Probe:
    """Call the candidate ABI, not the ordinary cooperative/phased comparator."""
    path = os.environ.get("GENERATIVEQC_BECKE_PARTITION_PROBE")
    if not path:
        pytest.skip("emitted partition primitive CUDA probe is not configured")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in {
        None,
        "batch",
        "extern",
    }:
        pytest.fail("real-GPU primitive gates require an explicit finite Slurm step")
    library = ct.CDLL(str(Path(path).resolve()))
    pointer = ct.POINTER(ct.c_double)
    library.probe_partition.argtypes = [
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
        ct.POINTER(ct.c_size_t),
    ]
    library.probe_partition.restype = ct.c_int

    def run(
        points: np.ndarray,
        centers: np.ndarray,
        owners: np.ndarray,
        seeds: np.ndarray,
        tile_points: int = 256,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Keep failure publication sentinel checks and actual membership counts."""
        baseline = np.full((len(points), len(centers), 3), 12345.0)
        candidate = np.full_like(baseline, 12345.0)
        counts = np.zeros(len(points), dtype=np.uintp)
        times = np.empty(8)
        as_pointer = lambda array: array.ctypes.data_as(pointer)
        status = library.probe_partition(
            len(centers),
            len(points),
            tile_points,
            as_pointer(centers),
            as_pointer(points),
            owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
            as_pointer(seeds),
            as_pointer(baseline),
            as_pointer(candidate),
            as_pointer(times),
            counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
        )
        if status == 0:
            np.testing.assert_allclose(candidate, baseline, rtol=5e-12, atol=2e-12)
            assert np.all(counts <= len(centers))
        else:
            np.testing.assert_array_equal(baseline, 12345.0)
            np.testing.assert_array_equal(candidate, 12345.0)
        return status, candidate, counts

    return run


@pytest.mark.parametrize("atoms", [1, 2, 24, 33, 48, 96, 128])
@pytest.mark.parametrize("tile_points", [17, 256])
def test_device_changed_geometry_tail_and_host_ad(
    probe: Probe, helper: ct.CDLL, atoms: int, tile_points: int
) -> None:
    if helper.iterations != 3:
        pytest.skip("the configured CUDA probe is specialized to iteration 3")
    rng = np.random.default_rng(1894100 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(257, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    moved = centers + rng.normal(size=centers.shape) * 0.01
    for geometry in (centers, moved, centers):
        expected = retained.run(helper, points, geometry, owners, seeds, True, 0)
        status, actual, _ = probe(points, geometry, owners, seeds, tile_points)
        assert status == expected[0] == 0
        np.testing.assert_allclose(
            actual.sum(axis=0), expected[1], rtol=5e-12, atol=2e-11
        )


@pytest.mark.parametrize(
    "case",
    ["saturated", "rounded_zero", "coincident", "collision", "nonfinite", "nan_seed"],
)
def test_device_zero_and_failure_publication(probe: Probe, case: str) -> None:
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
    status, _, _ = probe(points, centers, owners, seeds)
    assert (status == 0) == (case in {"saturated", "rounded_zero"})


def test_device_independent_decimal_directional_derivative(probe: Probe) -> None:
    owners = np.array([0, 1, 2], dtype=np.int64)
    seeds = np.array([0.3, -0.2, 0.7])
    status, gradient, _ = probe(POINTS, CENTERS, owners, seeds)
    assert status == 0
    with localcontext() as context:
        context.prec = 70
        step = Decimal("1e-16")

        def moved(
            values: np.ndarray, motion: np.ndarray, sign: int
        ) -> list[list[Decimal]]:
            return [
                [
                    Decimal(str(value)) + sign * step * Decimal(str(delta))
                    for value, delta in zip(row, direction, strict=True)
                ]
                for row, direction in zip(values, motion, strict=True)
            ]

        plus, minus = [
            decimal_partition(
                moved(POINTS, DC[owners], sign), moved(CENTERS, DC, sign), 3
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (positive[owner] - negative[owner]) / (2 * step)
            for seed, owner, positive, negative in zip(
                seeds, owners, plus, minus, strict=True
            )
        )
    np.testing.assert_allclose(
        np.einsum("pac,ac->", gradient, DC), float(expected), atol=3e-11, rtol=3e-10
    )
