"""Exercise the dense coefficient ABI with the independent existing device gates."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from test_becke_pair_coefficients import helper
from test_becke_partition_cuda import (
    test_device_changed_geometry_tail_and_host_ad,
    test_device_independent_decimal_directional_derivative,
    test_device_zero_and_failure_publication,
)

if TYPE_CHECKING:
    from test_becke_partition_cuda import Probe

__all__ = [
    "helper",
    "test_device_changed_geometry_tail_and_host_ad",
    "test_device_independent_decimal_directional_derivative",
    "test_device_zero_and_failure_publication",
]


@pytest.fixture(scope="module")
def probe() -> Probe:
    """Call actual coefficient kernels; reported counts are the dense contract."""
    path = os.environ.get("GENERATIVEQC_BECKE_COEFFICIENT_PROBE")
    if not path:
        pytest.skip("emitted coefficient CUDA probe is not configured")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") in {
        None,
        "batch",
        "extern",
    }:
        pytest.fail("real-GPU coefficient gates require an explicit finite Slurm step")
    library = ct.CDLL(str(Path(path).resolve()))
    pointer = ct.POINTER(ct.c_double)
    library.probe_coefficients.argtypes = [
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
    library.probe_coefficients.restype = ct.c_int

    def run(
        points: np.ndarray,
        centers: np.ndarray,
        owners: np.ndarray,
        seeds: np.ndarray,
        tile_points: int = 256,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        baseline = np.full((len(points), len(centers), 3), 12345.0)
        candidate = np.full_like(baseline, 12345.0)
        times = np.empty(8)
        as_pointer = lambda array: array.ctypes.data_as(pointer)
        status = library.probe_coefficients(
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
        )
        if status == 0:
            np.testing.assert_array_equal(candidate, baseline)
        else:
            np.testing.assert_array_equal(baseline, 12345.0)
            np.testing.assert_array_equal(candidate, 12345.0)
        return status, candidate, np.full(len(points), len(centers), dtype=np.uintp)

    return run
