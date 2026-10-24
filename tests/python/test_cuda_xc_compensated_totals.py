"""Independent sum oracles for the actual emitted, bounded XC total reduction."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.xc_totals_cuda import emit_compensated_xc_totals

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def totals_probe(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile production reduction emission through a verified compiler cache."""
    assert os.environ.get("SLURM_JOB_ID")
    cupy = pytest.importorskip("cupy")
    compiler = shutil.which("nvcc")
    cache = shutil.which("ccache") or shutil.which("sccache")
    assert compiler and cache, (
        "CUDA totals qualification needs nvcc and a compiler cache"
    )
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("cuda-xc-compensated-totals")
    source = directory / "probe.cu"
    source.write_text(
        "#include <cuda_runtime.h>\n#include <cmath>\nusing I = long long;\n"
        "__device__ double finite(double value, int* error, int code) {\n"
        "  if (!isfinite(value)) atomicExch(error, code);\n"
        "  return value;\n}\n"
        + emit_compensated_xc_totals()
        + '\nextern "C" int reduce_batch(const double* points, I count, I tile_points,\n'
        "    double* totals, int* error, cudaStream_t stream) {\n"
        "  accumulate_compensated_xc_totals<<<1,32,0,stream>>>(\n"
        "      points,count,tile_points,totals,error);\n"
        "  return cudaGetLastError();\n}\n"
        'extern "C" int fold_totals(double* totals, int* error, cudaStream_t stream) {\n'
        "  fold_compensated_xc_totals<<<1,32,0,stream>>>(totals,error);\n"
        "  return cudaGetLastError();\n}\n"
    )
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    architecture = f"sm_{properties['major']}{properties['minor']}"
    library_path = directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-O3",
            "-std=c++17",
            f"-arch={architecture}",
            "-shared",
            "-Xcompiler=-fPIC",
            str(source),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    library = ctypes.CDLL(str(library_path))
    library.reduce_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.reduce_batch.restype = ctypes.c_int
    library.fold_totals.argtypes = [ctypes.c_void_p] * 3
    library.fold_totals.restype = ctypes.c_int
    return cupy, library


def reduce_channels(
    probe: Any, values: np.ndarray, *, tile_points: int, batch_tiles: int
) -> tuple[np.ndarray, int]:
    """Retain low words across independently packed, partial final batches."""
    cupy, library = probe
    totals = cupy.zeros(6, dtype=cupy.float64)
    error = cupy.zeros(1, dtype=cupy.int32)
    batch_size = tile_points * batch_tiles
    for begin in range(0, max(1, values.shape[1]), batch_size):
        batch = values[:, begin : begin + batch_size]
        tiles = [
            batch[:, offset : offset + tile_points].ravel()
            for offset in range(0, batch.shape[1], tile_points)
        ]
        packed = np.concatenate(tiles) if tiles else np.empty(0, dtype=np.float64)
        device_points = cupy.asarray(packed)
        status = library.reduce_batch(
            device_points.data.ptr,
            batch.shape[1],
            tile_points,
            totals.data.ptr,
            error.data.ptr,
            cupy.cuda.Stream.null.ptr,
        )
        assert status == 0
        cupy.cuda.Stream.null.synchronize()
    assert (
        library.fold_totals(totals.data.ptr, error.data.ptr, cupy.cuda.Stream.null.ptr)
        == 0
    )
    cupy.cuda.Stream.null.synchronize()
    return totals.get()[:3], int(error.get()[0])


@pytest.mark.parametrize("count", (0, 1, 31, 32, 33, 255, 256, 257, 4097))
@pytest.mark.parametrize("batch_tiles", (1, 7))
def test_compensated_totals_match_independent_fsum(
    totals_probe: Any, count: int, batch_tiles: int
) -> None:
    """Every channel, tile tail and batch boundary retains its mathematical sum."""
    random = np.random.default_rng(2507 + count)
    values = random.uniform(0.0, 1.0, (3, count))
    values[0] *= -1
    expected = np.asarray([math.fsum(channel) for channel in values])
    actual, error = reduce_channels(
        totals_probe, values, tile_points=32, batch_tiles=batch_tiles
    )
    assert error == 0
    np.testing.assert_array_max_ulp(actual, expected, maxulp=2)


def test_low_words_survive_cross_tile_and_cross_batch_cancellation(
    totals_probe: Any,
) -> None:
    """A closed-form dyadic answer cannot be obtained by discarding tile tails."""
    values = np.tile(np.asarray([2.0**60, 1.0, -(2.0**60), 0.25]), 1025)
    channels = np.stack((values, -values, values / 2))
    actual, error = reduce_channels(
        totals_probe, channels, tile_points=31, batch_tiles=3
    )
    assert error == 0
    np.testing.assert_array_equal(actual, [1281.25, -1281.25, 640.625])


def test_complete_large_grid_totals_match_fsum(totals_probe: Any) -> None:
    """Exercise the 96-atom grid census without an O(grid) retained device state."""
    random = np.random.default_rng(2507)
    values = random.uniform(0.0, 1.0, (3, 2359296)) / 8192
    values[0] *= -2
    expected = np.asarray([math.fsum(channel) for channel in values])
    actual, error = reduce_channels(
        totals_probe, values, tile_points=256, batch_tiles=17
    )
    assert error == 0
    np.testing.assert_array_max_ulp(actual, expected, maxulp=2)


@pytest.mark.parametrize("invalid", (math.inf, -math.inf, math.nan))
def test_nonfinite_point_totals_remain_audited(
    totals_probe: Any, invalid: float
) -> None:
    values = np.ones((3, 33), dtype=np.float64)
    values[0, -1] = invalid
    actual, error = reduce_channels(totals_probe, values, tile_points=32, batch_tiles=1)
    assert error == 3
    assert not math.isfinite(actual[0])
