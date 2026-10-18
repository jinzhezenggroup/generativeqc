"""Exercise the actual FP32 Boys owner against independent high precision."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module", params=("cpu", "cuda", "cuda-no-fma"))
def mixed_boys(request: Any, tmp_path_factory: Any) -> Any:
    """Compile native headers; real-device execution requires explicit Slurm opt-in."""
    cuda = request.param != "cpu"
    if cuda and os.environ.get("GENERATIVEQC_TEST_MIXED_BOYS_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_MIXED_BOYS_CUDA=1 in a Slurm GPU job")
    if cuda and not (
        os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        pytest.fail("CUDA Boys qualification requires Slurm-assigned device visibility")
    compiler = shutil.which("nvcc" if cuda else "c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("native compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    root = Path(__file__).resolve().parents[2]
    folder = tmp_path_factory.mktemp(f"mixed-boys-{request.param}")
    source = r"""
#include "scf/cuda/boys_table.cuh"
using namespace generativeqc::scf::cuda_execution;
template<unsigned Order>
GENERATIVEQC_ERI_DEVICE void evaluate_values(float argument, float* output) {
  MixedPrecisionFloat values[Order + 1];
  boys_values<Order>(MixedPrecisionFloat{argument}, values);
  for (unsigned index = 0; index <= Order; ++index) output[index] = values[index].value;
}
"""
    if cuda:
        source += r"""
template<unsigned Order>
__global__ void evaluate_kernel(const float* arguments, float* output, unsigned count) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) evaluate_values<Order>(arguments[index], output + index * (Order + 1));
}
template<unsigned Order>
int evaluate_order(const float* arguments, float* output, unsigned count) {
  evaluate_kernel<Order><<<(count + 127) / 128, 128>>>(arguments, output, count);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
"""
    else:
        source += r"""
template<unsigned Order>
int evaluate_order(const float* arguments, float* output, unsigned count) {
  for (unsigned index = 0; index < count; ++index)
    evaluate_values<Order>(arguments[index], output + index * (Order + 1));
  return 0;
}
"""
    source += r"""
int dispatch(unsigned order, const float* arguments, float* output, unsigned count) {
  switch (order) {
#define RUN_ORDER(bound) case bound: return evaluate_order<bound>(arguments, output, count);
    RUN_ORDER(0) RUN_ORDER(1) RUN_ORDER(2) RUN_ORDER(3) RUN_ORDER(4)
    RUN_ORDER(5) RUN_ORDER(6) RUN_ORDER(7) RUN_ORDER(8) RUN_ORDER(9)
    RUN_ORDER(10) RUN_ORDER(11) RUN_ORDER(12) RUN_ORDER(13)
  }
  return -1;
}
"""
    if cuda:
        source += r"""
extern "C" int evaluate(unsigned order, const float* arguments, float* output, unsigned count) {
  if (order > 13 || count == 0) return -1;
  float *device_arguments = nullptr, *device_output = nullptr;
  if (cudaMalloc(&device_arguments, count * sizeof(float)) != cudaSuccess) return -1;
  if (cudaMalloc(&device_output, count * (order + 1) * sizeof(float)) != cudaSuccess) {
    cudaFree(device_arguments);
    return -1;
  }
  int result = -1;
  if (cudaMemcpy(device_arguments, arguments, count * sizeof(float), cudaMemcpyHostToDevice)
      == cudaSuccess && dispatch(order, device_arguments, device_output, count) == 0 &&
      cudaMemcpy(output, device_output, count * (order + 1) * sizeof(float),
                 cudaMemcpyDeviceToHost) == cudaSuccess) result = 0;
  cudaFree(device_output);
  cudaFree(device_arguments);
  return result;
}
"""
    else:
        source += r"""
extern "C" int evaluate(unsigned order, const float* arguments, float* output, unsigned count) {
  return count ? dispatch(order, arguments, output, count) : -1;
}
"""
    path = folder / ("probe.cu" if cuda else "probe.cpp")
    path.write_text(source)
    object_file, library = folder / "probe.o", folder / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-c",
            *(
                [
                    "-arch=sm_120",
                    "--fmad=false" if request.param == "cuda-no-fma" else "--fmad=true",
                    "-Xcompiler=-fPIC",
                ]
                if cuda
                else ["-fPIC", "-ffp-contract=off"]
            ),
            f"-I{root / 'src'}",
            f"-I{root / 'include'}",
            str(path),
            "-o",
            str(object_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    subprocess.run(
        [compiler, "-shared", str(object_file), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    function = ctypes.CDLL(str(library)).evaluate
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_uint, pointer, pointer, ctypes.c_uint]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("order", range(14))
def test_mixed_boys_against_incomplete_gamma(mixed_boys: Any, order: int) -> None:
    """Zero, cancellation transitions and float extremes cover every table bound."""
    mp = pytest.importorskip("mpmath")
    boundary = np.float32(order + 1.5)
    arguments = np.unique(
        np.concatenate(
            (
                np.logspace(-40, 38, 180, dtype=np.float32),
                np.linspace(0, order + 3, 120, dtype=np.float32),
                np.array(
                    [
                        np.nextafter(boundary, np.float32(0)),
                        boundary,
                        np.nextafter(boundary, np.float32(np.inf)),
                        np.finfo(np.float32).max,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    )
    actual = np.full((len(arguments), order + 1), np.nan, dtype=np.float32)
    assert mixed_boys(order, arguments, actual, len(arguments)) == 0
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
    subnormal_unit = float(np.nextafter(np.float32(0), np.float32(1)))
    with mp.workdps(80):
        for argument, values in zip(arguments, actual, strict=True):
            precise_argument = mp.mpf(float(argument))
            for index, value in enumerate(values):
                power = mp.mpf(index) + mp.mpf("0.5")
                expected = float(
                    1 / (2 * power)
                    if argument == 0
                    else mp.gammainc(power, 0, precise_argument)
                    / (2 * precise_argument**power)
                )
                assert abs(float(value) - expected) <= max(
                    2e-6 * expected, subnormal_unit
                )


def test_unsupported_bound_fails_without_writing(mixed_boys: Any) -> None:
    arguments = np.array([0.5], dtype=np.float32)
    output = np.full(15, 17, dtype=np.float32)
    assert mixed_boys(14, arguments, output, 1) == -1
    assert np.all(output == 17)


@pytest.mark.parametrize("order", range(1, 14))
def test_mixed_boys_subnormals_before_coulomb_scaling(
    mixed_boys: Any, order: int
) -> None:
    """A small table error must not grow through the established seed factor."""
    mp = pytest.importorskip("mpmath")
    subnormal = float(np.nextafter(np.float32(0), np.float32(1)))
    with mp.workdps(100):
        power = mp.mpf(order) + mp.mpf("0.5")
        arguments = np.unique(
            np.asarray(
                [
                    float(
                        (mp.gamma(power) / (2 * mp.mpf(midpoint + offset) * subnormal))
                        ** (1 / power)
                    )
                    for midpoint in (0.5, 1.5, 2.5, 8.5, 64.5)
                    for offset in (-5e-6, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 5e-6)
                ],
                dtype=np.float32,
            )
        )
        actual = np.full((len(arguments), order + 1), np.nan, dtype=np.float32)
        assert mixed_boys(order, arguments, actual, len(arguments)) == 0
        factor = np.float32((-4.0) ** order)
        for argument, values in zip(arguments, actual, strict=True):
            precise = mp.mpf(float(argument))
            moment = mp.gammainc(power, 0, precise) / (2 * precise**power)
            expected = float(factor * np.float32(float(moment)))
            observed = float(factor * values[order])
            assert abs(observed - expected) <= max(2e-6 * abs(expected), subnormal)
