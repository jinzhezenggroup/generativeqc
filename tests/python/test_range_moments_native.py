"""Native modified moments against independent adaptive interval quadrature."""

import ctypes
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.range_separation import (
    CoulombKernel,
    reference_moments,
)


@pytest.fixture(scope="module", params=("cpu", "cuda", "cuda-no-fma"))
def native_moments(request: typing.Any, tmp_path_factory: typing.Any) -> typing.Any:
    """Compile identical host/device arithmetic; CUDA execution is opt-in Slurm."""
    pytest.importorskip("scipy")
    cuda = request.param != "cpu"
    if cuda and os.environ.get("GENERATIVEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    compiler = shutil.which("nvcc" if cuda else "c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("native compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    folder = tmp_path_factory.mktemp(f"range-moments-{request.param}")
    source = '#include "integrals/range_moments.hpp"\n'
    if cuda:
        source += r"""
#include <cuda_runtime.h>
__global__ void kernel(bool second_order, unsigned order, double t, double rho, unsigned range,
                       double omega, double* out) {
  const auto radial = static_cast<generativeqc::integrals::CoulombRange>(range);
  out[15] = second_order
      ? generativeqc::integrals::bounded_range_moments<14>(order, t, rho, radial, omega, out)
      : generativeqc::integrals::range_moments(order, t, rho, radial, omega, out);
}
int evaluate_bound(bool second_order, unsigned order, double t, double rho, unsigned range,
                   double omega, double* out) {
  double* device = nullptr;
  if (cudaMalloc(&device, 16 * sizeof(double)) != cudaSuccess) return -1;
  auto status = cudaMemcpy(device, out, 16 * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    kernel<<<1,1>>>(second_order, order, t, rho, range, omega, device);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(out, device, 16 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(device);
  return status == cudaSuccess ? static_cast<int>(out[15]) : -1;
}
"""
    else:
        source += r"""
int evaluate_bound(bool second_order, unsigned order, double t, double rho, unsigned range,
                   double omega, double* out) {
  const auto radial = static_cast<generativeqc::integrals::CoulombRange>(range);
  return second_order
      ? generativeqc::integrals::bounded_range_moments<14>(order, t, rho, radial, omega, out)
      : generativeqc::integrals::range_moments(order, t, rho, radial, omega, out);
}
"""
    source += r"""
extern "C" int evaluate(unsigned order, double t, double rho, unsigned range,
                        double omega, double* out) {
  return evaluate_bound(false, order, t, rho, range, omega, out);
}
extern "C" int evaluate_second(unsigned order, double t, double rho, unsigned range,
                               double omega, double* out) {
  return evaluate_bound(true, order, t, rho, range, omega, out);
}
"""
    path = folder / ("probe.cu" if cuda else "probe.cpp")
    path.write_text(source)
    library = folder / "probe.so"
    object_file = folder / "probe.o"
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-O3",
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
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
            str(path),
            "-o",
            str(object_file),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(Path(__file__).resolve().parents[2])},
    )
    subprocess.run(
        [compiler, "-shared", str(object_file), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    owner = ctypes.CDLL(str(library))
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    owner.evaluate.restype = ctypes.c_int
    owner.evaluate_second.argtypes = owner.evaluate.argtypes
    owner.evaluate_second.restype = ctypes.c_int
    owner.evaluate.second_order = owner.evaluate_second
    return owner.evaluate


@pytest.mark.parametrize("family", ["long_range", "short_range"])
@pytest.mark.parametrize("rho", [1e-12, 0.73, 1e12])
def test_native_moments_resolve_limits_and_separate_parts(
    native_moments: typing.Any, family: typing.Any, rho: typing.Any
) -> None:
    """All orders retain relative accuracy, including tiny positive SR values."""
    tag = 1 if family == "long_range" else 2
    for omega in (0, 1e-12, 0.2, 1, 1e6, 1e150):
        for argument in (0, 1e-12, 0.1, 10, 100, 700, 1e6, 1e150, 1e300):
            actual = np.full(16, 123.0)
            assert native_moments(13, argument, rho, tag, omega, actual) == 1
            expected = reference_moments(
                13, argument, rho, CoulombKernel(family, omega)
            )
            np.testing.assert_allclose(
                actual[:14],
                expected,
                atol=1e-320,
                rtol=2e-12,
                err_msg=f"{family}, omega={omega}, T={argument}, rho={rho}",
            )
            assert np.all(actual[:14] >= 0)


def test_native_moment_controls_fail_without_touching_output(
    native_moments: typing.Any,
) -> None:
    cases = [
        (14, 1, 1, 1, 0.5),
        (0, float("nan"), 1, 1, 0.5),
        (0, float("inf"), 1, 1, 0.5),
        (0, -1, 1, 1, 0.5),
        (0, 1, 0, 1, 0.5),
        (0, 1, float("inf"), 1, 0.5),
        (0, 1, 1, 3, 0.5),
        (0, 1, 1, 0, 0.5),
        (0, 1, 1, 1, -1),
        (0, 1, 1, 1, float("nan")),
        (0, 1, 1, 1, float("inf")),
    ]
    for controls in cases:
        actual = np.full(16, 123.0)
        assert native_moments(*controls, actual) == 0
        np.testing.assert_array_equal(actual[:14], 123)


def test_native_moments_derivative_chain_and_complement(
    native_moments: typing.Any,
) -> None:
    values = []
    for tag in (0, 1, 2):
        omega = 0 if tag == 0 else 0.8
        output = np.zeros(16)
        assert native_moments(13, 1.3, 0.71, tag, omega, output) == 1
        values.append(output[:14].copy())
        errors = []
        for step in (0.01, 0.003, 0.001):
            plus, minus = np.zeros(16), np.zeros(16)
            assert native_moments(13, 1.3 + step, 0.71, tag, omega, plus) == 1
            assert native_moments(13, 1.3 - step, 0.71, tag, omega, minus) == 1
            errors.append(
                np.max(np.abs((plus[:13] - minus[:13]) / (2 * step) + output[1:14]))
            )
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < 2e-8
    np.testing.assert_allclose(values[1] + values[2], values[0], rtol=2e-13, atol=0)


@pytest.mark.parametrize("order", range(15))
def test_long_moments_branch_boundaries_and_extremes_against_gamma(
    native_moments: typing.Any, order: int
) -> None:
    """Independent 80-digit incomplete gamma covers every bound through f Hessians."""
    mp = pytest.importorskip("mpmath")
    function = native_moments.second_order if order == 14 else native_moments
    transition = order + 1.5
    cases = []
    for rho, omega in ((0.73, 0.03), (1.0, 0.3), (1e-12, 1e6)):
        boundary = omega / np.hypot(omega, np.sqrt(rho))
        for scaled in (
            0.0,
            1e-12,
            0.1,
            np.nextafter(transition, 0.0),
            transition,
            np.nextafter(transition, np.inf),
            100.0,
            1e6,
        ):
            cases.append((float(scaled / boundary**2), rho, omega))
    cases.extend(
        (
            (1e300, 1.0, 1e-160),  # T*a*a is normal although a*a is subnormal.
            (1e308, 1.0, 1e-200),  # Squaring a first would incorrectly give zero.
            (1e308, 1.0, 1e150),
            (0.0, 1e300, 1e-160),
            (1e-300, 1e-300, 1e-150),
        )
    )
    with mp.workdps(80):
        for argument, rho, omega in cases:
            actual = np.full(16, 123.0)
            assert function(order, argument, rho, 1, omega, actual) == 1
            t, r, w = mp.mpf(argument), mp.mpf(rho), mp.mpf(omega)
            a = w / mp.sqrt(w * w + r)
            expected = []
            for n in range(order + 1):
                power = mp.mpf(n) + mp.mpf("0.5")
                moment = (
                    a ** (2 * n + 1) / (2 * n + 1)
                    if t == 0
                    else mp.gammainc(power, 0, t * a * a) / (2 * t**power)
                )
                expected.append(float(moment))
            np.testing.assert_allclose(
                actual[: order + 1],
                expected,
                rtol=1e-13,
                atol=2e-323,
                err_msg=f"order={order}, T={argument}, rho={rho}, omega={omega}",
            )
            assert np.all(actual[: order + 1] >= 0)
            np.testing.assert_array_equal(actual[order + 1 : 15], 123.0)


def test_second_moment_bound_preserves_caller_storage(
    native_moments: typing.Any,
) -> None:
    """The fifteen-value opt-in must reject a sixteenth moment before any writes."""
    actual = np.full(16, 123.0)
    assert native_moments.second_order(15, 1.0, 1.0, 1, 0.3, actual) == 0
    np.testing.assert_array_equal(actual[:15], 123.0)
