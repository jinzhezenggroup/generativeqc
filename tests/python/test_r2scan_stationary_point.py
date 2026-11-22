"""Qualify the development continuation, without changing the native SCF policy."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc.boundary import bulk_feature_names
from generativeqc_compiler.xc.r2scan_stationary import emit_r2scan_stationary_point

from tools.generate_libxc_boundary_reference import _configure, evaluate_reference

if TYPE_CHECKING:
    from collections.abc import Callable

_EXCHANGE_CUTOFF = 1e-11
_CORRELATION_CUTOFF = 1e-15
_TAU_FLOOR = 1e-20
_NAMES = bulk_feature_names("mgga", "polarized")
_WIDE_REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "data/xc/r2scan-stationary-tail-reference.json"
    ).read_text()
)


def _reference_energy(library: ctypes.CDLL, point: np.ndarray) -> float:
    """Independent Libxc formula values at the specified work coordinates."""

    def libxc_energy(identifier: int, features: tuple[float, ...]) -> float:
        result = evaluate_reference(
            library,
            {"name": str(identifier), "id": identifier, "family": "mgga"},
            "polarized",
            _NAMES,
            features,
            sigma_threshold=1e-45,
        )
        assert result["oracle_finite"]
        return result["expected"][0]

    def floor(value: float, threshold: float) -> float:
        position = np.clip(value / threshold - 0.5, 0.0, 1.0)
        return max(value, threshold * (1 + 0.5 * position**3 * (2 - position)))

    def activation(value: float, threshold: float) -> float:
        position = np.clip(value / threshold - 1, 0, 1)
        return position**3 * (10 - 15 * position + 6 * position**2)

    densities = point[:2]
    alpha, beta = point[2:5], point[5:8]
    kinetic = tuple(floor(value, _TAU_FLOOR) for value in point[8:])
    energy = 0.0
    for density, gradient, tau in zip(densities, (alpha, beta), kinetic, strict=True):
        if density > _EXCHANGE_CUTOFF:
            exchange = libxc_energy(
                497, (density, 0, gradient @ gradient, 0, 0, 0, 0, tau, 0)
            )
            # Undo Libxc's physical/work total-density ratio for a single spin.
            energy += (
                activation(density, _EXCHANGE_CUTOFF)
                * exchange
                * (1 + _EXCHANGE_CUTOFF / density)
            )
    total = sum(densities)
    if total > _CORRELATION_CUTOFF:
        work_rho = tuple(floor(value, _CORRELATION_CUTOFF) for value in densities)
        # Use the two physical gradients in the upstream interface, including
        # its full-spin and low-kinetic-density continuation.
        correlation = libxc_energy(
            498,
            (*work_rho, alpha @ alpha, alpha @ beta, beta @ beta, 0, 0, *kinetic),
        )
        energy += (
            activation(total, _CORRELATION_CUTOFF) * correlation * total / sum(work_rho)
        )
    return energy


@pytest.fixture(scope="module")
def libxc_library() -> ctypes.CDLL:
    """Use the pinned independent oracle for all fixtures in this module."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("the stationary work oracle is pinned to Libxc 7.0.0")
    _configure(libxc._itrf)
    return libxc._itrf


@pytest.fixture(scope="module", params=("host", "cuda"))
def evaluate_point(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    libxc_library: ctypes.CDLL,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Compile the actual generated point with a verified compiler cache."""
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_R2SCAN_STATIONARY_POINT_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly allocated Slurm GPU")
    if cuda:
        assert os.environ.get("SLURM_JOB_ID")
    compiler = shutil.which("nvcc" if cuda else "c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or launcher is None:
        if cuda:
            pytest.fail(
                "allocated CUDA qualification requires nvcc and a compiler cache"
            )
        pytest.skip("requires a C++ compiler and a compiler cache")
    subprocess.run([launcher, "--version"], check=True, capture_output=True, timeout=10)
    root = tmp_path_factory.mktemp(request.param)
    source = root / ("point.cu" if cuda else "point.cpp")
    qualifier = "__device__ inline" if cuda else "inline"
    body = "\n".join(
        emit_r2scan_stationary_point(with_action=include_action, qualifier=qualifier)
        for include_action in (False, True)
    )
    stores = """
  const auto value = r2scan_stationary_action(point, direction);
  const auto first_order = r2scan_stationary_value(point);
  result[0] = value.energy;
  result[21] = first_order.energy;
  for (unsigned index = 0; index < 10; ++index) {
    result[1 + index] = value.potential[index];
    result[11 + index] = value.action[index];
    result[22 + index] = first_order.potential[index];
  }
"""
    if cuda:
        preamble = "#include <cuda_runtime.h>\n"
        wrapper = (
            """
__global__ void point_kernel(const double* point, const double* direction, double* result) {
"""
            + stores
            + """
}
extern "C" int evaluate(const double* point, const double* direction, double* result) {
  double* buffer = nullptr;
  if (cudaMalloc(&buffer, 52 * sizeof(double)) != cudaSuccess) return 1;
  auto error = cudaMemcpy(buffer, point, 10 * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffer + 10, direction, 10 * sizeof(double), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    point_kernel<<<1, 1>>>(buffer, buffer + 10, buffer + 20);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess)
    error = cudaMemcpy(result, buffer + 20, 32 * sizeof(double), cudaMemcpyDeviceToHost);
  const auto cleanup = cudaFree(buffer);
  return error == cudaSuccess ? cleanup : error;
}
"""
        )
        architecture = os.environ.get(
            "GENERATIVEQC_R2SCAN_STATIONARY_POINT_CUDA_ARCH", "native"
        )
        options = ["-O2", f"-arch={architecture}", "--fmad=false", "-Xcompiler=-fPIC"]
    else:
        preamble = ""
        wrapper = (
            'extern "C" int evaluate(const double* point, const double* direction, double* result) {\n'
            + stores
            + "  return 0;\n}\n"
        )
        options = ["-O1", "-fPIC", "-ffp-contract=off"]
    source.write_text("#include <cmath>\n" + preamble + body + wrapper)
    compiled, binary = root / "point.o", root / "point.so"
    for command in (
        [
            launcher,
            compiler,
            "-std=c++17",
            *options,
            "-c",
            str(source),
            "-o",
            str(compiled),
        ],
        [compiler, "-shared", str(compiled), "-o", str(binary)],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=180, check=False
        )
        assert result.returncode == 0, result.stderr
    evaluate = ctypes.CDLL(str(binary)).evaluate
    evaluate.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")] * 3
    evaluate.restype = ctypes.c_int

    def call(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        result = np.empty(32)
        assert (
            evaluate(
                np.ascontiguousarray(point), np.ascontiguousarray(direction), result
            )
            == 0
        )
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[21:], result[:11], atol=0, rtol=2e-13)
        return result[:21]

    return call


@pytest.mark.parametrize(
    "point,direction",
    (
        pytest.param(
            (0.17, 0.11, 0.025, -0.005, 0.018, 0.13, 0.07, 0.01, 0.13, 0.07),
            (0.02, -0.016, 0.01, -0.002, -0.005, 0.04, -0.02, 0.003, 0.04, -0.02),
            id="interior",
        ),
        pytest.param(
            (4e-10, 1.5e-11, 1e-16, 0, 2e-17, 3e-17, -1e-17, 0, 2e-18, 1e-19),
            (0, 2e-12, 1e-17, 0, -5e-18, -6e-18, 3e-18, 0, 1e-19, 2e-20),
            id="exchange-activation",
        ),
        pytest.param(
            (1.2e-11, 8e-12, 4e-16, 0, 0, 0, 0, 4e-16, 9e-21, 7e-21),
            (0, 0, 0, 0, 0, 0, 0, 0, 4e-21, 5e-21),
            id="kinetic-continuation",
        ),
        pytest.param(
            (0.21, 0, 0.018, 0, 0.02, 0, 0, 0, 0.11, 0),
            (0.02, 0, 0.004, 0, -0.001, 0, 0, 0, -0.025, 0),
            id="empty-spin",
        ),
        pytest.param(
            (0.23, 0.12, 0.125, 0, 0.03125, -0.125, 0, -0.03125, 0.21, 0.11),
            (0.023, -0.01, 0.01, 0.003, 0.007, 0.011, 0.004, 0.013, 0.01, -0.01),
            id="antiparallel-gradient",
        ),
        pytest.param(
            (0.23, 0.12, 0, 0, 0, 0, 0, 0, 0.21, 0.11),
            (0, 0, 0.012, 0.005, 0.007, 0.011, 0.004, 0.013, 0, 0),
            id="flat-gradient",
        ),
    ),
)
def test_stationary_action_matches_libxc_energy_differences(
    evaluate_point: Callable[[np.ndarray, np.ndarray], np.ndarray],
    libxc_library: ctypes.CDLL,
    point: tuple[float, ...],
    direction: tuple[float, ...],
) -> None:
    point = np.array(point, dtype=np.float64)
    direction = np.array(direction, dtype=np.float64)
    energy = _reference_energy(libxc_library, point)
    analytic = evaluate_point(point, direction)
    np.testing.assert_allclose(analytic[0], energy, rtol=5e-11, atol=0)

    derivatives = []
    for step in (0.02, 0.01):
        above = _reference_energy(libxc_library, point + step * direction)
        below = _reference_energy(libxc_library, point - step * direction)
        derivatives.append(
            np.array(
                ((above - below) / (2 * step), (above + below - 2 * energy) / step**2)
            )
        )
    reference = (4 * derivatives[1] - derivatives[0]) / 3
    np.testing.assert_allclose(
        (analytic[1:11] @ direction, analytic[11:] @ direction),
        reference,
        rtol=3e-6,
        atol=5e-12 * abs(energy),
    )

    # Exchange the full physical spin blocks, including tau and the tangent.
    swap = [1, 0, 5, 6, 7, 2, 3, 4, 9, 8]
    swapped = evaluate_point(point[swap], direction[swap])
    np.testing.assert_allclose(swapped[0], analytic[0], rtol=5e-11, atol=0)
    for start in (1, 11):
        np.testing.assert_allclose(
            swapped[start : start + 10],
            analytic[start : start + 10][swap],
            rtol=5e-9,
            atol=5e-12,
        )


def test_stationary_vacuum_is_inert(
    evaluate_point: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    point = np.array([1e-35, *np.zeros(9)])
    # The point is wholly below both activation windows.
    assert not np.any(evaluate_point(point, np.ones(10)))


def test_stationary_correlation_joins_vacuum_through_second_order(
    evaluate_point: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """At the lower join, E, dE, and ddE must vanish cubically/quadratically/linearly."""
    point = np.array([0.0, _CORRELATION_CUTOFF, *np.zeros(6), 1e-20, 1.3e-20])
    direction = np.array([0.0, _CORRELATION_CUTOFF, *np.zeros(8)])
    assert not np.any(evaluate_point(point, direction))
    assert not np.any(evaluate_point(point - direction / 256, direction))
    outside = []
    for scale in (1 / 256, 1 / 512):
        result = evaluate_point(point + direction * scale, direction)
        outside.append(
            np.array((result[0], result[1:11] @ direction, result[11:] @ direction))
        )
    assert np.all(outside[1] != 0)
    np.testing.assert_allclose(outside[0] / outside[1], (8, 4, 2), atol=0, rtol=0.02)


@pytest.mark.parametrize(
    "case", _WIDE_REFERENCE["points"], ids=lambda point: point["label"]
)
def test_stationary_correlation_activation_matches_quad_energy_differences(
    evaluate_point: Callable[[np.ndarray, np.ndarray], np.ndarray],
    case: dict,
) -> None:
    """Use retained independent upstream data where binary64 cancellation dominates."""
    point = np.array(case["inputs"], dtype=np.float64)
    direction = np.array(case["direction"], dtype=np.float64)
    result = evaluate_point(point, direction)
    np.testing.assert_allclose(
        (result[0], result[1:11] @ direction, result[11:] @ direction),
        case["reference"],
        **_WIDE_REFERENCE["oracle"]["tolerance"],
    )
