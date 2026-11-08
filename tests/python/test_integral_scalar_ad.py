"""Primal-only complex-step and radial-quadrature gates for scalar schedules."""

import ctypes
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.scalar_ad import emit_scalar_ad_header
from generativeqc_compiler.xc.point_ad import emit_point_ad_header

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=("host", "cuda"))
def scalar_probe(native_cxx: Any, tmp_path_factory: Any, request: Any) -> Any:
    """Exercise the actual native Dual3 adapters against freshly emitted code."""
    directory = tmp_path_factory.mktemp("integral-scalar-ad")
    (directory / "integrals").mkdir()
    (directory / "integrals/generated_scalar_ad.hpp").write_text(
        emit_scalar_ad_header()
    )
    (directory / "dft").mkdir()
    (directory / "dft/generated_point_ad.hpp").write_text(emit_point_ad_header())
    compiler, qualifier = native_cxx, ""
    flags = ("-std=c++20", "-O3", "-ffp-contract=off")
    cuda = request.param == "cuda"
    if cuda:
        if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
            pytest.skip("scheduled CUDA qualification is opt-in")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("device qualification requires finite srun")
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.fail("scheduled CUDA qualification requires nvcc")
        compiler = replace(native_cxx, compiler=nvcc)
        qualifier = "__device__ "
        flags = (
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O3",
            "--fmad=false",
            "-arch=native",
        )
    body = r"""
#include "scf/cuda/scalar_math.cuh"
#include "integrals/range_moments.hpp"
#include "dft/generated_point_ad.hpp"
@QUALIFIER@ void evaluate(int kind, const double* input, double* output) {
  using namespace generativeqc::scf::cuda_execution;
  if (kind < 0) {
    using namespace generativeqc::integrals::generated_scalar_ad;
    switch (-1 - kind) {
      case 0: output[0] = Add::primal(input[0], input[1]); break;
      case 1: output[0] = Subtract::primal(input[0], input[1]); break;
      case 2: output[0] = Multiply::primal(input[0], input[1]); break;
      case 3: output[0] = Divide::primal(input[0], input[1]); break;
      case 4: output[0] = Exp::primal(input[0]); break;
      case 5: output[0] = Sqrt::primal(input[0]); break;
      case 6: output[0] = Erf::primal(input[0]); break;
    }
    output[1] = output[2] = output[3] = 0;
    return;
  }
  if (kind >= 10) {
    double energy = input[1], derivative[7];
    for (unsigned index = 0; index < 7; ++index) derivative[index] = input[2 + index];
    generativeqc::dft::point::detail::apply_r2scan_tail(input[0], energy, derivative);
    output[0] = kind == 10 ? energy : derivative[3];
    for (unsigned index = 1; index < 4; ++index)
      output[index] = derivative[kind == 10 ? index - 1 : index + 3];
    return;
  }
  const Dual3 left{input[0], input[2], input[3], input[4]};
  const Dual3 right{input[1], input[5], input[6], input[7]};
  Dual3 result{};
  switch (kind) {
    case 0: result = left + right; break;
    case 1: result = left - right; break;
    case 2: result = left * right; break;
    case 3: result = left / right; break;
    case 4: result = qexp(left); break;
    case 5: result = qsqrt(left); break;
    case 6: result = qerf(left); break;
    default: {
      const unsigned order = static_cast<unsigned>(input[1]);
      double moments[16]{};
      if (!generativeqc::integrals::range_moments(order + 1, input[0], input[8],
          static_cast<generativeqc::integrals::CoulombRange>(kind - 7), input[9], moments)) {
        output[0] = NAN; return;
      }
      using generativeqc::integrals::generated_scalar_ad::moment_tangent;
      result = {moments[order], moment_tangent(moments[order + 1], input[2]),
                moment_tangent(moments[order + 1], input[3]),
                moment_tangent(moments[order + 1], input[4])};
    }
  }
  output[0] = result.value;
  output[1] = result.derivative_x;
  output[2] = result.derivative_y;
  output[3] = result.derivative_z;
}
""".replace("@QUALIFIER@", qualifier)
    if cuda:
        body += r"""
__global__ void run(int kind, const double* input, double* output) { evaluate(kind, input, output); }
extern "C" int probe(int kind, const double* input, double* output) {
  double* device{};
  if (cudaMalloc(&device, 14 * sizeof(double)) != cudaSuccess) return 1;
  int status = 1;
  if (cudaMemcpy(device, input, 10 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess) {
    run<<<1,1>>>(kind, device, device + 10);
    if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, device + 10, 4 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess)
      status = 0;
  }
  if (cudaFree(device) != cudaSuccess) status = 1;
  return status;
}
"""
    else:
        body += 'extern "C" int probe(int kind, const double* input, double* output) { evaluate(kind, input, output); return 0; }\n'
    source = directory / ("probe.cu" if cuda else "probe.cpp")
    source.write_text(body)
    library = directory / "probe.so"
    compiler.build_shared(
        [source], library, compile_args=(*flags, f"-I{directory}", f"-I{ROOT / 'src'}")
    )
    loaded = ctypes.CDLL(str(library))
    function = loaded.probe
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = (ctypes.c_int, pointer, pointer)
    function.restype = ctypes.c_int

    def evaluate(kind: int, inputs: np.ndarray) -> np.ndarray:
        output = np.empty(4)
        assert function(kind, np.ascontiguousarray(inputs), output) == 0
        return output

    return evaluate


@pytest.mark.parametrize("kind", range(7))
def test_native_channels_against_complex_primal(scalar_probe: Any, kind: int) -> None:
    """No derivative formulas are used by the independent point oracle."""
    special = pytest.importorskip("scipy.special")
    operations = (
        lambda left, right: left + right,
        lambda left, right: left - right,
        lambda left, right: left * right,
        lambda left, right: left / right,
        lambda left, right: np.exp(left),
        lambda left, right: np.sqrt(left),
        lambda left, right: special.erf(left),
    )
    generator = np.random.default_rng(20261107)
    for _ in range(80):
        inputs = generator.normal(size=10)
        inputs[:2] = np.exp(inputs[:2] * 2)
        actual = scalar_probe(kind, inputs)
        value_only = scalar_probe(-1 - kind, inputs)
        assert value_only[0] == actual[0]
        assert np.array_equal(value_only[1:], np.zeros(3))
        expected = [operations[kind](*inputs[:2])]
        for axis in range(3):
            perturbed = inputs[:2].astype(complex)
            perturbed += 1e-25j * inputs[[2 + axis, 5 + axis]]
            expected.append(operations[kind](*perturbed).imag / 1e-25)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_erf_extreme_finite_tangents_against_high_precision_primal(
    scalar_probe: Any,
) -> None:
    """Finite tangent seeds must not overflow before a small erf slope acts."""
    mp = pytest.importorskip("mpmath").mp.clone()
    mp.dps = 100
    limits = np.finfo(np.float64)
    for value in (-10.0, -0.5, 0.0, 0.5, 10.0):
        slope = mp.diff(mp.erf, mp.mpf(value))
        for magnitude in (limits.max, limits.tiny, np.nextafter(0.0, 1.0)):
            inputs = np.zeros(10)
            inputs[0] = value
            inputs[2:5] = (magnitude, -magnitude, 0.0)
            actual = scalar_probe(6, inputs)
            expected = np.array(
                [float(slope * mp.mpf(float(seed))) for seed in inputs[2:5]]
            )
            np.testing.assert_allclose(
                actual[1:], expected, rtol=4e-14, atol=2 * np.nextafter(0.0, 1.0)
            )


def test_primal_only_does_not_require_finite_tangent_invariants(
    scalar_probe: Any,
) -> None:
    """A value-only recurrence never evaluates singular or overflowed slopes."""
    inputs = np.zeros(10)
    inputs[:2] = (1e-310, 1e-310)
    assert scalar_probe(-4, inputs)[0] == 1
    inputs[0] = 0
    assert scalar_probe(-6, inputs)[0] == 0


@pytest.mark.parametrize("radial", range(3))
def test_radial_moment_channels_against_quadrature(
    scalar_probe: Any, radial: int
) -> None:
    """Integrate independently on full, LR and SR intervals at fixed exponents."""
    integrate = pytest.importorskip("scipy.integrate")
    for argument in (0.0, 1e-10, 0.3, 8.0, 120.0):
        for order in (0, 1, 4, 8):
            inputs = np.array([argument, order, 0.3, -1.2, 0.0, 0, 0, 0, 0.7, 0.43])
            boundary = inputs[9] / np.sqrt(inputs[8] + inputs[9] ** 2)
            lower, upper = ((0, 1), (0, boundary), (boundary, 1))[radial]
            if radial == 0:
                inputs[9] = 0.0
            actual = scalar_probe(7 + radial, inputs)
            primal = lambda argument, order=order, lower=lower, upper=upper: (
                integrate.quad_vec(
                    lambda point: (
                        point ** (2 * order) * np.exp(-argument * point * point)
                    ),
                    lower,
                    upper,
                    epsabs=1e-13,
                    epsrel=1e-13,
                )[0]
            )
            slope = primal(argument + 1e-25j).imag / 1e-25
            np.testing.assert_allclose(
                actual,
                np.r_[primal(argument), inputs[2:5] * slope],
                rtol=3e-12,
                atol=3e-14,
            )


def test_vacuum_energy_composition_against_beta_integral(scalar_probe: Any) -> None:
    """Use the independent incomplete-beta primal at both C2 boundaries."""
    mp = pytest.importorskip("mpmath")
    lower, upper = 1e-56, 1e-52
    width = upper - lower
    positions = [np.nextafter(lower, upper), np.nextafter(upper, lower)]
    positions += [
        lower + width * value for value in (1e-8, 1e-4, 0.2, 0.5, 0.8, 1 - 1e-8)
    ]
    with mp.workdps(130):
        for density in positions:
            inputs = np.array(
                [
                    density,
                    -2.3e-60,
                    -0.7e-8,
                    0.4e-8,
                    -0.3e-8,
                    0.1e-8,
                    0.6e-8,
                    -0.9e-8,
                    0.2e-8,
                    0.0,
                ]
            )
            actual = np.r_[scalar_probe(10, inputs), scalar_probe(11, inputs)]
            total, energy = map(mp.mpf, inputs[:2])

            def primal(
                displacement: Any,
                channel: int,
                total: Any = total,
                energy: Any = energy,
                inputs: Any = inputs,
            ) -> Any:
                density_shift = displacement if channel < 2 else 0
                coordinate = (total + density_shift - mp.mpf(lower)) / mp.mpf(width)
                return (
                    energy + mp.mpf(inputs[2 + channel]) * displacement
                ) * mp.betainc(3, 3, 0, coordinate, regularized=True)

            expected = [float(primal(mp.mpf(0), 0))]
            expected += [
                float(
                    mp.diff(
                        lambda displacement, channel=channel: primal(
                            displacement, channel
                        ),
                        0,
                    )
                )
                for channel in range(7)
            ]
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-300)
