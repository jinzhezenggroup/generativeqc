"""Generated Cartesian reassociation against independent Gaussian quadrature.

The oracle integrates a correlated two-electron Gaussian along the Coulomb
Laplace coordinate. Its centered Gaussian moments use Wick's identity, not the
production Hermite or Coulomb recurrence. Geometry derivatives independently
raise/lower primitive polynomial powers. Host compilation does not qualify CUDA
execution, scheduling, resource usage or endpoint performance.
"""

import ctypes
import functools
import math
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
    emit_direct_cartesian_contraction_headers,
)
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
FloatArray = NDArray[np.float64]
PowerArray = NDArray[np.uint32]


@pytest.fixture(scope="module")
def primitive(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., int]:
    """Compile the actual generated consumer and its shared native arithmetic."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], capture_output=True, check=True)
    folder = tmp_path_factory.mktemp("hermite-convolution")
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#define __host__\n#define __device__\n"
        "#define __forceinline__ inline\n#define __noinline__\n"
    )
    generated = emit_direct_cartesian_contraction_headers()
    (folder / "generated_direct_cartesian.cuh").write_text(
        generated["generated_direct_cartesian.cuh"]
    )
    source, obj, library = folder / "probe.cpp", folder / "probe.o", folder / "probe.so"
    source.write_text(PROBE)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-fPIC",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    owner = ctypes.CDLL(str(library))
    owner.evaluate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    owner.evaluate.restype = ctypes.c_int
    return owner.evaluate


NODES, WEIGHTS = np.polynomial.legendre.leggauss(96)


def gaussian_moment(
    degrees: tuple[int, ...],
    means: tuple[FloatArray, ...],
    origins: FloatArray,
    variance: tuple[FloatArray, FloatArray],
    covariance: FloatArray,
) -> FloatArray:
    """Integrate a shifted four-factor polynomial by the Gaussian pairing identity."""

    @functools.cache
    def moment(powers: tuple[int, ...]) -> FloatArray:
        if not any(powers):
            return np.ones_like(covariance)
        first = next(i for i, power in enumerate(powers) if power)
        reduced = list(powers)
        reduced[first] -= 1
        value = (means[first] - origins[first]) * moment(tuple(reduced))
        for other, power in enumerate(reduced):
            if power:
                lowered = reduced.copy()
                lowered[other] -= 1
                cov = variance[first // 2] if first // 2 == other // 2 else covariance
                value = value + power * cov * moment(tuple(lowered))
        return value

    return moment(degrees)


def quadrature(
    powers: PowerArray,
    positions: FloatArray,
    exponents: FloatArray,
    radial: int,
    omega: float,
) -> float:
    """Unnormalized Cartesian ERI from the positive Coulomb Laplace measure."""
    a, b, c, d = exponents
    p, q = a + b, c + d
    rho = p * q / (p + q)
    bra = (a * positions[0] + b * positions[1]) / p
    ket = (c * positions[2] + d * positions[3]) / q
    theta = omega / math.sqrt(rho + omega**2)
    lower, upper = ((0.0, 1.0), (0.0, theta), (theta, 1.0))[radial]
    points = lower + (NODES + 1) * (upper - lower) / 2
    weights = WEIGHTS * (upper - lower) / 2
    u2 = points**2
    mean_bra = bra[:, None] - (q / (p + q)) * (bra - ket)[:, None] * u2
    mean_ket = ket[:, None] + (p / (p + q)) * (bra - ket)[:, None] * u2
    variance = ((1 - q / (p + q) * u2) / (2 * p), (1 - p / (p + q) * u2) / (2 * q))
    covariance = u2 / (2 * (p + q))
    polynomial = np.ones_like(points)
    for axis in range(3):
        means = (mean_bra[axis], mean_bra[axis], mean_ket[axis], mean_ket[axis])
        polynomial *= gaussian_moment(
            tuple(int(powers[i, axis]) for i in range(4)),
            means,
            positions[:, axis],
            variance,
            covariance,
        )
    decay = math.exp(
        -a * b / p * np.sum((positions[0] - positions[1]) ** 2)
        - c * d / q * np.sum((positions[2] - positions[3]) ** 2)
    )
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q)) * decay
    return prefactor * np.dot(
        weights, polynomial * np.exp(-rho * np.sum((bra - ket) ** 2) * u2)
    )


def independent_jet(
    powers: PowerArray,
    positions: FloatArray,
    exponents: FloatArray,
    radial: int,
    omega: float,
    mask: int,
) -> FloatArray:
    """Analytic nuclear derivative from independently raised/lowered integrals."""
    result = [quadrature(powers, positions, exponents, radial, omega)]
    for axis in range(3):
        derivative = 0.0
        for center in range(4):
            if not (mask & (1 << center)):
                continue
            raised = powers.copy()
            raised[center, axis] += 1
            derivative += (
                2
                * exponents[center]
                * quadrature(raised, positions, exponents, radial, omega)
            )
            if powers[center, axis]:
                lowered = powers.copy()
                lowered[center, axis] -= 1
                derivative -= powers[center, axis] * quadrature(
                    lowered, positions, exponents, radial, omega
                )
        result.append(derivative)
    return np.asarray(result)


def normalization(powers: PowerArray, exponents: FloatArray) -> float:
    """Unit-norm Cartesian primitives make absolute gates meaningful for diffuse shells."""
    factor = 1.0
    for angular, alpha in zip(powers, exponents, strict=True):
        factorials = math.prod(math.prod(range(1, 2 * int(n), 2)) for n in angular)
        factor *= (2 * alpha / math.pi) ** 0.75 * math.sqrt(
            (4 * alpha) ** int(sum(angular)) / factorials
        )
    return factor


# Every shell is through-f; the total degrees cover 0 through 12. Components
# include balanced and one-axis powers, where the amount of reusable work differs.
POWERS = [
    ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0)),
    ((1, 1, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)),
    ((2, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
    ((1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 0)),
    ((2, 0, 0), (0, 1, 1), (0, 1, 0), (0, 0, 1)),
    ((2, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 1, 1), (1, 1, 0), (0, 0, 2)),
    ((2, 1, 0), (0, 1, 2), (1, 1, 1), (0, 0, 0)),
    ((2, 1, 0), (0, 1, 2), (1, 1, 0), (0, 0, 2)),
    ((2, 1, 0), (0, 1, 2), (1, 1, 1), (1, 0, 1)),
    ((2, 1, 0), (0, 1, 2), (1, 1, 1), (1, 1, 1)),
]


@pytest.mark.parametrize(
    "powers", POWERS, ids=lambda p: "order-" + str(sum(map(sum, p)))
)
@pytest.mark.parametrize("radial", range(3), ids=("full", "long", "short"))
@pytest.mark.parametrize("case", ("distinct", "repeated", "diffuse", "translation"))
def test_generated_convolution_against_independent_cartesian_integral_and_jet(
    primitive: Callable[..., int], powers: Any, radial: int, case: str
) -> None:
    positions = np.asarray(
        ((0.1, -0.2, -0.8), (0.3, 0.1, 0.7), (-0.5, 0.6, 0.2), (0.8, -0.4, 0.3))
    )
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    mask, omega = 1, 0.3
    if case == "repeated":
        positions[1] = positions[0]
        exponents = np.asarray((0.05, 1.8, 0.09, 0.8))
        mask = 3
    elif case == "diffuse":
        positions *= 4
        exponents = np.asarray((0.018, 0.041, 0.073, 0.012))
        mask = 8
    elif case == "translation":
        mask, omega = 15, 1.5
    powers = np.asarray(powers, dtype=np.uint32)
    norm = normalization(powers, exponents)
    expected = independent_jet(powers, positions, exponents, radial, omega, mask) * norm
    if case == "translation":
        np.testing.assert_allclose(expected[1:], 0.0, atol=2e-11)
    for selected in range(4):
        actual, value = np.empty(4), np.empty(4)
        assert (
            primitive(
                powers, positions, exponents, omega, radial, mask, selected, 1, actual
            )
            == 0
        )
        assert (
            primitive(
                powers, positions, exponents, omega, radial, mask, selected, 0, value
            )
            == 0
        )
        np.testing.assert_allclose(actual * norm, expected, rtol=3e-10, atol=2e-11)
        np.testing.assert_allclose(value[0] * norm, expected[0], rtol=3e-10, atol=2e-11)


def test_mixed_evaluator_does_not_inherit_new_reassociation(
    primitive: Callable[..., int],
) -> None:
    """The new switch cannot change the separate, unqualified FP32 consumer."""
    powers = np.asarray(POWERS[-1], dtype=np.uint32)
    positions = np.asarray(
        ((0.1, -0.2, -0.8), (0.3, 0.1, 0.7), (-0.5, 0.6, 0.2), (0.8, -0.4, 0.3))
    )
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    for reachable in (0, 1):
        off, on = np.empty(4), np.empty(4)
        assert (
            primitive(powers, positions, exponents, 0.3, 0, 1, reachable, 2, off) == 0
        )
        assert (
            primitive(powers, positions, exponents, 0.3, 0, 1, reachable | 2, 2, on)
            == 0
        )
        np.testing.assert_array_equal(off, on)


PROBE = r"""
#include "generated_direct_cartesian.cuh"
using namespace generativeqc::scf::cuda_execution;
template<unsigned L, typename Scalar> void run(const unsigned* powers, const double* positions,
    const double* exponent, double omega, unsigned radial, unsigned mask, unsigned selected,
    double* out) {
  Angular angular[4]; Vec3<Scalar> center[4];
  for (unsigned i=0;i<4;++i) {
    angular[i]={powers[3*i],powers[3*i+1],powers[3*i+2]};
    center[i]={scalar<Scalar>(positions[3*i]),scalar<Scalar>(positions[3*i+1]),scalar<Scalar>(positions[3*i+2])};
    if constexpr(std::is_same_v<Scalar,Dual3>) {
      const double seed=(mask & (1U<<i)) ? 1.0 : 0.0;
      center[i].x.derivative_x=seed; center[i].y.derivative_y=seed; center[i].z.derivative_z=seed;
    }
  }
  const double p=exponent[0]+exponent[1],q=exponent[2]+exponent[3],rho=p*q/(p+q);
  const auto bra=product_center(exponent[0],center[0],exponent[1],center[1]);
  const auto ket=product_center(exponent[2],center[2],exponent[3],center[3]);
  HermiteCoefficients<Scalar> first[3],second[3];
  for(unsigned axis=0;axis<3;++axis) {
    fill_hermite(angular_axis(angular[0],axis),angular_axis(angular[1],axis),vec_axis(bra,axis),
      vec_axis(center[0],axis),vec_axis(center[1],axis),exponent[0],exponent[1],first[axis]);
    fill_hermite(angular_axis(angular[2],axis),angular_axis(angular[3],axis),vec_axis(ket,axis),
      vec_axis(center[2],axis),vec_axis(center[3],axis),exponent[2],exponent[3],second[axis]);
  }
  const auto value=eri_cartesian_value<L,Scalar>(EvaluationReal<Scalar>{p},EvaluationReal<Scalar>{q},
    EvaluationReal<Scalar>{rho},bra,ket,angular[0],angular[1],angular[2],angular[3],first,second,
    static_cast<generativeqc::integrals::CoulombRange>(radial),omega,(selected&1)!=0,(selected&2)!=0);
  out[0]=scalar_value(value); out[1]=out[2]=out[3]=0;
  if constexpr(std::is_same_v<Scalar,Dual3>) {
    out[1]=value.derivative_x;out[2]=value.derivative_y;out[3]=value.derivative_z;
  }
}
template<typename Scalar> int dispatch(const unsigned* powers, const double* positions,
    const double* exponent, double omega, unsigned radial, unsigned mask, unsigned selected,double* out) {
  unsigned order=0;for(unsigned i=0;i<12;++i)order+=powers[i];
  switch(order) {
#define CASE(L) case L: run<L,Scalar>(powers,positions,exponent,omega,radial,mask,selected,out);return 0
    CASE(0);CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);CASE(9);CASE(10);CASE(11);CASE(12);
#undef CASE
  }
  return -1;
}
extern "C" int evaluate(const unsigned* powers,const double* positions,const double* exponent,
    double omega,unsigned radial,unsigned mask,unsigned selected,unsigned kind,double* out) {
  if(kind==0)return dispatch<double>(powers,positions,exponent,omega,radial,mask,selected,out);
  if(kind==1)return dispatch<Dual3>(powers,positions,exponent,omega,radial,mask,selected,out);
  return dispatch<MixedPrecisionFloat>(powers,positions,exponent,omega,radial,mask,selected,out);
}
"""
