"""Generated order-six derivatives against independent raised/lowered quadrature.

The oracle uses correlated-Gaussian moments, not the production Hermite/Boys
recurrence. Normalization keeps absolute gates meaningful for diffuse shells.
Host execution does not qualify CUDA resources or complete endpoint latency.
"""

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_pair_gradient_cuda import (
    emit_direct_high_order_pair_gradient_header,
)
from generativeqc_compiler.integral.direct_pair_support_cuda import (
    emit_direct_pair_support_headers,
)
from generativeqc_compiler.integral.direct_recurrence_cuda import (
    emit_direct_recurrence_headers,
)
from test_hermite_convolution import independent_jet, normalization

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = (
    ((2, 1, 0), (0, 1, 2), (0, 0, 0), (0, 0, 0)),
    ((3, 0, 0), (0, 2, 0), (0, 0, 1), (0, 0, 0)),
    ((2, 0, 0), (0, 1, 1), (1, 0, 0), (0, 0, 1)),
    ((2, 0, 0), (0, 1, 0), (0, 0, 2), (1, 0, 0)),
)


@pytest.fixture(scope="module")
def gradient(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: "NativeCxx"
) -> Callable[..., int]:
    """Compile both admitted and fallback consumers from the actual generator."""
    directory = tmp_path_factory.mktemp("order-six-response")
    headers = {
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        "gradient.cuh": emit_direct_high_order_pair_gradient_header(),
    }
    for name, source in headers.items():
        (directory / name).write_text(source)
    (directory / "cuda_runtime.h").write_text(
        "#pragma once\n#include <cmath>\n#include <algorithm>\n"
        "#define __device__\n#define __host__\n#define __forceinline__ inline\n"
        "#define __noinline__\nusing std::min; using std::max;\n"
        "inline unsigned __popc(unsigned value) { return __builtin_popcount(value); }\n"
    )
    source = directory / "probe.cpp"
    source.write_text(PROBE)
    library = native_cxx.build_shared(
        [source],
        directory / "probe.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            f"-I{directory}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    owner = ctypes.CDLL(str(library))
    owner.evaluate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_bool,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    owner.evaluate.restype = ctypes.c_int
    return owner.evaluate


@pytest.mark.parametrize("powers", COMPONENTS)
@pytest.mark.parametrize("rotation", (0, 1, 2))
@pytest.mark.parametrize(
    "radial,omega", ((0, 0.0), (1, 0.0), (1, 1e-8), (1, 0.3), (1, 1e4))
)
@pytest.mark.parametrize(
    "geometry", ("distinct", "repeated", "common", "diffuse", "separated")
)
def test_all_center_responses_against_independent_integrals(
    gradient: Callable[..., int],
    powers: tuple[tuple[int, ...], ...],
    rotation: int,
    radial: int,
    omega: float,
    geometry: str,
) -> None:
    """Check every center, pair partition and radial limit without AD or ERI reuse."""
    positions = np.asarray(
        ((0.1, -0.2, -0.8), (0.3, 0.1, 0.7), (-0.5, 0.6, 0.2), (0.8, -0.4, 0.3))
    )
    exponents = np.asarray((0.8, 0.6, 0.7, 0.9))
    if geometry == "repeated":
        positions[1] = positions[0]
        exponents = np.asarray((0.05, 1.8, 0.09, 0.8))
    elif geometry == "common":
        positions[:] = positions[0]
    elif geometry == "diffuse":
        positions *= 4
        exponents = np.asarray((0.018, 0.041, 0.073, 0.012))
    elif geometry == "separated":
        positions *= 8
    angular = np.ascontiguousarray(
        np.roll(np.asarray(powers, dtype=np.uint32), rotation, axis=1)
    )
    norm = normalization(angular, exponents)
    expected = (
        np.asarray(
            [
                independent_jet(
                    angular, positions, exponents, radial, omega, 1 << center
                )[1:]
                for center in range(4)
            ]
        )
        * norm
    )
    for spatial in (False, True):
        actual = np.empty((4, 3))
        assert (
            gradient(angular, positions, exponents, radial, omega, spatial, actual) == 0
        )
        np.testing.assert_allclose(actual * norm, expected, rtol=3e-10, atol=2e-11)
        np.testing.assert_allclose((actual * norm).sum(axis=0), 0.0, rtol=0, atol=2e-11)


PROBE = r"""
#include "gradient.cuh"
using namespace generativeqc::scf::cuda_execution;

template <unsigned FirstOrder, bool LongRange, bool Spatial>
void evaluate_partition(const double* exponents, const Vec3<double>* positions,
                        const Angular* angular, double omega, double* output) {
  double response[4][3];
  primitive_eri_order456_gradient<FirstOrder,6U-FirstOrder,LongRange,Spatial>(
      exponents[0],positions[0],angular[0],exponents[1],positions[1],angular[1],
      exponents[2],positions[2],angular[2],exponents[3],positions[3],angular[3],response,omega);
  for (unsigned center=0;center<4U;++center)
    for (unsigned axis=0;axis<3U;++axis) output[3U*center+axis]=response[center][axis];
}

template <bool LongRange, bool Spatial>
int dispatch(const unsigned* powers, const double* coordinates, const double* exponents,
             double omega, double* output) {
  Angular angular[4];
  Vec3<double> positions[4];
  for (unsigned center=0;center<4U;++center) {
    angular[center]={powers[3U*center],powers[3U*center+1U],powers[3U*center+2U]};
    positions[center]={coordinates[3U*center],coordinates[3U*center+1U],coordinates[3U*center+2U]};
  }
  unsigned first_order=0U;
  for (unsigned index=0;index<6U;++index) first_order+=powers[index];
  switch(first_order) {
    case 6: evaluate_partition<6U,LongRange,Spatial>(exponents,positions,angular,omega,output);break;
    case 5: evaluate_partition<5U,LongRange,Spatial>(exponents,positions,angular,omega,output);break;
    case 4: evaluate_partition<4U,LongRange,Spatial>(exponents,positions,angular,omega,output);break;
    case 3: evaluate_partition<3U,LongRange,Spatial>(exponents,positions,angular,omega,output);break;
    default:return 1;
  }
  return 0;
}

extern "C" int evaluate(const unsigned* powers, const double* coordinates,
                        const double* exponents, unsigned radial, double omega,
                        bool spatial, double* output) {
  if (radial==0U) {
    if (spatial) return dispatch<false,true>(powers,coordinates,exponents,omega,output);
    return dispatch<false,false>(powers,coordinates,exponents,omega,output);
  }
  if (radial==1U) {
    if (spatial) return dispatch<true,true>(powers,coordinates,exponents,omega,output);
    return dispatch<true,false>(powers,coordinates,exponents,omega,output);
  }
  return 2;
}
"""
