"""Independent analytic gates for the streamed DF precision policy (#1840)."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_cuda import emit_df_values_cpu

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compensated_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    """Compile the actual generated primitive and its shared FP64 arithmetic."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    directory = tmp_path_factory.mktemp("df-compensated")
    (directory / "values.hpp").write_text(emit_df_values_cpu())
    source = directory / "probe.cpp"
    source.write_text(r"""
#include "values.hpp"
namespace df=generativeqc::scf::generated_df;
extern "C" void primitive(const double* e,const double* r,const unsigned* l,double* out) {
  auto v=df::compensated::value(e[0],{r[0],r[1],r[2]},{l[0],l[1],l[2]},
      e[1],{r[3],r[4],r[5]},{l[3],l[4],l[5]},e[2],{r[6],r[7],r[8]},{l[6],l[7],l[8]});
  out[0]=v.hi;out[1]=v.lo;
}
extern "C" void dot(unsigned n,const double* a,const double* b,double* out) {
  df::fp64_expansion::Wide sum;
  for(unsigned i=0;i<n;++i)sum+=df::fp64_expansion::Wide(a[i])*b[i];
  out[0]=sum.hi;out[1]=sum.lo;
}
""")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    obj, library = directory / "probe.o", directory / "probe.so"
    subprocess.run(
        [
            cache,
            compiler,
            "-O2",
            "-fno-fast-math",
            "-std=c++17",
            "-fPIC",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
        timeout=180,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    dll = ct.CDLL(str(library))
    dll.primitive.argtypes = [ct.c_void_p] * 4
    dll.primitive.restype = None
    dll.dot.argtypes = [ct.c_uint] + [ct.c_void_p] * 3
    dll.dot.restype = None
    return dll


@pytest.mark.parametrize(
    "angular",
    (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
        (1, 0, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 1),
        (2, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 1),
    ),
)
@pytest.mark.parametrize("separation", (0.0, 5.31113, 20.0, 60.0))
def test_low_angular_expansion_against_analytic_center_derivatives(
    compensated_probe: ct.CDLL, angular: tuple[int, ...], separation: float
) -> None:
    """Differentiate the s/s/s Coulomb formula at 90 digits, without the IR.

    A squared Cartesian factor includes the lower-order 2*alpha term in the
    Gaussian derivative identity. The separated cases exercise both sides of
    the Boys positive-series/asymptotic branch and Gaussian pair decay.
    """
    mp = pytest.importorskip("mpmath")
    e = np.array([0.1285, 0.1513, 0.10153627861])
    r = np.array([0.1, -0.2, 0.3, -0.4, 0.2, -0.1, separation, 0.5, -0.7])
    l = np.array(angular, dtype=np.uint32)
    out = np.empty(2)
    compensated_probe.primitive(*[x.ctypes.data for x in (e, r, l, out)])
    with mp.workdps(90):
        a, b, g = map(mp.mpf, e)
        p, rho = a + b, (a + b) * g / (a + b + g)

        def base(*coordinates: Any) -> Any:
            A, B, C = coordinates[:3], coordinates[3:6], coordinates[6:]
            P = [(a * x + b * y) / p for x, y in zip(A, B, strict=True)]
            t = rho * mp.fsum((x - y) ** 2 for x, y in zip(P, C, strict=True))
            decay = mp.exp(
                -a * b / p * mp.fsum((x - y) ** 2 for x, y in zip(A, B, strict=True))
            )
            f0 = (
                mp.sqrt(mp.pi) * mp.erf(mp.sqrt(t)) / (2 * mp.sqrt(t))
                if t
                else mp.mpf(1)
            )
            return 2 * mp.pi ** mp.mpf("2.5") / (p * g * mp.sqrt(p + g)) * decay * f0

        args = tuple(map(mp.mpf, r))
        expected = mp.diff(base, args, angular)
        if 2 in angular:
            axis = angular.index(2)
            expected += 2 * (a, b, g)[axis // 3] * base(*args)
        expected /= mp.fprod(
            (2 * (a, b, g)[axis // 3]) ** power for axis, power in enumerate(angular)
        )
        actual = mp.mpf(out[0]) + mp.mpf(out[1])
        assert abs(actual - expected) < mp.mpf("5e-28") * max(1, abs(expected)), (
            actual,
            expected,
        )


def test_expansion_dot_retains_product_and_sum_residuals(
    compensated_probe: ct.CDLL,
) -> None:
    """Ordinary products/sums lose the entire signed result in both examples."""
    mp = pytest.importorskip("mpmath")
    for a, b in (
        ([1 + 2**-27, -1], [1 - 2**-27, 1]),
        ([1e16, 1, -1e16, 2**-40], [1, 1, 1, -1]),
    ):
        aa, bb, out = np.array(a), np.array(b, dtype=float), np.empty(2)
        compensated_probe.dot(len(a), *[x.ctypes.data for x in (aa, bb, out)])
        with mp.workdps(90):
            expected = mp.fsum(
                mp.mpf(x) * mp.mpf(y) for x, y in zip(aa, bb, strict=True)
            )
            assert mp.mpf(out[0]) + mp.mpf(out[1]) == expected
