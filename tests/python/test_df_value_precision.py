"""Independent strict-quadrature gates for condition-sensitive DF values."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import random
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral import rys
from generativeqc_compiler.integral.df_cuda import emit_df_values_cpu
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def value_precision_probe(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the actual generated value consumer, without a CUDA dependency."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    directory = tmp_path_factory.mktemp("df-value-precision")
    (directory / "values.hpp").write_text(emit_df_values_cpu())
    (directory / "cuda_runtime.h").write_text("")
    (directory / "derivatives.cuh").write_text(emit_df_derivatives_cuda())
    source = directory / "probe.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "values.hpp"
#include "derivatives.cuh"
namespace df = generativeqc::scf::generated_df;
namespace response = generativeqc::scf::generated_df_derivatives;
extern "C" void derivative(unsigned count,double a,double g,double r,
                           unsigned la,unsigned lb,unsigned lc,double omega,
                           bool default_kernel,double* output) {
  const response::Vec3 A{0,0,0},C{0,0,r};
  const response::Angular aa{0,0,la},bb{0,0,lb},cc{0,0,lc};
  const auto value=count==2
      ? (default_kernel ? response::metric(a,A,aa,g,C,cc)
                        : response::metric(a,A,aa,g,C,cc,omega))
      : (default_kernel ? response::three_center(a,A,aa,a,A,bb,g,C,cc)
                        : response::three_center(a,A,aa,a,A,bb,g,C,cc,omega));
  output[0]=value.value;
  const response::Vec3 centers[]{value.first,value.second,value.third};
  for(unsigned center=0;center<3;++center) {
    output[1+3*center]=centers[center].x;
    output[2+3*center]=centers[center].y;
    output[3+3*center]=centers[center].z;
  }
}
extern "C" void roots(unsigned n, double t, double* output, unsigned stride) {
  if(n==2) df::df_rys2_roots(t,output,stride);
  if(n==3) df::df_rys3_roots(t,output,stride);
  if(n==4) df::df_rys4_roots(t,output,stride);
  if(n==5) df::df_rys5_roots(t,output,stride);
}
extern "C" double primitive(double a, double g, double r,
                            unsigned la, unsigned lb, unsigned lc) {
  return df::three_center(a,{0,0,0},{0,0,la},a,{0,0,0},{0,0,lb},g,{0,0,r},{0,0,lc});
}
""")
    obj, library = directory / "probe.o", directory / "probe.so"
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    subprocess.run(
        [
            cache,
            compiler,
            "-O2",
            "-fno-fast-math",
            "-std=c++17",
            "-fPIC",
            f"-I{directory}",
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
    dll = ctypes.CDLL(str(library))
    dll.roots.argtypes = [
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint,
    ]
    dll.roots.restype = None
    dll.primitive.argtypes = [ctypes.c_double] * 3 + [ctypes.c_uint] * 3
    dll.primitive.restype = ctypes.c_double
    dll.derivative.argtypes = (
        [ctypes.c_uint]
        + [ctypes.c_double] * 3
        + [ctypes.c_uint] * 3
        + [ctypes.c_double, ctypes.c_bool, ctypes.POINTER(ctypes.c_double)]
    )
    dll.derivative.restype = None
    return dll


def _arguments(nroots: int) -> list[float]:
    rng = random.Random(1781)
    values = [0.0, 1e-20, 1e-12, 1e-8, 1e-7, 2.053027104306768, 1e3, 1e100, 1e300]
    values += [rng.uniform(0, 35 + 5 * nroots + 5) for _ in range(100)]
    for edge in (3e-7, *(2.5 * k for k in range(1, (35 + 5 * nroots) * 2 // 5 + 1))):
        values += [math.nextafter(edge, 0.0), edge, math.nextafter(edge, math.inf)]
    return values


@pytest.mark.parametrize("nroots", (2, 3, 4, 5))
def test_emitted_df_quadrature_preserves_all_defining_moments(
    value_precision_probe: ctypes.CDLL, nroots: int
) -> None:
    """Independent incomplete-gamma moments cover tables and both branch edges."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        for argument in _arguments(nroots):
            storage = np.full(2 * nroots * 3 + 4, np.nan)
            value_precision_probe.roots(
                nroots,
                argument,
                storage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                3,
            )
            selected = np.arange(2 * nroots) * 3
            values = storage[selected]
            assert np.isnan(np.delete(storage, selected)).all()
            nodes, weights = values[::2], values[1::2]
            assert np.all((nodes > 0) & (nodes < 1)) and np.all(weights > 0)
            t = mp.mpf(argument)
            for order in range(2 * nroots):
                exponent = order + mp.mpf(".5")
                expected = (
                    mp.mpf(1) / (2 * order + 1)
                    if not t
                    else mp.gammainc(exponent, 0, t) / (2 * t**exponent)
                )
                actual = mp.fsum(
                    mp.mpf(w) * mp.mpf(x) ** order
                    for x, w in zip(nodes, weights, strict=True)
                )
                assert abs(actual / expected - 1) < mp.mpf("5e-14"), (
                    nroots,
                    argument,
                    order,
                    actual,
                    expected,
                )


@pytest.mark.parametrize("angular", ((0, 0, 2), (1, 1, 2), (2, 2, 2), (3, 3, 3)))
def test_df_primitive_table_floor_against_analytic_center_derivatives(
    value_precision_probe: ctypes.CDLL, angular: tuple[int, int, int]
) -> None:
    """Center differentiation is independent of production Wick/Rys lowering.

    Coincident orbital Gaussians combine into exponent p=a+b and power la+lb.
    Applying the Gaussian center-derivative identity to the analytic s/s
    Coulomb interaction provides each requested Cartesian integral directly.
    The old tables miss these normalized values by roughly 1.5--2.5e-12.
    """
    mp = pytest.importorskip("mpmath")
    alpha, gamma = 0.1285, 0.10153627861
    distance = float.fromhex("0x1.53e982889df5bp+2")
    la, lb, lc = angular
    with mp.workdps(90):
        a, g, r = map(mp.mpf, (alpha, gamma, distance))
        p, total = 2 * a, la + lb
        rho = p * g / (p + g)

        def base(d: typing.Any) -> typing.Any:
            t = rho * d**2
            f0 = mp.sqrt(mp.pi) * mp.erf(mp.sqrt(t)) / (2 * mp.sqrt(t))
            return 2 * mp.pi ** mp.mpf("2.5") / (p * g * mp.sqrt(p + g)) * f0

        coefficients = mp.taylor(base, -r, total + lc)

        def center_terms(
            order: int, exponent: typing.Any
        ) -> list[tuple[int, typing.Any]]:
            return [
                (
                    order - 2 * k,
                    mp.factorial(order)
                    / mp.factorial(k)
                    / mp.factorial(order - 2 * k)
                    * exponent**k
                    / (2 * exponent) ** order,
                )
                for k in range(order // 2 + 1)
            ]

        expected = mp.fsum(
            wb * wc * (-1) ** dc * coefficients[db + dc] * mp.factorial(db + dc)
            for db, wb in center_terms(total, p)
            for dc, wc in center_terms(lc, g)
        )
        normalization = mp.fprod(
            (2 * e / mp.pi) ** mp.mpf(".75")
            * mp.sqrt((4 * e) ** l / mp.fprod(range(1, 2 * l, 2)))
            for l, e in zip(angular, (a, a, g), strict=True)
        )
        expected *= normalization
        actual = value_precision_probe.primitive(alpha, gamma, distance, la, lb, lc)
        actual *= float(normalization)
        assert abs(mp.mpf(actual) - expected) < mp.mpf("3e-14"), (actual, expected)


def test_df_value_precision_preserves_direct_rys_consumers() -> None:
    """Keep the existing Direct-table scope guard separate from derivative science."""
    expected = {
        2: "199549ba1728967e4ed6f73b9ae125cd6eed1d8bc880ddd08f1b506e776be4fb",
        3: "150c6b6e0eb76cee74fe34f1a2cfa7f9caef2360bab8d44aa848a58cec444df3",
        4: "f470a6b2d83c81b15407073f472d5290dfe4083054197b52cc285112237c4532",
        5: "4563f7243ca49bc4ef3852b56f87dcf4c98cf2b33785eaf2e4cde9b2d93b4025",
    }
    for nroots, digest in expected.items():
        source = getattr(rys, f"emit_rys{nroots}_roots_cuda")()
        assert hashlib.sha256(source.encode()).hexdigest() == digest


@pytest.mark.parametrize("omega", (0.0, 0.3))
@pytest.mark.parametrize(
    "angular", ((0, 2), (3, 3), (0, 0, 2), (1, 1, 2), (2, 2, 2), (3, 3, 3))
)
def test_emitted_df_response_matches_independent_analytic_center_derivatives(
    value_precision_probe: ctypes.CDLL, angular: tuple[int, ...], omega: float
) -> None:
    """Preserve Coulomb and LR derivative semantics without freezing emitted bytes.

    Differentiate the analytic s/s erf interaction at 90 digits and use Gaussian
    center raising/lowering, independently of emitted moment-polynomial algebra.
    Both real metric centers and all three ERI centers are checked, including
    internal fourth angular powers required for public f-shell derivatives.
    """
    mp = pytest.importorskip("mpmath")
    count = len(angular)
    la, lb, lc = (angular[0], 0, angular[1]) if count == 2 else angular
    alpha, gamma = 0.1285, 0.10153627861
    distance = float.fromhex("0x1.53e982889df5bp+2")
    actual = np.zeros(10)
    value_precision_probe.derivative(
        count,
        alpha,
        gamma,
        distance,
        la,
        lb,
        lc,
        omega,
        False,
        actual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if omega == 0.0:
        default = np.zeros(10)
        value_precision_probe.derivative(
            count,
            alpha,
            gamma,
            distance,
            la,
            lb,
            lc,
            omega,
            True,
            default.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        np.testing.assert_array_equal(default, actual)
    with mp.workdps(90):
        a, g, r, w = map(mp.mpf, (alpha, gamma, distance, omega))
        p, total = (a, la) if count == 2 else (2 * a, la + lb)
        rho = p * g / (p + g)
        scale = w / mp.sqrt(w * w + rho) if w else mp.mpf(1)

        def base(displacement: typing.Any) -> typing.Any:
            t = rho * scale * scale * displacement**2
            f0 = mp.sqrt(mp.pi) * mp.erf(mp.sqrt(t)) / (2 * mp.sqrt(t))
            return 2 * mp.pi ** mp.mpf("2.5") / (p * g * mp.sqrt(p + g)) * scale * f0

        coefficients = mp.taylor(base, -r, total + lc + 1)

        def center_terms(
            order: int, exponent: typing.Any
        ) -> list[tuple[int, typing.Any]]:
            return [
                (
                    order - 2 * k,
                    mp.factorial(order)
                    / mp.factorial(k)
                    / mp.factorial(order - 2 * k)
                    * exponent**k
                    / (2 * exponent) ** order,
                )
                for k in range(order // 2 + 1)
            ]

        def integral(first: int, last: int) -> typing.Any:
            if first < 0 or last < 0:
                return mp.mpf(0)
            return mp.fsum(
                wb * wc * (-1) ** dc * coefficients[db + dc] * mp.factorial(db + dc)
                for db, wb in center_terms(first, p)
                for dc, wc in center_terms(last, g)
            )

        value = integral(total, lc)
        first = 2 * a * integral(total + 1, lc) - la * integral(total - 1, lc)
        second = (
            (2 * a * integral(total + 1, lc) - lb * integral(total - 1, lc))
            if count == 3
            else 0
        )
        third = 2 * g * integral(total, lc + 1) - lc * integral(total, lc - 1)
        normalization = mp.fprod(
            (2 * exponent / mp.pi) ** mp.mpf(".75")
            * mp.sqrt((4 * exponent) ** power / mp.fprod(range(1, 2 * power, 2)))
            for power, exponent in zip(
                angular, (a, g) if count == 2 else (a, a, g), strict=True
            )
        )
        expected = (value, 0, 0, first, 0, 0, second, 0, 0, third)
        for result, reference in zip(actual, expected, strict=True):
            normalized_reference = reference * normalization
            # Force channels can exceed unity; keep the strict absolute floor
            # and the defining-moment relative gate for their natural scale.
            assert abs(mp.mpf(result) * normalization - normalized_reference) < (
                mp.mpf("3e-14") + mp.mpf("5e-14") * abs(normalized_reference)
            )
        np.testing.assert_allclose(
            actual[1:].reshape(3, 3).sum(axis=0), 0, atol=2e-12, rtol=0
        )
