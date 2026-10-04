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
    source = directory / "probe.cpp"
    source.write_text(r"""
#define __device__
#include "values.hpp"
namespace df = generativeqc::scf::generated_df;
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


def test_df_value_precision_preserves_other_generated_consumers() -> None:
    """The value precision choice must not silently change Direct/derivative math."""
    expected = {
        2: "199549ba1728967e4ed6f73b9ae125cd6eed1d8bc880ddd08f1b506e776be4fb",
        3: "150c6b6e0eb76cee74fe34f1a2cfa7f9caef2360bab8d44aa848a58cec444df3",
        4: "f470a6b2d83c81b15407073f472d5290dfe4083054197b52cc285112237c4532",
        5: "4563f7243ca49bc4ef3852b56f87dcf4c98cf2b33785eaf2e4cde9b2d93b4025",
    }
    for nroots, digest in expected.items():
        source = getattr(rys, f"emit_rys{nroots}_roots_cuda")()
        assert hashlib.sha256(source.encode()).hexdigest() == digest
    assert hashlib.sha256(emit_df_derivatives_cuda().encode()).hexdigest() == (
        "71430c3d8929868c10156d36578f719bbd56d595b35c4ae56131930395f4ce04"
    )
