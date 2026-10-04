"""Qualify the independent low-degree reference, including real d harmonics."""

from __future__ import annotations

import ctypes as ct
import itertools
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def analytic_reference(tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    """Build the validation oracle independently of every production header."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache or np.finfo(np.longdouble).nmant < 63:
        pytest.skip("host C++/ccache and a 64-bit extended mantissa required")
    directory = tmp_path_factory.mktemp("df-independent-oracle")
    obj, dll_path = directory / "oracle.o", directory / "oracle.so"
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-fno-fast-math",
            "-fPIC",
            "-fopenmp",
            "-c",
            str(ROOT / "tools/generativeqc_validation/df_precision_oracle.cpp"),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [compiler, "-shared", "-fopenmp", str(obj), "-o", str(dll_path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    dll = ct.CDLL(str(dll_path))
    dll.analytic_low_degree.argtypes = [ct.c_int] * 4 + [ct.c_void_p] * 8
    dll.analytic_low_degree.restype = None
    # Distinct, anisotropic centers prevent vanishing components from concealing
    # signs or d-harmonic normalization errors. Every shell has one primitive.
    info = np.array([[l, l, l, 1] for l in range(3)], dtype=np.int32)
    coeff = np.array([[0.1285, 1.0], [0.31, 1.0], [0.1117, 1.0]])
    ao = np.array([[l, k] for l in range(3) for k in range(2 * l + 1)], dtype=np.int32)
    xyz = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, -0.1], [5.31113, 0.5, -0.7]])
    raw = np.full((9, 9, 9), np.nan, dtype=np.longdouble)
    dll.analytic_low_degree(
        9,
        9,
        3,
        3,
        *[x.ctypes.data for x in (info, info, coeff, coeff, ao, ao, xyz, raw)],
    )
    degrees = ao[:, 0]
    mask = degrees[:, None, None] + degrees[None, :, None] + degrees[None, None, :] <= 2
    assert np.isfinite(raw[mask]).all()
    assert np.isnan(raw[~mask]).all(), (
        "the reference must preserve higher-degree libcint entries"
    )
    return raw


def _polynomial(l: int, component: int, mp: Any) -> list[tuple[tuple[int, ...], Any]]:
    """Independently normalized real harmonics multiplying the radial factor."""
    if l == 0:
        return [((0, 0, 0), mp.mpf(1))]
    if l == 1:
        return [(tuple(int(i == component) for i in range(3)), mp.mpf(1))]
    return [
        [((1, 1, 0), mp.mpf(1))],
        [((0, 1, 1), mp.mpf(1))],
        [
            ((2, 0, 0), -1 / (2 * mp.sqrt(3))),
            ((0, 2, 0), -1 / (2 * mp.sqrt(3))),
            ((0, 0, 2), 1 / mp.sqrt(3)),
        ],
        [((1, 0, 1), mp.mpf(1))],
        [((2, 0, 0), mp.mpf(".5")), ((0, 2, 0), -mp.mpf(".5"))],
    ][component]


@pytest.mark.parametrize(
    "angular", [a for a in itertools.product(range(3), repeat=3) if sum(a) <= 2]
)
def test_reference_against_90_digit_gaussian_derivatives(
    analytic_reference: np.ndarray, angular: tuple[int, ...]
) -> None:
    """All low-degree angular placements and every real component are checked."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        exponents = list(map(mp.mpf, (0.1285, 0.31, 0.1117)))
        centers = [
            tuple(map(mp.mpf, p))
            for p in ((0.1, -0.2, 0.3), (-0.4, 0.2, -0.1), (5.31113, 0.5, -0.7))
        ]
        a, b, g = (exponents[l] for l in angular)
        args = tuple(x for l in angular for x in centers[l])
        p, rho = a + b, (a + b) * g / (a + b + g)

        def base(*r: Any) -> Any:
            A, B, C = r[:3], r[3:6], r[6:]
            P = [(a * x + b * y) / p for x, y in zip(A, B, strict=True)]
            t = rho * mp.fsum((x - y) ** 2 for x, y in zip(P, C, strict=True))
            f0 = (
                mp.mpf(1)
                if not t
                else mp.sqrt(mp.pi) * mp.erf(mp.sqrt(t)) / (2 * mp.sqrt(t))
            )
            decay = mp.exp(
                -a * b / p * mp.fsum((x - y) ** 2 for x, y in zip(A, B, strict=True))
            )
            return 2 * mp.pi ** mp.mpf("2.5") / (p * g * mp.sqrt(p + g)) * f0 * decay

        normalization = mp.fprod(
            (2 * e / mp.pi) ** mp.mpf(".75") * (4 * e) ** (mp.mpf(l) / 2)
            for e, l in zip((a, b, g), angular, strict=True)
        )
        for components in itertools.product(*(range(2 * l + 1) for l in angular)):
            expected = mp.mpf(0)
            polynomials = [
                _polynomial(l, c, mp) for l, c in zip(angular, components, strict=True)
            ]
            for terms in itertools.product(*polynomials):
                powers = tuple(power for powers, _ in terms for power in powers)
                derivative = mp.diff(base, args, powers)
                if 2 in powers:
                    derivative += 2 * (a, b, g)[powers.index(2) // 3] * base(*args)
                derivative /= mp.fprod(
                    (2 * (a, b, g)[i // 3]) ** power for i, power in enumerate(powers)
                )
                expected += derivative * mp.fprod(weight for _, weight in terms)
            expected *= normalization
            index = tuple(l * l + c for l, c in zip(angular, components, strict=True))
            # Decimal conversion keeps every host extended-precision digit.
            actual = mp.mpf(str(analytic_reference[index]))
            assert abs(actual - expected) < mp.mpf("1e-17") * max(1, abs(expected)), (
                angular,
                components,
                actual,
                expected,
            )
