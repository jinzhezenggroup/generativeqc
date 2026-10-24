"""Physical and signed-response oracles for the shared PBE subnormal ratio."""

from __future__ import annotations

import ctypes as ct
import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import mpmath as mp
import numpy as np
import pytest

from tools import generate_xc_scf_references as original

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ((1.0, 1.0), (0.75, 1.0), (0.37, 0.91), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
TAIL = np.loadtxt(ROOT / "tests/data/xc/pbe0_restricted_point.tsv")


@pytest.fixture(scope="module")
def point(tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx) -> ct.CDLL:
    directory = tmp_path_factory.mktemp("pbe-subnormal-ratio")
    source = directory / "probe.cpp"
    source.write_text(
        r"""
#include "dft/xc_point_response.hpp"
using namespace generativeqc::dft::point;
static int pack(const Value& v, double* out) {
  out[0] = v.energy;
  for (unsigned s = 0; s < 2; ++s) {
    out[1+s] = v.rho[s];
    for (unsigned k = 0; k < 3; ++k) out[3+3*s+k] = v.gradient[s][k];
  }
  return v.valid;
}
extern "C" int physical(int layout, int pbe, double x, double c,
                         const double* q, double* out) {
  const auto* g = reinterpret_cast<const double (*)[3]>(q+2);
  const auto v = layout == 3 ? evaluate_interior(pbe, q, g)
      : layout == 2 ? evaluate_pbe0_restricted_bound(q[0], g[0])
      : layout == 1 ? evaluate_with_channels<4, true>(pbe, q, g, x, c)
      : evaluate(pbe, q, g, x, c);
  return pack(v, out);
}
extern "C" int response(int restricted, int pbe, double x, double c, const double* q,
                         const double* dq, double* out) {
  const auto* g = reinterpret_cast<const double (*)[3]>(q+2);
  const auto* dg = reinterpret_cast<const double (*)[3]>(dq+2);
  double total_g[3], total_dg[3];
  for (unsigned k=0; k<3; ++k) { total_g[k]=2*g[0][k]; total_dg[k]=2*dg[0][k]; }
  const auto v = restricted
      ? restricted_response(pbe, 2*q[0], total_g, 2*dq[0], total_dg, x, c)
      : unrestricted_response(pbe, q, g, dq, dg, x, c);
  return pack(v, out);
}
""",
        encoding="utf-8",
    )
    library = native_cxx.build_shared(
        [source],
        directory / "probe.so",
        compile_args=(
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            f"-I{ROOT / 'src'}",
        ),
    )
    lib = ct.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.physical.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        pointer,
        pointer,
    ]
    lib.response.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        pointer,
        pointer,
        pointer,
    ]
    return lib


def physical(
    point: ct.CDLL,
    q: Sequence[float],
    weights: tuple[float, float],
    layout: int = 0,
    pbe: bool = True,
) -> np.ndarray:
    out = np.empty(9)
    assert point.physical(layout, pbe, *weights, np.asarray(q, dtype=float), out)
    assert np.isfinite(out).all()
    return out


def response(
    point: ct.CDLL,
    q: Sequence[float],
    dq: Sequence[float],
    weights: tuple[float, float],
    restricted: bool = False,
    pbe: bool = True,
) -> np.ndarray:
    out = np.empty(9)
    assert point.response(
        restricted,
        pbe,
        *weights,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        out,
    )
    assert np.isfinite(out).all()
    return out[1:]


def assert_physical(actual: np.ndarray, expected: np.ndarray) -> None:
    tolerance = 5e-10 * np.abs(expected) + 1e-322
    assert np.all(np.abs(actual - expected) <= tolerance), (
        actual,
        expected,
        actual - expected,
    )


@pytest.mark.parametrize(
    "layout", (0, 1, 2), ids=("full-spin", "equal-spin", "bound-pbe0")
)
def test_all_87_physical_tail_oracles(point: ct.CDLL, layout: int) -> None:
    for row in TAIL:
        assert_physical(physical(point, row[:8].copy(), (0.75, 1.0), layout), row[8:])


def test_all_97_ordinary_independent_oracles(point: ct.CDLL) -> None:
    rows = np.loadtxt(ROOT / "tests/data/xc/scf_domain.tsv")
    assert len(rows) == 97
    for row in rows:
        assert_physical(
            physical(point, row[2:10].copy(), (1.0, 1.0), pbe=bool(row[0])), row[10:]
        )


def original_exchange(q: list[Any]) -> Any:
    """Original spin-scaled s² PBE expression, with no native ratio rewrite."""
    cx = mp.mpf(3) / 8 * (3 / mp.pi) ** (mp.mpf(1) / 3) * 4 ** (mp.mpf(2) / 3)
    kappa, beta = mp.mpf("0.804"), mp.mpf("0.06672455060314922")
    result = mp.mpf(0)
    for spin, density in enumerate(q[:2]):
        if not density:
            continue
        grad = q[2 + 3 * spin : 5 + 3 * spin]
        s2 = sum(g * g for g in grad) / (
            4 * (6 * mp.pi**2) ** (mp.mpf(2) / 3) * density ** (mp.mpf(8) / 3)
        )
        enhancement = 1 + kappa * (1 - kappa / (kappa + beta * mp.pi**2 / 3 * s2))
        result -= cx * density ** (mp.mpf(4) / 3) * enhancement
    return result


def original_weighted(q: list[Any], weights: tuple[float, float]) -> Any:
    x, c = map(mp.mpf, weights)
    if c == 0:
        return x * original_exchange(q)
    return c * original.energy(True, q) + (x - c) * original_exchange(q)


@cache
def oracle(
    q: tuple[float, ...],
    weights: tuple[float, float],
    direction: tuple[float, ...] | None = None,
    digits: int = 450,
) -> np.ndarray:
    """Differentiate independent original energy, never native potentials."""
    with mp.workdps(digits):
        coordinates = list(map(mp.mpf, q))
        scale = sum(coordinates[:2])
        dq = list(map(mp.mpf, direction)) if direction is not None else [mp.mpf(0)] * 8
        result = (
            [] if direction is not None else [original_weighted(coordinates, weights)]
        )
        for index in range(8):

            def potential(t: Any, coordinate: int = index) -> Any:
                shifted = [a + t * d for a, d in zip(coordinates, dq, strict=True)]

                def displaced(h: Any) -> Any:
                    moved = shifted.copy()
                    moved[coordinate] += h * scale
                    return original_weighted(moved, weights) / scale

                return mp.diff(displaced, 0)

            result.append(
                mp.diff(potential, 0) if direction is not None else potential(0)
            )
        return np.asarray([float(value) for value in result])


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("row", (8, 9, 65, 66, 67, 68))
def test_weighted_tail_physical_original_formula(
    point: ct.CDLL,
    weights: tuple[float, float],
    row: int,
    record_property: Callable[[str, object], None],
) -> None:
    q = tuple(TAIL[row, :8])
    expected = oracle(q, weights)
    exchange = oracle(q, (1.0, 0.0))
    correlation = oracle(q, (0.0, 1.0))
    magnitude = weights[0] * np.abs(exchange) + weights[1] * np.abs(correlation)
    # This NEW arbitrary-weight corpus includes severe PBE X/C cancellation.
    # Budget roundoff using independently differentiated component magnitudes,
    # as in the existing response corpus. The original 87/97 gates above have
    # no such term and remain unchanged. No native value sets this allowance.
    tolerance = 5e-10 * np.abs(expected) + 32 * np.finfo(float).eps * magnitude + 1e-322
    record_property(
        "components",
        json.dumps(
            {
                "exchange": exchange.tolist(),
                "correlation": correlation.tolist(),
                "reference": expected.tolist(),
                "tolerance": tolerance.tolist(),
            }
        ),
    )
    for layout in (0, 1):
        actual = physical(point, q, weights, layout)
        record_property(f"actual_{layout}", json.dumps(actual.tolist()))
        assert np.all(np.abs(actual - expected) <= tolerance), (actual, expected)


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("restricted", (False, True))
def test_tail_signed_response_original_formula(
    point: ct.CDLL,
    weights: tuple[float, float],
    restricted: bool,
    record_property: Callable[[str, object], None],
) -> None:
    q = TAIL[9, :8].copy()
    dq = np.array([3e-242, -7e-242, 1e-322, -2e-322, 3e-322, -2e-322, 1e-322, -1e-322])
    if restricted:
        dq[1], dq[5:] = dq[0], dq[2:5]
    expected = oracle(tuple(q), weights, tuple(dq))
    higher = oracle(tuple(q), weights, tuple(dq), 550)
    assert np.array_equal(expected, higher)
    record_property(
        "independent_direction",
        json.dumps(
            {
                "inputs": q.tolist(),
                "direction": dq.tolist(),
                "weights": weights,
                "reference_450": expected.tolist(),
                "reference_550": higher.tolist(),
            }
        ),
    )
    for sign in (-1, 1):
        actual = response(point, q, sign * dq, weights, restricted)
        record_property(f"actual_sign_{sign}", json.dumps(actual.tolist()))
        # Separate response gate on an independent Hessian action, not the E/V corpus.
        tolerance = 2e-9 * np.abs(expected) + 1e-100
        assert np.all(np.abs(actual - sign * expected) <= tolerance), (actual, expected)


@pytest.mark.parametrize("restricted", (False, True))
def test_existing_independent_response_corpus(point: ct.CDLL, restricted: bool) -> None:
    name = "rks_response" if restricted else "uks_response"
    rows = np.loadtxt(ROOT / f"tests/data/xc/{name}.tsv")
    assert len(rows) == (30 if restricted else 48)
    for row in rows:
        if restricted:
            q = np.r_[row[1] / 2, row[1] / 2, row[2:5] / 2, row[2:5] / 2]
            dq = np.r_[row[5] / 2, row[5] / 2, row[6:9] / 2, row[6:9] / 2]
            expected, magnitude = row[9:13], row[13:17]
        else:
            q, dq = row[1:9].copy(), row[9:17].copy()
            expected, magnitude = row[17:25], row[25:33]
        actual = response(point, q, dq, (1.0, 1.0), restricted, bool(row[0]))
        if restricted:
            actual = np.r_[actual[0], actual[2:5]]
        tolerance = (3e-11 if restricted else 3e-10) * np.abs(expected)
        tolerance += (32 if restricted else 64) * np.finfo(float).eps * magnitude
        tolerance += 8 * np.nextafter(0.0, 1.0)
        assert np.all(np.abs(actual - expected) <= tolerance), (actual, expected)


@pytest.mark.parametrize("weights", WEIGHTS)
def test_signed_interior_finite_difference(
    point: ct.CDLL, weights: tuple[float, float]
) -> None:
    q = np.array([0.4, 0.6, 0.15, -0.07, 0.12, -0.04, 0.05, 0.09])
    dq = np.array([0.08, -0.03, 0.03, 0.02, -0.01, -0.02, 0.04, 0.01])
    actual = response(point, q, dq, weights)
    for step in (1e-4, 3e-5):
        expected = (
            physical(point, q + step * dq, weights)[1:]
            - physical(point, q - step * dq, weights)[1:]
        ) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-6)
    np.testing.assert_array_equal(response(point, q, -dq, weights), -actual)


def test_interior_empty_minority_and_extreme_finite_controls(point: ct.CDLL) -> None:
    tiny = np.nextafter(0.0, 1.0)
    maximum = np.finfo(float).max
    controls = (
        ([1e-240] * 2 + [0.0] * 6, [3e-242] * 2 + [tiny, 0, 0] * 2),
        ([0.0, 1e-240] + [0.0] * 6, [0.0, 1e-242] + [0.0] * 3 + [tiny, 0, 0]),
        ([1e-300, 1e-240] + [0.0] * 6, [1e-302, 3e-242] + [0.0] * 3 + [tiny, 0, 0]),
        ([1e-240] * 2 + [maximum, 0, 0, -maximum, 0, 0], [0.0] * 8),
        ([1e230] * 2 + [0.0] * 6, [0.0] * 8),
    )
    for q, dq in controls:
        for weights in WEIGHTS:
            physical(point, q, weights)
            actual = response(point, q, dq, weights)
            np.testing.assert_array_equal(
                response(point, q, -np.asarray(dq), weights), -actual
            )
            np.testing.assert_array_equal(
                response(point, q, np.zeros(8), weights), np.zeros(8)
            )
    output = np.empty(9)
    q = np.array([0.0, 1.0] + [0.0] * 6)
    dq = np.zeros(8)
    dq[0] = 1e-30
    assert not point.response(0, True, 1.0, 1.0, q, dq, output)
    dq[0], dq[2] = 0.0, tiny
    assert not point.response(0, True, 1.0, 1.0, q, dq, output)
    q = np.array([maximum] * 2 + [0.0] * 6)
    assert not point.physical(0, True, 1.0, 1.0, q, output)
    assert not point.response(0, True, 1.0, 1.0, q, np.zeros(8), output)
    physical(point, [0.5, 0.5] + [0.0] * 6, (1.0, 1.0), layout=3)
    for q in (
        [4e-13] * 2 + [0.0] * 6,
        [1.0, 5e-11] + [0.0] * 6,
        [0.5, 0.5, 1e6 + 1, 0, 0, 0, 0, 0],
    ):
        assert not point.physical(3, True, 1.0, 1.0, np.array(q), output)


@pytest.mark.parametrize("zero_gradient", (False, True))
def test_subnormal_exchange_extreme_direction_oracle(
    point: ct.CDLL, zero_gradient: bool, record_property: Callable[[str, object], None]
) -> None:
    # Large *finite* relative density directions must not overflow a sum before
    # multiplication by the tiny reduced gradient. At grad=0 the nonzero
    # gradient Hessian action must remain, including subnormal directions.
    tiny = np.nextafter(0.0, 1.0)
    q = np.array([1e-240] * 2 + ([0.0] * 6 if zero_gradient else [tiny, 0, 0] * 2))
    dq = np.array(
        ([3e-242] * 2 + [tiny, 0, 0] * 2) if zero_gradient else [1.6e68] * 2 + [0.0] * 6
    )
    expected = oracle(tuple(q), (1.0, 0.0), tuple(dq))
    higher = oracle(tuple(q), (1.0, 0.0), tuple(dq), 550)
    assert np.array_equal(expected, higher)
    actual = response(point, q, dq, (1.0, 0.0))
    record_property(
        "independent_direction",
        json.dumps(
            {
                "inputs": q.tolist(),
                "direction": dq.tolist(),
                "weights": [1.0, 0.0],
                "reference_450": expected.tolist(),
                "reference_550": higher.tolist(),
                "actual": actual.tolist(),
            }
        ),
    )
    assert np.all(np.abs(actual - expected) <= 2e-9 * np.abs(expected) + 1e-100)
