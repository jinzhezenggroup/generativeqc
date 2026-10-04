"""Exact component reachability and native recurrence against interval quadrature.

The host probe compiles the real shared device arithmetic with CUDA annotations
removed. This proves dependency/jet semantics, not CUDA execution or speed.
"""

import ctypes
import itertools
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def recurrence(tmp_path_factory: pytest.TempPathFactory) -> Callable:
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], capture_output=True, check=True)
    folder = tmp_path_factory.mktemp("reachable-coulomb")
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#define __host__\n#define __device__\n"
        "#define __forceinline__ inline\n"
    )
    source, library = folder / "probe.cpp", folder / "probe.so"
    source.write_text(PROBE)
    built = subprocess.run(
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
            str(folder / "probe.o"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    assert built.returncode == 0, built.stderr
    subprocess.run(
        [compiler, "-shared", str(folder / "probe.o"), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    owner = ctypes.CDLL(str(library))
    owner.evaluate.argtypes = [ctypes.c_uint] * 6 + [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ]
    owner.evaluate.restype = ctypes.c_int
    return owner.evaluate


def _closure(x: int, y: int, z: int) -> set[tuple[int, ...]]:
    """Follow actual recurrence operands backwards from every consumer root."""
    seen = set()

    def visit(n: int, t: int, u: int, v: int) -> None:
        key = n, t, u, v
        if key in seen:
            return
        seen.add(key)
        for axis, power in enumerate((t, u, v)):
            if power:
                coordinates = [t, u, v]
                coordinates[axis] -= 1
                visit(n + 1, *coordinates)
                if power > 1:
                    coordinates[axis] -= 1
                    visit(n + 1, *coordinates)
                break

    for t, u, v in itertools.product(range(x + 1), range(y + 1), range(z + 1)):
        visit(0, t, u, v)
    return seen


def _oracle(t: int, u: int, v: int, moments: Sequence[float]) -> float:
    """Independent analytic derivative of the radial Gaussian integral."""
    powers = t, u, v
    result = 0.0
    for pairs in itertools.product(*(range(power // 2 + 1) for power in powers)):
        order = sum(powers) - sum(pairs)
        term = (-1.4) ** order * moments[order]
        for power, pair, coordinate in zip(powers, pairs, (0.19, -0.27, 0.11)):
            term *= (
                math.factorial(power)
                / (2**pair * math.factorial(pair) * math.factorial(power - 2 * pair))
                * coordinate ** (power - 2 * pair)
            )
        result += term
    return result


@pytest.mark.parametrize("radial", [0, 1, 2])
@pytest.mark.parametrize("order", range(13))
def test_actual_native_writes_match_dependency_closure_and_quadrature(
    recurrence: Callable, radial: int, order: int
) -> None:
    """Poison unused storage; check all 455 axis triples, full/LR/SR and jets."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    theta = 0.3 / math.sqrt(0.7 + 0.3**2)
    lower, upper = ((0.0, 1.0), (0.0, theta), (theta, 1.0))[radial]
    points = lower + (nodes + 1) * (upper - lower) / 2
    weights = weights * (upper - lower) / 2
    argument = 0.7 * (0.19**2 + 0.27**2 + 0.11**2)
    moments = [
        np.dot(weights, points ** (2 * n) * np.exp(-argument * points**2))
        for n in range(order + 2)
    ]
    for x in range(order + 1):
        for y in range(order - x + 1):
            z = order - x - y
            size = (x + 1) * (y + 1) * (z + 1) * 4
            dense, selected = np.empty(size), np.empty(size)
            assert recurrence(order, x, y, z, radial, 0, dense) == math.comb(
                order + 4, 4
            )
            assert recurrence(order, x, y, z, radial, 1, selected) == len(
                _closure(x, y, z)
            )
            # Identical dependencies and arithmetic preserve every output bit.
            np.testing.assert_array_equal(selected, dense)
            expected = []
            for t, u, v in itertools.product(range(x + 1), range(y + 1), range(z + 1)):
                expected.extend(
                    (
                        _oracle(t, u, v, moments),
                        _oracle(t + 1, u, v, moments),
                        _oracle(t, u + 1, v, moments),
                        _oracle(t, u, v + 1, moments),
                    )
                )
            np.testing.assert_allclose(selected, expected, rtol=2e-10, atol=2e-11)


def test_invalid_component_domain_retains_complete_recurrence(
    recurrence: Callable,
) -> None:
    """Malformed diagnostic extents must not underflow an unsigned loop bound."""
    expected, actual = np.empty(16), np.empty(16)
    assert recurrence(6, 1, 1, 0, 1, 0, expected) == 210
    assert recurrence(6, 1, 1, 0, 1, 1, actual) == 210
    np.testing.assert_array_equal(actual, expected)


PROBE = r"""
#include <cmath>
#include <limits>
#include "scf/cuda/coulomb_auxiliary.cuh"
using namespace generativeqc::scf::cuda_execution;
template<unsigned L> int run(unsigned x, unsigned y, unsigned z, unsigned radial,
                             unsigned selected, double* out) {
  CoulombAuxiliary<Dual3, L> auxiliary;
  const double poison = std::numeric_limits<double>::quiet_NaN();
  for (auto& item : auxiliary.data) item = {poison, poison, poison, poison};
  const Vec3<Dual3> p{{0.19,1,0,0},{-0.27,0,1,0},{0.11,0,0,1}};
  const Vec3<Dual3> q{{0,0,0,0},{0,0,0,0},{0,0,0,0}};
  const CoulombComponentDomain domain{x,y,z,selected != 0};
  if (radial == 0) fill_coulomb<L>(0.7, p, q, auxiliary, domain);
  else if (!fill_range_coulomb<L>(0.7, p, q,
      static_cast<generativeqc::integrals::CoulombRange>(radial), 0.3, auxiliary, domain))
    return -1;
  unsigned written = 0;
  for (const auto& item : auxiliary.data) written += !std::isnan(item.value);
  for (unsigned t=0;t<=x;++t) for (unsigned u=0;u<=y;++u) for (unsigned v=0;v<=z;++v) {
    const auto value = auxiliary.at(0,t,u,v);
    *out++=value.value; *out++=value.derivative_x;
    *out++=value.derivative_y; *out++=value.derivative_z;
  }
  return static_cast<int>(written);
}
extern "C" int evaluate(unsigned order, unsigned x, unsigned y, unsigned z,
                        unsigned radial, unsigned selected, double* out) {
  switch(order) {
#define CASE(L) case L: return run<L>(x,y,z,radial,selected,out)
    CASE(0); CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6);
    CASE(7); CASE(8); CASE(9); CASE(10); CASE(11); CASE(12);
#undef CASE
  }
  return -2;
}
"""
