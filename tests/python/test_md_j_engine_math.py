"""Execute compact MD math against full auxiliary and independent 80-digit roots.

These are compiled-host arithmetic contracts, not CUDA performance or device
qualification. The independent oracle differentiates the Boys integral in a
closed Cartesian polynomial using factorial multiplicities and mpmath moments;
it shares neither production recurrence states nor generated address tables.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import pytest
from generativeqc_compiler.integral.md_j_engine_cuda import (
    _emit_primitive_norms,
    emit_md_j_engine_header,
)

ROOT = Path(__file__).resolve().parents[2]
ANGULAR_PAIRS = tuple((bra, ket) for bra in range(5) for ket in range(5))


def _states(order: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (t, u, v)
        for t in range(order + 1)
        for u in range(order - t + 1)
        for v in range(order - t - u + 1)
    )


@pytest.fixture(scope="module")
def md_engine(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual emitted consumer and native reference, through a cache."""
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    requested_cache = os.environ.get("CCACHE")
    cache_launcher = (
        shutil.which(requested_cache)
        if requested_cache
        else shutil.which("sccache") or shutil.which("ccache")
    )
    if compiler is None or cache_launcher is None:
        pytest.fail(
            "MD arithmetic qualification requires a C++ compiler and sccache/ccache"
        )
    subprocess.run(
        [cache_launcher, "--version"], check=True, capture_output=True, text=True
    )
    folder = tmp_path_factory.mktemp("md-engine-math")
    (folder / "cuda_runtime_api.h").write_text(
        "#pragma once\n#define __host__\n#define __device__\n#define __constant__\n"
        "#define __forceinline__ inline\nusing cudaStream_t = void*;\nusing cudaError_t = int;\n"
    )
    (folder / "generated_md_j_engine.cuh").write_text(emit_md_j_engine_header())
    source, obj, library = folder / "probe.cpp", folder / "probe.o", folder / "probe.so"
    # Execute the emitted CUDA-only bound with outward host arithmetic doubles.
    # Independent high-precision Coulomb derivatives below supply its oracle.
    source.write_text(
        PROBE
        + NORM_INTRINSICS
        + _emit_primitive_norms().replace("#if defined(__CUDACC__)", "#if 1")
        + '\nextern "C" double primitive_norm(double p,unsigned l,unsigned i) {'
        + "return md_engine_hermite_norm(p,l,i);}\n"
    )
    subprocess.run(
        [
            cache_launcher,
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
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    owner.radial_values.argtypes = [ctypes.c_uint, ctypes.c_double, array, array, array]
    owner.radial_values.restype = ctypes.c_int
    owner.contract_values.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        array,
        array,
        array,
        ctypes.c_uint,
        ctypes.c_uint,
        array,
        array,
        array,
    ]
    owner.contract_values.restype = ctypes.c_int
    owner.primitive_norm.argtypes = [ctypes.c_double, ctypes.c_uint, ctypes.c_uint]
    owner.primitive_norm.restype = ctypes.c_double
    return owner


@cache
def _closed_radial(
    order: int, rho: float, displacement: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Closed derivatives of integral_0^1 exp(-rho*r*r*s*s) ds at 80 digits."""
    with mp.workdps(80):
        exponent = mp.mpf(rho)
        xyz = tuple(mp.mpf(value) for value in displacement)
        argument = exponent * sum(value * value for value in xyz)
        moments = tuple(
            mp.mpf(1) / (2 * n + 1)
            if argument == 0
            else mp.gammainc(mp.mpf(n) + mp.mpf("0.5"), 0, argument)
            / (2 * argument ** (mp.mpf(n) + mp.mpf("0.5")))
            for n in range(order + 1)
        )
        result, bounds = [], []
        for powers in _states(order):
            terms = []
            for kx in range(powers[0] // 2 + 1):
                for ky in range(powers[1] // 2 + 1):
                    for kz in range(powers[2] // 2 + 1):
                        pairs = (kx, ky, kz)
                        n = sum(powers) - sum(pairs)
                        term = (-2 * exponent) ** n * moments[n]
                        for degree, pair_count, coordinate in zip(
                            powers, pairs, xyz, strict=True
                        ):
                            coefficient = math.factorial(degree) // (
                                2**pair_count
                                * math.factorial(pair_count)
                                * math.factorial(degree - 2 * pair_count)
                            )
                            term *= coefficient * coordinate ** (
                                degree - 2 * pair_count
                            )
                        terms.append(term)
            result.append(float(mp.fsum(terms)))
            bounds.append(float(mp.fsum(abs(term) for term in terms)))
    return np.asarray(result), np.asarray(bounds)


# Cover diffuse/tight exponents, zero displacement and every shared Boys branch
# threshold, as well as the asymptotic high-T tail, with all Cartesian roots.
RADIAL_CASES = tuple(
    (rho, argument, direction)
    for rho in (1e-10, 0.03, 1.0, 100.0, 1e6)
    for argument, direction in (
        (0.0, (1.0, 0.0, 0.0)),
        (1e-12, (1.0, -2.0, 3.0)),
        (0.249999, (2.0, 0.0, -1.0)),
        (0.75, (1.0, 2.0, -3.0)),
        (1.25, (-3.0, 2.0, 1.0)),
        (2.0, (1.0, -2.0, 3.0)),
        (5.999999, (1.0, -2.0, 3.0)),
        (6.0, (1.0, -2.0, 3.0)),
        (100.0, (1.0, -2.0, 3.0)),
        (1e4, (1.0, 2.0, -3.0)),
    )
)


@pytest.mark.parametrize("p", [1e-20, 0.03, 1.0, 100.0, 1e20])
def test_hermite_norm_bounds_independent_coulomb_derivatives(
    md_engine: Any, p: float
) -> None:
    """Every s/p/d pair derivative, including mixed Cartesian components."""
    for angular in range(5):
        radial, _ = _closed_radial(2 * angular, p / 2, (0.0, 0.0, 0.0))
        positions = {state: i for i, state in enumerate(_states(2 * angular))}
        for index, state in enumerate(_states(angular)):
            squared = (
                2
                * math.pi**2.5
                / (p * p * math.sqrt(2 * p))
                * (-1) ** sum(state)
                * radial[positions[tuple(2 * n for n in state)]]
            )
            actual = md_engine.primitive_norm(p, angular, index)
            expected = math.sqrt(squared)
            assert expected <= actual <= expected * 1.001


@pytest.mark.parametrize("p", [0.0, -1.0, 1e-200, 1e200, math.inf, math.nan])
def test_unrepresentable_hermite_norm_fails_open(md_engine: Any, p: float) -> None:
    assert math.isinf(md_engine.primitive_norm(p, 4, 34))


NORM_INTRINSICS = r"""
using std::isfinite;
double __dmul_rd(double a,double b) { return std::nextafter(a*b,0.0); }
double __dmul_ru(double a,double b) { return std::nextafter(a*b,INFINITY); }
double __ddiv_ru(double a,double b) { return std::nextafter(a/b,INFINITY); }
double __dsqrt_rd(double a) { return std::nextafter(std::sqrt(a),0.0); }
double __dsqrt_ru(double a) { return std::nextafter(std::sqrt(a),INFINITY); }
"""


@pytest.mark.parametrize("order", range(9))
@pytest.mark.parametrize("rho,argument,direction", RADIAL_CASES)
def test_compact_radial_matches_full_auxiliary_and_high_precision(
    md_engine: Any,
    order: int,
    rho: float,
    argument: float,
    direction: tuple[float, ...],
) -> None:
    displacement = np.asarray(direction) * math.sqrt(
        argument / rho / sum(x * x for x in direction)
    )
    count = math.comb(order + 3, 3)
    compact = np.full(count + 2, np.nan)
    compact[[0, -1]] = (23.5, -71.25)
    full = np.empty(count)
    assert md_engine.radial_values(order, rho, displacement, compact, full) == 0
    np.testing.assert_array_equal(compact[[0, -1]], (23.5, -71.25))
    expected, bound = _closed_radial(order, rho, tuple(displacement))
    assert np.all(np.isfinite(compact[1:-1]))
    np.testing.assert_allclose(compact[1:-1], full, rtol=2e-14, atol=1e-290)
    error = np.abs(compact[1:-1] - expected)
    assert np.all(error <= 3e-12 * bound + 1e-290), (order, rho, argument, error.max())


@pytest.mark.parametrize("order", range(9))
def test_displacement_reversal_has_exact_cartesian_parity(
    md_engine: Any, order: int
) -> None:
    displacement = np.asarray((0.23, -0.51, 0.79))
    count = math.comb(order + 3, 3)
    forward, reverse, full = np.empty(count + 2), np.empty(count + 2), np.empty(count)
    assert md_engine.radial_values(order, 0.63, displacement, forward, full) == 0
    assert md_engine.radial_values(order, 0.63, -displacement, reverse, full) == 0
    phase = np.asarray([(-1) ** sum(state) for state in _states(order)])
    np.testing.assert_array_equal(forward[1:-1], phase * reverse[1:-1])


@pytest.mark.parametrize("bra,ket", ANGULAR_PAIRS)
@pytest.mark.parametrize(
    "directions", (0, 1, 2, 3), ids=("neither", "forward", "reverse", "both")
)
@pytest.mark.parametrize("lanes", (1, 7, 32))
def test_all_pair_contractions_match_independent_reciprocal_full_auxiliaries(
    md_engine: Any, bra: int, ket: int, directions: int, lanes: int
) -> None:
    """Includes all 15 canonical angular pairs and the ten reversed views."""
    counts = (math.comb(bra + 3, 3), math.comb(ket + 3, 3))
    size = sum(counts)
    positions = np.asarray((0.31, -0.17, 0.51, -0.19, 0.43, -0.07))
    exponents = np.asarray((0.17, 3.1))
    density = np.random.default_rng(71 + 5 * bra + ket).normal(size=size + 7)
    values, cooperative, reference = (np.full(size + 2, np.nan) for _ in range(3))
    for output in (values, cooperative, reference):
        output[[0, -1]] = (23.5, -71.25)
    assert (
        md_engine.contract_values(
            bra,
            ket,
            exponents,
            positions,
            density,
            directions,
            lanes,
            values,
            cooperative,
            reference,
        )
        == 0
    )
    for output in (values, cooperative, reference):
        np.testing.assert_array_equal(output[[0, -1]], (23.5, -71.25))
    np.testing.assert_allclose(values[1:-1], reference[1:-1], rtol=3e-13, atol=3e-13)
    np.testing.assert_array_equal(values, cooperative)
    if not directions & 1:
        np.testing.assert_array_equal(values[1 : 1 + counts[0]], np.zeros(counts[0]))
    if not directions & 2:
        np.testing.assert_array_equal(values[1 + counts[0] : -1], np.zeros(counts[1]))

    # A second reference uses closed high-precision roots and direct tensor
    # summation, sharing neither the native auxiliary nor generated lookup map.
    rho = exponents[0] * exponents[1] / exponents.sum()
    roots, _ = _closed_radial(bra + ket, rho, tuple(positions[:3] - positions[3:]))
    lookup = dict(zip(_states(bra + ket), roots, strict=True))
    bra_states, ket_states = _states(bra), _states(ket)
    matrix = np.asarray(
        [
            [
                (-1) ** sum(second)
                * lookup[tuple(a + b for a, b in zip(first, second, strict=True))]
                for second in ket_states
            ]
            for first in bra_states
        ]
    )
    prefactor = 2 * math.pi**2.5 / (np.prod(exponents) * math.sqrt(exponents.sum()))
    expected = np.concatenate(
        (
            prefactor * matrix @ density[2 + counts[0] + 3 : 2 + size + 3]
            if directions & 1
            else np.zeros(counts[0]),
            prefactor * matrix.T @ density[2 : 2 + counts[0]]
            if directions & 2
            else np.zeros(counts[1]),
        )
    )
    np.testing.assert_allclose(values[1:-1], expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("angular", range(5))
def test_self_source_uses_full_forward_weight_without_reciprocal_half(
    md_engine: Any, angular: int
) -> None:
    count = math.comb(angular + 3, 3)
    exponents = np.asarray((0.61, 0.61))
    positions = np.zeros(6)
    density = np.random.default_rng(819 + angular).normal(size=2 * count + 7)
    outputs = [np.full(2 * count + 2, np.nan) for _ in range(3)]
    assert (
        md_engine.contract_values(
            angular,
            angular,
            exponents,
            positions,
            density,
            1 | 4,
            32,
            *outputs,
        )
        == 0
    )
    np.testing.assert_allclose(
        outputs[0][1:-1], outputs[2][1:-1], rtol=3e-14, atol=1e-14
    )
    np.testing.assert_array_equal(outputs[0][1:-1], outputs[1][1:-1])
    np.testing.assert_array_equal(outputs[0][count + 1 : -1], np.zeros(count))
    assert np.linalg.norm(outputs[0][1 : count + 1]) > 0


def test_generation_is_standalone_deterministic_and_preserves_unchanged_output(
    tmp_path: Path,
) -> None:
    """Generated bytes are a build contract, independent of runtime importability."""
    output = tmp_path / "generated_md_j_engine.cuh"
    command = [
        sys.executable,
        str(ROOT / "tools/generate_md_j_engine.py"),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.read_text() == emit_md_j_engine_header()
    before = output.stat().st_mtime_ns
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.stat().st_mtime_ns == before


def test_generation_import_boundary_does_not_load_runtime_or_oracles() -> None:
    """The production compiler must remain usable before native installation."""
    script = r"""
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'torch', 'cupy'}:
            raise AssertionError('source generation attempted runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, sys.argv[1])
from generativeqc_compiler.integral.md_j_engine_cuda import emit_md_j_engine_header
assert emit_md_j_engine_header()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "python")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


PROBE = r"""
#include "generated_md_j_engine.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/md_hermite_index.cuh"
using namespace generativeqc::scf::cuda_execution;

template<unsigned Order>
void radial_impl(double rho, const double* d, double* compact, double* full) {
  Vec3<double> p{d[0], d[1], d[2]}, q{0.0, 0.0, 0.0};
  md_engine_radial<Order>(rho, p, q, compact + 1);
  CoulombAuxiliary<double, Order> auxiliary;
  fill_coulomb<Order>(rho, p, q, auxiliary);
  for (unsigned i = 0; i < md_j_hermite_count(Order); ++i) {
    const auto state = md_hermite_angular(Order, i);
    full[i] = auxiliary.at(0, state.x, state.y, state.z);
  }
}
extern "C" int radial_values(unsigned order, double rho, const double* d,
                              double* compact, double* full) {
  switch (order) {
#define CASE(N) case N: radial_impl<N>(rho, d, compact, full); return 0
    CASE(0); CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
#undef CASE
  }
  return -1;
}

template<unsigned B, unsigned K>
void contract_impl(const double* exponent, const double* position, const double* density,
                   unsigned flags, unsigned lanes, double* result, double* cooperative,
                   double* reference) {
  constexpr unsigned nb = md_j_hermite_count(B), nk = md_j_hermite_count(K);
  MdJPrimitive bra{}, ket{};
  bra.exponent = exponent[0]; ket.exponent = exponent[1];
  bra.product = {position[0], position[1], position[2]};
  ket.product = {position[3], position[4], position[5]};
  bra.hermite_offset = 2; ket.hermite_offset = 2 + nb + 3;
  if (flags & 4U) ket = bra;
  MdJView md{};
  md.density = const_cast<double*>(density);
  const bool forward = flags & 1U, reverse = flags & 2U;
  md_engine_contract<B,K>(bra, ket, md, forward, reverse, result + 1, result + 1 + nb);
  const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
  const double prefactor = 2.0 * pow(kPi, 2.5) /
      (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
  double radial[md_j_hermite_count(B + K)];
  md_engine_radial<B+K>(rho, bra.product, ket.product, radial);
  for (unsigned lane = 0; lane < lanes; ++lane)
    md_engine_contract_radial<B,K>(radial, density + bra.hermite_offset,
        density + ket.hermite_offset, forward, reverse, cooperative + 1,
        cooperative + 1 + nb, lane, lanes, prefactor);
  CoulombAuxiliary<double, B+K> first, second;
  fill_coulomb<B+K>(rho, bra.product, ket.product, first);
  fill_coulomb<B+K>(rho, ket.product, bra.product, second);
  for (unsigned i = 0; i < nb; ++i) {
    const auto a = md_hermite_angular(B, i);
    double value = 0.0;
    if (forward) for (unsigned j = 0; j < nk; ++j) {
      const auto b = md_hermite_angular(K, j);
      const double sign = (b.x + b.y + b.z) & 1U ? -1.0 : 1.0;
      value += sign * density[ket.hermite_offset + j] *
               first.at(0, a.x+b.x, a.y+b.y, a.z+b.z);
    }
    reference[1+i] = prefactor * value;
  }
  for (unsigned j = 0; j < nk; ++j) {
    const auto b = md_hermite_angular(K, j);
    double value = 0.0;
    if (reverse) for (unsigned i = 0; i < nb; ++i) {
      const auto a = md_hermite_angular(B, i);
      const double sign = (a.x + a.y + a.z) & 1U ? -1.0 : 1.0;
      value += sign * density[bra.hermite_offset + i] *
               second.at(0, a.x+b.x, a.y+b.y, a.z+b.z);
    }
    reference[1+nb+j] = prefactor * value;
  }
}
extern "C" int contract_values(unsigned bra, unsigned ket, const double* exponent,
    const double* position, const double* density, unsigned flags, unsigned lanes,
    double* result, double* cooperative, double* reference) {
  if (lanes == 0) return -2;
  switch (5 * bra + ket) {
#define CASE(B,K) case 5*B+K: contract_impl<B,K>(exponent, position, density, flags, lanes, result, cooperative, reference); return 0
#define ROW(B) CASE(B,0); CASE(B,1); CASE(B,2); CASE(B,3); CASE(B,4)
    ROW(0); ROW(1); ROW(2); ROW(3); ROW(4);
#undef ROW
#undef CASE
  }
  return -1;
}
"""
