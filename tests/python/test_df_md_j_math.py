"""Independent host math for DF density-before-integrals contraction.

These tests qualify radial addressing and transpose products, including f/f/g;
they do not establish CUDA execution, source admission or endpoint speed.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_md_j_cuda import emit_df_md_j_header
from generativeqc_compiler.integral.md_j_engine_cuda import emit_md_radial_function
from test_md_j_engine_math import _closed_radial, _states

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def df_md_math(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Any:
    """Compile the emitted math through the required cached native toolchain."""
    folder = tmp_path_factory.mktemp("df-md-math")
    (folder / "generated_df_md_j.cuh").write_text(emit_df_md_j_header())
    source = folder / "probe.cpp"
    source.write_text(_PROBE)
    library = folder / "probe.so"
    required_native_cxx.build_shared(
        [source],
        library,
        compile_args=[
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
        compile_timeout=180,
    )
    owner = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        array,
        array,
        array,
        array,
        array,
        array,
    ]
    owner.evaluate.restype = ctypes.c_int
    return owner


@pytest.mark.parametrize(
    ("pair", "auxiliary"),
    [(p, a) for p in range(8) for a in range(6) if p + a <= 11],
)
@pytest.mark.parametrize("stride", (1, 3))
@pytest.mark.parametrize("rho", (1e-6, 1.0, 1e6))
@pytest.mark.parametrize("argument", (0.0, 1e-12, 6.0, 100.0))
def test_pair_and_auxiliary_products_match_closed_roots(
    df_md_math: Any,
    pair: int,
    auxiliary: int,
    stride: int,
    rho: float,
    argument: float,
) -> None:
    """All roots and signed products use independent 80-digit derivative sums."""
    displacement = np.array((1.0, -2.0, 3.0)) * math.sqrt(argument / rho / 14)
    bra, ket, roots = _states(pair), _states(auxiliary), _states(pair + auxiliary)
    rng = np.random.default_rng(971 + 5 * pair + auxiliary)
    density = rng.normal(size=len(bra))
    potential = rng.normal(size=len(ket))
    radial = np.full(stride * len(roots) + 2, 19.75)
    pair_output = np.full(len(bra) + 2, -31.25)
    auxiliary_output = np.full(len(ket) + 2, -31.25)
    assert (
        df_md_math.evaluate(
            pair,
            auxiliary,
            stride,
            rho,
            displacement,
            density,
            potential,
            radial,
            pair_output,
            auxiliary_output,
        )
        == 0
    )
    untouched = np.ones(len(radial), dtype=bool)
    untouched[1 : 1 + stride * len(roots) : stride] = False
    np.testing.assert_array_equal(radial[untouched], 19.75)
    for output in (pair_output, auxiliary_output):
        np.testing.assert_array_equal(output[[0, -1]], -31.25)

    expected_roots, root_bounds = _closed_radial(
        pair + auxiliary, rho, tuple(displacement)
    )
    actual_roots = radial[1 : 1 + stride * len(roots) : stride]
    assert np.isfinite(actual_roots).all()
    assert np.all(np.abs(actual_roots - expected_roots) <= 3e-12 * root_bounds + 1e-290)
    positions = {state: index for index, state in enumerate(roots)}
    addresses = np.array(
        [
            [positions[tuple(a + b for a, b in zip(x, y, strict=True))] for y in ket]
            for x in bra
        ]
    )
    matrix = expected_roots[addresses] * np.array([(-1) ** sum(s) for s in ket])
    bounds = root_bounds[addresses]
    for actual, expected, allowance in (
        (pair_output[1:-1], matrix @ potential, bounds @ np.abs(potential)),
        (auxiliary_output[1:-1], matrix.T @ density, bounds.T @ np.abs(density)),
    ):
        assert np.isfinite(actual).all()
        assert np.all(np.abs(actual - expected) <= 5e-12 * allowance + 1e-290)


@pytest.mark.parametrize("order", (-1, 12, True, 1.5))
def test_radial_emitter_rejects_unqualified_orders(order: Any) -> None:
    with pytest.raises(ValueError, match="orders"):
        emit_md_radial_function(order)


@pytest.mark.parametrize("name", ("", "invalid-name", "a::b", "9radial"))
def test_radial_emitter_rejects_invalid_cpp_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        emit_md_radial_function(10, name)


_PROBE = r"""
#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline
#include "generated_df_md_j.cuh"
using namespace generativeqc::scf::cuda_execution;

template<unsigned Pair, unsigned Auxiliary, unsigned Stride>
void products(double rho, const double* d, const double* density, const double* potential,
              double* radial, double* pair_result, double* auxiliary_result) {
  Vec3<double> p{d[0], d[1], d[2]}, q{0,0,0};
  df_md_radial<Pair + Auxiliary, Stride>(rho, p, q, radial + 1);
  for (unsigned i = 0; i < df_md_count(Pair); ++i)
    pair_result[i + 1] = df_md_pair_potential<Pair, Auxiliary, Stride>(radial + 1, potential, i);
  for (unsigned i = 0; i < df_md_count(Auxiliary); ++i)
    auxiliary_result[i + 1] = df_md_auxiliary_charge<Pair, Auxiliary, Stride>(radial + 1, density, i);
}
extern "C" int evaluate(unsigned pair, unsigned auxiliary, unsigned stride, double rho,
                         const double* d, const double* density, const double* potential,
                         double* radial, double* pair_result, double* auxiliary_result) {
  switch (6 * pair + auxiliary) {
#define CASE(P,A) case 6*P+A: \
  if (stride == 1) products<P,A,1>(rho,d,density,potential,radial,pair_result,auxiliary_result); \
  else if (stride == 3) products<P,A,3>(rho,d,density,potential,radial,pair_result,auxiliary_result); \
  else return -2; return 0
#define ROW(P) CASE(P,0); CASE(P,1); CASE(P,2); CASE(P,3); CASE(P,4); CASE(P,5)
    ROW(0); ROW(1); ROW(2); ROW(3); ROW(4); ROW(5); ROW(6);
    CASE(7,0); CASE(7,1); CASE(7,2); CASE(7,3); CASE(7,4);
#undef ROW
#undef CASE
  }
  return -1;
}
"""
