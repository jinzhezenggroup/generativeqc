"""Bounded mathematical domains for generated density/Hermite projection."""

from __future__ import annotations

import ctypes
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.integral.direct_density_gradient_cuda import (
    density_hermite_states,
    emit_density_direct_force_support,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from conftest import NativeCxx


@pytest.mark.parametrize("maximum", range(5))
def test_density_projection_covers_complete_polynomial_space(maximum: int) -> None:
    """Every supported monomial appears once, including the scalar response."""
    states = density_hermite_states(maximum)
    expected = {
        powers
        for powers in product(range(maximum + 1), repeat=3)
        if sum(powers) <= maximum
    }
    assert set(states) == expected
    assert len(states) == len(expected)
    assert all(sum(state) <= maximum for state in states)


@pytest.mark.parametrize("maximum", (-1, 5, 9))
def test_density_projection_rejects_unbounded_pairs(maximum: int) -> None:
    """A larger shell pair needs an explicitly proved workspace contract."""
    with pytest.raises(ValueError, match="s/p/d"):
        density_hermite_states(maximum)


@pytest.mark.parametrize("maximum", (True, False, 2.0, "2", None))
def test_density_projection_rejects_noninteger_domains(maximum: object) -> None:
    """Boolean or coerced domains cannot silently select a resource frame."""
    with pytest.raises(TypeError, match="integer"):
        density_hermite_states(maximum)


@pytest.fixture(scope="module")
def response_contraction(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Callable[..., int]:
    """Execute the actual emitted contraction over independent response inputs.

    Mock only CUDA lane ownership and the already-qualified radial/projection
    producers. A separate long-double direct atomic contraction supplies the
    oracle without reusing the emitted accumulation code.
    """
    captured = []

    def traversal(contraction: str, **projections: str) -> str:
        captured.append(contraction)
        return ""

    emit_density_direct_force_support(traversal, "")
    states = ",".join(
        f"{axis_x | (axis_y << 4) | (axis_z << 8)}U"
        for axis_x, axis_y, axis_z in density_hermite_states()
    )
    source = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
using std::pow;
using std::sqrt;
constexpr double kPi = 3.141592653589793238462643383279502884;
struct Response { double value, derivative[3]; };
struct DensityShared { Response hermite[2][35]; };
struct Census { unsigned long long hermite_products{}; };
void atomicAdd(unsigned long long* destination, unsigned long long value) {
  *destination += value;
}
struct Radial {
  const double* values;
  double at(unsigned, unsigned axis_x, unsigned axis_y, unsigned axis_z) const {
    return values[100 * axis_x + 10 * axis_y + axis_z];
  }
};
struct Recurrence {
  double coefficients[4], p, q, response_scales[3][3];
  Radial coulomb;
};
const unsigned kDirectDensityHermiteStates[] = {@STATES@};
unsigned direct_density_hermite_count(unsigned degree) {
  return (degree + 1) * (degree + 2) * (degree + 3) / 6;
}
extern "C" unsigned long long run(unsigned bra_degree, unsigned ket_degree,
    unsigned centers, unsigned lanes, const double* projection, const double* radial,
    const double* scales, double exponent, double* output) {
  const auto& density_shared = *reinterpret_cast<const DensityShared*>(projection);
  const unsigned li = std::min(2U, bra_degree), lj = bra_degree - li;
  const unsigned lk = std::min(2U, ket_degree), ll = ket_degree - lk;
  Recurrence recurrence{{1.0, -0.5, 0.75, 0.25}, exponent, 1.5 * exponent, {}, {radial}};
  std::copy(scales, scales + 9, &recurrence.response_scales[0][0]);
  std::fill(output, output + 9, 0.0);
  Census work;
  auto* census = &work;
  struct Lane { unsigned x; } threadIdx{}, blockDim{lanes};
  for (unsigned lane = 0; lane < lanes; ++lane) {
    threadIdx.x = lane;
    double gradients[3][1][3]{};
@CONTRACTION@
    for (unsigned center = 0; center + 1 < centers; ++center)
      for (unsigned axis = 0; axis < 3; ++axis)
        output[3 * center + axis] += gradients[center][0][axis];
  }
  return work.hermite_products;
}
""".replace("@STATES@", states).replace("@CONTRACTION@", captured[0])
    directory = tmp_path_factory.mktemp("density-response-contraction")
    path = directory / "contraction.cpp"
    path.write_text(source)
    library = native_cxx.build_shared(
        [path],
        directory / "contraction.so",
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off"),
    )
    handle = ctypes.CDLL(str(library))
    function = handle.run
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ctypes.c_uint] * 4 + [pointer] * 3 + [ctypes.c_double, pointer]
    function.restype = ctypes.c_ulonglong
    return function


@pytest.mark.parametrize("degrees", ((2, 4), (3, 3), (4, 2), (3, 4), (4, 3)))
@pytest.mark.parametrize("centers", (2, 3, 4))
@pytest.mark.parametrize("lanes", (1, 32, 256))
def test_emitted_response_matches_direct_atomic_contraction(
    response_contraction: Callable[..., int],
    degrees: tuple[int, int],
    centers: int,
    lanes: int,
) -> None:
    """Preserve signed responses and lane tails for every order-six/seven domain."""
    random = np.random.default_rng(1907 + 10 * degrees[0] + degrees[1])
    projections = random.normal(size=(2, 35, 4))
    radial = random.normal(size=(10, 10, 10))
    scales = random.uniform(-1, 1, size=(3, 3))
    pointer = ctypes.POINTER(ctypes.c_double)
    first_states, last_states = map(density_hermite_states, degrees)
    for exponent in (1e-3, 1.0, 1e3):
        output = np.zeros((3, 3))
        count = response_contraction(
            *degrees,
            centers,
            lanes,
            projections.ctypes.data_as(pointer),
            radial.ctypes.data_as(pointer),
            scales.ctypes.data_as(pointer),
            exponent,
            output.ctypes.data_as(pointer),
        )
        expected = np.zeros((3, 3), dtype=np.longdouble)
        weight = np.longdouble(-0.5 * 0.75 * 0.25) * 2 * np.longdouble(np.pi) ** 2.5
        weight /= (
            np.longdouble(exponent)
            * (1.5 * exponent)
            * np.sqrt(np.longdouble(2.5 * exponent))
        )
        for first_index, first_state in enumerate(first_states):
            bra = projections[0, first_index].astype(np.longdouble)
            for last_index, last_state in enumerate(last_states):
                ket = projections[1, last_index].astype(np.longdouble)
                powers = tuple(
                    first + last
                    for first, last in zip(first_state, last_state, strict=True)
                )
                parity = (-1) ** sum(last_state)
                for axis in range(3):
                    derivative = list(powers)
                    derivative[axis] += 1
                    responses = np.array(
                        [
                            bra[axis + 1] * ket[0] * radial[powers],
                            bra[0] * ket[axis + 1] * radial[powers],
                            bra[0] * ket[0] * radial[tuple(derivative)],
                        ],
                        dtype=np.longdouble,
                    )
                    for center in range(centers - 1):
                        expected[center, axis] += (
                            parity
                            * weight
                            * np.dot(scales[center].astype(np.longdouble), responses)
                        )
        assert count == len(first_states) * len(last_states)
        np.testing.assert_allclose(output, expected, rtol=3e-12, atol=2e-11)
