"""Qualify bounded shared pair lowering against independent Decimal algebra."""

from __future__ import annotations

import ctypes as ct
import itertools
import math
import random
from decimal import Decimal, localcontext
from typing import Any

import pytest

from tools.generate_nonlocal_pair_native import native_header


@pytest.fixture(scope="module")
def mixed_pair(tmp_path_factory: Any, native_cxx: Any) -> Any:
    """Compile the real generated FP32 closures, with no runtime or GPU oracle."""
    root = tmp_path_factory.mktemp("nonlocal-mixed-pair")
    source = root / "pair.cpp"
    source.write_text(
        "#include <cmath>\nnamespace generativeqc::dft::nlc { "
        "enum class Vv10Variant { vv10, rvv10 }; }\n"
        + native_header()
        + r"""
using namespace generativeqc::dft::nlc;
template<Vv10Variant Variant, bool Features, bool Geometry>
bool evaluate(const double* input, double* output) {
  generated::PairValues value{};
  if (!generated::try_pair_values_fp32<Variant, Features, Geometry>(
      input[0], input[1], input[2], input[3], input[4], input[5], value)) return false;
  output[0] = value.phi; output[1] = value.dphi_domega;
  output[2] = value.dphi_dkappa; output[3] = value.dphi_dr2;
  return true;
}
extern "C" bool pair(int mode, const double* input, double* output) {
  switch (mode) {
"""
        + "\n".join(
            f"case {index}: return evaluate<Vv10Variant::{variant}, "
            f"{str(features).lower()}, {str(geometry).lower()}>(input, output);"
            for index, (variant, features, geometry) in enumerate(
                itertools.product(("vv10", "rvv10"), (False, True), (False, True))
            )
        )
        + r"""
  default: return false;
  }
}
extern "C" bool bilateral(const double* input, double* output) {
  generated::BilateralPairValues value{};
  if (!generated::try_pair_values_vv10_bilateral_fp32<true>(
      input[0], input[1], input[2], input[3], input[4], value)) return false;
  output[0] = value.phi; output[1] = value.dphi_domega;
  output[2] = value.dphi_dkappa; output[3] = value.dphi_dr2;
  output[4] = value.dphi_domega_j; output[5] = value.dphi_dkappa_j;
  return true;
}
"""
    )
    library = native_cxx.build_shared(
        [source],
        root / "pair.so",
        compile_args=["-std=c++17", "-O2", "-ffp-contract=off"],
    )
    dll = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    dll.pair.argtypes = [ct.c_int, pointer, pointer]
    dll.pair.restype = ct.c_bool
    dll.bilateral.argtypes = [pointer, pointer]
    dll.bilateral.restype = ct.c_bool
    return dll


def _reference(variant: str, values: tuple[float, ...]) -> list[float]:
    """Differentiate the pair energy with 90-digit scalar arithmetic."""
    with localcontext() as context:
        context.prec = 90
        r2, wi, wj, ki, kj, inverse = map(Decimal.from_float, values)
        if variant == "vv10":
            gi, gj = wi * r2 + ki, wj * r2 + kj
            phi = -Decimal("1.5") / (gi * gj * (gi + gj))
            di = -phi * (1 / gi + 1 / (gi + gj))
            dj = -phi * (1 / gj + 1 / (gi + gj))
            result = [phi, r2 * di, di, wi * di + wj * dj, r2 * dj, dj]
        else:
            zi, zj = 1 + wi * r2, 1 + wj * r2
            phi = -Decimal("1.5") / (ki * kj * zi * zj * (zi + zj))
            logarithmic_i = 1 / zi + 1 / (zi + zj)
            result = [
                phi,
                -phi * r2 * inverse * logarithmic_i,
                phi * inverse * (-Decimal("1.5") + (zi - 1) * logarithmic_i),
                -phi * (wi / zi + wj / zj + (wi + wj) / (zi + zj)),
            ]
        return list(map(float, result))


@pytest.mark.parametrize("mode", range(8))
def test_mixed_pair_matches_independent_decimal(mixed_pair: Any, mode: int) -> None:
    variant, features, geometry = tuple(
        itertools.product(("vv10", "rvv10"), (False, True), (False, True))
    )[mode]
    rng = random.Random(69281)
    cases = [
        (0.0, 1.0, 2.0, 0.5, 3.0, 1.0),
        (4096.0, 256.0, 256.0, 256.0, 256.0, 1 / 256),
        (0.0, 1 / 256, 1 / 256, 1 / 256, 1 / 256, 256.0),
    ]
    cases += [
        (distance, *scales)
        for distance in (0.0, 2**-16, 4096.0)
        for scales in itertools.product((1 / 256, 256.0), repeat=5)
    ]
    cases += [
        (2 ** rng.uniform(-12, 12), *(2 ** rng.uniform(-8, 8) for _ in range(5)))
        for _ in range(128)
    ]
    for values in cases:
        actual = (ct.c_double * 4)()
        assert mixed_pair.pair(mode, (ct.c_double * 6)(*values), actual)
        expected = _reference(variant, values)[:4]
        if not features:
            expected[1:3] = [0.0, 0.0]
        if not geometry:
            expected[3] = 0.0
        assert all(math.isfinite(value) for value in actual)
        assert list(actual) == pytest.approx(expected, rel=2e-4, abs=1e-30)
        if variant == "vv10" and features and geometry:
            bilateral = (ct.c_double * 6)()
            assert mixed_pair.bilateral((ct.c_double * 6)(*values), bilateral)
            assert list(bilateral) == pytest.approx(
                _reference(variant, values), rel=3e-6, abs=1e-30
            )


@pytest.mark.parametrize("slot", range(6))
@pytest.mark.parametrize("invalid", [0.0, -1.0, 1e-300, 1e300, math.inf, math.nan])
def test_mixed_pair_rejects_extremes_without_publication(
    mixed_pair: Any, slot: int, invalid: float
) -> None:
    values = [1.0] * 6
    values[slot] = invalid
    output = (ct.c_double * 4)(*([9182.5] * 4))
    admitted = mixed_pair.pair(7, (ct.c_double * 6)(*values), output)
    expected = slot == 0 and (invalid == 0 or 2**-16 <= invalid <= 4096)
    assert admitted == expected
    if not admitted:
        assert list(output) == [9182.5] * 4


@pytest.mark.parametrize("slot", range(5))
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("direction", [-math.inf, None, math.inf])
def test_bilateral_admission_keeps_fp64_domain_edges(
    mixed_pair: Any, slot: int, upper: bool, direction: float | None
) -> None:
    """Scale proofs must be decided before adjacent FP64 inputs round to float.

    Test both point legs and the independently varying radius. The first value
    outside a boundary must leave the caller's output untouched, even when its
    FP32 representation is identical to the admitted boundary's representation.
    """
    lower_bound, upper_bound = (2**-16, 4096.0) if slot == 0 else (2**-8, 256.0)
    boundary = upper_bound if upper else lower_bound
    value = boundary if direction is None else math.nextafter(boundary, direction)
    inputs = [1.0] * 6
    inputs[slot] = value
    output = (ct.c_double * 6)(*[9182.5] * 6)
    admitted = mixed_pair.bilateral((ct.c_double * 6)(*inputs), output)
    assert admitted == (lower_bound <= value <= upper_bound)
    if admitted:
        assert list(output) == pytest.approx(
            _reference("vv10", tuple(inputs)), rel=3e-6, abs=1e-30
        )
    else:
        assert list(output) == [9182.5] * 6
