"""Compiled exact predicate versus independent rational binary64 arithmetic."""

from __future__ import annotations

import ctypes
import math
import random
import struct
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.xc.physical_kinetic_predicate import (
    emit_exact_kinetic_comparison,
)

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture(scope="module")
def exact_comparison(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Any:
    """Compile the production emitter, not a copied implementation or stand-in."""
    folder = tmp_path_factory.mktemp("exact-physical-kinetic")
    source = folder / "predicate.cpp"
    source.write_text(
        "#include <cmath>\n"
        'extern "C" int compare(const double* input) {\n'
        "  const double rho[2]{input[0], input[1]};\n"
        "  const double gradient[2][3]{{input[2], input[3], input[4]},\n"
        "      {input[5], input[6], input[7]}};\n"
        "  const double tau[2]{input[8], input[9]};\n"
        f"  return {emit_exact_kinetic_comparison()};\n"
        "}\n"
    )
    binary = native_cxx.build_shared(
        [source], folder / "predicate.so", compile_args=("-std=c++20", "-O2")
    )
    library = ctypes.CDLL(str(binary))
    library.compare.argtypes = (ctypes.POINTER(ctypes.c_double),)
    library.compare.restype = ctypes.c_int
    return library


def _inside(inputs: np.ndarray) -> bool:
    """Use exact rationals of original bits, never rounded sigma intermediates."""
    rational = [Fraction(float(value)) for value in inputs]
    cap = 8 * (rational[0] + rational[1]) * (rational[8] + rational[9])
    sigma = sum((rational[2 + axis] + rational[5 + axis]) ** 2 for axis in range(3))
    return sigma <= cap


def _check(library: Any, inputs: list[float]) -> None:
    values = np.array(inputs, dtype=np.float64)
    actual = library.compare(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    assert bool(actual) == _inside(values), [float(value).hex() for value in values]


def test_exact_predicate_extremes_and_cancellation(exact_comparison: Any) -> None:
    """Subnormal squares and overflowing opposing gradients retain their sign."""
    tiny = math.ulp(0.0)
    huge = float.fromhex("0x1.fffffffffffffp1023")
    for value in (tiny, 1e-200, 1.0, 1e200, huge):
        _check(exact_comparison, [0, 0, value, 0, 0, -value, 0, 0, 0, 0])
        _check(exact_comparison, [0, 0, value, 0, 0, 0, 0, 0, 0, 0])
        _check(exact_comparison, [value, value, 0, 0, 0, 0, 0, 0, value, value])


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_exact_predicate_tie_and_adjacent_raw_sides(
    exact_comparison: Any, scale: float
) -> None:
    """An inclusive tie is not permission to admit its next representable exterior."""
    inputs = [scale * value for value in (0.1, 0.4, 0.2, 0, 0, 0.8, 0, 0, 0.05, 0.2)]
    assert _inside(np.array(inputs))
    _check(exact_comparison, inputs)
    for direction in (0.0, math.inf):
        displaced = inputs.copy()
        displaced[5] = math.nextafter(displaced[5], direction)
        assert _inside(np.array(displaced)) == (direction == 0.0)
        _check(exact_comparison, displaced)


def test_exact_predicate_random_binary64_inputs(exact_comparison: Any) -> None:
    """Deterministic independent bit patterns span the complete finite exponent range."""
    generator = random.Random(1118)

    def finite_value() -> float:
        while True:
            value = struct.unpack("d", struct.pack("Q", generator.getrandbits(64)))[0]
            if math.isfinite(value):
                return value

    for _ in range(512):
        inputs = [finite_value() for _ in range(10)]
        for index in (0, 1, 8, 9):
            inputs[index] = abs(inputs[index])
        _check(exact_comparison, inputs)
    for exponent in range(-500, 501, 25):
        density_a = math.ldexp(generator.random(), exponent)
        density_b = math.ldexp(generator.random(), exponent)
        inputs = [
            density_a,
            density_b,
            2 * density_a,
            0,
            0,
            2 * density_b,
            0,
            0,
            density_a / 2,
            density_b / 2,
        ]
        assert _inside(np.array(inputs))
        _check(exact_comparison, inputs)
