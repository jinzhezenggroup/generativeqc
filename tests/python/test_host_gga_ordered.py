"""Executable bitwise scheduling, range, and scratch-lifetime controls for GGA pairs."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
POINTER = ct.POINTER(ct.c_double)


@pytest.fixture(scope="module")
def contractions(tmp_path_factory: Any, required_native_cxx: Any) -> ct.CDLL:
    """Compile the real internal owner; scalar controls bypass ISA dispatch."""
    directory = tmp_path_factory.mktemp("host-gga-ordered")
    source = directory / "probe.cpp"
    source.write_text(
        r"""
#include "dft/host_gga_ordered.hpp"
namespace ordered = generativeqc::dft::host_gga_ordered;
extern "C" void fields(std::size_t basis, std::size_t count, std::size_t spins,
    const double* ao, const double* density, const double* direction,
    double* primal, double* tangent, double* packing, std::size_t capacity, int scalar) {
  if (scalar) ordered::scalar_fields(basis, count, spins, ao, density, direction, primal, tangent);
  else ordered::fields(basis, count, spins, ao, density, direction, primal, tangent,
                       {packing, capacity});
}
extern "C" void pullback(std::size_t basis, std::size_t count, std::size_t spins,
    const double* ao, const double* coefficients, const double* weights, double* result,
    int scalar) {
  if (scalar) ordered::scalar_pullback(basis, count, spins, ao, coefficients, weights, result);
  else ordered::pullback(basis, count, spins, ao, coefficients, weights, result);
}
extern "C" void batch_fields(std::size_t basis, std::size_t count, std::size_t spins,
    std::size_t inputs, const double* ao, const double* densities, double* fields,
    double* packing, std::size_t capacity, int scalar) {
  if (scalar) ordered::scalar_batch_fields(basis, count, spins, inputs, ao, densities, fields);
  else ordered::batch_fields(basis, count, spins, inputs, ao, densities, fields, {packing, capacity});
}
extern "C" void batch_pullback(std::size_t basis, std::size_t count, std::size_t spins,
    std::size_t directions, const double* ao, const double* coefficients,
    const double* weights, double* result, int scalar) {
  if (scalar) ordered::scalar_batch_pullback(basis, count, spins, directions, ao, coefficients, weights, result);
  else ordered::batch_pullback(basis, count, spins, directions, ao, coefficients, weights, result);
}
"""
    )
    compiler = replace(
        required_native_cxx, cache=shutil.which("sccache") or required_native_cxx.cache
    )
    subprocess.run([compiler.cache, "--version"], check=True, capture_output=True)
    library = compiler.build_shared(
        [source],
        directory / "ordered.so",
        compile_args=[
            "-std=c++20",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
        ],
    )
    loaded = ct.CDLL(str(library))
    loaded.fields.argtypes = [ct.c_size_t] * 3 + [POINTER] * 6 + [ct.c_size_t, ct.c_int]
    loaded.fields.restype = None
    loaded.pullback.argtypes = [ct.c_size_t] * 3 + [POINTER] * 4 + [ct.c_int]
    loaded.pullback.restype = None
    loaded.batch_fields.argtypes = (
        [ct.c_size_t] * 4 + [POINTER] * 4 + [ct.c_size_t, ct.c_int]
    )
    loaded.batch_fields.restype = None
    loaded.batch_pullback.argtypes = [ct.c_size_t] * 4 + [POINTER] * 4 + [ct.c_int]
    loaded.batch_pullback.restype = None
    return loaded


def _pointer(values: np.ndarray) -> POINTER:
    return values.ctypes.data_as(POINTER)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("directions", (2, 3, 4))
@pytest.mark.parametrize("basis,points", ((1, 1), (3, 3), (17, 5), (81, 257)))
@pytest.mark.parametrize("capacity", ("none", "short", "exact"))
@pytest.mark.parametrize(
    "powers",
    (
        (0, 0, 0, 0),
        (-400, 700, -200, 300),
        (400, -700, 200, -300),
        (-530, 900, 800, -700),
        (510, -1000, -300, -600),
    ),
)
def test_independent_batch_fields_and_pullbacks_keep_bits_and_bounds(
    contractions: ct.CDLL,
    spins: int,
    directions: int,
    basis: int,
    points: int,
    capacity: str,
    powers: tuple[int, int, int, int],
) -> None:
    """Scalar fallback, SIMD direction tails and signed nonsymmetric Ds agree."""
    generator = np.random.default_rng(2848 + spins + directions + basis)
    ao = generator.normal(size=(4, points, basis))
    matrices = generator.normal(size=(directions + 1, spins, basis, basis))
    coefficients = generator.normal(size=(directions, 4 * spins, points))
    weights = generator.normal(size=points)
    weights[0] = -0.0
    initial = generator.normal(size=(directions, spins, basis, basis))
    ao = np.ldexp(ao, powers[0])
    matrices = np.ldexp(matrices, powers[1])
    coefficients = np.ldexp(coefficients, powers[2])
    weights = np.ldexp(weights, powers[3])
    packed_matrices = np.moveaxis(matrices, 0, -1).copy()
    packed_coefficients = np.moveaxis(coefficients, 0, -1).copy()
    expected_fields = []
    expected_pullback = initial.copy()
    empty = np.empty(0)
    for index in range(directions + 1):
        fields = np.empty((4 * spins, points))
        tangent = np.empty_like(fields)
        contractions.fields(
            basis,
            points,
            spins,
            _pointer(ao),
            _pointer(matrices[index]),
            _pointer(matrices[index]),
            _pointer(fields),
            _pointer(tangent),
            _pointer(empty),
            0,
            1,
        )
        expected_fields.append(fields)
        if index < directions:
            contractions.pullback(
                basis,
                points,
                spins,
                _pointer(ao),
                _pointer(coefficients[index]),
                _pointer(weights),
                _pointer(expected_pullback[index]),
                1,
            )
    expected_fields = np.array(expected_fields)
    available = {"none": 0, "short": 4 * basis - 1, "exact": 4 * basis}[capacity]
    for scalar in (1, 0):
        packing = np.full(available + 2, -713.25)
        field_storage = np.full(expected_fields.size + 2, -781.25)
        fields = field_storage[1:-1].reshape(expected_fields.shape)
        result_storage = np.full(initial.size + 2, -793.25)
        result = result_storage[1:-1].reshape(spins, basis, basis, directions)
        result[:] = np.moveaxis(initial, 0, -1)
        contractions.batch_fields(
            basis,
            points,
            spins,
            directions + 1,
            _pointer(ao),
            _pointer(packed_matrices),
            _pointer(fields),
            _pointer(packing[1:-1]),
            available,
            scalar,
        )
        contractions.batch_pullback(
            basis,
            points,
            spins,
            directions,
            _pointer(ao),
            _pointer(packed_coefficients),
            _pointer(weights),
            _pointer(result),
            scalar,
        )
        np.testing.assert_array_equal(
            fields.view(np.uint64), expected_fields.view(np.uint64)
        )
        np.testing.assert_array_equal(
            np.moveaxis(result, -1, 0).view(np.uint64),
            expected_pullback.view(np.uint64),
        )
        np.testing.assert_array_equal(field_storage[[0, -1]], -781.25)
        np.testing.assert_array_equal(result_storage[[0, -1]], -793.25)
        np.testing.assert_array_equal(packing[[0, -1]], -713.25)
        if scalar or capacity != "exact":
            np.testing.assert_array_equal(packing, -713.25)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("basis,points", ((1, 1), (3, 3), (4, 17), (17, 3), (81, 257)))
@pytest.mark.parametrize("capacity", ("none", "short", "exact"))
def test_ordered_fields_and_pullback_use_bounded_scratch(
    contractions: ct.CDLL, spins: int, basis: int, points: int, capacity: str
) -> None:
    """Exercise scalar capacity fallback, SIMD tails, and arbitrary ordered Ds."""
    generator = np.random.default_rng(2848 + basis + spins + points)
    ao = generator.normal(size=(4, points, basis))
    density = generator.normal(size=(spins, basis, basis))
    direction = generator.normal(size=density.shape)
    coefficients = generator.normal(size=(4 * spins, points))
    weights = generator.normal(size=points)
    available = {"none": 0, "short": 4 * basis - 1, "exact": 4 * basis}[capacity]
    packing = np.full(available + 2, -713.25)
    results = []
    for scalar in (1, 0):
        primal = np.full((4 * spins, points), np.nan)
        tangent = np.full_like(primal, np.nan)
        contractions.fields(
            basis,
            points,
            spins,
            _pointer(ao),
            _pointer(density),
            _pointer(direction),
            _pointer(primal),
            _pointer(tangent),
            _pointer(packing[1:]),
            available,
            scalar,
        )
        result = np.zeros_like(density)
        contractions.pullback(
            basis,
            points,
            spins,
            _pointer(ao),
            _pointer(coefficients),
            _pointer(weights),
            _pointer(result),
            scalar,
        )
        results.append((primal, tangent, result))
    for scalar, dispatched in zip(*results, strict=True):
        np.testing.assert_array_equal(
            scalar.view(np.uint64), dispatched.view(np.uint64)
        )
    np.testing.assert_array_equal(packing[[0, -1]], -713.25)
    if capacity != "exact":
        np.testing.assert_array_equal(packing, -713.25)
    for spin in range(spins):
        for fields, matrix in ((results[1][0], density), (results[1][1], direction)):
            reference = np.einsum("pi,ij,pj->p", ao[0], matrix[spin], ao[0])
            np.testing.assert_allclose(
                fields[4 * spin], reference, atol=3e-12, rtol=3e-12
            )
            for axis in range(3):
                reference = np.einsum("pi,ij,pj->p", ao[axis + 1], matrix[spin], ao[0])
                reference += np.einsum("pi,ij,pj->p", ao[0], matrix[spin], ao[axis + 1])
                np.testing.assert_allclose(
                    fields[4 * spin + axis + 1], reference, atol=3e-12, rtol=3e-12
                )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize(
    "ao_scale,matrix_scale,coefficient_scale",
    (
        (1.0, 1.0, 1.0),
        (1e-160, 1e300, 1e300),
        (1e150, 1e-300, np.nextafter(0.0, 1.0)),
        (1e-150, 1e300, 1e-300),
        (1e150, 1e150, 1e150),
    ),
)
def test_ordered_range_zero_and_overflow_controls(
    contractions: ct.CDLL,
    spins: int,
    ao_scale: float,
    matrix_scale: float,
    coefficient_scale: float,
) -> None:
    """No half-coefficient underflow, FMA/reassociation, or hidden density floor."""
    basis, points = 17, 3
    generator = np.random.default_rng(2848 + spins)
    ao = generator.normal(size=(4, points, basis)) * ao_scale
    density = generator.normal(size=(spins, basis, basis)) * matrix_scale
    density[:, 0] = -0.0
    coefficients = generator.normal(size=(4 * spins, points)) * coefficient_scale
    weights = np.array((0.0, -1.0, 0.25))
    packing = np.zeros(4 * basis)
    results = []
    for scalar in (1, 0):
        primal = np.empty((4 * spins, points))
        tangent = np.empty_like(primal)
        contractions.fields(
            basis,
            points,
            spins,
            _pointer(ao),
            _pointer(density),
            _pointer(density),
            _pointer(primal),
            _pointer(tangent),
            _pointer(packing),
            packing.size,
            scalar,
        )
        result = np.zeros_like(density)
        contractions.pullback(
            basis,
            points,
            spins,
            _pointer(ao),
            _pointer(coefficients),
            _pointer(weights),
            _pointer(result),
            scalar,
        )
        results.append((primal, tangent, result))
    for scalar, dispatched in zip(*results, strict=True):
        np.testing.assert_array_equal(
            scalar.view(np.uint64), dispatched.view(np.uint64)
        )
