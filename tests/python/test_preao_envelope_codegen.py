"""Device-free checks of pre-AO domain arithmetic against the Python oracle."""

from __future__ import annotations

import ctypes as ct
import math
import typing
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.dft.envelope_cuda import emit_ao_region_screen_cuda
from generativeqc_compiler.dft.envelopes import _axis_bound

if TYPE_CHECKING:
    from conftest import NativeCxx


def _reference_axis_bound(
    power: int, derivative: int, alpha: float, lower: float, upper: float
) -> float:
    """Independent pre-migration directed-rounding recurrence, test-only."""

    def add(left: float, right: float) -> float:
        return (
            right
            if left == 0
            else left
            if right == 0
            else math.nextafter(left + right, math.inf)
        )

    def multiply(left: float, right: float) -> float:
        return (
            0.0
            if left == 0 or right == 0
            else math.nextafter(float(left) * float(right), math.inf)
        )

    coefficients = [0.0] * (power + derivative + 1)
    coefficients[power] = 1.0
    for order in range(derivative):
        following = [0.0] * len(coefficients)
        for degree in range(power + order + 1):
            if degree:
                following[degree - 1] = add(
                    following[degree - 1], multiply(degree, coefficients[degree])
                )
            following[degree + 1] = add(
                following[degree + 1],
                multiply(multiply(2.0, alpha), coefficients[degree]),
            )
        coefficients = following
    maximum = max(abs(lower), abs(upper))
    minimum = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
    polynomial, monomial = 0.0, 1.0
    for coefficient in coefficients:
        polynomial = add(polynomial, multiply(coefficient, monomial))
        monomial = multiply(monomial, maximum)
    if not math.isfinite(polynomial):
        return math.inf
    square = max(0.0, math.nextafter(minimum * minimum, -math.inf))
    exponent = max(0.0, math.nextafter(alpha * square, -math.inf))
    gaussian = multiply(
        math.nextafter(math.exp(-exponent), math.inf), 1 + 32 * np.finfo(float).eps
    )
    return multiply(polynomial, gaussian)


@pytest.fixture(scope="module")
def native_axis(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> typing.Any:
    """Compile the actual emitted scalar bound without a GPU or native library."""
    root = tmp_path_factory.mktemp("preao-bound")
    source, library = root / "bound.cpp", root / "bound.so"
    emitted = emit_ao_region_screen_cuda()
    scalar = emitted[: emitted.index("__global__ void ao_region_box_kernel")]
    source.write_text(
        "#include <cmath>\n#define __device__\nusing std::isfinite;\n"
        + scalar
        + '}\nextern "C" double bound(int power,int derivative,double alpha,double lower,double upper) {'
        + "return ao_region_axis_bound(power,derivative,alpha,lower,upper);}\n"
    )
    native_cxx.build_shared(
        (source,), library, compile_args=("-std=c++17", "-O2", "-ffp-contract=off")
    )
    function = ct.CDLL(str(library)).bound
    function.argtypes = (ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_double)
    function.restype = ct.c_double
    return function


def test_scalar_bounds_match_existing_oracle(native_axis: typing.Any) -> None:
    rng = np.random.default_rng(1893006)
    for power in range(9):
        for derivative in range(4):
            for alpha in (0.03, 1.0, 40.0):
                for lower, upper in (
                    (0.0, 0.0),
                    (-2.0, 3.0),
                    (17.0, 21.0),
                    (1e154, 1e155),
                ):
                    expected = _reference_axis_bound(
                        power, derivative, alpha, lower, upper
                    )
                    assert (
                        native_axis(power, derivative, alpha, lower, upper) == expected
                    )
                    assert (
                        _axis_bound(power, derivative, alpha, lower, upper) == expected
                    )
            for lower, upper in np.sort(rng.normal(size=(12, 2)), axis=1):
                expected = _reference_axis_bound(power, derivative, 0.7, lower, upper)
                assert native_axis(power, derivative, 0.7, lower, upper) == expected
                assert _axis_bound(power, derivative, 0.7, lower, upper) == expected


def test_unsupported_degree_and_nonfinite_regions_retain(
    native_axis: typing.Any,
) -> None:
    for values in ((12, 0, 1, 0, 1), (1, 4, 1, 0, 1), (1, 2, 1, -np.inf, np.inf)):
        assert np.isposinf(native_axis(*values))


def test_producer_does_not_sample_or_allocate_aos() -> None:
    source = emit_ao_region_screen_cuda()
    assert "scheduled_ao" not in source and "ao_kernel(" not in source
    assert "cudaMalloc" not in source
    assert "derivatives[jet][axis]" in source
    assert "nextafter(bounds[axis] - basis[3 * atom + axis], -HUGE_VAL)" in source


def test_preao_cache_preserves_capability_and_truthful_discovery_work() -> None:
    from generativeqc._resident_ao_maps import ResidentAoMapCache
    from test_resident_ao_map_cache import Grid, domain

    class PreAoGrid(Grid):
        def select_ao_device_points(
            self,
            pointer: int,
            count: int,
            *,
            cutoff: float,
            producer: str,
        ) -> np.ndarray:
            assert producer == "pre-ao-envelope"
            return super().select_ao_device_points(pointer, count, cutoff=cutoff)

    grid = PreAoGrid()
    cache = ResidentAoMapCache(
        grid, domain(), cutoff=1e-12, budget_bytes=512, producer="pre-ao-envelope"
    )
    selected, layout = cache.select_block(grid, domain(), 0, 4)
    layout.require_derivative_order(2)
    assert selected is grid.answer
    assert cache.work["discovery_ao_jet_values"] == 0
    assert cache.work["discovery_region_bounds"] == 100
    assert cache.work["discovery_producer"] == "pre-ao-envelope"
    cache.reset_work()
    assert cache.select_block(grid, domain(), 0, 4)[0] is selected
    assert len(grid.calls) == 1
    assert cache.work["discovery_region_bounds"] == 0
    with pytest.raises(AttributeError):
        cache.producer = "sampled-jets"
