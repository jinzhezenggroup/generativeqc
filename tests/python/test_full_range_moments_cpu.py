"""Independent Full Boys moments and the explicit CPU geometry admission domain."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
POINTER = ctypes.POINTER(ctypes.c_double)
SENTINEL = -781.25


@pytest.fixture(scope="module")
def full_moments_cpu(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: Any
) -> ctypes.CDLL:
    """Compile the real headers through the verified native compiler cache."""
    directory = tmp_path_factory.mktemp("full-range-moments-cpu")
    source = directory / "moments.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include "integrals/eri_geometry.hpp"

struct Geometry {
  double inverse_two_p, inverse_two_q, rho;
  double difference[3], shifts[4][3], product_scales[4], decay[4][3];
  double prefactor, boys[15];
};

extern "C" bool full_moments(unsigned order, double argument, double* output) {
  return generativeqc::integrals::bounded_full_range_moments<14>(order, argument, output);
}

template<unsigned UnitSlots, bool Positive> bool bind(
    const double* exponents, const double* centers, unsigned order,
    unsigned radial, double omega, double* output) {
  Geometry geometry{};
  if (!output || !generativeqc::integrals::make_eri_geometry<Geometry,14,UnitSlots,Positive>(
          exponents, centers, order,
          static_cast<generativeqc::integrals::CoulombRange>(radial), omega, geometry))
    return false;
  std::copy_n(geometry.boys, order + 1, output);
  return true;
}

template<bool Positive> bool dispatch(
    unsigned slots, const double* exponents, const double* centers,
    unsigned order, unsigned radial, double omega, double* output) {
  if (slots == 0) return bind<0,Positive>(exponents,centers,order,radial,omega,output);
  if (slots == 8) return bind<8,Positive>(exponents,centers,order,radial,omega,output);
  if (slots == 10) return bind<10,Positive>(exponents,centers,order,radial,omega,output);
  return false;
}

extern "C" bool geometry_moments(
    bool positive, unsigned slots, const double* exponents, const double* centers,
    unsigned order, unsigned radial, double omega, double* output) {
  if (positive) return dispatch<true>(slots,exponents,centers,order,radial,omega,output);
  return dispatch<false>(slots,exponents,centers,order,radial,omega,output);
}
"""
    )
    library_path = required_native_cxx.build_shared(
        [source],
        directory / "moments.so",
        compile_args=(
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            f"-I{ROOT / 'src'}",
        ),
    )
    library = ctypes.CDLL(str(library_path))
    library.full_moments.argtypes = (ctypes.c_uint, ctypes.c_double, POINTER)
    library.full_moments.restype = ctypes.c_bool
    library.geometry_moments.argtypes = (
        ctypes.c_bool,
        ctypes.c_uint,
        POINTER,
        POINTER,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_double,
        POINTER,
    )
    library.geometry_moments.restype = ctypes.c_bool
    return library


@pytest.mark.parametrize("order", range(15))
def test_full_moments_independent_incomplete_gamma(
    full_moments_cpu: ctypes.CDLL, order: int
) -> None:
    """Cover both recurrence regimes, transition neighbors and real underflow."""
    threshold = order + 1.5
    arguments = np.unique(
        np.r_[
            0.0,
            np.logspace(-300, 308, 161),
            np.linspace(0.01, 32, 97),
            threshold,
            np.nextafter(threshold, 0),
            np.nextafter(threshold, np.inf),
        ]
    )
    with mp.workdps(100):
        for argument in arguments:
            exact_argument = mp.mpf(float(argument))
            expected = np.array(
                [
                    float(
                        mp.mpf(1) / (2 * degree + 1)
                        if argument == 0
                        else mp.gammainc(degree + mp.mpf("0.5"), 0, exact_argument)
                        / (2 * exact_argument ** (degree + mp.mpf("0.5")))
                    )
                    for degree in range(order + 1)
                ]
            )
            output = np.full(order + 2, SENTINEL)
            assert full_moments_cpu.full_moments(
                order, argument, output.ctypes.data_as(POINTER)
            )
            assert output[-1] == SENTINEL
            assert np.isfinite(output).all() and (output[:-1] >= 0).all()
            np.testing.assert_allclose(output[:-1], expected, rtol=2e-14, atol=5e-324)


@pytest.mark.parametrize(
    ("order", "argument"),
    ((15, 1.0), (2**32 - 1, 1.0), (0, -1.0), (2, np.nan), (4, np.inf)),
)
def test_full_moments_reject_before_writing(
    full_moments_cpu: ctypes.CDLL, order: int, argument: float
) -> None:
    """Invalid domains cannot overrun or partially overwrite the caller buffer."""
    output = np.full(16, SENTINEL)
    assert not full_moments_cpu.full_moments(
        order, argument, output.ctypes.data_as(POINTER)
    )
    np.testing.assert_array_equal(output, SENTINEL)


def test_full_moments_null_output(full_moments_cpu: ctypes.CDLL) -> None:
    """A valid mathematical request still requires owned output storage."""
    assert not full_moments_cpu.full_moments(0, 0, None)


@pytest.mark.parametrize("slots", (0, 8, 10))
@pytest.mark.parametrize("radial", range(3))
def test_geometry_retains_default_operator_and_storage_contract(
    full_moments_cpu: ctypes.CDLL, slots: int, radial: int
) -> None:
    """The schedule changes Full evaluation, not geometry or LR/SR arithmetic."""
    exponents = np.array([0.6, 0.8, 1.1, 0.9])
    for slot in range(4):
        if slots & (1 << slot):
            exponents[slot] = 0
    omega = 0.0 if radial == 0 else 0.3
    for scale in (0, 0.1, 1, 10, 100):
        centers = scale * np.arange(12, dtype=float) * 0.1
        for order in (0, 2, 8, 14):
            outputs = []
            for positive in (False, True):
                output = np.full(16, SENTINEL)
                assert full_moments_cpu.geometry_moments(
                    positive,
                    slots,
                    exponents.ctypes.data_as(POINTER),
                    centers.ctypes.data_as(POINTER),
                    order,
                    radial,
                    omega,
                    output.ctypes.data_as(POINTER),
                )
                np.testing.assert_array_equal(output[order + 1 :], SENTINEL)
                outputs.append(output[: order + 1])
            if radial == 0:
                np.testing.assert_allclose(*outputs, atol=3e-15, rtol=2e-13)
            else:
                np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("slots", (0, 8, 10))
@pytest.mark.parametrize(
    "failure",
    (
        "order",
        "operator",
        "full_omega",
        "negative_omega",
        "nan_omega",
        "inf_omega",
        "zero_exponent",
        "negative_exponent",
        "nan_exponent",
        "inf_exponent",
        "unit_exponent",
        "nan_coordinate",
        "inf_coordinate",
        "overflow_distance",
        "overflow_exponent",
        "null_exponents",
        "null_centers",
        "null_output",
    ),
)
def test_geometry_rejects_original_invalid_domain(
    full_moments_cpu: ctypes.CDLL, slots: int, failure: str
) -> None:
    """The CPU opt-in cannot bypass Full omega, derived T/rho or DF slot checks."""
    exponents = np.array([0.6, 0.8, 1.1, 0.9])
    for slot in range(4):
        if slots & (1 << slot):
            exponents[slot] = 0
    centers = np.arange(12, dtype=float) * 0.1
    order, radial, omega = 14, 0, 0.0
    if failure == "order":
        order = 15
    if failure == "operator":
        radial = 3
    if failure == "full_omega":
        omega = 0.2
    if failure == "negative_omega":
        radial, omega = 1, -0.2
    if failure in ("nan_omega", "inf_omega"):
        omega = np.nan if failure == "nan_omega" else np.inf
    if failure.endswith("exponent"):
        exponents[0] = {
            "zero_exponent": 0,
            "negative_exponent": -0.2,
            "nan_exponent": np.nan,
            "inf_exponent": np.inf,
            "unit_exponent": 0 if slots == 0 else 0.2,
            "overflow_exponent": 1e308,
        }[failure]
        if failure == "unit_exponent" and slots:
            exponents[3] = 0.2
        if failure == "overflow_exponent":
            exponents[[index for index in range(4) if not slots & (1 << index)]] = 1e308
    if failure in ("nan_coordinate", "inf_coordinate", "overflow_distance"):
        centers[0] = {
            "nan_coordinate": np.nan,
            "inf_coordinate": np.inf,
            "overflow_distance": 1e200,
        }[failure]
    for positive in (False, True):
        output = np.full(16, SENTINEL)
        assert not full_moments_cpu.geometry_moments(
            positive,
            slots,
            None if failure == "null_exponents" else exponents.ctypes.data_as(POINTER),
            None if failure == "null_centers" else centers.ctypes.data_as(POINTER),
            order,
            radial,
            omega,
            None if failure == "null_output" else output.ctypes.data_as(POINTER),
        )
        np.testing.assert_array_equal(output, SENTINEL)
