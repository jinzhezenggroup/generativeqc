"""Independent stored-D cancellation, feature-budget and refusal contracts."""

import ctypes as ct
from decimal import Decimal, localcontext
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _cpp_source_support import cpp_function_definition
from _cuda_rounding_support import DIRECTED_ROUNDING_HOST
from generativeqc_compiler.dft.feature_policy import emit_feature_policy
from generativeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    _emit_occupied_features,
)

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.fixture
def certificate_probe(tmp_path: Path, native_cxx: "NativeCxx") -> ct.CDLL:
    emitted = _emit_occupied_features(DEFAULT_XC_MATRIX_SCHEDULE)
    source = tmp_path / "certificate.cpp"
    source.write_text(
        PREAMBLE
        + DIRECTED_ROUNDING_HOST
        + "namespace generativeqc_grid_policy {\n"
        + emit_feature_policy(device=False)
        + "}\n"
        + cpp_function_definition(emitted, "occupied_feature_certificate")
        + cpp_function_definition(
            emitted, "occupied_density_features_body", include_template=True
        )
        + DRIVER
    )
    library = native_cxx.build_shared(
        [source],
        tmp_path / "certificate.so",
        compile_args=("-std=c++17", "-O2", "-ffp-contract=off"),
    )
    native = ct.CDLL(str(library))
    native.admitted.argtypes = [ct.c_void_p, ct.c_void_p] + [ct.c_int64] * 3
    native.admitted.restype = ct.c_int
    native.evaluate.argtypes = [ct.c_void_p] * 4 + [ct.c_int64] * 5 + [ct.c_void_p] * 2
    native.evaluate.restype = ct.c_int
    return native


@pytest.mark.parametrize("term", range(5))
def test_every_feature_has_its_own_error_gate(
    certificate_probe: ct.CDLL, term: int
) -> None:
    """Synthetic helper inputs isolate rho, each gradient, and tau admission."""
    features = np.full(5, 1e6)
    squared_norms = np.ones(4)
    assert certificate_probe.admitted(
        features.ctypes.data, squared_norms.ctypes.data, 17, 1, 1
    )
    features[term] = 0.0
    assert not certificate_probe.admitted(
        features.ctypes.data, squared_norms.ctypes.data, 17, 1, 1
    )


def test_invalid_or_unbounded_certificates_fail_closed(
    certificate_probe: ct.CDLL,
) -> None:
    features, norms = np.zeros(5), np.zeros(4)
    assert certificate_probe.admitted(features.ctypes.data, norms.ctypes.data, 17, 1, 2)
    for invalid in (np.inf, np.nan, -1.0):
        norms[2] = invalid
        assert not certificate_probe.admitted(
            features.ctypes.data, norms.ctypes.data, 17, 1, 2
        )
    norms[:] = 0.0
    for invalid in (np.inf, np.nan):
        features[3] = invalid
        assert not certificate_probe.admitted(
            features.ctypes.data, norms.ctypes.data, 17, 1, 2
        )
    features[:] = 0.0
    for active, rank in ((0, 1), (17, 0), (1 << 51, 1), (1, 1 << 51)):
        assert not certificate_probe.admitted(
            features.ctypes.data, norms.ctypes.data, active, rank, 2
        )


def _stored_density_features(density: np.ndarray, ao: np.ndarray) -> np.ndarray:
    """Scalar FP64 oracle, consuming stored half-before-add D, not factor identities."""
    spins, active = density.shape[:2]
    points = ao.shape[1]
    result = np.zeros((spins, 5, points))
    for spin in range(spins):
        symmetric = 0.5 * density[spin] + 0.5 * density[spin].T
        for point in range(points):
            for row in range(active):
                products = np.zeros(4)
                for column in range(active):
                    for jet in range(4):
                        products[jet] += symmetric[row, column] * ao[jet, point, column]
                result[spin, 0, point] += ao[0, point, row] * products[0]
                kinetic = 0.0
                for axis in range(3):
                    derivative = ao[axis + 1, point, row]
                    result[spin, axis + 1, point] += 2.0 * derivative * products[0]
                    kinetic += derivative * products[axis + 1] * 0.5
                result[spin, 4, point] += kinetic
    return result


@pytest.mark.parametrize("occupations", [(1,), (1, 1), (0, 1), (1, 0)])
@pytest.mark.parametrize("mapped", [False, True])
def test_finite_cancellation_uses_original_density_for_all_features(
    certificate_probe: ct.CDLL, occupations: tuple[int, ...], mapped: bool
) -> None:
    active, points, rank = 17, 3, 1
    spins = len(occupations)
    full_n = 23 if mapped else active
    ids = np.arange(active, dtype=np.uintp) + (3 if mapped else 0)
    coefficients = np.zeros((full_n, rank))
    coefficients[ids[:2], 0] = [0.6, 0.8]
    density = np.zeros((spins, full_n, full_n))
    projections = np.zeros((spins, 4, points, rank))
    absolute = np.zeros_like(projections)
    ao = np.zeros((4, points, active))
    for jet in range(4):
        ao[jet, :, 0] = 4e6 * (jet + 1)
        ao[jet, :, 1] = -3e6 * (jet + 1)
    for spin, occupied in enumerate(occupations):
        if occupied:
            density[spin] = (2.0 if spins == 1 else 1.0) * coefficients @ coefficients.T
            projections[spin] = ao @ coefficients[ids]
            absolute[spin] = np.nextafter(
                np.abs(ao) @ np.abs(coefficients[ids]), np.inf
            )
    actual = np.zeros((spins, 5, points))
    status = certificate_probe.evaluate(
        density.ctypes.data,
        projections.ctypes.data,
        absolute.ctypes.data,
        ao.ctypes.data,
        active,
        points,
        rank,
        spins,
        full_n,
        ids.ctypes.data if mapped else None,
        actual.ctypes.data,
    )
    assert status == 0
    expected = _stored_density_features(density[:, ids[:, None], ids], ao)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-14)
    assert np.max(np.abs(expected)) > 0.018
    assert not np.any(projections)


def test_well_conditioned_small_features_are_admitted_and_accurate(
    certificate_probe: ct.CDLL,
) -> None:
    rng = np.random.default_rng(20261018)
    active, rank, points = 17, 2, 3
    coefficients = np.ascontiguousarray(rng.normal(size=(active, rank)) * 0.1)
    ao = np.ascontiguousarray(rng.normal(size=(4, points, active)) * 1e-4)
    density = np.ascontiguousarray((2.0 * coefficients) @ coefficients.T)[None]
    projections = np.ascontiguousarray(ao @ coefficients)[None]
    absolute = np.ascontiguousarray(
        np.nextafter(np.abs(ao) @ np.abs(coefficients), np.inf)
    )[None]
    for point in range(points):
        panel = projections[0, :, point]
        features = np.array(
            [
                np.sum(panel[0] ** 2),
                *(2.0 * np.sum(panel[0] * panel[jet]) for jet in range(1, 4)),
                0.5 * np.sum(panel[1:] ** 2),
            ]
        )
        norms = np.sum(absolute[0, :, point] ** 2, axis=-1)
        assert certificate_probe.admitted(
            features.ctypes.data, norms.ctypes.data, active, rank, 1
        )
    actual = np.zeros((1, 5, points))
    assert (
        certificate_probe.evaluate(
            density.ctypes.data,
            projections.ctypes.data,
            absolute.ctypes.data,
            ao.ctypes.data,
            active,
            points,
            rank,
            1,
            active,
            None,
            actual.ctypes.data,
        )
        == 0
    )
    np.testing.assert_allclose(
        actual, _stored_density_features(density, ao), rtol=5e-13, atol=5e-14
    )


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("exponent", [-250, -40, 0, 40, 250])
def test_certificate_encloses_both_routes_against_exact_decimal_features(
    certificate_probe: ct.CDLL, spins: int, exponent: int
) -> None:
    """Check the bound independently over balanced, widely separated C/AO scales."""
    rng = np.random.default_rng(20261019)
    active, rank = 17, 3
    coefficients = np.ldexp(rng.uniform(0.5, 1.0, (active, rank)), exponent)
    coefficients *= rng.choice([-1.0, 1.0], coefficients.shape)
    ao = np.ldexp(rng.uniform(-1.0, 1.0, (4, 1, active)), -exponent - 12)
    weight = 2.0 if spins == 1 else 1.0
    density = np.repeat(((weight * coefficients) @ coefficients.T)[None], spins, axis=0)
    projected = np.repeat((ao @ coefficients)[None], spins, axis=0)
    absolute = np.zeros_like(projected)
    with localcontext() as context:
        context.prec = 800
        exact = [
            [
                sum(
                    Decimal(float(coefficients[row, orbital]))
                    * Decimal(float(ao[jet, 0, row]))
                    for row in range(active)
                )
                for orbital in range(rank)
            ]
            for jet in range(4)
        ]
        norms = []
        for jet in range(4):
            magnitudes = [
                sum(
                    abs(
                        Decimal(float(coefficients[row, orbital]))
                        * Decimal(float(ao[jet, 0, row]))
                    )
                    for row in range(active)
                )
                for orbital in range(rank)
            ]
            absolute[:, jet, 0] = [
                np.nextafter(float(value), np.inf) for value in magnitudes
            ]
            norms.append(sum(value * value for value in magnitudes))
        features = [sum(value * value for value in exact[0])]
        features += [
            2
            * sum(
                left * right for left, right in zip(exact[0], exact[jet], strict=True)
            )
            for jet in range(1, 4)
        ]
        features += [
            sum(sum(value * value for value in exact[jet]) for jet in range(1, 4)) / 2
        ]
        majorants = [
            norms[0],
            *(2 * norms[0].sqrt() * norms[jet].sqrt() for jet in range(1, 4)),
            sum(norms[1:]) / 2,
        ]
        steps = Decimal(8 * (active + rank) + 128)
        scaled = steps * Decimal(2) ** -53
        gamma = scaled / (1 - scaled)
        bounds = [
            2 * gamma * Decimal(weight) * value + steps * steps * Decimal(2) ** -810
            for value in majorants
        ]
        panel = projected[0, :, 0]
        raw_features = np.array(
            [
                np.sum(panel[0] ** 2),
                *(2.0 * np.sum(panel[0] * panel[jet]) for jet in range(1, 4)),
                0.5 * np.sum(panel[1:] ** 2),
            ]
        )
        upper_norms = np.nextafter(np.sum(absolute[0, :, 0] ** 2, axis=-1), np.inf)
        assert certificate_probe.admitted(
            raw_features.ctypes.data, upper_norms.ctypes.data, active, rank, spins
        )
        candidate = np.zeros((spins, 5, 1))
        assert (
            certificate_probe.evaluate(
                density.ctypes.data,
                projected.ctypes.data,
                absolute.ctypes.data,
                ao.ctypes.data,
                active,
                1,
                rank,
                spins,
                active,
                None,
                candidate.ctypes.data,
            )
            == 0
        )
        stored = _stored_density_features(density, ao)
        for spin in range(spins):
            for term in range(5):
                exact_feature = Decimal(weight) * features[term]
                candidate_value = Decimal(float(candidate[spin, term, 0]))
                stored_value = Decimal(float(stored[spin, term, 0]))
                assert abs(candidate_value - exact_feature) <= bounds[term] / 2
                assert abs(stored_value - exact_feature) <= bounds[term] / 2
                assert abs(candidate_value - stored_value) <= bounds[term]


def test_missing_certificate_and_original_matrix_failures_are_not_hidden(
    certificate_probe: ct.CDLL,
) -> None:
    active, points = 17, 1
    density = np.zeros((1, active, active))
    density[0, 0, 0] = np.nan
    ao = np.ones((4, points, active))
    projections = np.zeros((1, 4, points, 1))
    actual = np.zeros((1, 5, points))
    assert (
        certificate_probe.evaluate(
            density.ctypes.data,
            projections.ctypes.data,
            None,
            ao.ctypes.data,
            active,
            points,
            1,
            1,
            active,
            None,
            actual.ctypes.data,
        )
        == 1
    )
    assert np.any(~np.isfinite(actual))


PREAMBLE = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#define __device__
using I = std::int64_t;
using std::isfinite;
struct Dim { I x; };
Dim threadIdx{0}, blockIdx{0}, blockDim{1}, gridDim{1};
double __shfl_down_sync(unsigned, double value, int) { return value; }
bool __any_sync(unsigned, bool value) { return value; }
double finite(double value, int* error, int code) {
  if (!isfinite(value)) *error = code;
  return value;
}
"""

DRIVER = r"""
extern "C" int admitted(const double* features, const double* norms, I active, I rank, I spins) {
  return occupied_feature_certificate(features, norms, active, rank, spins);
}
extern "C" int evaluate(const double* density, const double* work, const double* absolute,
    const double* ao, I active, I count, I rank, I spins, I full_n, const size_t* ids,
    double* output) {
  int error = 0;
  occupied_density_features_body<false>(work, rank, count, spins, 5, output, &error,
                                        density, ao, active, full_n, ids, absolute);
  return error;
}
"""
