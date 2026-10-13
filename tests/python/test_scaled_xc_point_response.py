"""Weighted SCF point-response jets against independent Libxc potentials."""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np
import pytest
from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def point_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> typing.Any:
    directory = tmp_path_factory.mktemp("scaled-point-response")
    source = directory / "point.cpp"
    source.write_text(
        r"""
#include "api/ks_snapshot.hpp"
#include "dft/xc_point_response.hpp"
extern "C" std::size_t controls_size() { return sizeof(generativeqc_ks_response_controls); }
extern "C" int evaluate(int pbe, int spins, double exchange, double correlation,
                        const double* rho, const double* gradient, const double* drho,
                        const double* dgradient, double* output) {
  const auto* gradients = reinterpret_cast<const double (*)[3]>(gradient);
  const auto* directions = reinterpret_cast<const double (*)[3]>(dgradient);
  const auto value = spins == 1
      ? generativeqc::dft::point::restricted_response(
            pbe, rho[0], gradients[0], drho[0], directions[0], exchange, correlation)
      : generativeqc::dft::point::unrestricted_response(
            pbe, rho, gradients, drho, directions, exchange, correlation);
  for (unsigned spin = 0; spin < 2; ++spin) {
    output[4 * spin] = value.rho[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      output[4 * spin + axis + 1] = value.gradient[spin][axis];
  }
  return value.valid ? 0 : 1;
}
""",
        encoding="utf-8",
    )
    library = directory / "point.so"
    native_cxx.build_shared(
        [source],
        library,
        compile_args=[
            "-std=c++20",
            "-fPIC",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
        ],
    )
    loaded = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    loaded.evaluate.argtypes = [ct.c_int, ct.c_int, ct.c_double, ct.c_double] + [
        pointer
    ] * 5
    loaded.evaluate.restype = ct.c_int
    loaded.controls_size.restype = ct.c_size_t
    return loaded


def test_resident_controls_match_native_abi(point_probe: typing.Any) -> None:
    from generativeqc._resident_cpks import _Controls

    assert point_probe.controls_size() == ct.sizeof(_Controls)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("pbe", (False, True))
@pytest.mark.parametrize(
    "scales", ((1.0, 1.0), (0.75, 1.0), (0.37, 0.91), (0.0, 1.0), (1.0, 0.0))
)
def test_weighted_point_response_matches_libxc(
    point_probe: typing.Any, spins: int, pbe: bool, scales: tuple[float, float]
) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    rho = np.array([1.0, 0.0]) if spins == 1 else np.array([0.4, 0.6])
    drho = np.array([0.05, 0.0]) if spins == 1 else np.array([0.08, -0.03])
    gradient = np.array([[0.15, -0.07, 0.12], [-0.04, 0.05, 0.09]])
    dgradient = np.array([[0.03, 0.02, -0.01], [-0.02, 0.04, 0.01]])
    output = np.empty((2, 4))
    pointer = ct.POINTER(ct.c_double)
    assert (
        point_probe.evaluate(
            pbe,
            spins,
            *scales,
            *(
                value.ctypes.data_as(pointer)
                for value in (rho, gradient, drho, dgradient, output)
            ),
        )
        == 0
    )
    identifiers = ("GGA_X_PBE", "GGA_C_PBE") if pbe else ("LDA_X", "LDA_C_PW")
    code = "+".join(
        f"{weight}*{name}"
        for weight, name in zip(scales, identifiers, strict=True)
        if weight != 0.0
    )

    def potential(step: float) -> np.ndarray:
        density = rho[:spins] + step * drho[:spins]
        gradients = gradient[:spins] + step * dgradient[:spins]
        features = (
            np.column_stack((density, gradients))[:, :, None]
            if pbe
            else density[:, None]
        )
        _, values, _, _ = libxc.eval_xc(
            code, features[0] if spins == 1 else features, spin=spins - 1, deriv=1
        )
        result = np.zeros((spins, 4))
        result[:, 0] = np.asarray(values[0]).reshape(spins)
        if pbe:
            sigma = np.asarray(values[1]).ravel()
            if spins == 1:
                result[:, 1:] = 2 * sigma[0] * gradients
            else:
                result[0, 1:] = 2 * sigma[0] * gradients[0] + sigma[1] * gradients[1]
                result[1, 1:] = sigma[1] * gradients[0] + 2 * sigma[2] * gradients[1]
        return result

    for step in (1e-4, 3e-5):
        reference = (potential(step) - potential(-step)) / (2 * step)
        np.testing.assert_allclose(output[:spins], reference, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize(
    "scales", ((-1.0, 1.0), (1.0, float("nan")), (float("inf"), 1.0))
)
def test_invalid_weights_fail_closed(
    point_probe: typing.Any, scales: tuple[float, float]
) -> None:
    rho, drho = np.ones(2), np.zeros(2)
    gradient = np.zeros((2, 3))
    output = np.empty((2, 4))
    pointer = ct.POINTER(ct.c_double)
    assert (
        point_probe.evaluate(
            True,
            1,
            *scales,
            *(
                value.ctypes.data_as(pointer)
                for value in (rho, gradient, drho, gradient, output)
            ),
        )
        == 1
    )
