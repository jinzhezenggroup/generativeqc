"""Exercise compiled energy, physical potential and point action for split hybrids."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.geometry_cuda import _emit_stationary_point
from generativeqc_compiler.xc.split_hybrid_stationary import (
    emit_split_hybrid_stationary_point,
    split_hybrid_stationary_symbol,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from conftest import NativeCxx

_PHYSICAL_ORDER = [0, 5, 1, 2, 3, 6, 7, 8, 4, 9]
_SPIN_ORDER = np.argsort(_PHYSICAL_ORDER)
_POINT = np.array([[0.2, 0.04, 0.03, 0.01, 0.3], [0.15, 0.02, -0.03, 0.02, 0.2]])
_TANGENT = np.array([[0.1, 0.3, -0.1, 0.05, 0.12], [-0.02, -0.2, 0.35, 0.01, -0.09]])


@pytest.fixture(scope="module", params=["M06-2X", "MN15"])
def functional(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def evaluate_point(
    functional: str, native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Compile the directional action and the force's physical point binding."""
    work = tmp_path_factory.mktemp(f"{functional.lower()}-point")
    source = work / "point.cpp"
    wrappers = """
extern "C" void evaluate(const double* point, const double* tangent, double* out) {
  const auto response = POINT_SYMBOL_action(point, tangent);
  const double gradient[2][3] = {{point[2], point[3], point[4]},
                                 {point[5], point[6], point[7]}};
  const auto force = stationary_evaluate_point(point, gradient, point + 8);
  out[0] = response.energy;
  out[21] = force.valid ? force.energy : NAN;
  for (unsigned spin = 0; spin < 2; ++spin) {
    out[22 + spin] = force.rho[spin];
    out[30 + spin] = 2.0 * force.kinetic[spin];
    for (unsigned axis = 0; axis < 3; ++axis)
      out[24 + 3 * spin + axis] = force.gradient[spin][axis];
  }
  for (unsigned index = 0; index < 10; ++index) {
    out[1 + index] = response.potential[index];
    out[11 + index] = response.action[index];
  }
}
""".replace("POINT_SYMBOL", split_hybrid_stationary_symbol(functional))
    semilocal = resolve_method(functional, spin="polarized").primitives[0].functional
    source.write_text(
        "\n".join(
            (
                "#include <cmath>",
                "#define __device__",
                "using std::isfinite;",
                _emit_stationary_point(
                    SPLIT_HYBRIDS[functional]["functional_code"],
                    semilocal=semilocal,
                    variational_split_hybrid=True,
                ),
                emit_split_hybrid_stationary_point(functional, with_action=True),
                wrappers,
            )
        )
    )
    path = native_cxx.build_shared(
        (source,),
        work / "point.so",
        compile_args=("-std=c++20", "-O3", "-ffp-contract=off"),
    )
    library = ctypes.CDLL(str(path))
    evaluate = library.evaluate
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    evaluate.argtypes = (pointer, pointer, pointer)
    evaluate.restype = None

    def point_action(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        out = np.empty(32)
        evaluate(
            point.ravel()[_PHYSICAL_ORDER], direction.ravel()[_PHYSICAL_ORDER], out
        )
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[:11], out[21:], atol=1e-12, rtol=1e-10)
        return np.r_[out[0], out[1:11][_SPIN_ORDER], out[11:21][_SPIN_ORDER]]

    return point_action


def test_compiled_point_matches_independent_libxc(
    functional: str,
    evaluate_point: Callable,
) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("requires the pinned Libxc 7.0.0 oracle")
    numint = pytest.importorskip("pyscf.dft.numint").NumInt()
    rng = np.random.default_rng(632)
    random = rng.uniform(-0.12, 0.12, size=(32, 2, 5))
    random[:, :, 0] = rng.uniform(0.03, 1.4, size=(32, 2))
    random[:, :, 4] = rng.uniform(0.06, 0.9, size=(32, 2))
    # Respect the occupied-orbital Cauchy bound for each physical spin.
    random[:, :, 4] += np.sum(random[:, :, 1:4] ** 2, axis=2) / (8 * random[:, :, 0])
    flat = _POINT.copy()
    flat[:, 1:4] = 0
    for point in (_POINT, _POINT[[0, 0]], flat, *random):
        exc, vxc, fxc, _ = numint.eval_xc_eff(
            functional, point[:, :, None], deriv=2, spin=1
        )
        reference = np.r_[
            exc[0] * point[:, 0].sum(),
            vxc.ravel(),
            fxc.reshape(10, 10) @ _TANGENT.ravel(),
        ]
        np.testing.assert_allclose(
            evaluate_point(point, _TANGENT), reference, atol=5e-10, rtol=2e-8
        )


@pytest.mark.parametrize(
    "boundary", ["empty-spin", "flat-gradient", "tail", "extreme-spin-ratio"]
)
def test_boundary_is_variational(
    functional: str, evaluate_point: Callable, boundary: str
) -> None:
    point = _POINT.copy()
    direction = _TANGENT.copy()
    if boundary in {"empty-spin", "extreme-spin-ratio"}:
        point[1] = direction[1] = 0
        if boundary == "extreme-spin-ratio":
            point[0, (0, 4)] *= 1e7
            direction[0, (0, 4)] *= 1e7
    elif boundary == "flat-gradient":
        point[:, 1:4] = 0
    elif boundary == "tail":
        # Each method's continuation follows the tabulated source threshold.
        density_scale = 1e-3 if functional == "MN15" else 1.0
        point = np.array(
            [
                [1.4e-12 * density_scale, 2e-18, 0, 0, 1.6e-20],
                [4e-13 * density_scale, -3e-18, 0, 0, 8e-21],
            ]
        )
        direction = 0.1 * point

    central = evaluate_point(point, direction)
    step = 1e-4
    plus = evaluate_point(point + step * direction, direction)
    minus = evaluate_point(point - step * direction, direction)
    np.testing.assert_allclose(
        (plus[0] - minus[0]) / (2 * step),
        central[1:11] @ direction.ravel(),
        atol=1e-12,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        (plus[1:11] - minus[1:11]) / (2 * step), central[11:], atol=1e-7, rtol=1e-5
    )
    matrix = np.column_stack(
        [evaluate_point(point, basis.reshape(2, 5))[11:] for basis in np.eye(10)]
    )
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-8)
    swapped = evaluate_point(point[::-1], direction[::-1])
    spin_flip = np.r_[0, 6:11, 1:6, 16:21, 11:16]
    np.testing.assert_allclose(swapped, central[spin_flip], atol=1e-8, rtol=2e-8)
