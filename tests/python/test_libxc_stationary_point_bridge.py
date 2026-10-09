"""Real native E/vxc point bridge for automatically imported Libxc XC (#1122).

These fixed-feature checks do not certify a converged molecular force endpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc import _native
from generativeqc._ks_snapshot import _scf_xc_points
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN


@pytest.mark.parametrize(
    ("name", "ingredients"),
    (
        ("LDA_C_VWN_4", ("rho",)),
        ("GGA_X_APBE", ("rho", "sigma")),
        ("MGGA_X_LTA", ("rho", "sigma", "tau")),
    ),
)
def test_imported_native_point_and_cartesian_derivatives(
    name: str, ingredients: tuple[str, ...]
) -> None:
    library = _native.load_library(device="cpu")
    functional = automatic_functional_code(name)
    rho = np.array([[0.42, 0.27], [0.31, 0.19]])
    gradient = np.array(
        [
            [[0.030, -0.020, 0.010], [0.020, 0.010, -0.010]],
            [[-0.015, 0.022, 0.012], [0.015, -0.010, 0.009]],
        ]
    )
    tau = np.array([[0.35, 0.24], [0.29, 0.18]])
    drho = np.array([[0.10, -0.05], [-0.06, 0.09]])
    dgradient = np.array(
        [
            [[0.02, 0.01, -0.01], [-0.01, 0.03, 0.01]],
            [[-0.02, 0.01, 0.02], [0.02, -0.01, 0.01]],
        ]
    )
    dtau = np.array([[0.04, 0.03], [-0.02, 0.01]])
    needs_tau = "tau" in ingredients

    def evaluate(step: float) -> dict[str, np.ndarray]:
        return _scf_xc_points(
            library,
            functional,
            rho + step * drho,
            gradient + step * dgradient,
            tau + step * dtau if needs_tau else None,
            required_ingredients=ingredients,
        )

    actual = evaluate(0.0)
    assert actual["energy"].shape == (2,)
    assert actual["rho"].shape == (2, 2)
    assert actual["gradient"].shape == (2, 2, 3)
    assert actual["kinetic"].shape == (2, 2)
    assert all(np.isfinite(value).all() for value in actual.values())
    if not needs_tau:
        np.testing.assert_array_equal(actual["kinetic"], np.zeros((2, 2)))

    # Independent displaced energies also test the Cartesian sigma conversion
    # and the native vtau/2 convention used by the tau geometry pullback.
    predicted = (
        np.einsum("sp,sp->p", actual["rho"], drho)
        + np.einsum("spa,spa->p", actual["gradient"], dgradient)
        + (
            2 * np.einsum("sp,sp->p", actual["kinetic"], dtau)
            if needs_tau
            else 0
        )
    )
    step = 1.0e-5
    displaced = (evaluate(step)["energy"] - evaluate(-step)["energy"]) / (2 * step)
    np.testing.assert_allclose(predicted, displaced, atol=2.0e-6, rtol=2.0e-5)

    # The imported compiler-owned work domain is distinct from the curated PBE
    # stationary point model. This only constructs the internal derivative plan.
    method = resolve_bulk_ks(name, spin="polarized", backend="cpu").method
    plan = StationaryGradientPlan(method, StationaryMeanField(LIBXC_WORK_DOMAIN))
    assert plan.mean_field.point_model == LIBXC_WORK_DOMAIN


def test_imported_point_bridge_keeps_unsupported_requests_closed() -> None:
    library = _native.load_library(device="cpu")
    rho = np.array([[0.42], [0.31]])
    gradient = np.zeros((2, 1, 3))
    gga = automatic_functional_code("GGA_X_APBE")
    mgga = automatic_functional_code("MGGA_X_LTA")

    with pytest.raises(ValueError, match="exact ingredients"):
        _scf_xc_points(library, gga, rho, gradient)
    with pytest.raises(ValueError, match=r"requires tau\\[2,n\\]"):
        _scf_xc_points(
            library,
            mgga,
            rho,
            gradient,
            required_ingredients=("rho", "sigma", "tau"),
        )
    with pytest.raises(RuntimeError):
        _scf_xc_points(
            library,
            gga,
            rho,
            gradient,
            required_ingredients=("rho", "sigma"),
            scales=(0.5, 1.0),
        )
    with pytest.raises(RuntimeError):
        _scf_xc_points(
            library,
            0x3FFFF,
            rho,
            gradient,
            required_ingredients=("rho", "sigma"),
        )
