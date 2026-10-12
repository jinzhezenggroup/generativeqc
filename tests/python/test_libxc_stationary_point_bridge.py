"""Real native E/vxc point bridge for automatically imported Libxc XC (#1122).

These fixed-feature checks do not certify a converged molecular force endpoint.
"""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native, _stationary_cpu
from generativeqc._dft_gradient import StationaryKsState
from generativeqc._ks_snapshot import _scf_xc_points
from generativeqc.ks import scf_domain_for_method
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.xc.automatic_semilocal import automatic_functional_code
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN

if TYPE_CHECKING:
    from pathlib import Path

    from generativeqc_compiler.method.spec import MethodIR


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
        + (2 * np.einsum("sp,sp->p", actual["kinetic"], dtau) if needs_tau else 0)
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
    with pytest.raises(ValueError, match=r"requires tau\[2,n\]"):
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
    with pytest.raises(ValueError, match="not registered"):
        _scf_xc_points(
            library,
            0x3FFFF,
            rho,
            gradient,
            required_ingredients=("rho", "sigma"),
        )


@pytest.mark.parametrize("ingredients", (("rho",), ("rho", "sigma")))
def test_imported_mgga_cannot_hide_required_tau(ingredients: tuple[str, ...]) -> None:
    library = _native.load_library(device="cpu")
    with pytest.raises(ValueError, match="exact ingredients"):
        _scf_xc_points(
            library,
            automatic_functional_code("MGGA_X_LTA"),
            np.array([[0.42], [0.31]]),
            np.zeros((2, 1, 3)),
            required_ingredients=ingredients,
        )


@pytest.mark.parametrize("code", (0x1000300B8, 0x100000001))
def test_imported_point_code_cannot_wrap_to_registered_program(code: int) -> None:
    library = _native.load_library(device="cpu")
    with pytest.raises(ValueError, match="not registered"):
        _scf_xc_points(
            library,
            code,
            np.array([[0.42], [0.31]]),
            np.zeros((2, 1, 3)),
            required_ingredients=("rho", "sigma"),
        )


@pytest.mark.parametrize(
    ("name", "tail_energy", "tail_potential", "interior_energy", "interior_potential"),
    (
        (
            "LDA_C_VWN_4",
            -7.507157336220036e-21,
            -9.08794956105623e-6,
            -0.04193907322617088,
            -0.07581761675210714,
        ),
        (
            "LDA_C_CHACHIYO",
            -5.81986732777399e-21,
            -7.054113899082287e-6,
            -0.04063514652527078,
            -0.0736011811837693,
        ),
        (
            "LDA_C_PZ_MOD",
            -7.7127975777928e-21,
            -9.333867975378661e-6,
            -0.04143055425676998,
            -0.07470320169925493,
        ),
    ),
)
def test_live_restricted_snapshot_uses_unpolarized_worker_in_active_tail(
    name: str,
    tail_energy: float,
    tail_potential: float,
    interior_energy: float,
    interior_potential: float,
) -> None:
    """PySCF 2.14.0 / Libxc 7.0.0 spin-zero references, not a P2 fold."""
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(
        method=f"libxc:{name}",
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calculator.prepare_batch([atoms]) as batch,
        NativeAO(atoms, basis="sto-3g") as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        source = StationaryKsState.from_native(batch, basis)._source
        assert source.functional.spin == "unpolarized"
        total_rho = np.array([0.0, 2.5e-16, 1.1e-15, 0.62])
        rho = np.stack((total_rho / 2, total_rho / 2))
        values = source.evaluate_xc_points(
            source.functional, rho, np.zeros((2, total_rho.size, 3))
        )
        np.testing.assert_array_equal(values["energy"][:2], 0.0)
        np.testing.assert_array_equal(values["rho"][:, :2], 0.0)
        np.testing.assert_allclose(values["energy"][2], tail_energy, atol=2e-32, rtol=0)
        np.testing.assert_allclose(
            values["rho"][:, 2], tail_potential, atol=2e-14, rtol=0
        )
        np.testing.assert_allclose(
            values["energy"][3], interior_energy, atol=2e-14, rtol=0
        )
        np.testing.assert_allclose(
            values["rho"][:, 3], interior_potential, atol=2e-14, rtol=0
        )
        np.testing.assert_array_equal(values["gradient"], 0.0)
        np.testing.assert_array_equal(values["kinetic"], 0.0)


@pytest.mark.parametrize("name", ("GGA_X_APBE", "MGGA_X_LTA"))
def test_restricted_legacy_domain_preserves_its_scf_continuation(name: str) -> None:
    library = _native.load_library(device="cpu")
    ingredients = ("rho", "sigma", "tau") if name == "MGGA_X_LTA" else ("rho", "sigma")
    rho = np.full((2, 2), 0.31)
    gradient = np.tile([0.015, -0.01, 0.005], (2, 2, 1))
    tau = np.full((2, 2), 0.29)
    arguments = (library, automatic_functional_code(name), rho, gradient, tau)
    polarized = _scf_xc_points(*arguments, required_ingredients=ingredients)
    restricted = _scf_xc_points(
        *arguments, spin="unpolarized", required_ingredients=ingredients
    )
    for component in polarized:
        np.testing.assert_array_equal(restricted[component], polarized[component])


@pytest.mark.parametrize("unequal", ("rho", "gradient", "tau"))
def test_restricted_transport_rejects_unequal_half_features(unequal: str) -> None:
    features = {
        "rho": np.full((2, 1), 0.31),
        "gradient": np.zeros((2, 1, 3)),
        "tau": np.full((2, 1), 0.29),
    }
    features[unequal][0] += 0.01
    with pytest.raises(ValueError, match="equal half-feature channels"):
        _scf_xc_points(
            _native.load_library(device="cpu"),
            automatic_functional_code("LDA_C_VWN_4"),
            **features,
            spin="unpolarized",
            required_ingredients=("rho",),
        )


def test_restricted_transport_does_not_replay_a_legacy_polarized_library() -> None:
    library = _native.load_library(device="cpu")
    legacy = SimpleNamespace(
        generativeqc_xc_point_batch_v2=library.generativeqc_xc_point_batch_v2,
        generativeqc_xc_point_batch_v3=library.generativeqc_xc_point_batch_v3,
    )
    arguments = (
        legacy,
        automatic_functional_code("LDA_C_VWN_4"),
        np.full((2, 1), 5.5e-16),
        np.zeros((2, 1, 3)),
    )
    assert np.isfinite(
        _scf_xc_points(*arguments, required_ingredients=("rho",))["energy"]
    ).all()
    with pytest.raises(NotImplementedError, match="required semilocal XC point bridge"):
        _scf_xc_points(*arguments, spin="unpolarized", required_ingredients=("rho",))


def test_native_layout_abi_keeps_equal_uks_polarized_and_rejects_bad_rks() -> None:
    library = _native.load_library(device="cpu")
    functional = automatic_functional_code("LDA_C_VWN_4")
    rho = np.full((2, 1), 5.5e-16)
    gradient = np.zeros((2, 1, 3))
    tau = np.zeros_like(rho)
    output = np.empty(11)
    pointer = ct.POINTER(ct.c_double)
    bridge = library.generativeqc_xc_point_batch_v4
    bridge.argtypes = [
        ct.c_uint32,
        ct.c_uint32,
        ct.c_double,
        ct.c_double,
        pointer,
        pointer,
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
    ]
    bridge.restype = ct.c_int

    def evaluate(spins: int) -> int:
        return bridge(
            functional,
            spins,
            1.0,
            1.0,
            rho.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            tau.ctypes.data_as(pointer),
            1,
            output.ctypes.data_as(pointer),
            11,
        )

    assert evaluate(2) == 0
    legacy = _scf_xc_points(
        library, functional, rho, gradient, required_ingredients=("rho",)
    )
    np.testing.assert_array_equal(
        output[:3], [legacy["energy"][0], *legacy["rho"][:, 0]]
    )
    assert evaluate(1) == 0
    assert abs(output[0] - (-7.507157336220036e-21)) <= 2e-32
    for spins in (0, 3):
        assert evaluate(spins) == 1
    for feature in (rho, gradient, tau):
        original = feature.copy()
        feature[0] += 0.01
        assert evaluate(1) == 1
        feature[...] = original


def test_imported_live_snapshot_reaches_actual_stationary_point_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(
        method="libxc:GGA_X_APBE",
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    assert scf_domain_for_method("libxc:GGA_X_APBE") == LIBXC_WORK_DOMAIN

    class PlanObserved(Exception):
        pass

    def observe_plan(
        method: MethodIR, mean_field: StationaryMeanField
    ) -> StationaryGradientPlan:
        plan = StationaryGradientPlan(method, mean_field)
        assert plan.mean_field.point_model == LIBXC_WORK_DOMAIN
        # Stop at the real diagnostic's plan boundary. This tests the live
        # handoff without claiming a qualified molecular force endpoint.
        raise PlanObserved

    with (
        calculator.prepare_batch([atoms]) as batch,
        NativeAO(atoms, basis="sto-3g") as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        source = state._source
        assert source.functional_code == automatic_functional_code("GGA_X_APBE")
        values = source.evaluate_xc_points(
            source.functional, np.array([[0.365], [0.365]]), np.zeros((2, 1, 3))
        )
        assert all(np.isfinite(value).all() for value in values.values())
        monkeypatch.setattr(_stationary_cpu, "StationaryGradientPlan", observe_plan)
        with pytest.raises(PlanObserved):
            _stationary_cpu.complete_rks_gradient_diagnostic(
                state, basis, cache=tmp_path
            )
