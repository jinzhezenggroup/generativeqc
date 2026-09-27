"""Public automatic-Libxc Calculator discovery and fail-closed construction."""

from __future__ import annotations

import typing

import pytest

from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._api_types import MethodCapabilities
from vibeqc_compiler.method import MethodIR, SemilocalXCPrimitive
from vibeqc_compiler.xc.spec import functional


GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)


def _options(name: str, spin: str) -> KsOptions:
    spec = functional(name, spin=spin)
    ir = MethodIR(
        identifier=f"LIBXC:{name}",
        spin=spin,
        primitives=(SemilocalXCPrimitive(spec),),
    )
    options = KsOptions(
        functional=spec,
        grid=GRID,
        scf_domain="libxc-bulk-production-candidate/v2",
        xc_schedule="host_unfused",
    )
    object.__setattr__(options, "_method_ir", ir)
    return options


class CaptureCalculator(Calculator):
    def __init__(self, method: str, **kwargs: typing.Any) -> None:
        self.captured_method = method
        self.captured_kwargs = kwargs
        self._capabilities = MethodCapabilities(
            method=method,
            family="density_functional",
            available=True,
            supports_batch=True,
            supported_properties=frozenset(("energy", "forces")),
        )


@pytest.mark.parametrize(
    ("name", "spin", "carrier"),
    (
        ("LDA_C_BR78", "unpolarized", "lda-rks"),
        ("GGA_X_APBE", "polarized", "pbe-uks"),
        ("MGGA_X_LTA", "unpolarized", "r2scan-rks"),
    ),
)
def test_from_libxc_uses_only_ingredient_carrier(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    spin: str,
    carrier: str,
) -> None:
    import vibeqc.ks as ks

    options = _options(name, spin)
    seen: dict[str, object] = {}

    def resolve(
        identifier: str,
        *,
        spin: str,
        grid: GridSpec,
        tile_points: int,
        xc_schedule: str,
    ) -> KsOptions:
        seen.update(
            identifier=identifier,
            spin=spin,
            grid=grid,
            tile_points=tile_points,
            xc_schedule=xc_schedule,
        )
        return options

    monkeypatch.setattr(ks, "resolve_public_libxc_ks_options", resolve)
    calculator = CaptureCalculator.from_libxc(
        name,
        spin=spin,
        grid=GRID,
        basis="sto-3g",
    )
    assert calculator.captured_method == carrier
    assert calculator.captured_kwargs["ks_options"] is options
    assert calculator.captured_kwargs["device"] == "cpu"
    assert seen == {
        "identifier": name,
        "spin": spin,
        "grid": GRID,
        "tile_points": 256,
        "xc_schedule": "host_unfused",
    }
    assert calculator._capabilities.method == (
        f"libxc:{name}:{'uks' if spin == 'polarized' else 'rks'}"
    )
    assert calculator._capabilities.supported_properties == frozenset(("energy",))
    assert not calculator._capabilities.supports_batch


def test_from_libxc_rejects_unqualified_execution_modes_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.ks as ks

    monkeypatch.setattr(
        ks,
        "resolve_public_libxc_ks_options",
        lambda *args, **kwargs: pytest.fail("unqualified request reached resolution"),
    )
    with pytest.raises(NotImplementedError, match="requires CPU"):
        CaptureCalculator.from_libxc(
            "GGA_X_APBE", spin="unpolarized", grid=GRID, device="cuda"
        )
    with pytest.raises(NotImplementedError, match="direct Coulomb"):
        CaptureCalculator.from_libxc(
            "GGA_X_APBE",
            spin="unpolarized",
            grid=GRID,
            density_fitting="cpu",
        )
    with pytest.raises(NotImplementedError, match="strict FP64"):
        CaptureCalculator.from_libxc(
            "GGA_X_APBE",
            spin="unpolarized",
            grid=GRID,
            precision="auto",
        )


def test_generic_libxc_options_never_infer_force_coefficients() -> None:
    options = _options("GGA_X_APBE", "unpolarized")
    assert options.generic_libxc_registration == "GGA_X_APBE"
    assert options.coefficients == (1.0, 1.0, 0.0)


def test_available_libxc_functionals_delegates_to_installed_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc_compiler.method as method

    monkeypatch.setattr(
        method,
        "installed_public_functionals",
        lambda: ("GGA_X_APBE", "MGGA_X_LTA"),
    )
    assert Calculator.available_libxc_functionals() == (
        "GGA_X_APBE",
        "MGGA_X_LTA",
    )
