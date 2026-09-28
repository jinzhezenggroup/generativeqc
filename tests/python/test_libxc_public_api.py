from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Calculator, method_capabilities
from vibeqc.ks import (
    AUTOMATIC_SCF_DOMAIN,
    native_xc_functional_code,
    parse_automatic_libxc_selector,
)
from vibeqc_compiler.xc.automatic_semilocal import automatic_functional_code

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


def test_public_libxc_selector_executes_automatic_cpu_scf() -> None:
    calculator = Calculator(
        method="libxc:GGA_X_APBE",
        basis="sto-3g",
        device="cpu",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )

    assert calculator._method_name == "libxc:gga_x_apbe"
    assert calculator._native_method_name == "pbe-rks"
    assert calculator._automatic_libxc_name == "GGA_X_APBE"
    assert calculator.ks_options.scf_domain == AUTOMATIC_SCF_DOMAIN
    assert calculator.ks_options.functional.components == (("GGA_X_APBE", 1),)
    assert calculator.method_ir.spin == "unpolarized"

    result = calculator.singlepoint(H2, properties=("energy",))
    assert result.converged
    assert np.isfinite(result.energy)
    assert result.forces is None
    assert result.executed_backend == "cpu_reference"

    pbe = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device="cpu",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    ).singlepoint(H2, properties=("energy",))
    assert abs(result.energy - pbe.energy) > 1e-6


def test_public_libxc_capability_stays_energy_only() -> None:
    capability = method_capabilities("libxc:GGA_X_APBE")

    assert capability.family == "density_functional"
    assert capability.available
    assert capability.supports_batch
    assert capability.supported_properties == frozenset(("energy",))

    calculator = Calculator(method="libxc:GGA_X_APBE", basis="sto-3g")
    with pytest.raises(ValueError, match="does not support properties.*forces"):
        calculator.singlepoint(H2, properties=("energy", "forces"))


@pytest.mark.parametrize(
    ("selector", "transport", "spin"),
    (
        ("libxc:LDA_C_BR78", "lda-rks", "unpolarized"),
        ("libxc-rks:GGA_X_APBE", "pbe-rks", "unpolarized"),
        ("libxc-uks:GGA_X_APBE", "pbe-uks", "polarized"),
        ("libxc:MGGA_X_LTA", "r2scan-rks", "unpolarized"),
    ),
)
def test_public_libxc_selector_uses_ingredient_transport_only(
    selector: str, transport: str, spin: str
) -> None:
    calculator = Calculator(method=selector, basis="sto-3g", device="cpu")

    assert calculator._native_method_name == transport
    assert calculator.method_ir.spin == spin
    assert calculator.ks_options.scf_domain == AUTOMATIC_SCF_DOMAIN


def test_public_libxc_selector_and_snapshot_code_share_identity() -> None:
    assert native_xc_functional_code("libxc:GGA_X_APBE") == automatic_functional_code(
        "GGA_X_APBE"
    )


@pytest.mark.parametrize("name", ("GGA_C_AM05", "GGA_X_AK13"))
def test_shared_boundary_classes_are_public_without_name_blacklist(name: str) -> None:
    calculator = Calculator(method=f"libxc:{name}", basis="sto-3g", device="cpu")

    assert calculator._automatic_libxc_name == name
    assert calculator.ks_options.scf_domain == AUTOMATIC_SCF_DOMAIN


def test_automatic_libxc_cuda_remains_explicitly_unavailable() -> None:
    with pytest.raises(ValueError, match="supports CPU only"):
        Calculator(method="libxc:GGA_X_APBE", basis="sto-3g", device="cuda")


def test_libxc_selector_parser_is_explicit_about_spin() -> None:
    assert parse_automatic_libxc_selector("libxc:GGA_X_APBE") == (
        "GGA_X_APBE",
        "unpolarized",
    )
    assert parse_automatic_libxc_selector("libxc-rks:GGA_X_APBE") == (
        "GGA_X_APBE",
        "unpolarized",
    )
    assert parse_automatic_libxc_selector("libxc-uks:GGA_X_APBE") == (
        "GGA_X_APBE",
        "polarized",
    )
    assert parse_automatic_libxc_selector("pbe-rks") is None
