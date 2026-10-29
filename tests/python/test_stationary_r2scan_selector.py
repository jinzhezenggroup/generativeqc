"""The R2SCAN stationary selector must never generate another XC model."""

from dataclasses import replace
from fractions import Fraction
from typing import NoReturn

import pytest
from generativeqc_compiler.dft._generated_native_semilocal import SEMILOCAL_FAMILIES
from generativeqc_compiler.xc import geometry_cuda
from generativeqc_compiler.xc.spec import FunctionalSpec, functional

R2SCAN_CODE = next(
    item["code"] for item in SEMILOCAL_FAMILIES if item["name"] == "R2SCAN"
)


@pytest.mark.parametrize(
    "spec",
    [
        functional("PBE"),
        replace(
            functional("R2SCAN"),
            components=(
                ("MGGA_X_R2SCAN", Fraction(1, 2)),
                ("MGGA_C_R2SCAN", Fraction(1)),
            ),
        ),
        replace(functional("R2SCAN"), exact_exchange=Fraction(1, 4)),
        replace(functional("R2SCAN"), range_omega=Fraction(1, 3)),
    ],
)
def test_r2scan_selector_rejects_changed_components_or_parameters(
    spec: FunctionalSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_codegen(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("mismatched model reached point code generation")

    monkeypatch.setattr(geometry_cuda, "emit_r2scan_program", unexpected_codegen)
    with pytest.raises(ValueError, match="canonical semilocal"):
        geometry_cuda._emit_stationary_point(R2SCAN_CODE, semilocal=spec)


@pytest.fixture(scope="module")
def canonical_r2scan_point() -> str:
    """Generate the actual SCF-domain point artifact used by both spin ABIs."""
    return geometry_cuda._emit_stationary_point(
        R2SCAN_CODE, semilocal=functional("R2SCAN", spin="polarized")
    )


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_r2scan_alias_and_spin_use_the_canonical_point_model(
    spin: str, canonical_r2scan_point: str
) -> None:
    supplied = replace(
        functional("R2SCAN", spin=spin), identifier="custom-r2scan-alias"
    )
    emitted = geometry_cuda._emit_stationary_point(R2SCAN_CODE, semilocal=supplied)
    assert emitted == canonical_r2scan_point
