"""The R2SCAN stationary selector must never generate another XC model."""

from dataclasses import replace
from fractions import Fraction
from typing import NoReturn

import pytest
from vibeqc_compiler.xc import geometry_cuda
from vibeqc_compiler.xc._generated_native_semilocal import SEMILOCAL_FAMILIES
from vibeqc_compiler.xc.spec import FunctionalSpec, functional

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

    monkeypatch.setattr(geometry_cuda, "energy_expression", unexpected_codegen)
    with pytest.raises(ValueError, match="canonical semilocal"):
        geometry_cuda._emit_stationary_point(R2SCAN_CODE, semilocal=spec)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_r2scan_alias_and_spin_use_the_canonical_point_model(
    spin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = functional("R2SCAN", spin="polarized")
    supplied = replace(
        functional("R2SCAN", spin=spin), identifier="custom-r2scan-alias"
    )

    class CapturedPoint(Exception):
        pass

    def capture(spec: FunctionalSpec, *, production: bool) -> NoReturn:
        assert dict(spec.components) == dict(canonical.components)
        assert spec.spin == "polarized"
        assert spec.range_omega == canonical.range_omega
        assert production
        raise CapturedPoint

    monkeypatch.setattr(geometry_cuda, "energy_expression", capture)
    with pytest.raises(CapturedPoint):
        geometry_cuda._emit_stationary_point(R2SCAN_CODE, semilocal=supplied)
