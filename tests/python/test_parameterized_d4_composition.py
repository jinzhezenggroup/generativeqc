"""Parameterized D4 composition from the pinned registry and existing MethodIR."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, evaluate_d4_correction
from generativeqc.ks import public_dft_selectors
from generativeqc_compiler.method import (
    D4Spec,
    DispersionCorrectionPrimitive,
    UnsupportedMethod,
    d4_composite_method_identifiers,
    d4_eeq_spec_for_method,
    resolve_method,
)


@pytest.mark.parametrize("base", ("PBE0", "B3LYP", "R2SCAN"))
def test_parameterized_d4_reuses_electronic_method_ir(base: str) -> None:
    electronic = resolve_method(base)
    combined = resolve_method(f"{base}-D4(BJ-EEQ-ATM)")
    corrections = tuple(
        primitive
        for primitive in combined.primitives
        if isinstance(primitive, DispersionCorrectionPrimitive)
    )
    assert len(corrections) == 1
    assert isinstance(corrections[0].specification, D4Spec)
    assert corrections[0].specification == d4_eeq_spec_for_method(base)
    assert (
        tuple(
            primitive
            for primitive in combined.primitives
            if not isinstance(primitive, DispersionCorrectionPrimitive)
        )
        == electronic.primitives
    )


def test_parameterized_d4_reuses_generated_xc_aliases() -> None:
    assert d4_eeq_spec_for_method("CAM-B3LYP") == d4_eeq_spec_for_method("CAMB3LYP")
    names = set(d4_composite_method_identifiers())
    assert "CAM-B3LYP-D4(BJ-EEQ-ATM)" in names
    assert "CAMB3LYP-D4(BJ-EEQ-ATM)" in names
    assert "PBEH-D4(BJ-EEQ-ATM)" in names


def test_parameterized_d4_inventory_is_intersection_not_cartesian_product() -> None:
    names = set(d4_composite_method_identifiers())
    assert "PBE0-D4(BJ-EEQ-ATM)" in names
    assert "B3LYP-D4(BJ-EEQ-ATM)" in names
    assert "R2SCAN-D4(BJ-EEQ-ATM)" in names
    assert "R2SCAN-3c-D4(BJ-EEQ-ATM)" not in names
    with pytest.raises(UnsupportedMethod, match="unknown DFT method"):
        resolve_method("LDA_XC_PW-D4(BJ-EEQ-ATM)")


def test_public_d4_discovery_remains_lowerer_gated() -> None:
    selectors = set(public_dft_selectors())
    assert "pbe0-d4-rks" in selectors
    assert "pbe0-d4-uks" in selectors
    assert "pbeh-d4-rks" in selectors
    assert "b3lyp-d4-rks" in selectors
    assert "r2scan-d4-rks" in selectors
    assert "scan-d4-rks" not in selectors


@pytest.mark.parametrize(
    ("selector", "electronic_selector", "parameter_name"),
    (
        ("pbe0-d4-rks", "pbe0-rks", "PBE0-D4(BJ-EEQ-ATM)"),
        ("b3lyp-d4-rks", "b3lyp-rks", "B3LYP-D4(BJ-EEQ-ATM)"),
    ),
)
def test_parameterized_d4_public_energy_is_exact_once(
    selector: str, electronic_selector: str, parameter_name: str
) -> None:
    atoms = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64)
    grid = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)

    electronic = Calculator(
        method=electronic_selector,
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(grid=grid),
    ).singlepoint(atoms, properties=("energy",))
    correction = evaluate_d4_correction(
        parameter_name, numbers, positions, device="cpu"
    )
    combined = Calculator(
        method=selector,
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(grid=grid),
    ).singlepoint(atoms, properties=("energy",))

    assert combined.dispersion is not None
    assert combined.dispersion.energy == pytest.approx(correction.energy, abs=2.0e-13)
    assert combined.energy == pytest.approx(
        electronic.energy + correction.energy, abs=2.0e-11
    )
