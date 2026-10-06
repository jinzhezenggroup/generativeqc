"""Generated-metadata lookup for curated native semilocal execution families."""

from __future__ import annotations

import typing
from fractions import Fraction

from ._generated_native_semilocal import SEMILOCAL_FAMILIES
from .spec import FunctionalSpec, UnsupportedXC, functional


def native_semilocal_spec(
    value: FunctionalSpec | str, *, spin: str = "polarized"
) -> FunctionalSpec:
    """Resolve a curated semilocal spec from generated family metadata."""
    if isinstance(value, FunctionalSpec):
        return value
    if not isinstance(value, str) or spin not in ("polarized", "unpolarized"):
        raise UnsupportedXC(
            "native semilocal selector requires a name and supported spin"
        )
    try:
        return functional(value, spin=spin)
    except UnsupportedXC:
        pass
    record = next((item for item in SEMILOCAL_FAMILIES if item["name"] == value), None)
    if record is None:
        raise UnsupportedXC(f"unknown native semilocal family {value!r}")
    return FunctionalSpec(
        record["name"],
        tuple(
            (name, Fraction(coefficient)) for name, coefficient in record["components"]
        ),
        spin,
        range_omega=Fraction(record["range_omega"]),
    )


def native_semilocal_record(
    value: FunctionalSpec | str, *, spin: str = "polarized"
) -> typing.Mapping[str, typing.Any]:
    """Return the generated family row for an exact curated semilocal graph."""
    spec = native_semilocal_spec(value, spin=spin)
    components = dict(spec.components)
    for record in SEMILOCAL_FAMILIES:
        expected = {
            name: Fraction(coefficient) for name, coefficient in record["components"]
        }
        if components != expected:
            continue
        if spec.range_omega != Fraction(record["range_omega"]):
            continue
        return record
    raise UnsupportedXC("semilocal composition has no exact native family record")


def legacy_grid_xc_selector(
    value: FunctionalSpec | str, *, spin: str = "polarized"
) -> int:
    """Return the two-state legacy grid-XC ABI selector from family metadata."""
    record = native_semilocal_record(value, spin=spin)
    kernel = record["stationary_kernel"]
    if kernel not in ("lda", "pbe"):
        raise UnsupportedXC("legacy CUDA grid-XC ABI has no lowerer for this family")
    return int(kernel != "lda")


def device_feature_ingredients(
    value: FunctionalSpec | str, *, spin: str = "polarized"
) -> tuple[str, ...]:
    """Return device-resident feature inputs from FunctionalSpec semantics."""
    spec = native_semilocal_spec(value, spin=spin)
    required = ["rho"]
    if "sigma" in spec.ingredients:
        required.append("gradient")
    if "tau" in spec.ingredients:
        required.append("tau")
    return tuple(required)
