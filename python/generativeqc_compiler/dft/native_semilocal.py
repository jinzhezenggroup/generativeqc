"""Generated-metadata lookup for curated native semilocal execution families."""

from __future__ import annotations

import typing
from fractions import Fraction

from ._generated_native_semilocal import SEMILOCAL_FAMILIES


def _selector_record(value: str) -> typing.Mapping[str, typing.Any] | None:
    return next(
        (
            record
            for record in SEMILOCAL_FAMILIES
            if value == record["name"] or value in record["aliases"]
        ),
        None,
    )


def native_semilocal_record(
    value: typing.Any, *, spin: str | None = None
) -> typing.Mapping[str, typing.Any]:
    """Resolve one exact native semilocal execution row without importing XC."""
    if spin is not None and spin not in ("polarized", "unpolarized"):
        raise ValueError("native semilocal selector requires a supported spin")
    if isinstance(value, str):
        record = _selector_record(value)
        if record is None:
            raise ValueError(f"unknown native semilocal family {value!r}")
        return record

    actual_spin = getattr(value, "spin", None)
    if spin is not None and actual_spin is not None and actual_spin != spin:
        raise ValueError("native semilocal functional spin disagrees with its consumer")
    components = getattr(value, "components", None)
    range_omega = getattr(value, "range_omega", None)
    if components is None or range_omega is None:
        raise TypeError(
            "native semilocal lookup requires a selector or functional-like object"
        )
    try:
        actual = {name: Fraction(coefficient) for name, coefficient in components}
        omega = Fraction(range_omega)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid native semilocal functional metadata") from error
    for record in SEMILOCAL_FAMILIES:
        expected = {
            name: Fraction(coefficient) for name, coefficient in record["components"]
        }
        if actual == expected and omega == Fraction(record["range_omega"]):
            return record
    raise ValueError("semilocal composition has no exact native family record")


def legacy_grid_xc_selector(value: typing.Any, *, spin: str | None = None) -> int:
    """Return the two-state legacy grid-XC ABI selector from family metadata."""
    record = native_semilocal_record(value, spin=spin)
    kernel = record["stationary_kernel"]
    if kernel not in ("lda", "pbe"):
        raise ValueError("legacy CUDA grid-XC ABI has no lowerer for this family")
    return int(kernel != "lda")


def device_feature_ingredients(
    value: typing.Any, *, spin: str | None = None
) -> tuple[str, ...]:
    """Return resident feature inputs from generated native execution metadata."""
    record = native_semilocal_record(value, spin=spin)
    required = ["rho"]
    if record["requires_gradient"]:
        required.append("gradient")
    if record["requires_tau"]:
        required.append("tau")
    return tuple(required)
