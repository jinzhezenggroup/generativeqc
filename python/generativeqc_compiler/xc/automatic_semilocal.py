"""Shared structural identity for default-allow automatic Libxc semilocal XC."""

from __future__ import annotations

from functools import cache

from .libxc_bulk_capabilities import available_capabilities, functional_capability
from .libxc_work import LIBXC_WORK_DOMAIN
from .spec import AUTO_BULK_COMPONENTS, UnsupportedXC

AUTOMATIC_FUNCTIONAL_CODE_BASE = 0x30000
AUTOMATIC_SCF_DOMAIN = LIBXC_WORK_DOMAIN
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


def automatic_functional_code(name: str) -> int:
    """Return the stable native code for one admitted automatic Libxc component."""
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedXC("automatic Libxc functional requires a nonempty name")
    key = name.upper()
    if key not in AUTO_BULK_COMPONENTS:
        raise UnsupportedXC(
            f"Libxc functional {key!r} is not an automatic semilocal registration"
        )
    capability = functional_capability(key)
    unsupported = tuple(
        ingredient
        for ingredient in capability.required_ingredients
        if ingredient not in _SUPPORTED_INGREDIENTS
    )
    if unsupported:
        raise UnsupportedXC(
            f"automatic Libxc functional {key} requires unsupported ingredients "
            f"{unsupported!r}"
        )
    if not 0 < capability.libxc_id < 0x10000:
        raise UnsupportedXC(f"automatic Libxc ID is outside the encoded range: {key}")
    return AUTOMATIC_FUNCTIONAL_CODE_BASE | capability.libxc_id


@cache
def _automatic_ingredient_registry() -> dict[int, tuple[str, ...]]:
    return {
        AUTOMATIC_FUNCTIONAL_CODE_BASE
        | capability.libxc_id: capability.required_ingredients
        for capability in available_capabilities()
        if capability.name in AUTO_BULK_COMPONENTS
        and 0 < capability.libxc_id < 0x10000
        and not set(capability.required_ingredients) - _SUPPORTED_INGREDIENTS
    }


def automatic_functional_ingredients(code: int) -> tuple[str, ...]:
    """Resolve an exact registered native code without integer ABI truncation."""
    if type(code) is not int or not (
        AUTOMATIC_FUNCTIONAL_CODE_BASE < code < AUTOMATIC_FUNCTIONAL_CODE_BASE + 0x10000
    ):
        raise UnsupportedXC("automatic Libxc point code is not registered")
    try:
        return _automatic_ingredient_registry()[code]
    except KeyError as error:
        raise UnsupportedXC("automatic Libxc point code is not registered") from error


__all__ = [
    "AUTOMATIC_FUNCTIONAL_CODE_BASE",
    "AUTOMATIC_SCF_DOMAIN",
    "automatic_functional_code",
    "automatic_functional_ingredients",
]
