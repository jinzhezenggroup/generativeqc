"""Shared structural identity for default-allow automatic Libxc semilocal XC."""

from __future__ import annotations

from .libxc_blacklist import blacklist_reason
from .libxc_bulk_capabilities import functional_capability
from .spec import AUTO_BULK_COMPONENTS, UnsupportedXC

AUTOMATIC_FUNCTIONAL_CODE_BASE = 0x30000
AUTOMATIC_SCF_DOMAIN = "libxc-bulk-production-candidate/v2"
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
    reason = blacklist_reason(key)
    if reason is not None:
        raise UnsupportedXC(
            f"automatic Libxc functional {key} is blacklisted: {reason}"
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


__all__ = [
    "AUTOMATIC_FUNCTIONAL_CODE_BASE",
    "AUTOMATIC_SCF_DOMAIN",
    "automatic_functional_code",
]
