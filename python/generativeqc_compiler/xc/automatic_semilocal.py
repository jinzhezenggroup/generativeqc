"""Shared structural identity for default-allow automatic Libxc semilocal XC."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from .density_work import DENSITY_WORK_DOMAIN
from .libxc_bulk_capabilities import available_capabilities, functional_capability
from .libxc_work import LIBXC_WORK_DOMAIN, automatic_work_policy
from .spec import AUTO_BULK_COMPONENTS, UnsupportedXC

AUTOMATIC_FUNCTIONAL_CODE_BASE = 0x30000
AUTOMATIC_SCF_DOMAIN = LIBXC_WORK_DOMAIN
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


def automatic_scf_domain(name: str) -> str:
    """Select the exact native worker domain by ingredient family, not alias.

    Density-only registrations use distinct total-density RKS and per-spin UKS
    layouts under version 5. Other ingredients retain their existing worker.
    This numerical identity alone does not promote a backend or product.
    """
    return (
        DENSITY_WORK_DOMAIN
        if automatic_work_policy(name).family == "lda"
        else AUTOMATIC_SCF_DOMAIN
    )


@dataclass(frozen=True)
class AutomaticRegistryEntry:
    """One structural registration, independent of backend/product admission."""

    name: str
    libxc_id: int
    code: int

    @property
    def stem(self) -> str:
        """Return the deterministic native symbol stem, not a science selector."""
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


def automatic_registry_entries() -> tuple[AutomaticRegistryEntry, ...]:
    """Enumerate CPU/CUDA plumbing without a positive functional whitelist.

    Structural representability does not grant endpoint, domain or derivative
    admission. These retain their separate evidence-backed capability gates.
    """
    result = []
    codes = set()
    for name in sorted(AUTO_BULK_COMPONENTS):
        capability = functional_capability(name)
        if set(capability.required_ingredients) - _SUPPORTED_INGREDIENTS:
            continue
        code = automatic_functional_code(name)
        if code in codes:
            raise ValueError(f"duplicate automatic Libxc functional code: {code:#x}")
        codes.add(code)
        result.append(AutomaticRegistryEntry(name, capability.libxc_id, code))
    if not result:
        raise RuntimeError("automatic Libxc semilocal registry is empty")
    return tuple(result)


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
    "AutomaticRegistryEntry",
    "automatic_functional_code",
    "automatic_functional_ingredients",
    "automatic_registry_entries",
    "automatic_scf_domain",
]
