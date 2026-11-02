"""Shared structural identity for default-allow automatic Libxc semilocal XC."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from ._generated_semilocal_compositions import COMPOSITIONS
from .density_work import DENSITY_WORK_DOMAIN
from .libxc_bulk_capabilities import available_capabilities, functional_capability
from .libxc_work import LIBXC_WORK_DOMAIN, automatic_work_policy
from .spec import POINTWISE_BULK_COMPONENTS, UnsupportedXC

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
        if all(
            automatic_work_policy(component).family == "lda"
            for component, _ in automatic_components(name)
        )
        else AUTOMATIC_SCF_DOMAIN
    )


@dataclass(frozen=True)
class AutomaticRegistryEntry:
    """One structural registration, independent of backend/product admission."""

    name: str
    libxc_id: int
    code: int

    @property
    def components(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the rational semilocal graph carried by this registration."""
        return automatic_components(self.name)

    @property
    def exact_exchange(self) -> Fraction:
        """Full-range exchange is a Fock contract, never part of point energy."""
        return Fraction(COMPOSITIONS.get(self.name, {}).get("exact_exchange", "0"))

    @property
    def stem(self) -> str:
        """Return the deterministic native symbol stem, not a science selector."""
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


def automatic_registry_entries() -> tuple[AutomaticRegistryEntry, ...]:
    """Enumerate CPU/CUDA plumbing without a positive functional whitelist.

    Structural representability does not grant endpoint, domain or derivative
    admission. These retain their separate evidence-backed capability gates.
    """
    # Capture the catalog and source identity once for this enumeration only.
    # A later call must observe changed catalog/compiler inputs, not a global memo.
    capabilities = {item.name: item for item in available_capabilities()}
    result = []
    codes = set()
    for name in sorted(POINTWISE_BULK_COMPONENTS):
        capability = capabilities.get(name)
        if capability is None:
            # Preserve the individual lookup's fail-closed diagnostic for a stale
            # selection whose registration is now missing or no longer imported.
            capability = functional_capability(name)
        if set(capability.required_ingredients) - _SUPPORTED_INGREDIENTS:
            continue
        if not 0 < capability.libxc_id < 0x10000:
            raise UnsupportedXC(
                f"automatic Libxc ID is outside the encoded range: {name}"
            )
        code = AUTOMATIC_FUNCTIONAL_CODE_BASE | capability.libxc_id
        if code in codes:
            raise ValueError(f"duplicate automatic Libxc functional code: {code:#x}")
        codes.add(code)
        result.append(AutomaticRegistryEntry(name, capability.libxc_id, code))
    for name, record in COMPOSITIONS.items():
        if record["code"] in codes:
            raise ValueError("duplicate composed Libxc functional code")
        if any(
            component not in {entry.name for entry in result}
            for component, _ in record["components"]
        ):
            raise UnsupportedXC(f"stale composed Libxc registration: {name}")
        codes.add(record["code"])
        result.append(AutomaticRegistryEntry(name, 0, record["code"]))
    if not result:
        raise RuntimeError("automatic Libxc semilocal registry is empty")
    return tuple(result)


def automatic_functional_code(name: str) -> int:
    """Return the stable native code for one admitted automatic Libxc component."""
    if not isinstance(name, str) or not name.strip():
        raise UnsupportedXC("automatic Libxc functional requires a nonempty name")
    key = name.upper()
    if key in COMPOSITIONS:
        return COMPOSITIONS[key]["code"]
    if key not in POINTWISE_BULK_COMPONENTS:
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
    result = {
        AUTOMATIC_FUNCTIONAL_CODE_BASE
        | capability.libxc_id: capability.required_ingredients
        for capability in available_capabilities()
        if capability.name in POINTWISE_BULK_COMPONENTS
        and 0 < capability.libxc_id < 0x10000
        and not set(capability.required_ingredients) - _SUPPORTED_INGREDIENTS
    }
    for record in COMPOSITIONS.values():
        ingredients = set().union(
            *(
                result[automatic_functional_code(component)]
                for component, _ in record["components"]
            )
        )
        result[record["code"]] = tuple(
            key for key in ("rho", "sigma", "tau") if key in ingredients
        )
    return result


def automatic_functional_ingredients(code: int) -> tuple[str, ...]:
    """Resolve an exact registered native code without integer ABI truncation."""
    if type(code) is not int:
        raise UnsupportedXC("automatic Libxc point code is not registered")
    try:
        return _automatic_ingredient_registry()[code]
    except KeyError as error:
        raise UnsupportedXC("automatic Libxc point code is not registered") from error


@cache
def _automatic_name_registry() -> dict[int, str]:
    return {entry.code: entry.name for entry in automatic_registry_entries()}


def automatic_functional_name(code: int) -> str:
    """Authenticate a code without granting a backend or derivative capability."""
    automatic_functional_ingredients(code)
    return _automatic_name_registry()[code]


def automatic_components(name: str) -> tuple[tuple[str, Fraction], ...]:
    """Expose a composition without replacing its components' worker policies."""
    record = COMPOSITIONS.get(name)
    if record is None:
        return ((name, Fraction(1)),)
    return tuple(
        (component, Fraction(weight)) for component, weight in record["components"]
    )


def automatic_composition_name(
    components: tuple[tuple[str, Fraction], ...],
    exact_exchange: Fraction = Fraction(0),
) -> str | None:
    """Match the canonical rational electronic graph, independent of its alias."""
    actual = tuple(sorted(components))
    if len(actual) == 1 and actual[0][1] == 1 and not exact_exchange:
        try:
            automatic_functional_code(actual[0][0])
            return actual[0][0]
        except UnsupportedXC:
            return None
    for name, record in COMPOSITIONS.items():
        if (
            automatic_components(name) == actual
            and Fraction(record["exact_exchange"]) == exact_exchange
        ):
            return name
    return None


__all__ = [
    "AUTOMATIC_FUNCTIONAL_CODE_BASE",
    "AUTOMATIC_SCF_DOMAIN",
    "AutomaticRegistryEntry",
    "automatic_components",
    "automatic_composition_name",
    "automatic_functional_code",
    "automatic_functional_ingredients",
    "automatic_functional_name",
    "automatic_registry_entries",
    "automatic_scf_domain",
]
