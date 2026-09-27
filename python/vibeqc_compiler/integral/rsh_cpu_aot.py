"""Compatibility surface for the packaged CPU range-exchange derivative AOT.

New code should use :mod:`derivative_aot_registry`. This module preserves the
existing build-time API and private symbol layout while the package inventory
migrates to the method-neutral registry.
"""

from __future__ import annotations

from itertools import product

from .derivative_aot_registry import (
    AOT_ANGULAR_DOMAIN,
    component_groups,
    entry_prefix_for_key,
    make_key,
)
from .derivative_aot_registry import (
    program_source as _program_source,
)
from .range_separation import CoulombKernel, CoulombKernelFamily


def _require_range(radial: CoulombKernel) -> None:
    if not isinstance(radial, CoulombKernel) or radial.family not in (
        CoulombKernelFamily.SHORT_RANGE,
        CoulombKernelFamily.LONG_RANGE,
    ):
        raise ValueError("CPU RSH AOT requires short- or long-range Coulomb")


def entry_prefix(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
) -> str:
    """Return the legacy-compatible symbol for one exact range AOT identity."""

    _require_range(radial)
    return entry_prefix_for_key(make_key(radial, angular, group_index, backend="cpu"))


def program_source(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
) -> tuple[str, tuple[int, ...], str]:
    """Emit one build-time CPU range derivative program."""

    _require_range(radial)
    return _program_source(radial, angular, group_index, backend="cpu")


def inventory_size() -> int:
    """Number of programs for one short/long radial pair over the s/p domain."""

    per_family = sum(
        len(component_groups((a, b, c, d)))
        for a, b, c, d in product(AOT_ANGULAR_DOMAIN, repeat=4)
    )
    return 2 * per_family


assert inventory_size() == 34
