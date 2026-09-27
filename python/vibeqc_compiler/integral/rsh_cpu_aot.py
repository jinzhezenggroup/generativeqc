"""Build-time AOT layout for bounded CPU range-exchange derivatives."""

from __future__ import annotations

import math
from itertools import product

from vibeqc_compiler.common.provenance import canonical_hash

from .ir import four_center_eri_operator
from .range_separation import CoulombKernel, CoulombKernelFamily
from .shell_spec import ShellClassSpec
from .weighted_eri import build_weighted_eri_ir, build_weighted_eri_kernel
from .weighted_eri_native import emit_weighted_eri_runtime

AOT_ANGULAR_DOMAIN = (0, 1)
AOT_COMPONENT_CAPACITY = 64


def component_groups(angular: tuple[int, int, int, int]) -> tuple[tuple[int, ...], ...]:
    """Partition one Cartesian s/p shell class into <=64-component programs."""

    if (
        not isinstance(angular, tuple)
        or len(angular) != 4
        or any(type(value) is not int or value not in AOT_ANGULAR_DOMAIN for value in angular)
    ):
        raise ValueError("CPU RSH AOT supports four s/p angular orders")
    spec = ShellClassSpec("".join("sp"[value] for value in angular), angular)
    return tuple(
        tuple(range(begin, min(begin + AOT_COMPONENT_CAPACITY, spec.component_count)))
        for begin in range(0, spec.component_count, AOT_COMPONENT_CAPACITY)
    )


def component_group(
    angular: tuple[int, int, int, int], component: int
) -> tuple[int, tuple[int, ...]]:
    groups = component_groups(angular)
    if type(component) is not int or component < 0:
        raise ValueError("CPU RSH AOT component must be nonnegative")
    for index, group in enumerate(groups):
        if component in group:
            return index, group
    raise ValueError("CPU RSH AOT component is outside the shell class")


def entry_prefix(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
) -> str:
    """Return a deterministic C symbol prefix including exact radial identity."""

    if not isinstance(radial, CoulombKernel) or radial.family not in (
        CoulombKernelFamily.SHORT_RANGE,
        CoulombKernelFamily.LONG_RANGE,
    ):
        raise ValueError("CPU RSH AOT requires short- or long-range Coulomb")
    groups = component_groups(angular)
    if type(group_index) is not int or not 0 <= group_index < len(groups):
        raise ValueError("CPU RSH AOT group index is out of range")
    radial_key = canonical_hash(radial.to_payload())[:12]
    family = "sr" if radial.family == CoulombKernelFamily.SHORT_RANGE else "lr"
    shell = "".join(str(value) for value in angular)
    return f"vibeqc_rsh_cpu_{family}_{shell}_{group_index}_{radial_key}"


def program_source(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
) -> tuple[str, tuple[int, ...], str]:
    """Emit one build-time weighted-ERI program and its runtime symbol prefix."""

    groups = component_groups(angular)
    if type(group_index) is not int or not 0 <= group_index < len(groups):
        raise ValueError("CPU RSH AOT group index is out of range")
    selected = groups[group_index]
    integral = build_weighted_eri_ir(
        angular, operator=four_center_eri_operator(radial)
    )
    kernel = build_weighted_eri_kernel(integral, selected)
    prefix = entry_prefix(radial, angular, group_index)
    return (
        emit_weighted_eri_runtime(
            kernel, backend="cpu", entry_prefix=prefix
        ),
        selected,
        prefix,
    )


def inventory_size() -> int:
    """Number of programs for one short/long radial pair over the s/p domain."""

    per_family = sum(
        len(component_groups(tuple(angular)))
        for angular in product(AOT_ANGULAR_DOMAIN, repeat=4)
    )
    return 2 * per_family


assert inventory_size() == 34
