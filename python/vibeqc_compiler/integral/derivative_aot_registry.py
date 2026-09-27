"""Method-neutral packaged first-derivative AOT identity and selection.

This module owns the runtime/build identity for generated four-center first
derivative programs. Scientific contraction weights remain in method/TensorIR
owners; this registry only maps an explicit radial operator and shell/component
domain onto a packaged generated artifact.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .ir import four_center_eri_operator
from .range_separation import CoulombKernel, CoulombKernelFamily
from .shell_spec import ShellClassSpec
from .weighted_eri import build_weighted_eri_ir, build_weighted_eri_kernel
from .weighted_eri_native import emit_weighted_eri_runtime

AOT_ANGULAR_DOMAIN = (0, 1)
AOT_COMPONENT_CAPACITY = 64
AOT_DERIVATIVE_ORDER = 1


def component_groups(
    angular: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], ...]:
    """Partition one Cartesian s/p shell class into bounded component programs."""

    if (
        not isinstance(angular, tuple)
        or len(angular) != 4
        or any(
            type(value) is not int or value not in AOT_ANGULAR_DOMAIN
            for value in angular
        )
    ):
        raise ValueError("derivative AOT supports four s/p angular orders")
    spec = ShellClassSpec("".join("sp"[value] for value in angular), angular)
    return tuple(
        tuple(
            range(
                begin,
                min(begin + AOT_COMPONENT_CAPACITY, spec.component_count),
            )
        )
        for begin in range(0, spec.component_count, AOT_COMPONENT_CAPACITY)
    )


def component_group(
    angular: tuple[int, int, int, int], component: int
) -> tuple[int, tuple[int, ...]]:
    """Return the bounded AOT component group containing one shell component."""

    groups = component_groups(angular)
    if type(component) is not int or component < 0:
        raise ValueError("derivative AOT component must be nonnegative")
    for index, group in enumerate(groups):
        if component in group:
            return index, group
    raise ValueError("derivative AOT component is outside the shell class")


@dataclass(frozen=True, slots=True)
class DerivativeAotKey:
    """Scientific/runtime identity for one packaged derivative program."""

    backend: str
    radial: CoulombKernel
    angular: tuple[int, int, int, int]
    group_index: int
    derivative_order: int = AOT_DERIVATIVE_ORDER

    def __post_init__(self) -> None:
        if self.backend not in ("cpu", "cuda"):
            raise ValueError("derivative AOT backend must be cpu or cuda")
        if not isinstance(self.radial, CoulombKernel):
            raise TypeError("derivative AOT requires an explicit CoulombKernel")
        groups = component_groups(self.angular)
        if (
            type(self.group_index) is not int
            or not 0 <= self.group_index < len(groups)
        ):
            raise ValueError("derivative AOT group index is out of range")
        if self.derivative_order != AOT_DERIVATIVE_ORDER:
            raise ValueError("derivative AOT currently packages first derivatives only")

    @property
    def component_indices(self) -> tuple[int, ...]:
        return component_groups(self.angular)[self.group_index]

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a path-independent identity payload for manifests/caches."""

        return {
            "version": 1,
            "backend": self.backend,
            "derivative_order": self.derivative_order,
            "radial": self.radial.to_payload(),
            "angular": list(self.angular),
            "group_index": self.group_index,
            "component_indices": list(self.component_indices),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True, slots=True)
class PackagedDerivativeAot:
    """Resolved package entry; method coefficients are deliberately absent."""

    key: DerivativeAotKey
    entry_prefix: str

    @property
    def component_indices(self) -> tuple[int, ...]:
        return self.key.component_indices


def _family_tag(family: CoulombKernelFamily) -> str:
    return {
        CoulombKernelFamily.FULL_RANGE: "full",
        CoulombKernelFamily.SHORT_RANGE: "sr",
        CoulombKernelFamily.LONG_RANGE: "lr",
    }[family]


def entry_prefix_for_key(key: DerivativeAotKey) -> str:
    """Return a stable private C symbol prefix for one exact AOT identity.

    Existing CPU SR/LR prefixes are intentionally retained so installed package
    layout can migrate to this registry without invalidating the current
    WB97M-V range-exchange artifacts. New operator/backend combinations use the
    method-neutral derivative stem.
    """

    radial_key = canonical_hash(key.radial.to_payload())[:12]
    shell = "".join(str(value) for value in key.angular)
    family = _family_tag(key.radial.family)
    if key.backend == "cpu" and key.radial.family in (
        CoulombKernelFamily.SHORT_RANGE,
        CoulombKernelFamily.LONG_RANGE,
    ):
        stem = "vibeqc_rsh_cpu"
    else:
        stem = f"vibeqc_derivative_{key.backend}_d{key.derivative_order}"
    return f"{stem}_{family}_{shell}_{key.group_index}_{radial_key}"


def make_key(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
    *,
    backend: str,
) -> DerivativeAotKey:
    return DerivativeAotKey(
        backend=backend,
        radial=radial,
        angular=angular,
        group_index=group_index,
    )


def select_packaged_derivative_aot(
    library: typing.Any,
    *,
    backend: str,
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    component: int,
) -> PackagedDerivativeAot | None:
    """Resolve a packaged program without consulting a method/function name."""

    group_index, _ = component_group(angular, component)
    key = make_key(radial, angular, group_index, backend=backend)
    prefix = entry_prefix_for_key(key)
    if library is None:
        return None
    try:
        getattr(library, f"{prefix}_identity_v2")
    except AttributeError:
        return None
    return PackagedDerivativeAot(key=key, entry_prefix=prefix)


def program_source(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
    *,
    backend: str = "cpu",
) -> tuple[str, tuple[int, ...], str]:
    """Emit one generated weighted-ERI derivative program for a registry key."""

    key = make_key(radial, angular, group_index, backend=backend)
    integral = build_weighted_eri_ir(
        angular, operator=four_center_eri_operator(radial)
    )
    kernel = build_weighted_eri_kernel(integral, key.component_indices)
    prefix = entry_prefix_for_key(key)
    return (
        emit_weighted_eri_runtime(kernel, backend=backend, entry_prefix=prefix),
        key.component_indices,
        prefix,
    )
