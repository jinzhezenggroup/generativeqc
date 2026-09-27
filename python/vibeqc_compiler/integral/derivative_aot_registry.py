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
AOT_GENERATOR_ABI = 1
AOT_SPIN_CONTRACT = "spin-neutral"
AOT_WEIGHTED_OUTPUT_CONTRACT = "weighted-eri-value-center-gradient-v2"
AOT_WEIGHTED_CONTRACTION_CONTRACT = "packed-component-weights-v1"
AOT_COMPONENT_OUTPUT_CONTRACT = "primitive-center-gradient-v1"
AOT_COMPONENT_CONTRACTION_CONTRACT = "record-scalar-weight-v1"


DERIVATIVE_AOT_RADIAL_MANIFEST_SCHEMA = "vibeqc.derivative-aot.radials.v1"


def radial_inventory_from_payload(
    payload: typing.Any, *, backend: str
) -> tuple[CoulombKernel, ...]:
    """Validate and return the packaged radial inventory for one backend."""

    if backend not in ("cpu", "cuda"):
        raise ValueError("derivative AOT backend must be cpu or cuda")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != DERIVATIVE_AOT_RADIAL_MANIFEST_SCHEMA
    ):
        raise ValueError("invalid derivative AOT radial manifest schema")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("derivative AOT radial manifest requires entries")
    selected: list[CoulombKernel] = []
    seen: set[tuple[str, float]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"backend", "family", "omega"}:
            raise ValueError("invalid derivative AOT radial manifest entry")
        if entry["backend"] not in ("cpu", "cuda"):
            raise ValueError("invalid derivative AOT radial manifest backend")
        kernel = CoulombKernel(entry["family"], entry["omega"])
        identity = (CoulombKernelFamily(kernel.family).value, kernel.omega)
        if entry["backend"] == backend:
            if identity in seen:
                raise ValueError("duplicate derivative AOT radial manifest entry")
            seen.add(identity)
            selected.append(kernel)
    return tuple(selected)


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
        if type(self.group_index) is not int or not 0 <= self.group_index < len(groups):
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
            "generator_abi": AOT_GENERATOR_ABI,
            "derivative_order": self.derivative_order,
            "output_contract": AOT_WEIGHTED_OUTPUT_CONTRACT,
            "spin_contract": AOT_SPIN_CONTRACT,
            "contraction_contract": AOT_WEIGHTED_CONTRACTION_CONTRACT,
            "radial": self.radial.to_payload(),
            "angular": list(self.angular),
            "group_index": self.group_index,
            "component_indices": list(self.component_indices),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True, slots=True)
class DerivativeAotPackageKey:
    """Target/package identity layered over one scientific derivative identity."""

    backend: str
    target: str
    scientific_identity: str
    output_contract: str
    spin_contract: str
    contraction_contract: str
    generator_abi: int = AOT_GENERATOR_ABI

    def __post_init__(self) -> None:
        if self.backend not in ("cpu", "cuda"):
            raise ValueError("derivative AOT package backend must be cpu or cuda")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("derivative AOT package target must be nonempty")
        if (
            not isinstance(self.scientific_identity, str)
            or not self.scientific_identity
        ):
            raise ValueError(
                "derivative AOT package scientific identity must be nonempty"
            )
        if self.generator_abi != AOT_GENERATOR_ABI:
            raise ValueError("derivative AOT package generator ABI mismatch")
        for value, label in (
            (self.output_contract, "output"),
            (self.spin_contract, "spin"),
            (self.contraction_contract, "contraction"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"derivative AOT {label} contract must be nonempty")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "version": 1,
            "backend": self.backend,
            "target": self.target,
            "scientific_identity": self.scientific_identity,
            "generator_abi": self.generator_abi,
            "output_contract": self.output_contract,
            "spin_contract": self.spin_contract,
            "contraction_contract": self.contraction_contract,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True, slots=True)
class PackagedDerivativeAot:
    """Resolved shell-program entry; method coefficients are deliberately absent."""

    key: DerivativeAotKey
    entry_prefix: str
    target: str

    @property
    def component_indices(self) -> tuple[int, ...]:
        return self.key.component_indices

    @property
    def package_key(self) -> DerivativeAotPackageKey:
        return DerivativeAotPackageKey(
            backend=self.key.backend,
            target=self.target,
            scientific_identity=self.key.identity,
            output_contract=AOT_WEIGHTED_OUTPUT_CONTRACT,
            spin_contract=AOT_SPIN_CONTRACT,
            contraction_contract=AOT_WEIGHTED_CONTRACTION_CONTRACT,
        )


@dataclass(frozen=True, slots=True)
class DerivativeAotBundleKey:
    """Identity for a packaged derivative inventory shared by many consumers."""

    backend: str
    radial: CoulombKernel
    component_domain: tuple[str, ...]
    derivative_order: int = AOT_DERIVATIVE_ORDER

    def __post_init__(self) -> None:
        if self.backend not in ("cpu", "cuda"):
            raise ValueError("derivative AOT backend must be cpu or cuda")
        if not isinstance(self.radial, CoulombKernel):
            raise TypeError("derivative AOT requires an explicit CoulombKernel")
        if (
            not self.component_domain
            or tuple(sorted(set(self.component_domain))) != self.component_domain
        ):
            raise ValueError(
                "derivative AOT component domain must be sorted and unique"
            )
        if self.derivative_order != AOT_DERIVATIVE_ORDER:
            raise ValueError("derivative AOT currently packages first derivatives only")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "version": 1,
            "backend": self.backend,
            "generator_abi": AOT_GENERATOR_ABI,
            "derivative_order": self.derivative_order,
            "output_contract": AOT_COMPONENT_OUTPUT_CONTRACT,
            "spin_contract": AOT_SPIN_CONTRACT,
            "contraction_contract": AOT_COMPONENT_CONTRACTION_CONTRACT,
            "radial": self.radial.to_payload(),
            "component_domain": list(self.component_domain),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True, slots=True)
class PackagedDerivativeAotBundle:
    """Resolved fixed derivative inventory and its native entry symbols."""

    key: DerivativeAotBundleKey
    symbols: tuple[typing.Any, ...]
    target: str

    @property
    def package_key(self) -> DerivativeAotPackageKey:
        return DerivativeAotPackageKey(
            backend=self.key.backend,
            target=self.target,
            scientific_identity=self.key.identity,
            output_contract=AOT_COMPONENT_OUTPUT_CONTRACT,
            spin_contract=AOT_SPIN_CONTRACT,
            contraction_contract=AOT_COMPONENT_CONTRACTION_CONTRACT,
        )


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
    family = _family_tag(CoulombKernelFamily(key.radial.family))
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


def select_packaged_component_derivative_aot(
    library: typing.Any,
    *,
    backend: str,
    radial: CoulombKernel,
    target: str | None = None,
) -> PackagedDerivativeAotBundle | None:
    """Resolve the shared full-range component inventory through this registry.

    The current component bundle is the packaged CPU s/p/d first-derivative
    inventory. CUDA stationary artifacts package equivalent derivative objects
    behind a different library contract, so this selector deliberately returns
    no CUDA bundle until that native package boundary is exported.
    """

    if library is None or backend != "cpu":
        return None
    selected_target = "native-host" if target is None else target
    if not isinstance(selected_target, str) or not selected_target:
        raise ValueError("derivative AOT package target must be nonempty")
    full = CoulombKernel(CoulombKernelFamily.FULL_RANGE, 0.0)
    if radial != full:
        return None
    from .first_derivative_schedule import (
        COMPONENT_LABELS,
        CPU_AOT_SHARDS,
        cpu_aot_symbol,
    )

    symbols = []
    for shard in range(CPU_AOT_SHARDS):
        try:
            symbols.append(getattr(library, cpu_aot_symbol(shard)))
        except AttributeError:
            return None
    key = DerivativeAotBundleKey(
        backend=backend,
        radial=radial,
        component_domain=COMPONENT_LABELS,
    )
    return PackagedDerivativeAotBundle(
        key=key, symbols=tuple(symbols), target=selected_target
    )


def select_packaged_derivative_aot(
    library: typing.Any,
    *,
    backend: str,
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    component: int,
    target: str | None = None,
) -> PackagedDerivativeAot | None:
    """Resolve a packaged program without consulting a method/function name."""

    group_index, _ = component_group(angular, component)
    key = make_key(radial, angular, group_index, backend=backend)
    prefix = entry_prefix_for_key(key)
    if library is None:
        return None
    selected_target = "native-host" if backend == "cpu" and target is None else target
    if selected_target is None:
        return None
    if not isinstance(selected_target, str) or not selected_target:
        raise ValueError("derivative AOT package target must be nonempty")
    try:
        getattr(library, f"{prefix}_identity_v2")
    except AttributeError:
        return None
    return PackagedDerivativeAot(key=key, entry_prefix=prefix, target=selected_target)


def program_source(
    radial: CoulombKernel,
    angular: tuple[int, int, int, int],
    group_index: int,
    *,
    backend: str = "cpu",
) -> tuple[str, tuple[int, ...], str]:
    """Emit one generated weighted-ERI derivative program for a registry key."""

    key = make_key(radial, angular, group_index, backend=backend)
    integral = build_weighted_eri_ir(angular, operator=four_center_eri_operator(radial))
    kernel = build_weighted_eri_kernel(integral, key.component_indices)
    prefix = entry_prefix_for_key(key)
    return (
        emit_weighted_eri_runtime(kernel, backend=backend, entry_prefix=prefix),
        key.component_indices,
        prefix,
    )
