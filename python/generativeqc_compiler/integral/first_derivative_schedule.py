"""Bounded s/p/d primitive dispatch using exact center and axis permutations.

Only compiled code is shared. Every ordered consumer weight is still evaluated
once, and derivative center/axis slots are restored before physical scattering.
The existing first-derivative graphs remain the sole mathematical lowering.
"""

import typing
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path
from time import perf_counter

from generativeqc_compiler.common.cuda_target import CudaTargetInfo
from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.common.semantic_source_cache import cached_sources

from .eri_weights import eri_weight_orbit
from .first_derivative_native import (
    emit_first_derivative_cpu,
    emit_first_derivative_cuda,
)

AXES = tuple(permutations(range(3)))
ERI_CENTERS = eri_weight_orbit((0, 1, 2, 3))
REQUESTS_PER_UNIT = 8
CUDA_REQUESTS_PER_UNIT = 16
MAX_UNIT_BYTES = 4 << 20
MAX_PROGRAM_BYTES = 64 << 20
DerivativeRequest = tuple[str, tuple[str, ...]]
COMPONENT_LABELS = ("", "x", "xx", "xy", "xz", "y", "yy", "yz", "z", "zz")
COMPONENT_OPERATORS = ("overlap", "kinetic", "nuclear_attraction", "four_center_eri")
DISPATCH_ROWS = 3 * len(COMPONENT_LABELS) ** 2 + len(COMPONENT_LABELS) ** 4
DISPATCH_WIDTH = 9
CPU_AOT_COMPONENTS = COMPONENT_LABELS
CPU_AOT_SHARDS = 46
CPU_AOT_SYMBOL_PREFIX = "generativeqc_first_derivative_cpu_shard_"


@dataclass(frozen=True)
class DerivativeBinding:
    request: DerivativeRequest
    # Native center/axis slot -> original center/axis slot.
    centers: tuple[int, ...]
    axes: tuple[int, ...]


@lru_cache(maxsize=16384)
def derivative_binding(operator: str, components: tuple[str, ...]) -> DerivativeBinding:
    """Canonicalize a scalar Cartesian integral, preserving all derivative slots."""
    if operator == "nuclear":
        if components:
            raise ValueError("nuclear derivative has no AO components")
        return DerivativeBinding((operator, ()), (0, 1), (0, 1, 2))
    rank = 4 if operator == "four_center_eri" else 2
    if operator not in ("four_center_eri", "overlap", "kinetic", "nuclear_attraction"):
        raise ValueError("unsupported first derivative operator")
    if len(components) != rank or any(
        not isinstance(c, str)
        or len(c) > 2
        or c != "".join(sorted(c))
        or any(a not in "xyz" for a in c)
        for c in components
    ):
        raise ValueError("first derivative schedule requires s/p/d Cartesian labels")
    centers = ERI_CENTERS if rank == 4 else ((0, 1), (1, 0))
    transformed = {
        c: tuple(
            "".join("xyz"[j] * c.count("xyz"[i]) for j, i in enumerate(axes))
            for axes in AXES
        )
        for c in components
    }
    labels, order, axes = min(
        (tuple(transformed[components[i]][a] for i in order), order, axes)
        for order in centers
        for a, axes in enumerate(AXES)
    )
    if operator == "nuclear_attraction":
        order += (2,)
    return DerivativeBinding((operator, labels), order, axes)


@lru_cache(maxsize=4)
def derivative_requests(
    domain: tuple[str, ...],
) -> tuple[DerivativeRequest, ...]:
    """Finite metadata-only plan; full s/p/d needs 313 ERI representatives."""
    if not domain or len(domain) > 10 or tuple(sorted(set(domain))) != domain:
        raise ValueError(
            "component domain must contain one to ten sorted unique labels"
        )
    requests = {("nuclear", ())}
    for operator, rank in (
        ("overlap", 2),
        ("kinetic", 2),
        ("nuclear_attraction", 2),
        ("four_center_eri", 4),
    ):
        for components in product(domain, repeat=rank):
            requests.add(derivative_binding(operator, components).request)
    return tuple(sorted(requests))


@lru_cache(maxsize=4)
def derivative_sources(
    domain: tuple[str, ...],
) -> tuple[tuple[tuple[DerivativeRequest, ...], str], ...]:
    """Bound each translation unit and the whole finite generated source set.

    The small in-process source cache avoids regenerating graphs on warm calls.
    Native artifact consumers must still revalidate source/header/binary and
    toolchain identities through the common compile_runtime cache each time.
    """
    requests = derivative_requests(domain)
    units, total = [], 0
    for begin in range(0, len(requests), REQUESTS_PER_UNIT):
        selected = requests[begin : begin + REQUESTS_PER_UNIT]
        source = emit_first_derivative_cpu(selected)
        size = len(source.encode("utf-8"))
        total += size
        if size > MAX_UNIT_BYTES or total > MAX_PROGRAM_BYTES:
            raise ValueError("first derivative generated source budget exceeded")
        units.append((selected, source))
    return tuple(units)


def cpu_aot_symbol(shard: int) -> str:
    if type(shard) is not int or not 0 <= shard < CPU_AOT_SHARDS:
        raise ValueError("CPU derivative AOT shard is out of range")
    return f"{CPU_AOT_SYMBOL_PREFIX}{shard}"


@lru_cache(maxsize=1)
def derivative_cpu_aot_sources() -> tuple[
    tuple[tuple[DerivativeRequest, ...], str], ...
]:
    """Emit the full s/p/d CPU inventory with link-safe exported dispatchers."""

    requests = derivative_requests(CPU_AOT_COMPONENTS)
    units, total = [], 0
    for shard, begin in enumerate(range(0, len(requests), REQUESTS_PER_UNIT)):
        selected = requests[begin : begin + REQUESTS_PER_UNIT]
        source = emit_first_derivative_cpu(selected, symbol=cpu_aot_symbol(shard))
        size = len(source.encode("utf-8"))
        total += size
        if size > MAX_UNIT_BYTES or total > MAX_PROGRAM_BYTES:
            raise ValueError("first derivative generated source budget exceeded")
        units.append((selected, source))
    if len(units) != CPU_AOT_SHARDS:
        raise RuntimeError("stationary CPU derivative AOT shard contract drift")
    return tuple(units)


@lru_cache(maxsize=4)
def derivative_cuda_sources(
    domain: tuple[str, ...],
) -> tuple[tuple[tuple[DerivativeRequest, ...], str], ...]:
    """Bound the s/p/d CUDA derivative inventory across relocatable objects."""

    requests = derivative_requests(domain)
    units, total = [], 0
    for unit, begin in enumerate(range(0, len(requests), CUDA_REQUESTS_PER_UNIT)):
        selected = requests[begin : begin + CUDA_REQUESTS_PER_UNIT]
        source = emit_first_derivative_cuda(
            selected, symbol=f"first_derivative_shard_{unit}"
        )
        size = len(source.encode("utf-8"))
        total += size
        if size > MAX_UNIT_BYTES or total > MAX_PROGRAM_BYTES:
            raise ValueError("first derivative generated CUDA source budget exceeded")
        units.append((selected, source))
    return tuple(units)


@lru_cache(maxsize=4)
def derivative_dispatch_table(domain: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    """Immutable runtime dispatch: library, kind, four centers and three axes.

    Slots use base-ten Cartesian label codes, independently of public AO order.
    Unrequested slots remain invalid. This is metadata, never consumer weights.
    """
    requests = {request: i for i, request in enumerate(derivative_requests(domain))}
    table = [(-1,) * DISPATCH_WIDTH] * DISPATCH_ROWS
    for op, operator in enumerate(COMPONENT_OPERATORS):
        rank = 4 if op == 3 else 2
        for labels in product(domain, repeat=rank):
            slot = 0
            for label in labels:
                slot = len(COMPONENT_LABELS) * slot + COMPONENT_LABELS.index(label)
            slot += op * len(COMPONENT_LABELS) ** 2
            binding = derivative_binding(operator, labels)
            kernel = requests[binding.request]
            table[slot] = (
                kernel // REQUESTS_PER_UNIT,
                kernel % REQUESTS_PER_UNIT,
                *binding.centers,
                *([-1] * (4 - len(binding.centers))),
                *binding.axes,
            )
    return tuple(table)


def cached_derivative_cuda_source(
    requests: tuple[DerivativeRequest, ...],
    *,
    cache: Path,
    target: CudaTargetInfo,
    component_domain: tuple[str, ...] | None = None,
    enabled: bool = True,
) -> tuple[str | tuple[str, ...], dict[str, typing.Any]]:
    """Reuse exact ordered primitive lowering without changing fallback demand.

    Demand projection stays with the qualified runtime capability owner. This
    provider only caches the existing full bounded or nuclear-only recipe and
    returns sources for the unchanged source/header/toolchain binary cache.
    """
    started = perf_counter()
    if type(enabled) is not bool:
        raise TypeError("stationary source cache selection must be boolean")
    requests = tuple(requests)
    if not requests or len(requests) != len(set(requests)):
        raise ValueError("stationary CUDA requires unique nonempty primitive requests")
    if component_domain is not None and requests != derivative_requests(
        component_domain
    ):
        raise ValueError(
            "stationary CUDA component domain differs from primitive demand"
        )
    recipe = {
        "product": "first-derivative-cuda.v1",
        "dependencies": source_hashes(
            "common",
            "integral",
            assets=(
                "src/integrals/eri_geometry.hpp",
                "src/integrals/range_moments.hpp",
            ),
        ),
        "target": asdict(target),
        "precision": "FP64/strict/no-fmad",
        "requests": requests,
        "component_domain": component_domain,
        "shard_width": CUDA_REQUESTS_PER_UNIT if component_domain is not None else None,
    }
    recipe_seconds = perf_counter() - started

    def produce() -> tuple[str, ...]:
        if component_domain is not None:
            return tuple(
                source for _, source in derivative_cuda_sources(component_domain)
            )
        return (emit_first_derivative_cuda(requests),)

    sources, work = cached_sources(
        cache if enabled else None,
        recipe,
        produce,
        max_unit_bytes=MAX_UNIT_BYTES
        if component_domain is not None
        else MAX_PROGRAM_BYTES,
        max_total_bytes=MAX_PROGRAM_BYTES,
    )
    work.update(requests=len(requests), recipe_seconds=recipe_seconds)
    return (sources if component_domain is not None else sources[0]), work
