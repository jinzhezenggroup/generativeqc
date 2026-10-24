"""ResourcePlan adapter for the production two-body D3(BJ) owner.

The discovered native query shares src/dft/dispersion/d3_runtime.cpp with the
production owner. Old libraries and library-free dry runs retain an explicit
portable layout. Neither path creates a context, correction batch or GPU state.
"""

from __future__ import annotations

import ctypes
import json
import typing

from generativeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)
from generativeqc_compiler.method import D3Spec, DispersionCorrectionPrimitive, MethodIR

_D3_MAXIMUM_ATOMS_PER_SYSTEM = 4096
_D3_ELEMENT_COUNT = 86
_D3_PAIR_COUNT = 86 * 87 // 2
_D3_REFERENCE_CN_COUNT = 237
_D3_REFERENCE_C6_COUNT = 28455


class _ElementData(ctypes.Structure):
    _fields_ = [
        ("reference_offset", ctypes.c_uint32),
        ("reference_count", ctypes.c_uint8),
        ("covalent_radius", ctypes.c_double),
        ("r4r2", ctypes.c_double),
    ]


class _PairData(ctypes.Structure):
    _fields_ = [
        ("c6_offset", ctypes.c_uint32),
        ("first_reference_count", ctypes.c_uint8),
        ("second_reference_count", ctypes.c_uint8),
        ("vdw_radius", ctypes.c_double),
    ]


_D3_RESOURCE_KEYS = (
    "plan_host_bytes",
    "execution_host_bytes",
    "device_bytes",
    "table_bytes",
    "workspace_bytes",
    "retained_bytes",
)


def _portable_d3_inventory(counts: tuple[int, ...], backend: str) -> dict[str, int]:
    """Legacy metadata-only layout; never retried after native rejection."""
    systems, atoms, maximum_atoms = len(counts), sum(counts), max(counts)
    table_bytes = (
        _D3_ELEMENT_COUNT * ctypes.sizeof(_ElementData)
        + _D3_PAIR_COUNT * ctypes.sizeof(_PairData)
        + byte_product(_D3_REFERENCE_CN_COUNT, 8)
        + byte_product(_D3_REFERENCE_C6_COUNT, 8)
    )
    plan_host_bytes = (
        byte_product(systems + 1, 4)
        + byte_product(atoms, 4)
        + byte_product(3, atoms, 8)
    )
    execution_host_bytes = (
        byte_product(3, atoms, 8)
        + byte_product(2, systems)
        + byte_product(systems, 4)
        + byte_product(systems, 8)
        + byte_product(3, atoms, 8)
    )
    workspace_bytes = byte_product(16, maximum_atoms if backend == "cpu" else atoms, 8)
    device_bytes = 0
    if backend == "cpu":
        execution_host_bytes += workspace_bytes + byte_product(3, maximum_atoms, 8)
    else:
        device_bytes = (
            byte_product(systems + 1, 4)
            + byte_product(atoms, 4)
            + byte_product(3, atoms, 8)
            + byte_product(2, systems)
            + byte_product(systems, 4)
            + byte_product(systems, 8)
            + byte_product(3, atoms, 8)
            + workspace_bytes
            + table_bytes
        )
    return {
        key: checked_bytes(value, "D3 " + key)
        for key, value in zip(
            _D3_RESOURCE_KEYS,
            (
                plan_host_bytes,
                execution_host_bytes,
                device_bytes,
                table_bytes,
                workspace_bytes,
                plan_host_bytes + execution_host_bytes + device_bytes,
            ),
            strict=True,
        )
    }


def _d3_inventory(
    counts: tuple[int, ...], backend: str, library: typing.Any
) -> dict[str, int]:
    """Query already-loaded native capacity without discovering a provider."""
    query = getattr(library, "generativeqc_resource_d3_layout_v1", None)
    if query is None:
        return _portable_d3_inventory(counts, backend)
    query.argtypes = (
        [ctypes.c_int]
        + [ctypes.c_uint64] * 3
        + [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
    )
    query.restype = ctypes.c_int
    values = (ctypes.c_uint64 * len(_D3_RESOURCE_KEYS))()
    if query(
        int(backend == "cuda"),
        len(counts),
        sum(counts),
        max(counts),
        values,
        len(values),
    ):
        raise ValueError("native D3 inventory rejected shape or capacity")
    inventory = {
        key: checked_bytes(int(value), "D3 " + key)
        for key, value in zip(_D3_RESOURCE_KEYS, values, strict=True)
    }
    if inventory["retained_bytes"] != sum(
        inventory[key] for key in _D3_RESOURCE_KEYS[:3]
    ):
        raise ValueError("native D3 inventory returned inconsistent retained capacity")
    return inventory


def _d3_spec(graph: MethodIR) -> D3Spec:
    corrections = tuple(
        node
        for node in graph.primitives
        if isinstance(node, DispersionCorrectionPrimitive)
    )
    if len(corrections) != 1 or not isinstance(corrections[0].specification, D3Spec):
        raise NotImplementedError(
            "global ResourcePlan currently supports exactly one production D3(BJ) correction"
        )
    return corrections[0].specification


def _normalized_atomic_numbers(systems: typing.Any) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(system) for system in systems)
    if not normalized or any(not system for system in normalized):
        raise ValueError("D3 resource planning requires nonempty systems")
    for system in normalized:
        for atomic_number in system:
            if type(atomic_number) is not int:
                raise TypeError("D3 atomic numbers must be integers")
    return normalized


def d3_resource_request(
    systems: typing.Any,
    *,
    method: MethodIR,
    backend: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
    name: str = "d3",
    first_phase: int = 0,
    last_phase: int = 0,
    library: typing.Any = None,
) -> ResourceRequest:
    """Describe the retained production D3(BJ) owner without allocating it.

    An already-loaded ``library`` supplies native sizing when its additive
    query exists. No library is loaded here; absent symbols retain portable
    metadata sizing, while discovered native failures are authoritative.
    """
    if not isinstance(method, MethodIR):
        raise TypeError("D3 resource planning requires a MethodIR")
    spec = _d3_spec(method)
    if backend not in {"cpu", "cuda"}:
        raise ValueError("D3 backend must be 'cpu' or 'cuda'")
    checked_bytes(device_id, "device ordinal")
    checked_bytes(first_phase, "first phase")
    checked_bytes(last_phase, "last phase")
    if last_phase < first_phase:
        raise ValueError("D3 resource lifetime ends before its first use")
    if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
        raise ValueError("D3 maximum_bytes must be a positive uint64 integer")

    atomic_numbers = _normalized_atomic_numbers(systems)
    counts = tuple(len(system) for system in atomic_numbers)
    total_atoms = checked_bytes(sum(counts), "D3 total atom count")
    maximum_atoms = max(counts)

    controls = {
        "correction_identity": spec.identity,
        "device_id": device_id,
        "inventory_version": 1,
        "inventory_owner": (
            "native-d3-layout-v1"
            if getattr(library, "generativeqc_resource_d3_layout_v1", None) is not None
            else "portable-d3-layout-v1"
        ),
        "maximum_bytes": maximum_bytes,
        "method_ir_identity": method.identity,
        "schedule": "retained-native-d3-bj",
    }
    identity = ResourceIdentity(
        method.identifier,
        "native-d3-bj-v1",
        backend,
        "fp64",
        json.dumps(
            {
                "atomic_numbers": atomic_numbers,
                "counts": counts,
            }
        ),
        ("energy", "forces"),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and Python objects",
        "allocator bookkeeping, arenas and page rounding",
    )
    if backend == "cuda":
        exclusions += ("CUDA driver/context/modules and pool/page retention",)

    unsupported = None
    if any(z < 1 or z > 86 for system in atomic_numbers for z in system):
        unsupported = "production D3(BJ) supports atomic numbers 1 through 86"
    elif maximum_atoms > _D3_MAXIMUM_ATOMS_PER_SYSTEM:
        unsupported = (
            "production D3(BJ) supports at most "
            f"{_D3_MAXIMUM_ATOMS_PER_SYSTEM} atoms per system"
        )
    elif len(counts) > 2**32 - 1 or total_atoms > 2**32 - 1:
        unsupported = "production D3(BJ) ragged offsets require uint32 atom counts"
    if unsupported is not None:
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            unsupported_reason=unsupported,
        )

    inventory = _d3_inventory(counts, backend, library)
    plan_host_bytes = inventory["plan_host_bytes"]
    execution_host_bytes = inventory["execution_host_bytes"]
    device_bytes = inventory["device_bytes"]
    retained_total = inventory["retained_bytes"]
    if retained_total > maximum_bytes:
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            infeasible_reason=(
                f"production D3(BJ) retained owner requires {retained_total} bytes, "
                f"exceeding dispersion maximum_bytes={maximum_bytes}"
            ),
        )

    estimates = [
        ResourceEstimate(
            "D3 retained plan host",
            plan_host_bytes,
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
        ResourceEstimate(
            "D3 retained execution host",
            execution_host_bytes,
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
    ]
    if backend == "cuda":
        estimates.append(
            ResourceEstimate(
                "D3 retained CUDA owner",
                device_bytes,
                f"device:{device_id}",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    candidate = ResourceCandidate(
        f"{backend}-d3-resident",
        "resident",
        tuple(estimates),
        decisions=(
            ("backend", backend),
            ("correction_identity", spec.identity),
            ("maximum_bytes", str(maximum_bytes)),
            ("plan_host_bytes", str(plan_host_bytes)),
            ("execution_host_bytes", str(execution_host_bytes)),
            ("device_bytes", str(device_bytes)),
            ("table_bytes", str(inventory["table_bytes"])),
            ("workspace_bytes", str(inventory["workspace_bytes"])),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)
