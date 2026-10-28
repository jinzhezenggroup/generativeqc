"""Present native D4 correction capacity without owning a sizing formula.

Global D4 plans require the complete named KS owner and its additive inventory
protocol. An older library remains usable for ordinary D4 execution, but cannot
claim global D4 capacity. Composite gCP is outside this correction-only scope.
"""

from __future__ import annotations

import ctypes as ct
import json
from typing import Any

from generativeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    checked_bytes,
)
from generativeqc_compiler.method import (
    D4Spec,
    DispersionCorrectionPrimitive,
    GeometricCounterpoisePrimitive,
    MethodIR,
)

from . import _native

_KEYS = (
    "plan_host_bytes",
    "execution_host_bytes",
    "device_bytes",
    "table_bytes",
    "workspace_bytes",
)


def _ks_d4_inventory(
    counts: tuple[int, ...], backend: str, library: Any
) -> dict[str, int]:
    """Read per-item retained owners; errors and malformed receipts are final."""
    query = library.generativeqc_ks_d4_resource_query_v1
    query.argtypes = [
        ct.c_int32,
        ct.POINTER(ct.c_uint32),
        ct.c_uint32,
        ct.POINTER(_native.D4RuntimeDiagnostic),
    ]
    query.restype = ct.c_int
    receipt = _native.D4RuntimeDiagnostic(
        ct.sizeof(_native.D4RuntimeDiagnostic), _native.ABI_VERSION
    )
    _native.check(
        library,
        query(
            int(backend == "cuda"),
            (ct.c_uint32 * len(counts))(*counts),
            len(counts),
            ct.byref(receipt),
        ),
    )
    inventory = {
        key: checked_bytes(int(getattr(receipt, key)), "KS D4 " + key) for key in _KEYS
    }
    inventory["retained_bytes"] = checked_bytes(
        sum(inventory[key] for key in _KEYS[:3]), "KS D4 retained bytes"
    )
    inventory["minimum_maximum_bytes"] = checked_bytes(
        int(receipt.maximum_bytes), "KS D4 correction cap"
    )
    if (
        receipt.backend != int(backend == "cuda")
        or receipt.system_count != len(counts)
        or receipt.total_atoms != sum(counts)
        or receipt.maximum_atoms != max(counts)
        or receipt.worker_blocks != len(counts)
        or receipt.workspace_slots != len(counts)
        or not 0 < inventory["retained_bytes"] <= inventory["minimum_maximum_bytes"]
        or (backend == "cpu" and inventory["device_bytes"] != 0)
    ):
        raise ValueError("native KS D4 inventory returned inconsistent capacity")
    return inventory


def _standalone_d4_inventory(
    counts: tuple[int, ...], backend: str, library: Any, maximum_bytes: int
) -> dict[str, int]:
    """Reserve the existing CPU compatibility force consumer by native sizing."""
    query = library.generativeqc_resource_d4_layout_v1
    query.argtypes = [
        ct.c_int,
        ct.c_uint64,
        ct.c_uint64,
        ct.c_uint64,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
        ct.c_size_t,
    ]
    query.restype = ct.c_int
    values = (ct.c_uint64 * 7)()
    if query(
        int(backend == "cuda"),
        len(counts),
        sum(counts),
        max(counts),
        maximum_bytes,
        values,
        7,
    ):
        raise ValueError("native standalone D4 inventory rejected shape or capacity")
    result = {
        key: checked_bytes(int(value), "D4 " + key)
        for key, value in zip(
            (*_KEYS, "retained_bytes", "workspace_slots"), values, strict=True
        )
    }
    if (
        result["retained_bytes"] != sum(result[key] for key in _KEYS[:3])
        or not 0 < result["retained_bytes"] <= maximum_bytes
        or not 0 < result["workspace_slots"] <= min(len(counts), 32)
        or (
            backend == "cpu"
            and (result["device_bytes"] != 0 or result["workspace_slots"] != 1)
        )
    ):
        raise ValueError(
            "native standalone D4 inventory returned inconsistent capacity"
        )
    return result


def d4_resource_request(
    systems: Any,
    *,
    method: MethodIR,
    backend: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
    name: str = "d4",
    first_phase: int = 0,
    last_phase: int = 0,
    library: Any = None,
    _native_ks: bool = False,
) -> ResourceRequest:
    """Describe admitted per-item D4 owners using the already selected library.

    No context/provider is created. Missing native capacity fails closed; there
    is no Python sizing fallback. Host estimates include a possible standalone
    correction-gradient owner for CPU electronic force domains not admitted by
    the complete native force endpoint. Global CUDA plans already require native
    stationary capacity and cannot create that extra correction owner.
    """
    if not isinstance(method, MethodIR):
        raise TypeError("D4 resource planning requires a MethodIR")
    specs = [
        node.specification
        for node in method.primitives
        if isinstance(node, DispersionCorrectionPrimitive)
    ]
    if (
        len(specs) != 1
        or not isinstance(specs[0], D4Spec)
        or any(
            isinstance(node, GeometricCounterpoisePrimitive)
            for node in method.primitives
        )
    ):
        raise NotImplementedError(
            "D4 resource planning requires an individual D4 correction without gCP"
        )
    if backend not in {"cpu", "cuda"}:
        raise ValueError("D4 backend must be cpu or cuda")
    checked_bytes(device_id, "device ordinal")
    checked_bytes(first_phase, "first phase")
    checked_bytes(last_phase, "last phase")
    if last_phase < first_phase:
        raise ValueError("D4 resource lifetime ends before its first use")
    if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
        raise ValueError("D4 maximum_bytes must be a positive uint64 integer")
    if (
        not _native_ks
        or not hasattr(library, "generativeqc_ks_d4_resource_query_v1")
        or (
            backend == "cpu"
            and not hasattr(library, "generativeqc_resource_d4_layout_v1")
        )
    ):
        raise NotImplementedError(
            "global D4 ResourcePlan requires native correction capacity"
        )
    numbers = tuple(tuple(system) for system in systems)
    if not numbers or any(not system for system in numbers):
        raise ValueError("D4 resource planning requires nonempty systems")
    if any(type(z) is not int for system in numbers for z in system):
        raise TypeError("D4 atomic numbers must be integers")
    counts = tuple(len(system) for system in numbers)
    spec = specs[0]
    controls = {
        "correction_identity": spec.identity,
        "device_id": device_id,
        "inventory_version": 1,
        "inventory_owner": "native-ks-d4-layout-v1",
        "maximum_bytes": maximum_bytes,
        "method_ir_identity": method.identity,
        "schedule": "retained-native-ks-d4",
    }
    identity = ResourceIdentity(
        method.identifier,
        "native-d4-bj-eeq-v1",
        backend,
        "fp64",
        json.dumps({"atomic_numbers": numbers, "counts": counts}),
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
    if any(z < 1 or z > 86 for system in numbers for z in system):
        unsupported = "production D4 supports atomic numbers 1 through 86"
    elif max(counts) > 256:
        unsupported = "production D4 supports at most 256 atoms per system"
    elif len(counts) > 2**32 - 1 or sum(counts) > 2**32 - 1:
        unsupported = "D4 compatibility topology requires uint32 aggregate atom counts"
    if unsupported:
        return ResourceRequest(
            name, identity, (), exclusions, unsupported_reason=unsupported
        )
    inventory = _ks_d4_inventory(counts, backend, library)
    required = inventory["minimum_maximum_bytes"]
    if required > maximum_bytes:
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            infeasible_reason=f"native KS D4 retained owners require {required} bytes, exceeding dispersion maximum_bytes={maximum_bytes}",
        )
    estimates = [
        ResourceEstimate(
            "D4 retained plan host",
            inventory["plan_host_bytes"],
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
        ResourceEstimate(
            "D4 retained execution host",
            inventory["execution_host_bytes"],
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
    ]
    if backend == "cpu":
        compatibility = _standalone_d4_inventory(
            counts, backend, library, maximum_bytes
        )
        estimates.append(
            ResourceEstimate(
                "D4 CPU compatibility force owner allowance",
                compatibility["retained_bytes"],
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    else:
        estimates.append(
            ResourceEstimate(
                "D4 retained CUDA owner",
                inventory["device_bytes"],
                f"device:{device_id}",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    candidate = ResourceCandidate(
        f"{backend}-d4-resident",
        "resident",
        tuple(estimates),
        decisions=(
            ("backend", backend),
            ("correction_identity", spec.identity),
            ("maximum_bytes", str(maximum_bytes)),
            *((key, str(inventory[key])) for key in _KEYS),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)
