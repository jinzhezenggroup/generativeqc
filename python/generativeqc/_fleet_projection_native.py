"""Borrow prepared HF owners for native fleet projection and seed publication.

The frontend transports explicit geometry/policy and presents detached metadata.
It never exports a source density or prepares overlap/projection work on this path.
Missing protocols/custom policies/unsupported metadata retain explicit compatibility;
an admitted native failure never retries it.
"""

from __future__ import annotations

import ctypes as ct
import time
import typing
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from . import _native, _projection_native
from ._system_projection_native import _check
from .checkpoint import _controls, _pointer, _resource_check


class Controls(ct.Structure):
    """Current-ABI transfer policy; spin/domain come from the prepared owners."""

    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in ("struct_size", "abi_version", "strict", "reserved")
        ]
        + [(name, ct.c_uint64) for name in ("maximum_ao", "maximum_host_bytes")]
        + [
            (name, ct.c_double)
            for name in (
                "relative_threshold",
                "validation_tolerance",
                "maximum_residual",
            )
        ]
    )


class Capacity(ct.Structure):
    """Conservative native peak, retained candidates and physical-validation shape."""

    _fields_ = [(name, ct.c_uint32) for name in ("struct_size", "abi_version")] + [
        (name, ct.c_uint64)
        for name in ("numeric_peak_bytes", "retained_bytes", "largest_density_bytes")
    ]


class Spin(ct.Structure):
    """Existing native projection invariants, without density/coefficient buffers."""

    _fields_ = [
        (name, ct.c_uint64)
        for name in ("source_metric_rank", "target_metric_rank", "occupied_rank")
    ] + [
        (name, ct.c_double)
        for name in (
            "minimum_projected_norm_squared",
            "projection_residual",
            "normal_equation_residual",
            "metric_orthogonality_error",
            "electron_trace",
            "metric_idempotency_error",
        )
    ]


class Item(ct.Structure):
    """One detached input-ordered publication receipt; unused fields stay zero."""

    _fields_ = [
        (name, ct.c_uint32)
        for name in ("struct_size", "abi_version", "accepted", "spin_count")
    ] + [
        ("source_iterations", ct.c_int32),
        ("reserved", ct.c_uint32),
        ("numeric_workspace_bytes", ct.c_uint64),
        ("source_energy", ct.c_double),
        ("reason", ct.c_char * 512),
        ("spins", Spin * 2),
    ]


def bind(library: typing.Any) -> bool:
    """Bind only the selected owner library, without discovering another backend."""
    query = getattr(library, "generativeqc_hf_fleet_projection_query_v1", None)
    operation = getattr(library, "generativeqc_hf_fleet_initialize_cpu_v1", None)
    if query is None or operation is None:
        return False
    common = [
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(Controls),
        ct.POINTER(_native.BatchInputDescriptor),
        ct.c_uint32,
    ]
    query.argtypes = [*common, ct.POINTER(Capacity)]
    query.restype = ct.c_int
    operation.argtypes = [*common, ct.POINTER(Item), ct.c_uint32]
    operation.restype = ct.c_int
    return True


def _publish(
    target: typing.Any,
    source: typing.Any,
    records: typing.Any,
    policy: typing.Any,
    started: float,
) -> dict:
    """Present native decisions and install only the frontend's provenance tags."""
    reports = []
    metadata = list(target._warm_state.metadata)
    accepted = set()
    expected_spins = 2 if target._calculator._method == _native.METHOD_UHF else 1
    for index, row in enumerate(records):
        if (
            row.struct_size != ct.sizeof(Item)
            or row.abi_version != _native.ABI_VERSION
            or row.accepted > 1
            or row.reserved
            or row.spin_count != (expected_spins if row.accepted else 0)
            or row.source_iterations < 0
            or bool(row.reason) == bool(row.accepted)
        ):
            raise RuntimeError(
                "native fleet projection returned an inconsistent receipt"
            )
        report = {
            "index": index,
            "accepted": bool(row.accepted),
            "reason": row.reason.decode() if row.reason else None,
            "source_model_identity": source._basis_metadata[index]["model_identity"],
            "target_model_identity": target._basis_metadata[index]["model_identity"],
            "spins": [],
        }
        if row.accepted:
            accepted.add(index)
            report.update(
                spins=[
                    {name: getattr(row.spins[spin], name) for name, _ in Spin._fields_}
                    for spin in range(row.spin_count)
                ],
                projection_owners=["native_cpu_system"] * row.spin_count,
                numeric_workspace_bound_bytes=row.numeric_workspace_bytes,
                source_energy=row.source_energy,
                source_iterations=row.source_iterations,
            )
            metadata[index] = {
                "controls": _controls(target._calculator),
                "backend": "cpu",
            }
        reports.append(report)
    report = {
        "schema": "generativeqc.basis_projection",
        "version": 1,
        "policy": asdict(policy),
        "items": reports,
        "seconds": time.perf_counter() - started,
        "placement": "native prepared CPU fleet projection and seed installation",
        "execution_owner": "native_cpu_fleet",
        "target_verification": "pending_execute",
    }
    detached = deepcopy(report)
    target._warm_state.metadata = metadata
    target._warm_state.projection_indices = accepted
    target._warm_state.projection_diagnostics = report
    return detached


def initialize(
    target: typing.Any,
    source: typing.Any,
    *,
    policy: typing.Any,
    strict: bool,
    maximum_host_bytes: int,
    started: float,
) -> dict | None:
    """Run the prepared transaction, or decline before any scientific work."""
    library = target._library
    handle = getattr(library, "_handle", None)
    if (
        not handle
        or getattr(source._library, "_handle", None) != handle
        or target._calculator._device_name != "cpu"
        or source._calculator._device_name != "cpu"
        or not _projection_native._supported_policy(policy)
        or not 0 < maximum_host_bytes < 2**64
        or not bind(library)
    ):
        return None
    controls = Controls(
        ct.sizeof(Controls),
        _native.ABI_VERSION,
        int(strict),
        0,
        policy.maximum_ao,
        maximum_host_bytes,
        policy.relative_threshold,
        policy.validation_tolerance,
        policy.maximum_residual,
    )
    # Coordinates are ordinary request metadata, never exported scientific state.
    storage = [
        np.asarray([atom.position for atom in atoms], dtype=np.float64).reshape(-1)
        for atoms in target._systems
    ]
    inputs = (_native.BatchInputDescriptor * target.system_count)(
        *(
            _native.BatchInputDescriptor(
                ct.sizeof(_native.BatchInputDescriptor),
                _native.ABI_VERSION,
                _pointer(values),
                values.size,
            )
            for values in storage
        )
    )
    arguments = (target._batch, source._batch, ct.byref(controls), inputs, len(inputs))
    capacity = Capacity(ct.sizeof(Capacity), _native.ABI_VERSION)
    status = library.generativeqc_hf_fleet_projection_query_v1(
        *arguments, ct.byref(capacity)
    )
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _check(library, status, target._context)
    if (
        capacity.struct_size != ct.sizeof(Capacity)
        or capacity.abi_version != _native.ABI_VERSION
    ):
        raise RuntimeError("native fleet projection returned an inconsistent capacity")
    if capacity.numeric_peak_bytes:
        _resource_check(
            target, capacity.numeric_peak_bytes, capacity.largest_density_bytes
        )
    records = (Item * target.system_count)(
        *(
            Item(ct.sizeof(Item), _native.ABI_VERSION)
            for _ in range(target.system_count)
        )
    )
    status = library.generativeqc_hf_fleet_initialize_cpu_v1(
        *arguments, records, len(records)
    )
    _check(library, status, target._context)
    try:
        return _publish(target, source, records, policy, started)
    except BaseException as error:
        # The native API has committed, but frontend publication did not finish.
        # The required fresh target had no prior seeds, so clearing restores it
        # without touching the source or rerunning projection/science.
        try:
            _native.check(
                library, library.generativeqc_batch_clear_warm_starts(target._batch)
            )
            target._warm_state.clear()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
