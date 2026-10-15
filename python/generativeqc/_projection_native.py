"""Optional installed CPU basis transport; missing protocol is explicit compatibility."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Any

import numpy as np

from . import _native


class Descriptor(ctypes.Structure):
    """Borrowed row-major arrays and versioned native transport policy."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("struct_size", "abi_version", "input_kind", "occupation")
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "source_ao",
                "target_ao",
                "occupied",
                "maximum_ao",
                "maximum_workspace_bytes",
            )
        ]
        + [
            (name, ctypes.c_double)
            for name in (
                "relative_threshold",
                "validation_tolerance",
                "maximum_residual",
            )
        ]
        + [
            (name, ctypes.POINTER(ctypes.c_double))
            for name in (
                "source_overlap",
                "target_overlap",
                "cross_overlap",
                "source_state",
            )
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "source_overlap_count",
                "target_overlap_count",
                "cross_overlap_count",
                "source_state_count",
            )
        ]
    )


class Capacity(ctypes.Structure):
    """Metadata-only native dimensions and owner-visible numeric bound."""

    _fields_ = [(name, ctypes.c_uint32) for name in ("struct_size", "abi_version")] + [
        (name, ctypes.c_uint64)
        for name in ("coefficient_count", "density_count", "numeric_workspace_bytes")
    ]


class Output(ctypes.Structure):
    """Caller-owned detached result buffers and native invariant diagnostics."""

    _fields_ = (
        [(name, ctypes.c_uint32) for name in ("struct_size", "abi_version")]
        + [
            ("coefficients", ctypes.POINTER(ctypes.c_double)),
            ("coefficient_capacity", ctypes.c_uint64),
            ("density", ctypes.POINTER(ctypes.c_double)),
            ("density_capacity", ctypes.c_uint64),
        ]
        + [
            (name, ctypes.c_uint64)
            for name in ("source_metric_rank", "target_metric_rank", "occupied_rank")
        ]
        + [
            (name, ctypes.c_double)
            for name in (
                "minimum_projected_norm_squared",
                "projection_residual",
                "normal_equation_residual",
                "metric_orthogonality_error",
                "electron_trace",
                "metric_idempotency_error",
            )
        ]
    )


@lru_cache(maxsize=8)
def _load(path: str) -> Any:
    """Bind an immutable selected path once; never convert loader failures to fallback."""
    library = ctypes.CDLL(path)
    query = getattr(library, "generativeqc_basis_projection_query_v1", None)
    operation = getattr(library, "generativeqc_basis_project_cpu_v1", None)
    if query is None or operation is None:
        return None
    library.generativeqc_get_abi_version.argtypes = []
    library.generativeqc_get_abi_version.restype = ctypes.c_uint32
    if library.generativeqc_get_abi_version() != _native.ABI_VERSION:
        raise RuntimeError("GENERATIVEQC Python/native ABI version mismatch")
    query.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Capacity)]
    query.restype = ctypes.c_int
    operation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(Output),
    ]
    operation.restype = ctypes.c_int
    library.generativeqc_context_create.argtypes = [
        ctypes.POINTER(_native.ContextDescriptor),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.generativeqc_context_create.restype = ctypes.c_int
    library.generativeqc_context_destroy.argtypes = [ctypes.c_void_p]
    library.generativeqc_context_destroy.restype = None
    library.generativeqc_context_get_last_detail.argtypes = [ctypes.c_void_p]
    library.generativeqc_context_get_last_detail.restype = ctypes.c_char_p
    library.generativeqc_status_message.argtypes = [ctypes.c_int]
    library.generativeqc_status_message.restype = ctypes.c_char_p
    return library


def optional_library() -> Any:
    """Only the first existing selected library path is eligible for this call."""
    for candidate in _native._candidate_paths():
        if candidate.exists():
            return _load(str(candidate.resolve()))
    return None


def _supported_policy(policy: Any) -> bool:
    from .projection import ProjectionPolicy

    scalar_types = (int, float, np.float16, np.float32, np.float64)
    return (
        type(policy) is ProjectionPolicy
        and type(policy.maximum_ao) is int
        and 0 < policy.maximum_ao <= 2**64 - 1
        and all(
            type(getattr(policy, name)) in scalar_types
            for name in (
                "relative_threshold",
                "validation_tolerance",
                "maximum_residual",
            )
        )
    )


def numeric_capacity(
    library: Any, source_ao: int, target_ao: int, occupied: int, policy: Any
) -> int | None:
    """Use the prepared owner's additive query; old/custom metadata stays portable.

    A discovered query error is not evidence for Python fallback. This numeric
    subplan excludes the surrounding prepared fleet/provider/retained budgets.
    """
    if not _supported_policy(policy) or any(
        type(value) is not int or not 0 <= value <= 2**64 - 1
        for value in (source_ao, target_ao, occupied)
    ):
        return None
    query = getattr(library, "generativeqc_basis_projection_query_v1", None)
    if (
        query is None
        or getattr(library, "generativeqc_basis_project_cpu_v1", None) is None
    ):
        return None
    query.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Capacity)]
    query.restype = ctypes.c_int
    descriptor = Descriptor(
        struct_size=ctypes.sizeof(Descriptor),
        abi_version=_native.ABI_VERSION,
        source_ao=source_ao,
        target_ao=target_ao,
        occupied=occupied,
        occupation=1,
        maximum_ao=policy.maximum_ao,
        **{
            name: getattr(policy, name)
            for name in (
                "relative_threshold",
                "validation_tolerance",
                "maximum_residual",
            )
        },
    )
    result = Capacity(
        struct_size=ctypes.sizeof(Capacity), abi_version=_native.ABI_VERSION
    )
    _native.check(library, query(ctypes.byref(descriptor), ctypes.byref(result)))
    return result.numeric_workspace_bytes


def project(
    source_overlap: Any,
    target_overlap: Any,
    cross_overlap: Any,
    source_state: Any,
    *,
    occupied: int | None,
    occupation: int,
    policy: Any,
    maximum_bytes: int = 0,
) -> Any:
    """Marshal plain built-in policies; custom/unbounded policy domains remain compatible.

    Only shape/finite conversion and immutable presentation remain in Python.
    Metric spectra, purity, rank, projection, diagnostics and scientific refusal
    belong to the shared native transport, with no retry after native failure.
    """
    from .projection import (
        OccupiedProjection,
        ProjectionDiagnostics,
        ProjectionRejected,
        _array,
        _immutable,
    )

    if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= 2**64 - 1:
        raise ValueError("maximum_bytes must be a uint64 byte capacity")
    if not _supported_policy(policy) or (occupied is not None and occupied > 2**64 - 1):
        return None
    library = optional_library()
    if library is None:
        return None
    metrics = []
    for value, name in (
        (source_overlap, "source overlap"),
        (target_overlap, "target overlap"),
    ):
        raw = np.asarray(value)
        if (
            raw.ndim != 2
            or raw.shape[0] != raw.shape[1]
            or not 0 < len(raw) <= policy.maximum_ao
        ):
            raise ProjectionRejected(
                f"{name} must be a bounded nonempty square AO metric"
            )
        metrics.append(np.ascontiguousarray(_array(raw, raw.shape, name)))
    source, target = metrics
    cross = np.ascontiguousarray(
        _array(cross_overlap, (len(target), len(source)), "cross overlap")
    )
    density_input = occupied is not None
    if density_input:
        state = np.ascontiguousarray(
            _array(source_state, source.shape, "source density")
        )
    else:
        raw = np.asarray(source_state)
        if raw.ndim != 2 or raw.shape[0] != len(source):
            raise ProjectionRejected(
                "source occupied coefficients have invalid dimensions"
            )
        occupied = raw.shape[1]
        state = np.ascontiguousarray(_array(raw, raw.shape, "source coefficients"))
    if occupied > min(len(source), len(target)):
        raise ProjectionRejected(
            "source or target metric has insufficient occupied rank"
        )
    buffers = {
        "source_overlap": source,
        "target_overlap": target,
        "cross_overlap": cross,
        "source_state": state,
    }
    descriptor = Descriptor(
        struct_size=ctypes.sizeof(Descriptor),
        abi_version=_native.ABI_VERSION,
        input_kind=int(density_input),
        occupation=occupation,
        source_ao=len(source),
        target_ao=len(target),
        occupied=occupied,
        maximum_ao=policy.maximum_ao,
        maximum_workspace_bytes=maximum_bytes,
        **{
            name: getattr(policy, name)
            for name in (
                "relative_threshold",
                "validation_tolerance",
                "maximum_residual",
            )
        },
        **{
            name: array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            for name, array in buffers.items()
        },
        **{name + "_count": array.size for name, array in buffers.items()},
    )
    capacity = Capacity(
        struct_size=ctypes.sizeof(Capacity), abi_version=_native.ABI_VERSION
    )
    status = library.generativeqc_basis_projection_query_v1(
        ctypes.byref(descriptor), ctypes.byref(capacity)
    )
    if status == _native.STATUS_INVALID_ARGUMENT:
        raise ProjectionRejected(
            "invalid or overflowing native basis projection metadata"
        )
    _native.check(library, status)
    coefficients = np.empty(capacity.coefficient_count, dtype=np.float64).reshape(
        len(target), occupied
    )
    density = np.empty(capacity.density_count, dtype=np.float64).reshape(target.shape)
    output = Output(
        struct_size=ctypes.sizeof(Output),
        abi_version=_native.ABI_VERSION,
        coefficients=coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coefficient_capacity=coefficients.size,
        density=density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        density_capacity=density.size,
    )
    context_descriptor = _native.ContextDescriptor(
        struct_size=ctypes.sizeof(_native.ContextDescriptor),
        abi_version=_native.ABI_VERSION,
        backend=_native.BACKEND_CPU_REFERENCE,
    )
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(context_descriptor), ctypes.byref(context)
        ),
    )
    try:
        status = library.generativeqc_basis_project_cpu_v1(
            context, ctypes.byref(descriptor), ctypes.byref(output)
        )
        if status in (_native.STATUS_INVALID_ARGUMENT, _native.STATUS_OUT_OF_MEMORY):
            detail = library.generativeqc_context_get_last_detail(context)
            message = (detail or library.generativeqc_status_message(status)).decode(
                "utf-8"
            )
            if status == _native.STATUS_OUT_OF_MEMORY:
                raise MemoryError(message)
            raise ProjectionRejected(message)
        _native.check(library, status, context=context)
    finally:
        library.generativeqc_context_destroy(context)
    diagnostics = ProjectionDiagnostics(
        **{
            name: getattr(output, name)
            for name in ProjectionDiagnostics.__dataclass_fields__
        }
    )
    return OccupiedProjection(
        _immutable(coefficients), _immutable(density), diagnostics, "native_cpu"
    )
