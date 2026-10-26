"""Typed transport for native CPU overlap preparation and all-spin HF projection.

Only missing protocol or explicitly unsupported metadata uses compatibility.
An admitted execution failure never retries Python scientific preparation.
"""

from __future__ import annotations

import ctypes as ct
import typing

import numpy as np

from . import _native, _projection_native
from .checkpoint import _pointer, _resource_check
from .projection import (
    OccupiedProjection,
    ProjectionDiagnostics,
    ProjectionRejected,
    _immutable,
)


class Controls(ct.Structure):
    """Versioned policy without scientific arrays or opaque Python callbacks."""

    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in ("struct_size", "abi_version", "unrestricted", "reserved")
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
    """Detached native dimensions and the composed numeric-matrix bound."""

    _fields_ = [(name, ct.c_uint32) for name in ("struct_size", "abi_version")] + [
        (name, ct.c_uint64)
        for name in (
            "source_ao",
            "target_ao",
            "spin_count",
            "alpha_occupied",
            "beta_occupied",
            "source_density_count",
            "target_density_count",
            "numeric_workspace_bytes",
        )
    ]


def bind(library: typing.Any) -> bool:
    """Bind only the caller-selected library; do not discover another backend."""
    query = getattr(library, "generativeqc_hf_system_projection_query_v1", None)
    operation = getattr(library, "generativeqc_hf_system_project_cpu_v1", None)
    if query is None or operation is None:
        return False
    query.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(Controls),
        ct.POINTER(Capacity),
    ]
    query.restype = ct.c_int
    operation.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(Controls),
        ct.POINTER(ct.c_double),
        ct.c_uint64,
        ct.POINTER(_projection_native.Output),
        ct.c_uint64,
    ]
    operation.restype = ct.c_int
    return True


def _check(library: typing.Any, status: int, context: typing.Any = None) -> None:
    if status in (_native.STATUS_INVALID_ARGUMENT, _native.STATUS_OUT_OF_MEMORY):
        detail = (
            library.generativeqc_context_get_last_detail(context) if context else None
        )
        message = (detail or library.generativeqc_status_message(status)).decode()
        if status == _native.STATUS_OUT_OF_MEMORY:
            raise MemoryError(message)
        raise ProjectionRejected(message)
    _native.check(library, status, context=context)


def project(
    target: typing.Any,
    source: typing.Any,
    index: int,
    density: np.ndarray,
    *,
    policy: typing.Any,
    maximum_host_bytes: int,
    retained_bytes: int,
    coordinate_count: int,
) -> tuple[list[OccupiedProjection], int] | None:
    """Prepare native metrics and project every spin, or decline before science.

    Batch validity, current geometry and source-success checks precede this call.
    Source and target must use the same selected CPU library; native systems are
    created in that context and destroyed even if query/evaluation fails.
    """
    library = target._library
    # Calculators may hold separate ctypes wrappers for the same live dlopen
    # mapping. Both wrappers remain alive during this call; no handle is cached.
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
        target._calculator._method == _native.METHOD_UHF,
        0,
        policy.maximum_ao,
        maximum_host_bytes,
        policy.relative_threshold,
        policy.validation_tolerance,
        policy.maximum_residual,
    )
    context = ct.c_void_p()
    descriptor = _native.ContextDescriptor(
        ct.sizeof(_native.ContextDescriptor),
        _native.ABI_VERSION,
        0,
        _native.BACKEND_CPU_REFERENCE,
    )
    _native.check(
        library,
        library.generativeqc_context_create(ct.byref(descriptor), ct.byref(context)),
    )
    left = right = None
    try:
        atoms = target._systems[index]
        charge, multiplicity = target.charges[index], target.multiplicities[index]
        left = target._calculator._create_native_system(
            context, atoms, charge, multiplicity
        )
        right = source._calculator._create_native_system(
            context, atoms, charge, multiplicity
        )
        required = Capacity(
            struct_size=ct.sizeof(Capacity), abi_version=_native.ABI_VERSION
        )
        status = library.generativeqc_hf_system_projection_query_v1(
            left, right, ct.byref(controls), ct.byref(required)
        )
        if status == _native.STATUS_NOT_IMPLEMENTED:
            return None
        _check(library, status, context)
        workspace = required.numeric_workspace_bytes + 4 * coordinate_count * 8
        if retained_bytes + workspace > maximum_host_bytes:
            raise MemoryError("native system projection exceeds maximum_host_bytes")
        _resource_check(
            target, retained_bytes + workspace, 8 * required.target_density_count
        )
        controls.maximum_host_bytes = required.numeric_workspace_bytes
        if density.shape != (
            required.spin_count,
            required.source_ao,
            required.source_ao,
        ):
            raise ProjectionRejected(
                "native system projection source-density shape mismatch"
            )
        outputs = (_projection_native.Output * required.spin_count)()
        storage = []
        for spin, occupied in enumerate(
            (required.alpha_occupied, required.beta_occupied)[: required.spin_count]
        ):
            coefficients = np.empty((required.target_ao, occupied), dtype=np.float64)
            projected = np.empty(
                (required.target_ao, required.target_ao), dtype=np.float64
            )
            outputs[spin] = _projection_native.Output(
                struct_size=ct.sizeof(_projection_native.Output),
                abi_version=_native.ABI_VERSION,
                coefficients=_pointer(coefficients),
                coefficient_capacity=coefficients.size,
                density=_pointer(projected),
                density_capacity=projected.size,
            )
            storage.append((coefficients, projected))
        status = library.generativeqc_hf_system_project_cpu_v1(
            context,
            left,
            right,
            ct.byref(controls),
            _pointer(density),
            density.size,
            outputs,
            required.spin_count,
        )
        _check(library, status, context)
        results = [
            OccupiedProjection(
                _immutable(coefficients),
                _immutable(projected),
                ProjectionDiagnostics(
                    **{
                        name: getattr(output, name)
                        for name in ProjectionDiagnostics.__dataclass_fields__
                    }
                ),
                "native_cpu_system",
            )
            for (coefficients, projected), output in zip(storage, outputs, strict=True)
        ]
        return results, workspace
    finally:
        if right:
            library.generativeqc_system_destroy(right)
        if left:
            library.generativeqc_system_destroy(left)
        library.generativeqc_context_destroy(context)
