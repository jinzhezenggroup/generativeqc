"""Execution-result translation independent of model resolution.

Only the narrow native descriptor/diagnostic boundary lives here.  The
single-system and batch facades can therefore test status, backend, force, and
correlation translation without constructing a model or a native execution
context.
"""

from __future__ import annotations

import ctypes
import typing

import numpy as np

from . import _native
from ._api_types import CcPerformanceResult, CorrelationResult


def read_correlation_result(
    library: ctypes.CDLL,
    owner: ctypes.c_void_p,
    *,
    index: int | None = None,
    context: ctypes.c_void_p | None = None,
    mp2_component: bool = False,
) -> CorrelationResult | None:
    """Decode the optional post-HF diagnostic ABI into a stable record."""

    name = (
        "generativeqc_calculation_get_correlation_diagnostic"
        if index is None
        else "generativeqc_batch_get_correlation_diagnostic"
    )
    if mp2_component:
        if index is not None:
            raise ValueError("MP2 companions are single-system results")
        name = "generativeqc_calculation_get_mp2_component_diagnostic_v1"
    getter = getattr(library, name)
    diag = _native.CorrelationDiagnostic()
    diag.struct_size = ctypes.sizeof(diag)
    diag.abi_version = _native.ABI_VERSION
    args = (
        (owner, ctypes.byref(diag))
        if index is None
        else (owner, index, ctypes.byref(diag))
    )
    status = getter(*args)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status, context=context)
    values = {
        field_name: getattr(diag, field_name)
        for field_name, _ in diag._fields_
        if field_name not in ("struct_size", "abi_version")
    }
    values["mo_host_staging"] = bool(values["mo_host_staging"])
    values["reference_execution_plan_reused"] = bool(
        values["reference_execution_plan_reused"]
    )
    for field_name in (
        "equation_hash",
        "response_operator_hash",
        "ccsd_replay_equation_hash",
        "ccsd_t_equation_hash",
    ):
        values[field_name] = values[field_name].decode("ascii")
    return CorrelationResult(**values)


def read_cc_performance_result(
    library: ctypes.CDLL,
    owner: ctypes.c_void_p,
    *,
    index: int | None = None,
    context: ctypes.c_void_p | None = None,
) -> CcPerformanceResult | None:
    """Decode the additive RCCSD/RCCSD(T) phase/work diagnostic."""

    name = (
        "generativeqc_calculation_get_cc_performance_diagnostic"
        if index is None
        else "generativeqc_batch_get_cc_performance_diagnostic"
    )
    getter = getattr(library, name)
    diag = _native.CcPerformanceDiagnostic()
    diag.struct_size = ctypes.sizeof(diag)
    diag.abi_version = _native.ABI_VERSION
    args = (
        (owner, ctypes.byref(diag))
        if index is None
        else (owner, index, ctypes.byref(diag))
    )
    status = getter(*args)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status, context=context)
    values = {
        field_name: getattr(diag, field_name)
        for field_name, _ in diag._fields_
        if field_name not in ("struct_size", "abi_version")
    }
    precision = getattr(library, "generativeqc_calculation_get_cc_precision_v1", None)
    if index is None and precision is not None:
        precision.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
        ]
        precision.restype = ctypes.c_int
        counters = (ctypes.c_uint64 * 6)()
        status = precision(owner, counters, len(counters))
        if status != _native.STATUS_NOT_IMPLEMENTED:
            _native.check(library, status, context=context)
            for position, field in enumerate(
                (
                    "ccsd_fp32_requested",
                    "ccsd_fp32_executed",
                    "ccsd_fp32_refused",
                    "ccsd_fp32_evaluations",
                    "ccsd_fp32_gemm_calls",
                    "ccsd_fp32_cast_elements",
                )
            ):
                values[field] = (
                    bool(counters[position]) if position < 3 else counters[position]
                )
    return CcPerformanceResult(**values)


def backend_name(executed_backend: int) -> str:
    """Translate the native backend enum without consulting model metadata."""

    return {
        _native.BACKEND_CPU_REFERENCE: "cpu_reference",
        _native.BACKEND_CUDA: "cuda",
        _native.BACKEND_HYBRID_CUDA: "hybrid_cuda",
    }.get(executed_backend, "unknown")


def copy_force_array(force_storage: typing.Any, atom_count: int) -> np.ndarray | None:
    """Copy a native ``3*N`` force buffer into the public ``(N, 3)`` shape."""

    if force_storage is None:
        return None
    return np.ctypeslib.as_array(force_storage).copy().reshape(atom_count, 3)


def status_message(library: ctypes.CDLL, status: int) -> str:
    """Decode a native status without requiring a calculator or model."""

    return library.generativeqc_status_message(status).decode("utf-8")


__all__ = [
    "backend_name",
    "copy_force_array",
    "read_cc_performance_result",
    "read_correlation_result",
    "status_message",
]
