"""Low-level decoder for versioned native precision-work records.

This module deliberately does not infer events from aggregate counters. The
Calculator/PreparedBatch facade can merge this dictionary with its aggregate
record after the active execution-owner changes have landed.
"""

from __future__ import annotations

import ctypes
from typing import Any

from . import _native

_EVENT_KINDS = {
    _native.PRECISION_EVENT_MIXED_FOCK: "mixed_fock",
    _native.PRECISION_EVENT_STRICT_FOCK: "strict_fock",
    _native.PRECISION_EVENT_POST_SCF_FOCK: "post_scf_fock",
    _native.PRECISION_EVENT_FINAL_AUDIT: "final_audit",
    _native.PRECISION_EVENT_RETRY: "retry",
    _native.PRECISION_EVENT_FALLBACK: "fallback",
    _native.PRECISION_EVENT_CONVERSION: "conversion",
}
_PHASES = {
    _native.PRECISION_PHASE_SCF: "scf",
    _native.PRECISION_PHASE_REFINEMENT: "refinement",
    _native.PRECISION_PHASE_FINALIZATION: "finalization",
    _native.PRECISION_PHASE_RETRY: "retry",
}
_OPERATOR_NAMES = {
    _native.PRECISION_OPERATOR_COULOMB_J: "coulomb_j",
    _native.PRECISION_OPERATOR_EXCHANGE_K: "exchange_k",
    _native.PRECISION_OPERATOR_XC: "xc",
    _native.PRECISION_OPERATOR_FOCK_ASSEMBLY: "fock_assembly",
    _native.PRECISION_OPERATOR_PHYSICAL_RESIDUAL: "physical_residual",
    _native.PRECISION_OPERATOR_EIGENSOLVER: "eigensolver",
    _native.PRECISION_OPERATOR_DENSITY_BUILD: "density_build",
    _native.PRECISION_OPERATOR_DIIS: "diis",
    _native.PRECISION_OPERATOR_MATRIX_PRODUCT: "matrix_product",
    _native.PRECISION_OPERATOR_DIAGNOSTICS: "diagnostics",
    _native.PRECISION_OPERATOR_OCCUPATION_STABILIZATION: "occupation_stabilization",
    _native.PRECISION_OPERATOR_COULOMB_RECURRENCE: "coulomb_recurrence",
    _native.PRECISION_OPERATOR_EXCHANGE_RECURRENCE: "exchange_recurrence",
}
_DTYPES = {
    _native.PRECISION_DTYPE_FP64: "fp64",
    _native.PRECISION_DTYPE_FP32: "fp32",
    _native.PRECISION_DTYPE_TF32: "tf32",
    _native.PRECISION_DTYPE_FP16: "fp16",
    _native.PRECISION_DTYPE_BF16: "bf16",
}
_ARITHMETIC_MODES = {
    _native.PRECISION_ARITHMETIC_STRICT: "strict",
    _native.PRECISION_ARITHMETIC_MIXED: "mixed",
    _native.PRECISION_ARITHMETIC_TF32: "tf32",
    _native.PRECISION_ARITHMETIC_FP16: "fp16",
    _native.PRECISION_ARITHMETIC_BF16: "bf16",
}


def _unknown(value: int) -> str:
    return f"unknown:{int(value)}"


def _unsupported() -> dict[str, Any]:
    return {
        "detail_version": None,
        "complete": False,
        "operator_inventory_complete": False,
        "conversion_count": None,
        "fallback_count": None,
        "returned_state_identity": None,
        "scf_fock_timeline": [],
        "operators": [],
    }


def query_precision_work(
    library: object, handle: object, index: int | None = None
) -> dict[str, Any] | None:
    """Return verbatim decoded detail, or None when no completed run exists.

    An older library or unsupported detail version returns an explicit
    unversioned/incomplete dictionary. Unknown enum values remain visible as
    ``unknown:<integer>`` so downstream acceptance fails closed.
    """

    name = (
        "generativeqc_calculation_get_precision_work"
        if index is None
        else "generativeqc_batch_get_precision_work"
    )
    getter = getattr(library, name, None)
    if getter is None:
        return _unsupported()
    detail = _native.PrecisionWorkDetail(
        ctypes.sizeof(_native.PrecisionWorkDetail), _native.ABI_VERSION
    )
    arguments = (
        (handle, _native.PRECISION_WORK_DETAIL_VERSION)
        if index is None
        else (handle, index, _native.PRECISION_WORK_DETAIL_VERSION)
    )
    status = getter(*arguments, ctypes.byref(detail), None, 0, None, 0)
    if status == _native.STATUS_PRECISION_UNAVAILABLE:
        return None
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return _unsupported()
    _native.check(library, status)

    events = (_native.PrecisionWorkEvent * detail.event_count)()
    for event in events:
        event.struct_size = ctypes.sizeof(_native.PrecisionWorkEvent)
        event.abi_version = _native.ABI_VERSION
    operators = (_native.PrecisionOperatorRecord * detail.operator_count)()
    for operator in operators:
        operator.struct_size = ctypes.sizeof(_native.PrecisionOperatorRecord)
        operator.abi_version = _native.ABI_VERSION
    status = getter(
        *arguments,
        ctypes.byref(detail),
        events if detail.event_count else None,
        detail.event_count,
        operators if detail.operator_count else None,
        detail.operator_count,
    )
    _native.check(library, status)

    identity = (
        int(detail.owner_id),
        int(detail.returned_solve_epoch),
        int(detail.returned_state_generation),
    )
    returned_state_identity = (
        f"cuda-ks:{identity[0]}:{identity[1]}:{identity[2]}" if all(identity) else None
    )
    timeline = [
        {
            "kind": _EVENT_KINDS.get(event.kind, _unknown(event.kind)),
            "sequence": int(event.sequence),
            "iteration": int(event.iteration),
            "state": (
                f"cuda-ks:{int(event.owner_id)}:{int(event.solve_epoch)}:"
                f"{int(event.state_generation)}"
            ),
            "phase": _PHASES.get(event.phase, _unknown(event.phase)),
            "count": 1,
        }
        for event in events
    ]
    census = [
        {
            "name": _OPERATOR_NAMES.get(operator.kind, _unknown(operator.kind)),
            "storage": _DTYPES.get(
                operator.storage_dtype, _unknown(operator.storage_dtype)
            ),
            "compute": _DTYPES.get(
                operator.compute_dtype, _unknown(operator.compute_dtype)
            ),
            "accumulation": _DTYPES.get(
                operator.accumulation_dtype,
                _unknown(operator.accumulation_dtype),
            ),
            "reduction": _DTYPES.get(
                operator.reduction_dtype, _unknown(operator.reduction_dtype)
            ),
            "arithmetic_mode": _ARITHMETIC_MODES.get(
                operator.arithmetic_mode, _unknown(operator.arithmetic_mode)
            ),
            "count": int(operator.count),
        }
        for operator in operators
    ]
    return {
        "detail_version": int(detail.detail_version),
        "complete": bool(detail.complete),
        "operator_inventory_complete": bool(detail.operator_inventory_complete),
        "conversion_count": int(detail.conversion_count),
        "fallback_count": int(detail.fallback_count),
        "returned_state_identity": returned_state_identity,
        "scf_fock_timeline": timeline,
        "operators": census,
    }
