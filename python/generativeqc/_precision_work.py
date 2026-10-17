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
    _native.PRECISION_OPERATOR_NONLOCAL_CORRELATION: "nonlocal_correlation",
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


def merge_precision_work(
    aggregate: dict[str, Any] | None, detail: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Merge execution detail without inventing or upgrading any evidence."""

    if aggregate is None:
        return detail
    if detail is None:
        return aggregate
    return {
        **aggregate,
        **detail,
        "native_provenance": dict(aggregate),
    }


def query_gradient_precision(
    library: object, handle: object, index: int | None = None
) -> dict[str, Any] | None:
    """Decode actual completed mixed-gradient work, never infer it from SCF.

    Uninstrumented KS calls report weighted dispatch policy rather than
    inventing counts or zero retries. V3 can publish explicitly collected
    per-item counts, including the direct-HF production consumer.
    Whole-gradient retries replace all sources at the identical strict token.
    """
    name = (
        "generativeqc_calculation_gradient_precision"
        if index is None
        else "generativeqc_batch_gradient_precision"
    )
    version = 3
    getter = getattr(library, f"{name}_v3", None)
    if getter is None:
        version = 2
        getter = getattr(library, f"{name}_v2", None)
    if getter is None:
        version = 1
        getter = getattr(library, f"{name}_v1", None)
    if getter is None:
        return None
    values = (ctypes.c_uint64 * {1: 5, 2: 6, 3: 10}[version])()
    arguments = (handle,) if index is None else (handle, index)
    status = getter(*arguments, values, len(values))
    if status in (_native.STATUS_PRECISION_UNAVAILABLE, _native.STATUS_NOT_IMPLEMENTED):
        return None
    _native.check(library, status)
    raw_qualification = int(values[5]) if version >= 2 else 1
    # 100+model identifies an executable candidate outside the independently
    # qualified one-endpoint population (or a relaxed-budget Fock density).
    # It MUST be compared with a separate FP64 E+F endpoint before publication.
    requires_independent_audit = 100 < raw_qualification < 200
    qualification = (
        raw_qualification - 100 if requires_independent_audit else raw_qualification
    )
    collected = version == 3 and bool(values[6])
    all_weighted_retried = (
        collected and values[7] > 0 and values[7] == values[9] and values[2] == 0
    )
    return {
        "schema_version": version,
        "qualification": (
            {
                1: "pbe0-rks-budget-fp32-v1",
                2: "pbe-rks-budget-fp32-v1",
                3: "b3lyp-rks-budget-fp32-v1",
                4: "r2scan-rks-budget-fp32-v1",
                5: "rhf-budget-fp32-v1",
                6: "m06-2x-rks-budget-fp32-v1",
                7: "rhf-df-shell-budget-fp32-v1",
            }.get(qualification, "unknown")
            + ("-audit-required" if requires_independent_audit else "")
        ),
        "requires_independent_audit": requires_independent_audit,
        "complete": bool(values[0]),
        "weighted_fp32_orders": (
            [0, 1, 2, 3, 4, 5] if values[1] and qualification != 7 else []
        ),
        **(
            {"weighted_fp32_consumer": "df-shell-polynomial-and-rys"}
            if qualification == 7 and values[1]
            else {}
        ),
        "weighted_task_retries": int(values[9]) if collected else None,
        "weighted_task_counters_collected": collected,
        "attempted_fp32_weighted_tasks": int(values[7]) if collected else None,
        "attempted_fp32_primitive_sources": int(values[8]) if collected else None,
        "attempted_fp32_geometry_points": int(values[2]),
        "attempted_fp32_grid_points": int(values[3]),
        "fp64_gradient_retries": int(values[4]),
        "returned_mode": "fp64" if values[4] or all_weighted_retried else "mixed",
        "fp64_components": [
            "final_scf",
            "one_electron_and_pulay",
            "boys",
            "integral_geometry_and_density_weights",
            *(["weighted_orders_above_5"] if qualification != 7 else []),
            "global_reductions",
            "numerical_fallbacks",
            *(["unselected_weighted_schedules"] if qualification == 5 else []),
            *(
                [
                    "df_metric_and_response_weights",
                    "generic_df_derivatives",
                ]
                if qualification == 7
                else []
            ),
            *(
                ["df_shell_sp_bra_contractions"]
                if qualification == 7 and values[1] == 2
                else []
            ),
            *(
                ["xc_work_input_conditioning", "xc_cartesian_conversion"]
                if qualification in (3, 4, 6)
                else []
            ),
        ],
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
