"""Typed transport and presentation vocabulary for native candidate decisions."""

from __future__ import annotations

import ctypes as ct
from enum import IntEnum
from typing import Any

from . import _native
from ._precision_native import _Context as _AdmissionContext
from ._precision_native import _uint, evidence


class Disposition(IntEnum):
    AUDIT = 0
    STRICT = 1
    FOCK = 2
    WARM_FOCK = 3
    MIXED_GRADIENT = 4
    STRICT_GRADIENT = 5
    STRICT_XC = 6
    STANDARD = 7


class _Context(ct.Structure):
    """Copied categorical facts and inline strings; no borrowed Python memory."""

    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in (
                "struct_size",
                "abi_version",
                "fast_profile",
                "explicit_target",
                "fast_native",
                "fast_fock",
                "forces_requested",
                "gradient_present",
                "gradient_schema_version",
                "gradient_complete",
                "gradient_owner",
                "gradient_returned_mode",
                "qualification_present",
                "gradient_capability_fallback",
                "weighted_task_counters_collected",
                "correction_present",
                "correction_result_flags",
                "mixed_fock_count_present",
                "reserved",
            )
        ]
        + [
            (name, ct.c_uint64)
            for name in (
                "geometry_points",
                "grid_points",
                "weighted_tasks",
                "primitive_sources",
                "gradient_retries",
                "execution_retries",
            )
        ]
        + [
            ("qualification", ct.c_char * 96),
            ("correction_model_identity", ct.c_char * 65),
        ]
    )


class _Decision(ct.Structure):
    """One checked disposition, reason bitset and native-bound gradient label."""

    _fields_ = [
        (name, ct.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "policy_version",
            "disposition",
            "qualified_gradient",
            "reasons",
        )
    ] + [("gradient_admission", ct.c_char * 96)]


REASONS = (
    (2, "fp32_geometry_capability_unavailable"),
    (4, "geometry_correction_not_qualified"),
    (8, "native_gradient_retried_fp64"),
    (16, "fp32_grid_storage_unavailable"),
    (32, "native_mixed_execution_retried_fp64"),
    (64, "fast_route_unavailable_using_standard"),
    (128, "budget_or_domain_requires_fp64_audit"),
)


def require_candidate_classification(library: Any) -> Any:
    """Require the selected owner before scalar or fleet electronic execution."""
    query = getattr(library, "generativeqc_precision_candidate_query_v1", None)
    if query is None:
        raise NotImplementedError(
            "native library lacks required precision-candidate query "
            "generativeqc_precision_candidate_query_v1; rebuild or install a current library"
        )
    return query


def _text(value: Any, capacity: int) -> bytes:
    """Unknown labels cannot become recognized qualifications by truncation."""
    encoded = value.encode("ascii") if isinstance(value, str) else b""
    if b"\0" in encoded or len(encoded) >= capacity:
        raise ValueError("precision evidence label exceeds its native inline field")
    return encoded


def classify_candidate(
    calculator: Any,
    result: Any,
    *,
    profile: str,
    target: Any,
    fast_native: bool,
    fast_fock: bool,
    forces: bool,
    precision: str,
    budget: Any,
) -> tuple[Disposition, int, str]:
    """Translate facts, never qualification tables or audit-shortcut predicates."""
    from .dispersion import R2SCAN3CCorrectionResult

    library = calculator._library
    query = require_candidate_classification(library)
    work = result.precision or {}
    gradient = work.get("gradient") or {}
    context = _Context(ct.sizeof(_Context), _native.ABI_VERSION)
    context.fast_profile, context.explicit_target = (
        profile == "fast",
        target is not None,
    )
    context.fast_native, context.fast_fock = fast_native, fast_fock
    context.forces_requested, context.gradient_present = forces, bool(gradient)
    context.gradient_schema_version = _uint(gradient.get("schema_version") or 0, 32)
    context.gradient_complete = gradient.get("complete") is True
    context.gradient_owner = {
        "python-stationary-cuda": 1,
        "python-composite-stationary-cuda": 2,
    }.get(gradient.get("owner"), 0)
    context.gradient_returned_mode = {"mixed": 1, "fp64": 2}.get(
        gradient.get("returned_mode"), 0
    )
    context.qualification_present = "qualification" in gradient
    context.qualification = _text(gradient.get("qualification"), 96)
    context.gradient_capability_fallback = bool(gradient.get("capability_fallback"))
    context.weighted_task_counters_collected = (
        gradient.get("weighted_task_counters_collected") is True
    )
    for name, source in (
        ("geometry_points", "attempted_fp32_geometry_points"),
        ("grid_points", "attempted_fp32_grid_points"),
        ("weighted_tasks", "attempted_fp32_weighted_tasks"),
        ("primitive_sources", "attempted_fp32_primitive_sources"),
        ("gradient_retries", "fp64_gradient_retries"),
    ):
        setattr(context, name, _uint(gradient.get(source) or 0))
    context.execution_retries = _uint(work.get("execution_retries") or 0)
    context.mixed_fock_count_present = "mixed_stage_fock_builds" in work
    correction_model = getattr(calculator, "_dispersion_method_ir", None)
    context.correction_present = correction_model is not None
    if correction_model is not None:
        context.correction_model_identity = _text(correction_model.identity, 65)
        correction = result.dispersion
        if isinstance(correction, R2SCAN3CCorrectionResult):
            context.correction_result_flags = (
                1
                | (bool(correction.ok) << 1)
                | (bool(correction.d4.ok) << 2)
                | (bool(correction.gcp.ok) << 3)
                | ((correction.backend == "cuda+cpu") << 4)
                | ((correction.gradient is not None) << 5)
            )
    model, provenance, detail, events, operators = evidence(
        calculator,
        work,
        result=result,
        precision=precision,
        budget=budget,
    )
    output = _Decision(ct.sizeof(_Decision), _native.ABI_VERSION)
    explicit_target, fast_profile = (
        bool(context.explicit_target),
        bool(context.fast_profile),
    )
    query.argtypes = [
        ct.POINTER(_Context),
        ct.POINTER(_AdmissionContext),
        ct.POINTER(_native.PrecisionProvenance),
        ct.POINTER(_native.PrecisionWorkDetail),
        ct.POINTER(_native.PrecisionWorkEvent),
        ct.POINTER(_native.PrecisionOperatorRecord),
        ct.POINTER(_Decision),
    ]
    query.restype = ct.c_int
    _native.check(
        library,
        query(
            ct.byref(context),
            ct.byref(model),
            ct.byref(provenance),
            ct.byref(detail),
            events,
            operators,
            ct.byref(output),
        ),
    )
    if (
        output.struct_size != ct.sizeof(_Decision)
        or output.abi_version != _native.ABI_VERSION
        or output.policy_version != 1
        or output.disposition not in Disposition._value2member_map_
        or output.qualified_gradient not in (0, 1)
        or output.reasons & ~255
        or bool(output.reasons & 1) != bool(output.qualified_gradient)
        or (explicit_target and output.disposition != Disposition.AUDIT)
        or (
            not fast_profile
            and not explicit_target
            and output.disposition != Disposition.STRICT
        )
        or (
            fast_profile
            and not explicit_target
            and output.disposition == Disposition.STRICT
        )
        or (
            output.disposition
            in (Disposition.MIXED_GRADIENT, Disposition.STRICT_GRADIENT)
            and not output.qualified_gradient
        )
        or bool(output.reasons & 64) != (output.disposition == Disposition.STANDARD)
        or (
            output.disposition == Disposition.AUDIT
            and not explicit_target
            and not output.reasons & 128
        )
        or (
            output.reasons & 128
            and (explicit_target or output.disposition != Disposition.AUDIT)
        )
        or not output.gradient_admission
        or len(output.gradient_admission) >= 96
    ):
        raise RuntimeError("native precision candidate returned an invalid receipt")
    return (
        Disposition(output.disposition),
        output.reasons,
        output.gradient_admission.decode("ascii"),
    )
