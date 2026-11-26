"""CBS stopping limits queried from the native correlated convergence owner.

Budgets select stopping criteria, not an a posteriori observable certificate.
The native owner independently validates the transport and solver ceilings.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _native
from ._cbs_plan import CBSPlan, _Plan

if TYPE_CHECKING:
    from .force_aware_numerics import TargetErrorBudget


_LIMIT_NAMES = (
    "energy_tolerance",
    "density_tolerance",
    "ccsd_energy_tolerance",
    "ccsd_residual_tolerance",
    "lambda_residual_tolerance",
    "response_absolute_tolerance",
    "orbital_residual_tolerance",
    "stationarity_tolerance",
)


class _Budget(ctypes.Structure):
    """Copied native allowance descriptor; zero RMS means unspecified."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)] + [
        (name, ctypes.c_double) for name in ("energy_abs", "force_max_abs", "force_rms")
    ]


class _Limits(ctypes.Structure):
    """Detached numerical limits; no solver or model ownership."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)] + [
        (name, ctypes.c_double) for name in ("scale", *_LIMIT_NAMES)
    ]


class _CBSDescriptor(ctypes.Structure):
    """Borrowed inputs for combined CBS allocation and stopping policy."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "correction_basis",
            "default_standard",
        )
    ] + [
        ("extrapolation", ctypes.POINTER(_Plan)),
        ("total_budget", ctypes.POINTER(_Budget)),
    ]


class _CBSResult(ctypes.Structure):
    """One native receipt; budget records are owned values, not handles."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("coefficient_l1", ctypes.c_double),
        ("total_budget", _Budget),
        ("component_budget", _Budget),
        ("limits", _Limits),
    ]


@dataclass(frozen=True)
class CBSConvergence:
    """Detached native budget and stopping receipt, not an error estimate."""

    coefficient_l1: float
    total_budget: TargetErrorBudget | None
    component_budget: TargetErrorBudget | None
    scale: float
    limits: dict[str, float]


def _encode_budget(budget: TargetErrorBudget | None) -> _Budget:
    return _Budget(
        ctypes.sizeof(_Budget),
        _native.ABI_VERSION,
        budget.energy_abs if budget else 0.0,
        budget.force_max_abs if budget else 0.0,
        budget.force_rms if budget and budget.force_rms is not None else 0.0,
    )


def _valid_header(value: ctypes.Structure) -> bool:
    return (
        value.struct_size == ctypes.sizeof(value)
        and value.abi_version == _native.ABI_VERSION
    )


def _decode_budget(value: _Budget) -> TargetErrorBudget | None:
    from .force_aware_numerics import TargetErrorBudget

    if not _valid_header(value):
        raise RuntimeError("native CBS convergence returned an invalid budget header")
    if value.energy_abs == value.force_max_abs == value.force_rms == 0:
        return None
    try:
        return TargetErrorBudget(
            value.energy_abs, value.force_max_abs, value.force_rms or None
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "native CBS convergence returned an invalid budget"
        ) from error


def _decode_limits(result: _Limits) -> tuple[float, dict[str, float]]:
    limits = {name: getattr(result, name) for name in _LIMIT_NAMES}
    if (
        not _valid_header(result)
        or not math.isfinite(result.scale)
        or not 0 <= result.scale <= 1
        or any(not math.isfinite(value) or value <= 0 for value in limits.values())
    ):
        raise RuntimeError("native correlated convergence returned an invalid receipt")
    return result.scale, limits


def cbs_convergence_policy(
    library: Any,
    plan: CBSPlan,
    correction_basis: int,
    budget: TargetErrorBudget | None,
    *,
    default_standard: bool,
) -> CBSConvergence:
    """Send the selected native plan and basis identity role without allocating in Python."""
    operation = getattr(library, "generativeqc_cbs_convergence_query_v1", None)
    if operation is None:
        raise NotImplementedError(
            "CBS requires the selected native CBS convergence query"
        )
    extrapolation = _Plan(
        ctypes.sizeof(_Plan),
        _native.ABI_VERSION,
        (ctypes.c_double * 2)(*plan.hf_weights),
        (ctypes.c_double * 2)(*plan.mp2_weights),
        (ctypes.c_double * 6)(*plan.coefficients),
    )
    allowance = _encode_budget(budget)
    descriptor = _CBSDescriptor(
        ctypes.sizeof(_CBSDescriptor),
        _native.ABI_VERSION,
        correction_basis,
        int(default_standard),
        ctypes.pointer(extrapolation),
        ctypes.pointer(allowance) if budget is not None else None,
    )
    result = _CBSResult(ctypes.sizeof(_CBSResult), _native.ABI_VERSION)
    operation.argtypes = [ctypes.POINTER(_CBSDescriptor), ctypes.POINTER(_CBSResult)]
    operation.restype = ctypes.c_int
    status = operation(ctypes.byref(descriptor), ctypes.byref(result))
    if status == _native.STATUS_INVALID_ARGUMENT:
        raise ValueError(
            "CBS convergence parameters produce an invalid amplification or allowance"
        )
    _native.check(library, status)
    total = _decode_budget(result.total_budget)
    component = _decode_budget(result.component_budget)
    scale, limits = _decode_limits(result.limits)
    if (
        not _valid_header(result)
        or not math.isfinite(result.coefficient_l1)
        or result.coefficient_l1 < 1
        or (total is None) != (not default_standard and budget is None)
        or (total is None) != (component is None)
        or (total is None and scale != 0)
        or (
            total is not None
            and component is not None
            and (total.force_rms is None) != (component.force_rms is None)
        )
    ):
        raise RuntimeError("native CBS convergence returned an invalid receipt")
    return CBSConvergence(result.coefficient_l1, total, component, scale, limits)


def convergence_policy(
    budget: TargetErrorBudget | None, *, library: Any = None
) -> tuple[float, dict[str, float]]:
    """Require one shared native policy, including during standalone CBS planning.

    Calculator admission supplies its selected library. Standalone CBS metadata
    uses the normal native loader. A refused query or malformed receipt is an
    error, never authorization to retry Python stopping-limit arithmetic.
    """
    selected = _native.load_library() if library is None else library
    operation = getattr(selected, "generativeqc_correlation_convergence_query_v1", None)
    if operation is None:
        raise NotImplementedError(
            "native library lacks required correlated-convergence query "
            "generativeqc_correlation_convergence_query_v1; rebuild or install a current library"
        )
    descriptor = _encode_budget(budget)
    result = _Limits(ctypes.sizeof(_Limits), _native.ABI_VERSION)
    operation.argtypes = [ctypes.POINTER(_Budget), ctypes.POINTER(_Limits)]
    operation.restype = ctypes.c_int
    _native.check(selected, operation(ctypes.byref(descriptor), ctypes.byref(result)))
    return _decode_limits(result)


def convergence_scale(budget: TargetErrorBudget | None) -> float:
    """Return the shared scale; zero preserves strict convergence."""
    return convergence_policy(budget)[0]


def convergence_limits(
    budget: TargetErrorBudget | None, *, library: Any = None
) -> dict[str, float]:
    """Return shared ceilings using the caller's library when one is selected."""
    return convergence_policy(budget, library=library)[1]
