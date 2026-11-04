"""CBS stopping limits queried from the native correlated convergence owner.

Budgets select stopping criteria, not an a posteriori observable certificate.
The native owner independently validates the transport and solver ceilings.
"""

from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING, Any

from . import _native

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
    descriptor = _Budget(
        ctypes.sizeof(_Budget),
        _native.ABI_VERSION,
        budget.energy_abs if budget else 0.0,
        budget.force_max_abs if budget else 0.0,
        budget.force_rms if budget and budget.force_rms is not None else 0.0,
    )
    result = _Limits(ctypes.sizeof(_Limits), _native.ABI_VERSION)
    operation.argtypes = [ctypes.POINTER(_Budget), ctypes.POINTER(_Limits)]
    operation.restype = ctypes.c_int
    _native.check(selected, operation(ctypes.byref(descriptor), ctypes.byref(result)))
    limits = {name: getattr(result, name) for name in _LIMIT_NAMES}
    if (
        result.struct_size != ctypes.sizeof(_Limits)
        or result.abi_version != _native.ABI_VERSION
        or not math.isfinite(result.scale)
        or not 0 <= result.scale <= 1
        or any(not math.isfinite(value) or value <= 0 for value in limits.values())
    ):
        raise RuntimeError("native correlated convergence returned an invalid receipt")
    return result.scale, limits


def convergence_scale(budget: TargetErrorBudget | None) -> float:
    """Return the shared scale; zero preserves strict convergence."""
    return convergence_policy(budget)[0]


def convergence_limits(
    budget: TargetErrorBudget | None, *, library: Any = None
) -> dict[str, float]:
    """Return shared ceilings using the caller's library when one is selected."""
    return convergence_policy(budget, library=library)[1]
