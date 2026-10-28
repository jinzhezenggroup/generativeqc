"""CBS stopping limits queried from the native correlated convergence owner.

Budgets select stopping criteria, not an a posteriori observable certificate.
The native owner independently validates the transport and solver ceilings.
"""

from __future__ import annotations

import ctypes
import math
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from . import _native

if TYPE_CHECKING:
    from .force_aware_numerics import TargetErrorBudget


def _compatibility_scale(budget: TargetErrorBudget | None) -> float:
    """Retain strict convergence below the supported relaxation bracket."""
    if budget is None:
        return 0.0
    scale = min(
        1.0,
        budget.energy_abs / 1e-5,
        budget.force_max_abs / 1e-4,
        budget.force_rms / 3e-5 if budget.force_rms is not None else 1.0,
    )
    return scale if scale >= 1e-3 else 0.0


def _compatibility_limits(budget: TargetErrorBudget | None) -> dict[str, float]:
    """Return native ceilings; explicit tighter SCF/CC controls still win."""
    scale = _compatibility_scale(budget)
    return {
        "energy_tolerance": max(1e-11, scale * 1e-8),
        "density_tolerance": max(1e-11, scale * 1e-9),
        "ccsd_energy_tolerance": 1e-8,
        "ccsd_residual_tolerance": max(1e-9, scale * 1e-6),
        "lambda_residual_tolerance": max(1e-9, scale * 1e-7),
        "response_absolute_tolerance": max(1e-12, scale * 1e-8),
        "orbital_residual_tolerance": max(1e-10, scale * 1e-7),
        "stationarity_tolerance": max(1e-8, scale * 1e-6),
    }


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


@lru_cache(maxsize=8)
def _load(path: str) -> Any:
    """Discover one selected library; do not search beyond a missing protocol."""
    library = ctypes.CDLL(path)
    if getattr(library, "generativeqc_correlation_convergence_query_v1", None) is None:
        return None
    library.generativeqc_get_abi_version.argtypes = []
    library.generativeqc_get_abi_version.restype = ctypes.c_uint32
    if library.generativeqc_get_abi_version() != _native.ABI_VERSION:
        raise RuntimeError("GENERATIVEQC Python/native ABI version mismatch")
    library.generativeqc_status_message.argtypes = [ctypes.c_int]
    library.generativeqc_status_message.restype = ctypes.c_char_p
    return library


def _optional_library() -> Any:
    """Allow native-free typed CBS construction when no library is installed."""
    for candidate in _native._candidate_paths():
        if candidate.exists():
            return _load(str(candidate.resolve()))
    return None


def convergence_policy(
    budget: TargetErrorBudget | None, *, library: Any = None
) -> tuple[float, dict[str, float]]:
    """Query one shared policy, retaining only missing-library/protocol compatibility.

    Calculator admission supplies its selected library. Standalone CBS metadata
    uses the first configured path. A refused query or malformed receipt is an
    error, never authorization to retry Python stopping-limit arithmetic.
    """
    selected = _optional_library() if library is None else library
    operation = getattr(selected, "generativeqc_correlation_convergence_query_v1", None)
    if operation is None:
        return _compatibility_scale(budget), _compatibility_limits(budget)
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
