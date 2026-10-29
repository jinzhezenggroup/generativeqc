"""Numerical plan admission through the selected native HF runtime.

Standalone typed plans discover only the first configured library. Running
calculators supply their own library. Bounded built-in inputs require the current
protocol; integers outside the ABI keep explicit Python arithmetic. Native
rejection never retries that policy.
"""

from __future__ import annotations

import ctypes
import typing
from functools import lru_cache

from . import _native


class Descriptor(ctypes.Structure):
    """Borrowed two-stage numerical controls; identities remain typed metadata."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("struct_size", "abi_version", "flags", "reserved")
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "source_iterations",
                "target_iterations",
                "source_iteration_limit",
                "total_iteration_limit",
            )
        ]
        + [
            (name, ctypes.c_double)
            for name in (
                "source_energy_tolerance",
                "source_density_tolerance",
                "target_energy_tolerance",
                "target_density_tolerance",
                "source_cost",
                "target_cost",
                "cost_limit",
            )
        ]
    )


class Decision(ctypes.Structure):
    """Detached failure mask, with no owning solver or device state."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("struct_size", "abi_version", "failures", "reserved")
    ]


@lru_cache(maxsize=8)
def _load(path: str) -> typing.Any:
    """Bind one selected library without searching past a missing protocol."""
    library = ctypes.CDLL(path)
    if getattr(library, "generativeqc_hf_plan_query_v1", None) is None:
        raise RuntimeError("HF planning requires the current native plan protocol")
    library.generativeqc_get_abi_version.restype = ctypes.c_uint32
    library.generativeqc_get_abi_version.argtypes = []
    if library.generativeqc_get_abi_version() != _native.ABI_VERSION:
        raise RuntimeError("GENERATIVEQC Python/native ABI version mismatch")
    library.generativeqc_status_message.argtypes = [ctypes.c_int]
    library.generativeqc_status_message.restype = ctypes.c_char_p
    return library


def require_library() -> typing.Any:
    """Use the first configured image; missing support cannot change policy ownership."""
    for candidate in _native._candidate_paths():
        if candidate.exists():
            return _load(str(candidate.resolve()))
    raise RuntimeError("HF planning requires the current native library")


def failures(
    source: typing.Any,
    target: typing.Any,
    budget: typing.Any,
    *,
    library: typing.Any = None,
) -> int:
    """Query numerical gates; do not convert a refused query into compatibility."""
    counts = (
        source.convergence.maximum_iterations,
        target.convergence.maximum_iterations,
        budget.maximum_source_iterations,
        budget.maximum_total_iterations,
    )
    if any(type(value) is not int or not 0 < value < 2**64 for value in counts):
        return _compatibility_failures(source, target, budget)
    selected = require_library() if library is None else library
    operation = getattr(selected, "generativeqc_hf_plan_query_v1", None)
    if operation is None:
        raise RuntimeError("HF planning requires the current native plan protocol")
    costs = (
        source.estimated_cost_units,
        target.estimated_cost_units,
        budget.maximum_estimated_cost_units,
    )
    descriptor = Descriptor(
        ctypes.sizeof(Descriptor),
        _native.ABI_VERSION,
        sum(int(value is not None) << index for index, value in enumerate(costs)),
        0,
        *counts,
        source.convergence.energy_change,
        source.convergence.density_rms,
        target.convergence.energy_change,
        target.convergence.density_rms,
        *(value if value is not None else 0.0 for value in costs),
    )
    result = Decision(ctypes.sizeof(Decision), _native.ABI_VERSION, 0, 0)
    operation.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Decision)]
    operation.restype = ctypes.c_int
    _native.check(selected, operation(ctypes.byref(descriptor), ctypes.byref(result)))
    if (
        result.struct_size != ctypes.sizeof(Decision)
        or result.abi_version != _native.ABI_VERSION
        or result.failures >> 5
        or result.reserved
        or result.failures & 24 == 24
    ):
        raise RuntimeError("native HF planning returned an inconsistent receipt")
    return result.failures


def _compatibility_failures(
    source: typing.Any, target: typing.Any, budget: typing.Any
) -> int:
    """Preserve arbitrary-width Python integer planning without truncating limits."""
    failed = 0
    if (
        source.convergence.energy_change < target.convergence.energy_change
        or source.convergence.density_rms < target.convergence.density_rms
    ):
        failed |= 1
    if source.convergence.maximum_iterations > budget.maximum_source_iterations:
        failed |= 2
    if (
        source.convergence.maximum_iterations + target.convergence.maximum_iterations
        > budget.maximum_total_iterations
    ):
        failed |= 4
    if budget.maximum_estimated_cost_units is not None:
        costs = (source.estimated_cost_units, target.estimated_cost_units)
        if any(value is None for value in costs):
            failed |= 8
        elif (
            sum(typing.cast("float", value) for value in costs)
            > budget.maximum_estimated_cost_units
        ):
            failed |= 16
    return failed
