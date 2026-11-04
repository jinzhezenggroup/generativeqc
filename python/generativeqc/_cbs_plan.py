"""Transport the selected native owner's geometry-independent CBS plan."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any

from . import _native


class _Descriptor(ctypes.Structure):
    """Copied scientific extrapolation inputs; no runtime resources are owned."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "low_cardinal",
            "high_cardinal",
            "hf_extrapolation",
            "reserved",
        )
    ] + [(name, ctypes.c_double) for name in ("correlation_exponent", "hf_exponent")]


class _Plan(ctypes.Structure):
    """Complete ABI receipt; coefficient order is the public native contract."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("hf_weights", ctypes.c_double * 2),
        ("mp2_weights", ctypes.c_double * 2),
        ("coefficients", ctypes.c_double * 6),
    ]


@dataclass(frozen=True)
class CBSPlan:
    """Detached native coefficients, not a convergence or observable certificate."""

    hf_weights: tuple[float, float]
    mp2_weights: tuple[float, float]
    coefficients: tuple[float, ...]


def native_cbs_plan(
    library: Any,
    low: int,
    high: int,
    correlation_exponent: float,
    hf_exponent: float,
    hf_extrapolation: str,
) -> CBSPlan:
    """Require the caller's selected owner; never calculate fallback coefficients."""
    if hf_extrapolation not in ("exponential", "highest"):
        raise ValueError("hf_extrapolation must be 'exponential' or 'highest'")
    query = getattr(library, "generativeqc_cbs_extrapolation_query_v1", None)
    if query is None:
        raise RuntimeError("CBS requires the selected native extrapolation plan query")
    query.argtypes = [ctypes.POINTER(_Descriptor), ctypes.POINTER(_Plan)]
    query.restype = ctypes.c_int
    descriptor = _Descriptor(
        ctypes.sizeof(_Descriptor),
        _native.ABI_VERSION,
        low,
        high,
        int(hf_extrapolation == "highest"),
        0,
        correlation_exponent,
        hf_exponent,
    )
    output = _Plan(ctypes.sizeof(_Plan), _native.ABI_VERSION)
    status = query(ctypes.byref(descriptor), ctypes.byref(output))
    if status == _native.STATUS_INVALID_ARGUMENT:
        raise ValueError(
            "CBS extrapolation parameters produce singular or nonfinite weights"
        )
    _native.check(library, status)
    if (
        output.struct_size < ctypes.sizeof(_Plan)
        or output.abi_version != _native.ABI_VERSION
        or not all(
            math.isfinite(value)
            for value in (*output.hf_weights, *output.mp2_weights, *output.coefficients)
        )
    ):
        raise RuntimeError("native CBS extrapolation returned an invalid receipt")
    return CBSPlan(
        tuple(output.hf_weights), tuple(output.mp2_weights), tuple(output.coefficients)
    )
