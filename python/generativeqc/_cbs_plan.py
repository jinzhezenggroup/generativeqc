"""Transport native CBS coefficients and completed component observables."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

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


_DoublePointer = ctypes.POINTER(ctypes.c_double)


class _Observables(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("energy", ctypes.c_double * 6),
        ("forces", _DoublePointer * 6),
        ("force_count", ctypes.c_size_t),
    ]


class _Composition(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("energy", ctypes.c_double * 4),
        ("force_blocks", _DoublePointer),
        ("force_capacity", ctypes.c_size_t),
    ]


class NativeCbsComposer:
    """Use the selected plan and library for every completed CBS replay."""

    def __init__(self, library: ctypes.CDLL, plan: CBSPlan) -> None:
        compose = getattr(library, "generativeqc_cbs_compose_v1", None)
        if compose is None:
            raise RuntimeError("CBS requires the selected native composition owner")
        compose.argtypes = [
            ctypes.POINTER(_Plan),
            ctypes.POINTER(_Observables),
            ctypes.POINTER(_Composition),
        ]
        compose.restype = ctypes.c_int
        self._library = library
        self._compose = compose
        self._plan = _Plan(
            ctypes.sizeof(_Plan),
            _native.ABI_VERSION,
            (ctypes.c_double * 2)(*plan.hf_weights),
            (ctypes.c_double * 2)(*plan.mp2_weights),
            (ctypes.c_double * 6)(*plan.coefficients),
        )

    def __call__(
        self, energies: tuple[float, ...], forces: tuple[np.ndarray | None, ...] | None
    ) -> tuple[tuple[float, ...], np.ndarray | None]:
        """Return total/HF/MP2-correlation/delta blocks without Python arithmetic."""
        inputs = _Observables(ctypes.sizeof(_Observables), _native.ABI_VERSION)
        output = _Composition(ctypes.sizeof(_Composition), _native.ABI_VERSION)
        inputs.energy[:] = energies
        blocks = None
        if forces is not None:
            if len(forces) != 6 or forces[0] is None:
                raise ValueError("CBS composition requires six matching force blocks")
            shape = forces[0].shape
            if any(item is None or item.shape != shape for item in forces):
                raise ValueError("CBS composition requires six matching force blocks")
            arrays = tuple(
                np.require(item, dtype=np.float64, requirements=("C", "A"))
                for item in forces
            )
            inputs.force_count = arrays[0].size
            inputs.forces[:] = [item.ctypes.data_as(_DoublePointer) for item in arrays]
            blocks = np.empty((4, *arrays[0].shape), dtype=np.float64)
            output.force_blocks = blocks.ctypes.data_as(_DoublePointer)
            output.force_capacity = blocks.size
        _native.check(
            self._library,
            self._compose(
                ctypes.byref(self._plan), ctypes.byref(inputs), ctypes.byref(output)
            ),
        )
        return tuple(output.energy), blocks


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
