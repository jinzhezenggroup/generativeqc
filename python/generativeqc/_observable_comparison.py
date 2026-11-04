"""Transport complete observable arrays to the selected native numerical owner.

Typed shape/representation validation stays here; reductions, relative allowances
and comparison decisions have no portable production fallback.
"""

from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native

if TYPE_CHECKING:
    from .accuracy import ObservableTarget, TargetAccuracy


class _Context(ctypes.Structure):
    """Copied requirements and scalar values; Cartesian storage is borrowed."""

    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("struct_size", "abi_version", "requested_mask", "reserved")
        ]
        + [("atom_count", ctypes.c_uint64)]
        + [
            ("energy", ctypes.c_double),
            ("reference_energy", ctypes.c_double),
            ("absolute", ctypes.c_double * 3),
            ("relative", ctypes.c_double * 3),
        ]
    )


class _Comparison(ctypes.Structure):
    """Fixed three-slot numerical receipt, never a convergence certificate."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "struct_size",
            "abi_version",
            "policy_version",
            "measured_mask",
            "met_mask",
            "reserved",
        )
    ] + [
        (name, ctypes.c_double * 3)
        for name in ("differences", "reference_norms", "allowances")
    ]


def require_observable_comparison(library: Any) -> Any:
    """Reject incompatible selected libraries without performing numerical work."""
    query = getattr(library, "generativeqc_observable_comparison_query_v1", None)
    if query is None:
        raise NotImplementedError(
            "native library lacks required observable comparison; rebuild or install a current library"
        )
    return query


def _observable_array(name: str, supplied: Any) -> np.ndarray:
    """Preserve typed finite FP64 scalar/Cartesian validation without reducing."""
    raw = np.asarray(supplied)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("observables must be finite real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("observables exceed finite FP64 representation")
    if name == "energy" and value.shape != ():
        raise ValueError("energy must be a scalar")
    if name == "forces" and (value.ndim != 2 or value.shape[1] != 3 or not len(value)):
        raise ValueError("forces must have shape (Natom, 3)")
    return value


def measure_observables(
    target: TargetAccuracy, values: dict, reference_values: dict, *, library: Any = None
) -> tuple[tuple[ObservableTarget, float, float, float], ...]:
    """Require one atomic native norm/allowance receipt for the entire request.

    Model/scope/convergence admission and strict-reference uncertainty remain
    endpoint concerns. The public offline evidence view retains its typed scalar
    assessment; it does not supply a replacement runtime array reduction.
    """
    if library is None:
        library = _native.load_library()
    query = require_observable_comparison(library)
    context = _Context(ctypes.sizeof(_Context), _native.ABI_VERSION)
    slots = {("energy", "absolute"): 0, ("forces", "max_abs"): 1, ("forces", "rms"): 2}
    arrays = {}
    for requirement in target.observables:
        slot = slots[(requirement.observable, requirement.norm)]
        context.requested_mask |= 1 << slot
        context.absolute[slot] = requirement.absolute
        context.relative[slot] = requirement.relative
        name = requirement.observable
        if name not in arrays:
            value = _observable_array(name, values[name])
            reference = _observable_array(name, reference_values[name])
            if value.shape != reference.shape:
                raise ValueError("observable dimensions differ")
            arrays[name] = tuple(
                np.require(array, dtype=np.float64, requirements=("C", "A"))
                for array in (value, reference)
            )
    if "energy" in arrays:
        context.energy, context.reference_energy = (
            float(array) for array in arrays["energy"]
        )
    pointer = ctypes.POINTER(ctypes.c_double)
    forces = reference_forces = None
    if "forces" in arrays:
        force_array, reference_array = arrays["forces"]
        context.atom_count = len(force_array)
        forces, reference_forces = (
            array.ctypes.data_as(pointer) for array in (force_array, reference_array)
        )
    query.argtypes = [
        ctypes.POINTER(_Context),
        pointer,
        pointer,
        ctypes.POINTER(_Comparison),
    ]
    query.restype = ctypes.c_int
    comparison = _Comparison(ctypes.sizeof(_Comparison), _native.ABI_VERSION)
    requested_mask = context.requested_mask
    status = query(
        ctypes.byref(context), forces, reference_forces, ctypes.byref(comparison)
    )
    if status == _native.STATUS_INVALID_ARGUMENT:
        raise ValueError(
            "native observable comparison rejected values, dimensions or allowances"
        )
    _native.check(library, status)
    if (
        comparison.struct_size != ctypes.sizeof(_Comparison)
        or comparison.abi_version != _native.ABI_VERSION
        or comparison.policy_version != 1
        or comparison.measured_mask != requested_mask
        or comparison.met_mask & ~requested_mask
        or comparison.reserved
    ):
        raise RuntimeError(
            "native observable comparison returned an inconsistent receipt"
        )
    for slot in range(3):
        difference, reference_norm, allowance = (
            comparison.differences[slot],
            comparison.reference_norms[slot],
            comparison.allowances[slot],
        )
        selected = bool(requested_mask & (1 << slot))
        if (
            any(
                not math.isfinite(value) or value < 0
                for value in (difference, reference_norm, allowance)
            )
            or (not selected and (difference or reference_norm or allowance))
            or (
                selected
                and bool(comparison.met_mask & (1 << slot)) != (difference <= allowance)
            )
        ):
            raise RuntimeError(
                "native observable comparison returned invalid numerical evidence"
            )
    return tuple(
        (
            requirement,
            comparison.differences[slots[(requirement.observable, requirement.norm)]],
            comparison.reference_norms[
                slots[(requirement.observable, requirement.norm)]
            ],
            comparison.allowances[slots[(requirement.observable, requirement.norm)]],
        )
        for requirement in target.observables
    )
