"""Marshalling for the native CPU retained-vector owner, without vector arithmetic."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from . import _native
from .elements import checked_integer


def _check(library: Any, status: int) -> None:
    """Keep invalid public vector data a ValueError without numerical retry."""
    if status == 1:
        raise ValueError("invalid native recycle input, extent or generation")
    _native.check(library, status)


def capacity(library: Any, dimension: int, max_vectors: int, max_bytes: int) -> Any:
    """Query native sizing before creating a handle or dimension-sized buffers."""
    descriptor = _native.ResponseRecycleDescriptor(
        struct_size=ctypes.sizeof(_native.ResponseRecycleDescriptor),
        abi_version=_native.ABI_VERSION,
        dimension=checked_integer(
            dimension, "recycle dimension", low=1, high=2**64 - 1
        ),
        max_vectors=checked_integer(
            min(max_vectors, dimension), "recycle vectors", low=1, high=2**64 - 1
        ),
        max_retained_bytes=checked_integer(
            min(max_bytes, 2**64 - 1), "recycle bytes", low=1, high=2**64 - 1
        ),
    )
    result = _native.ResponseRecycleCapacity(
        struct_size=ctypes.sizeof(_native.ResponseRecycleCapacity),
        abi_version=_native.ABI_VERSION,
    )
    _check(
        library,
        library.generativeqc_response_recycle_capacity_v1_get(
            ctypes.byref(descriptor), ctypes.byref(result)
        ),
    )
    return descriptor, result


def _columns(values: np.ndarray) -> Any:
    """Borrow signed element strides without reordering an entire Arnoldi panel."""
    if values.ndim != 2 or values.dtype != np.float64 or not values.flags.aligned:
        raise ValueError("native recycle columns require an aligned FP64 matrix")
    if any(stride % values.itemsize for stride in values.strides):
        raise ValueError("native recycle strides must address complete FP64 elements")
    rows, count = values.shape
    row_stride, column_stride = (stride // values.itemsize for stride in values.strides)
    for stride in (row_stride, column_stride):
        checked_integer(stride, "recycle stride", low=-(2**63), high=2**63 - 1)
    low = (
        min(0, (rows - 1) * row_stride) + min(0, (count - 1) * column_stride)
        if rows and count
        else 0
    )
    high = (
        max(0, (rows - 1) * row_stride) + max(0, (count - 1) * column_stride)
        if rows and count
        else -1
    )
    elements = checked_integer(high - low + 1, "recycle extent", low=0, high=2**64 - 1)
    return _native.ResponseColumns(
        struct_size=ctypes.sizeof(_native.ResponseColumns),
        abi_version=_native.ABI_VERSION,
        storage=ctypes.cast(
            values.ctypes.data + low * values.itemsize, ctypes.POINTER(ctypes.c_double)
        ),
        element_count=elements,
        rows=rows,
        columns=count,
        origin=-low,
        row_stride=row_stride,
        column_stride=column_stride,
    )


class NativeRecycle:
    """Move-style handle; only explicit diagnostic copies cross back into Python."""

    def __init__(
        self, library: Any, dimension: int, max_vectors: int, max_bytes: int
    ) -> None:
        self.library = library
        self.dimension = dimension
        self.limits = (max_vectors, max_bytes)
        descriptor, self.capacity = capacity(library, dimension, max_vectors, max_bytes)
        self.owner = ctypes.c_void_p()
        try:
            _check(
                library,
                library.generativeqc_response_recycle_create_v1(
                    ctypes.byref(descriptor), ctypes.byref(self.owner)
                ),
            )
        except BaseException:
            self.close()
            raise

    def state(self) -> Any:
        """Read scalar telemetry without materializing retained vectors."""
        if not self.owner:
            raise RuntimeError("native recycle owner is closed")
        result = _native.ResponseRecycleState(
            struct_size=ctypes.sizeof(_native.ResponseRecycleState),
            abi_version=_native.ABI_VERSION,
        )
        _check(
            self.library,
            self.library.generativeqc_response_recycle_state_v1_get(
                self.owner, ctypes.byref(result)
            ),
        )
        return result

    def project(self, rhs: np.ndarray) -> np.ndarray:
        """Publish a detached guess after one synchronous native projection."""
        columns = _columns(rhs.reshape((-1, 1)))
        state = self.state()
        result = np.empty(self.dimension, dtype=np.float64)
        _check(
            self.library,
            self.library.generativeqc_response_recycle_project_v1(
                self.owner,
                state.generation,
                ctypes.byref(columns),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                result.size,
                self.capacity.projection_workspace_bytes,
            ),
        )
        return result

    def update(
        self, solution: np.ndarray, basis: np.ndarray, max_vectors: int, max_bytes: int
    ) -> None:
        """Borrow solution/Arnoldi views; native state replaces itself atomically."""
        solved = _columns(solution.reshape((-1, 1)))
        columns = _columns(basis)
        state = self.state()
        descriptor, next_capacity = capacity(
            self.library, self.dimension, max_vectors, max_bytes
        )
        _check(
            self.library,
            self.library.generativeqc_response_recycle_update_v1(
                self.owner,
                state.generation,
                ctypes.byref(solved),
                ctypes.byref(columns),
                descriptor.max_vectors,
                descriptor.max_retained_bytes,
                next_capacity.update_workspace_bytes,
            ),
        )
        self.capacity = next_capacity
        self.limits = (max_vectors, max_bytes)

    def iter_vectors(self) -> Any:
        """Internal stream through one reusable buffer; callers must consume before advancing."""
        state = self.state()
        if not state.vector_count:
            return
        value = np.empty(self.dimension, dtype=np.float64)
        for column in range(state.vector_count):
            _check(
                self.library,
                self.library.generativeqc_response_recycle_copy_v1(
                    self.owner,
                    state.generation,
                    column,
                    1,
                    value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    value.size,
                ),
            )
            yield value

    def vectors(self) -> list[np.ndarray]:
        """Detach one immutable column at a time, bounding copy/freeze overlap."""
        return [immutable(value) for value in self.iter_vectors()]

    def close(self) -> None:
        """Release native vectors while leaving detached publications valid."""
        if getattr(self, "owner", None):
            self.library.generativeqc_response_recycle_destroy(self.owner)
            self.owner = ctypes.c_void_p()

    def __del__(self) -> None:
        self.close()
