"""Typed metadata transport for prepared warm-seed numeric capacity."""

from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np

from . import _native
from ._metadata_native import bind


class Capacity(ct.Structure):
    """Copied topology envelope; no seed or device storage is borrowed."""

    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("density_bytes", ct.c_uint64),
        ("coordinate_bytes", ct.c_uint64),
        ("total_bytes", ct.c_uint64),
        ("largest_density_bytes", ct.c_uint64),
    ]


_QUERY_ARGUMENTS = (ct.c_void_p, ct.POINTER(Capacity))
_QUERY_TYPE = ct.CFUNCTYPE(ct.c_int, *_QUERY_ARGUMENTS)


def seed_capacity(batch: Any) -> tuple[int, int] | None:
    """Return native bounds; only a missing protocol permits compatibility."""
    query = bind(
        batch._library, "generativeqc_batch_query_warm_seed_capacity_v1", _QUERY_TYPE
    )
    if query is None:
        return None
    query.argtypes = _QUERY_ARGUMENTS
    query.restype = ct.c_int
    output = Capacity(ct.sizeof(Capacity), _native.ABI_VERSION)
    _native.check(
        batch._library,
        query(batch._batch, ct.byref(output)),
        context=batch._context,
    )
    if (
        output.struct_size != ct.sizeof(Capacity)
        or output.abi_version != _native.ABI_VERSION
        or output.total_bytes != output.density_bytes + output.coordinate_bytes
        or output.largest_density_bytes > output.density_bytes
        or any(
            getattr(output, field) % 8
            for field in (
                "density_bytes",
                "coordinate_bytes",
                "total_bytes",
                "largest_density_bytes",
            )
        )
    ):
        raise RuntimeError("native warm-seed capacity returned an invalid receipt")
    return output.total_bytes, output.largest_density_bytes


class _SnapshotInfo(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("system_count", ct.c_uint32),
        ("reserved", ct.c_uint32),
        ("numeric_bytes", ct.c_uint64),
        ("largest_density_bytes", ct.c_uint64),
    ]


class _SeedView(ct.Structure):
    # The immutable native view preserves the established density layout. ctypes
    # cannot express const pointers; never expose these borrowed arrays to users.
    _fields_ = _native.HfWarmState._fields_


_CAPTURE = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    ct.c_uint64,
    ct.POINTER(_SnapshotInfo),
    ct.POINTER(ct.c_void_p),
)
_RESTORE = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)
_READ = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint32, ct.POINTER(_SeedView))
_DESTROY = ct.CFUNCTYPE(None, ct.c_void_p)


class SeedSnapshot:
    """Own one immutable native seed snapshot, independent of its source context.

    Same-model eligibility and resource reservations remain in the precision
    facade. Native code owns the arrays, copying and physical restore validation.
    """

    def __init__(self, batch: Any, maximum_bytes: int, capture: Any) -> None:
        """Capture only after all required lifecycle symbols are available."""
        self._handle = ct.c_void_p()
        self._library = batch._library
        functions = []
        for name, prototype in (
            ("generativeqc_batch_restore_warm_seeds_v1", _RESTORE),
            ("generativeqc_warm_seed_snapshot_read_v1", _READ),
            ("generativeqc_warm_seed_snapshot_destroy", _DESTROY),
        ):
            function = bind(self._library, name, prototype)
            if function is None:
                raise RuntimeError("native seed snapshot protocol is incomplete")
            function.argtypes, function.restype = (
                prototype._argtypes_,
                prototype._restype_,
            )
            functions.append(function)
        self._restore, self._read, self._destroy = functions
        info = _SnapshotInfo(ct.sizeof(_SnapshotInfo), _native.ABI_VERSION)
        _native.check(
            self._library,
            capture(
                batch._batch, maximum_bytes, ct.byref(info), ct.byref(self._handle)
            ),
            context=batch._context,
        )
        if (
            not self._handle
            or info.struct_size != ct.sizeof(_SnapshotInfo)
            or info.abi_version != _native.ABI_VERSION
            or info.system_count != batch.system_count
            or info.reserved
            or info.numeric_bytes > maximum_bytes
            or info.largest_density_bytes > info.numeric_bytes
        ):
            self.close()
            raise RuntimeError("native seed snapshot returned an invalid receipt")
        self.system_count = info.system_count
        self.numeric_bytes = info.numeric_bytes
        self.largest_density_bytes = info.largest_density_bytes

    def close(self) -> None:
        """Release host storage exactly once, including after source teardown."""
        handle, self._handle = self._handle, None
        if handle:
            self._destroy(handle)

    def __del__(self) -> None:
        """Release snapshots when their precision history slot is retired."""
        self.close()

    def restore(self, batch: Any) -> None:
        """Restore through the same loaded native runtime; retain reusable history."""
        if not self._handle:
            raise RuntimeError("native seed snapshot is closed")
        if getattr(self._library, "_handle", id(self._library)) != getattr(
            batch._library, "_handle", id(batch._library)
        ):
            raise ValueError("seed snapshot belongs to a different native library")
        from .checkpoint import _restore_seed_call

        status = _restore_seed_call(
            batch,
            lambda: self._restore(batch._batch, self._handle),
            native_managed=getattr(batch, "_resource_prepare", None) is not None,
        )
        _native.check(
            self._library,
            status,
            context=batch._context,
        )

    def diagnostic_arrays(self) -> list[np.ndarray]:
        """Return detached read-only copies for diagnostics, never production staging."""
        if not self._handle:
            raise RuntimeError("native seed snapshot is closed")
        arrays = []
        for index in range(self.system_count):
            view = _SeedView(ct.sizeof(_SeedView), _native.ABI_VERSION)
            _native.check(
                self._library, self._read(self._handle, index, ct.byref(view))
            )
            if not view.present:
                continue
            for pointer, count in (
                (view.density, view.density_count),
                (view.coordinates, view.coordinate_count),
            ):
                array = np.ctypeslib.as_array(pointer, shape=(count,)).copy()
                array.setflags(write=False)
                arrays.append(array)
        return arrays


def capture_snapshot(batch: Any, maximum_bytes: int) -> SeedSnapshot | None:
    """Use descriptor compatibility only when the capture protocol is absent."""
    capture = bind(batch._library, "generativeqc_batch_capture_warm_seeds_v1", _CAPTURE)
    if capture is None:
        return None
    # An explicitly looked-up CDLL symbol may have ctypes' default C-int
    # signature. Preserve its selection while declaring the uint64 limit ABI.
    capture.argtypes, capture.restype = _CAPTURE._argtypes_, _CAPTURE._restype_
    return SeedSnapshot(batch, maximum_bytes, capture)
