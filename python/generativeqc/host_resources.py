"""Opt-in native shared reservations for qualified prepared CPU endpoints."""

from __future__ import annotations

import ctypes as ct
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Self
from weakref import finalize

from . import _native

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class HostResourceSnapshot:
    """Modeled retained/workspace reservations, not measured allocator bytes."""

    maximum_bytes: int
    retained_bytes: int
    workspace_bytes: int
    owner_count: int


class _Snapshot(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("maximum_bytes", ct.c_uint64),
        ("retained_bytes", ct.c_uint64),
        ("workspace_bytes", ct.c_uint64),
        ("owner_count", ct.c_uint64),
    ]


class HostResources:
    """Share a modeled host cap across qualified CPU KS prepared batches.

    Pass it to ``Calculator.prepare_batch`` with a matching native resource
    plan. Retained capacity adds across live batches; preparations and replays
    on the shared cap serialize their scratch. Closing this metadata handle
    does not close a prepared batch or revoke its lease. This does not cap RSS.
    """

    def __init__(self, maximum_bytes: int) -> None:
        """Create a shared cap in modeled bytes, without preparing a calculation."""
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes < 2**64:
            raise ValueError("maximum_bytes must be a nonnegative uint64")
        library = _native.load_library(device="cpu")
        try:
            create = library.generativeqc_host_resources_create_v1
            destroy = library.generativeqc_host_resources_destroy_v1
            snapshot = library.generativeqc_host_resources_snapshot_v1
        except AttributeError as error:
            raise NotImplementedError(
                "native shared host resources are unavailable"
            ) from error
        create.argtypes = [ct.c_uint64, ct.POINTER(ct.c_void_p)]
        create.restype = ct.c_int
        destroy.argtypes = [ct.c_void_p]
        destroy.restype = None
        snapshot.argtypes = [ct.c_void_p, ct.POINTER(_Snapshot)]
        snapshot.restype = ct.c_int
        self._library = library
        self._handle = ct.c_void_p()
        self._lock = Lock()
        self._snapshot = snapshot
        _native.check(library, create(maximum_bytes, ct.byref(self._handle)))
        self._close = finalize(self, destroy, self._handle)

    @contextmanager
    def _borrow(self, library: ct.CDLL) -> Iterator[ct.c_void_p]:
        """Pin the metadata handle for one synchronous native borrow."""
        with self._lock:
            if not self._close.alive:
                raise RuntimeError("host resources are closed")
            if self._library._handle != library._handle:
                raise ValueError(
                    "host resources and calculator use different native libraries"
                )
            yield self._handle

    def snapshot(self) -> HostResourceSnapshot:
        """Read one native snapshot of the live leases."""
        output = _Snapshot(ct.sizeof(_Snapshot), _native.ABI_VERSION)
        with self._borrow(self._library) as handle:
            _native.check(self._library, self._snapshot(handle, ct.byref(output)))
        return HostResourceSnapshot(
            output.maximum_bytes,
            output.retained_bytes,
            output.workspace_bytes,
            output.owner_count,
        )

    def close(self) -> None:
        """Release metadata; prepared owners retain their native reservations."""
        with self._lock:
            self._close()

    def __enter__(self) -> Self:
        """Return the open cap for use in a context manager."""
        with self._borrow(self._library):
            return self

    def __exit__(self, *_exc: object) -> None:
        """Close metadata without revoking leases or suppressing exceptions."""
        self.close()
