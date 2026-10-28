"""Typed metadata transport for prepared warm-seed numeric capacity."""

from __future__ import annotations

import ctypes as ct
from typing import Any

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
