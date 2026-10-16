"""Compact shell-count marshalling, not basis expansion or provider admission."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

from generativeqc_compiler.common.resources import checked_bytes

from .elements import checked_integer

if TYPE_CHECKING:
    from collections.abc import Sequence

_SHAPE_KEYS = ("nbf", "cartesian_nbf", "shells", "primitives", "maximum_angular")


def _portable_basis_shape(
    angular: Sequence[int], primitives: Sequence[int], representation: str
) -> dict[str, int]:
    """Retain scalar metadata sizing for absent/older libraries and CUDA plans."""
    cartesian = sum((value + 1) * (value + 2) // 2 for value in angular)
    public = (
        cartesian
        if representation == "cartesian"
        else sum(2 * value + 1 for value in angular)
    )
    return dict(
        zip(
            _SHAPE_KEYS,
            map(
                checked_bytes,
                (public, cartesian, len(angular), sum(primitives), max(angular)),
            ),
            strict=True,
        )
    )


def _basis_shape(
    angular: Sequence[int],
    primitives: Sequence[int],
    representation: str,
    *,
    library: Any = None,
) -> dict[str, int]:
    """Query the already-loaded native owner without discovering a backend.

    A discovered query's rejection is authoritative; only missing symbols permit
    portable sizing. Rows describe expanded contractions, including repeated
    atoms and zero-coefficient primitive entries retained by basis expansion.
    """
    if representation not in ("cartesian", "spherical"):
        raise ValueError("invalid basis topology representation")
    if not len(angular) or len(angular) != len(primitives):
        raise ValueError("basis topology requires matching nonempty shell rows")
    maximum = min(2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1, 2**63 - 1)
    checked_integer(len(angular), "shell count", low=1, high=maximum)
    angular = tuple(
        checked_integer(value, "shell angular momentum", low=0, high=2**32 - 1)
        for value in angular
    )
    primitives = tuple(
        checked_integer(value, "shell primitive count", low=1, high=maximum)
        for value in primitives
    )
    query = getattr(library, "generativeqc_resource_basis_shape_v1", None)
    if query is None:
        return _portable_basis_shape(angular, primitives, representation)
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    output = (ctypes.c_uint64 * len(_SHAPE_KEYS))()
    if query(
        (ctypes.c_uint32 * len(angular))(*angular),
        (ctypes.c_uint64 * len(primitives))(*primitives),
        len(angular),
        int(representation == "spherical"),
        output,
        len(output),
    ):
        raise ValueError("native basis topology rejected shape or count overflow")
    return dict(zip(_SHAPE_KEYS, map(checked_bytes, output), strict=True))
