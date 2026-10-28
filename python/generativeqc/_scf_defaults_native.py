"""Typed transport for the shared native default-convergence policy."""

from __future__ import annotations

import ctypes as ct
import math
from functools import lru_cache
from typing import Any

from . import _native
from ._metadata_native import bind
from .basis import BasisSet


class _Context(ct.Structure):
    """Backend/method facts supplied by the already selected Calculator."""

    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("backend", ct.c_int32),
        ("method", ct.c_int32),
        ("has_ecp", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class _Controls(ct.Structure):
    """Copied solver defaults, never a replacement for explicit caller controls."""

    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("policy_version", ct.c_uint32),
        ("qualified_defaults", ct.c_uint32),
        ("energy_tolerance", ct.c_double),
        ("density_tolerance", ct.c_double),
        ("screening_tolerance", ct.c_double),
    ]


_QUERY_ARGUMENTS = (
    ct.POINTER(_Context),
    ct.POINTER(_native.KsOptionsDescriptor),
    ct.c_char_p,
    ct.c_size_t,
    ct.POINTER(_Controls),
)
_QUERY_TYPE = ct.CFUNCTYPE(ct.c_int, *_QUERY_ARGUMENTS)


@lru_cache(maxsize=16)
def _grid_options(grid: Any, schedule: str) -> Any:
    """Copy immutable metadata once; retain its borrowed radius array explicitly.

    Only the grid and schedule fields are consumed by this native query. This
    avoids lowering a scientific KS graph merely to ask for default controls.
    The bounded cache contains no geometry, density or device resource owner.
    """
    radii = (ct.c_double * 119)()
    for number, radius in grid.element_radii:
        radii[number] = radius
    descriptor = _native.KsOptionsDescriptor(
        struct_size=ct.sizeof(_native.KsOptionsDescriptor),
        abi_version=_native.ABI_VERSION,
        grid_version=grid.version,
        radial_points=grid.radial_points,
        angular_polar=grid.angular_polar,
        angular_azimuth=grid.angular_azimuth,
        partition_iterations=grid.partition_iterations,
        coincident_tolerance=grid.coincident_tolerance,
        element_radii=radii,
        element_radius_count=len(radii),
        xc_execution_schedule=_native.XC_EXECUTION_DEVICE_FUSED
        if schedule == "device_fused"
        else _native.XC_EXECUTION_HOST_UNFUSED,
    )
    descriptor._radii_owner = radii
    return descriptor


def default_controls(calculator: Any) -> dict[str, float] | None:
    """Use native defaults, retaining one immutable metadata answer per owner.

    Repeated precision decisions need not allocate fresh ctypes descriptors.
    The key includes the selected query and every consumed scientific fact;
    replacing a backend, model, grid or library invalidates the copied answer.
    This cache holds no geometry, seed or device resource.
    """
    library = getattr(calculator, "_library", None)
    query = bind(library, "generativeqc_scf_defaults_query_v1", _QUERY_TYPE)
    if query is None:
        return None
    cuda = getattr(calculator, "_device_name", None) == "cuda"
    method = getattr(calculator, "_method", -1)
    basis = getattr(calculator, "_basis", None)
    has_ecp = isinstance(basis, BasisSet) and any(
        element.ecp_core_electrons for element in basis.elements
    )
    options = getattr(calculator, "_ks_options", None) if cuda else None
    key = (
        query,
        cuda,
        method,
        has_ecp,
        None if options is None else options.grid,
        None if options is None else options.xc_schedule,
        None if options is None else options.method_ir.identity,
    )
    cached = getattr(calculator, "_native_scf_defaults_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1].copy()
    context = _Context(ct.sizeof(_Context), _native.ABI_VERSION)
    context.backend = _native.BACKEND_CUDA if cuda else _native.BACKEND_CPU_REFERENCE
    context.method = method if type(method) is int else -1
    context.has_ecp = has_ecp
    descriptor = None
    identity = b""
    if options is not None:
        descriptor = _grid_options(options.grid, options.xc_schedule)
        identity = options.method_ir.identity.encode("ascii")
    output = _Controls(ct.sizeof(_Controls), _native.ABI_VERSION)
    query.argtypes = _QUERY_ARGUMENTS
    query.restype = ct.c_int
    _native.check(
        library,
        query(
            ct.byref(context),
            None if descriptor is None else ct.byref(descriptor),
            identity,
            len(identity),
            ct.byref(output),
        ),
    )
    result = {
        name: getattr(output, name)
        for name in ("energy_tolerance", "density_tolerance", "screening_tolerance")
    }
    if (
        output.struct_size != ct.sizeof(_Controls)
        or output.abi_version != _native.ABI_VERSION
        or output.policy_version != 1
        or output.qualified_defaults not in (0, 1)
        or any(not math.isfinite(value) or value <= 0 for value in result.values())
    ):
        raise RuntimeError("native SCF default policy returned an invalid receipt")
    calculator._native_scf_defaults_cache = (key, result.copy())
    return result
