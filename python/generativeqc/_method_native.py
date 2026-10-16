"""Method registry presentation bound to the caller's already-selected library."""

from __future__ import annotations

import ctypes
from typing import Any

from . import _generated_methods as _manifest
from . import _native
from ._api_types import MethodCapabilities


def capabilities(library: Any, public_name: str, method_id: int) -> MethodCapabilities:
    """Query one native carrier without discovering or loading another library.

    Python still owns public aliases, explicit compiler compositions and metadata
    preflight. The existing native named selector verifies the expected carrier;
    an older library without that optional ABI retains numeric-ID compatibility.
    A selected lookup/query failure or inconsistent publication never falls back
    to another library, cached public discovery or a Python capability table.
    This backend-neutral registry view is not per-system scientific admission.
    """
    lookup = getattr(library, "generativeqc_method_from_name", None)
    if lookup is not None:
        lookup.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        lookup.restype = ctypes.c_int
        resolved = ctypes.c_int32()
        canonical = _manifest.METHOD_ID_TO_NAME[method_id].encode("ascii")
        _native.check(library, lookup(canonical, ctypes.byref(resolved)))
        if resolved.value != method_id:
            raise RuntimeError(
                "selected native method selector disagrees with ABI metadata"
            )
        method_id = resolved.value
    native = _native.MethodCapabilitiesDescriptor(
        ctypes.sizeof(_native.MethodCapabilitiesDescriptor), _native.ABI_VERSION
    )
    _native.check(
        library,
        library.generativeqc_method_get_capabilities(method_id, ctypes.byref(native)),
    )
    families = {
        _native.METHOD_FAMILY_HARTREE_FOCK: "hartree_fock",
        _native.METHOD_FAMILY_DENSITY_FUNCTIONAL: "density_functional",
        _native.METHOD_FAMILY_COUPLED_CLUSTER: "coupled_cluster",
        _native.METHOD_FAMILY_PERTURBATION: "perturbation",
        _native.METHOD_FAMILY_SEMIEMPIRICAL: "semiempirical",
    }
    if (
        native.struct_size < ctypes.sizeof(native)
        or native.abi_version != _native.ABI_VERSION
        or native.method != method_id
        or native.family not in families
    ):
        raise RuntimeError(
            "selected native method registry returned inconsistent metadata"
        )
    properties = set()
    if native.supported_properties & _native.PROPERTY_ENERGY:
        properties.add("energy")
    if native.supported_properties & _native.PROPERTY_FORCES:
        properties.add("forces")
    return MethodCapabilities(
        method=public_name,
        family=families[native.family],
        available=bool(native.available),
        supports_batch=bool(native.supports_batch),
        supported_properties=frozenset(properties),
    )
