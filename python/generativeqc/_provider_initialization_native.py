"""Optional intent transport on an already loaded native library; no loader."""

from __future__ import annotations

import ctypes
from typing import Any

from .provider_initialization import ProviderInitializationIntent


class ProviderInitializationIntentDescriptor(ctypes.Structure):
    """Standalone v1 C ABI; all identity fields deliberately remain native-only."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("allowance_bytes", ctypes.c_uint64),
    ]


def prepare(
    library: Any,
    context: Any,
    system: Any,
    method: Any,
    output: Any,
    intent: ProviderInitializationIntent | None,
) -> int:
    """Copy finite intent to the optional ABI or use the unchanged legacy call.

    This function never loads a library, certifies a device or creates an owner.
    An explicit request cannot silently fall back when transport is unavailable.
    """
    if intent is None:
        return int(
            library.generativeqc_calculation_prepare(context, system, method, output)
        )
    if not isinstance(intent, ProviderInitializationIntent):
        raise TypeError(
            "provider initialization intent must be an immutable intent value"
        )
    try:
        version = library.generativeqc_provider_initialization_intent_version
        capabilities = library.generativeqc_provider_initialization_capabilities_v1
        native_prepare = (
            library.generativeqc_calculation_prepare_with_provider_initialization_v1
        )
    except AttributeError as error:
        raise NotImplementedError(
            "native provider initialization intent transport is unavailable"
        ) from error
    version.argtypes = []
    version.restype = ctypes.c_uint32
    capabilities.argtypes = []
    capabilities.restype = ctypes.c_uint32
    if int(version()) != 1 or not int(capabilities()) & 3:
        raise NotImplementedError(
            "native provider initialization intent ABI/domain is unavailable"
        )
    native_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ProviderInitializationIntentDescriptor),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    native_prepare.restype = ctypes.c_int
    value = ProviderInitializationIntentDescriptor(
        ctypes.sizeof(ProviderInitializationIntentDescriptor),
        1,
        1,
        0,
        intent.allowance_bytes,
    )
    return int(native_prepare(context, system, method, ctypes.byref(value), output))
