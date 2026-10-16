"""Native-free registry ABI doubles for facade policy tests, not scientific evidence."""

from __future__ import annotations

import ctypes
from typing import Any

from generativeqc import _generated_methods as manifest
from generativeqc import _native


def install_registry_stub(
    library: Any, properties: Any, *, supports_batch: bool = True
) -> None:
    """Supply selected-library metadata without replacing global public discovery.

    Carrier IDs/families follow the public manifest. A property set, or a callable
    from carrier ID to a property set, controls the test's declared provider view.
    The absent named-selector protocol deliberately exercises old-library numeric
    compatibility; real named lookup/provenance is covered by compiled tests.
    """
    families = {
        "hartree_fock": _native.METHOD_FAMILY_HARTREE_FOCK,
        "density_functional": _native.METHOD_FAMILY_DENSITY_FUNCTIONAL,
        "coupled_cluster": _native.METHOD_FAMILY_COUPLED_CLUSTER,
        "perturbation": _native.METHOD_FAMILY_PERTURBATION,
        "semiempirical": _native.METHOD_FAMILY_SEMIEMPIRICAL,
    }

    def query(method: int, output: Any) -> int:
        metadata = manifest.METHOD_METADATA[manifest.METHOD_ID_TO_NAME[method]]
        enabled = properties(method) if callable(properties) else properties
        record = ctypes.cast(
            output, ctypes.POINTER(_native.MethodCapabilitiesDescriptor)
        ).contents
        record.method = method
        record.family = families[metadata["family"]]
        record.available = 1
        record.supports_batch = int(supports_batch)
        record.supported_properties = (
            _native.PROPERTY_ENERGY if "energy" in enabled else 0
        ) | (_native.PROPERTY_FORCES if "forces" in enabled else 0)
        return _native.STATUS_SUCCESS

    library.generativeqc_method_from_name = None
    library.generativeqc_method_get_capabilities = query
