"""Native-free registry ABI doubles for facade policy tests, not scientific evidence."""

from __future__ import annotations

import ctypes
from typing import Any

from generativeqc import _generated_methods as manifest
from generativeqc import _native


def install_correlation_policy_stub(library: Any) -> None:
    """Inject a fixed strict receipt, never infer stopping limits from a budget.

    Registry/facade fixtures do not qualify numerical policy. Tests of actual
    stopping selection and molecular endpoints use the compiled native owner.
    """
    from generativeqc._correlation_convergence import _Limits

    def query(budget: Any, output: Any) -> int:
        record = ctypes.cast(output, ctypes.POINTER(_Limits)).contents
        record.scale = 0
        for name, value in (
            ("energy_tolerance", 1e-11),
            ("density_tolerance", 1e-11),
            ("ccsd_energy_tolerance", 1e-8),
            ("ccsd_residual_tolerance", 1e-9),
            ("lambda_residual_tolerance", 1e-9),
            ("response_absolute_tolerance", 1e-12),
            ("orbital_residual_tolerance", 1e-10),
            ("stationarity_tolerance", 1e-8),
        ):
            setattr(record, name, value)
        return _native.STATUS_SUCCESS

    library.generativeqc_correlation_convergence_query_v1 = query


def install_scf_defaults_stub(
    library: Any,
    controls: tuple[float, float, float] = (1e-10, 1e-8, 1e-12),
    *,
    qualified: bool = False,
) -> None:
    """Inject a fixed metadata receipt, not a second model/grid selection policy.

    Facade tests choose their answer explicitly. Real native policy admission
    and numerical endpoints are exercised by the compiled companion tests.
    """
    from generativeqc._scf_defaults_native import _Controls

    def query(context: Any, options: Any, identity: Any, size: int, output: Any) -> int:
        record = ctypes.cast(output, ctypes.POINTER(_Controls)).contents
        record.policy_version = 1
        record.qualified_defaults = int(qualified)
        (
            record.energy_tolerance,
            record.density_tolerance,
            record.screening_tolerance,
        ) = controls
        return _native.STATUS_SUCCESS

    library.generativeqc_scf_defaults_query_v1 = query


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
    if not hasattr(library, "generativeqc_correlation_convergence_query_v1"):
        install_correlation_policy_stub(library)
    if not hasattr(library, "generativeqc_scf_defaults_query_v1"):
        install_scf_defaults_stub(library)
