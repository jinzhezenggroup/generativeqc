"""Hessian helpers and a bounded native CPU RHF integration (issue #180).

NativeRHFState and analytic_hessian form the native tools path. Numerical and
PySCF reference oracles are separate explicitly imported validation modules;
none of them supplies a state or derivative to the native Hessian calculation.
"""

import typing

from generativeqc.stationary_nuclear import (
    build_stationary_nuclear_rhs,
    metric_density_response_mo,
)

from .assembly import assemble_frozen_skeleton, validate_hessian_component
from .numerical import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)
from .weights import two_electron_energy, two_electron_weight, weight_energy

# Loading mathematical helpers does not load the optional native library.
_LAZY = {
    "NativeRHFState": "native",
    "DirectionalRHFResponse": "directional",
    "directional_rhf_response": "directional",
    "DirectionalRKSBatchResponse": "generativeqc.rks_hessian_directional",
    "DirectionalRKSResponse": "generativeqc.rks_hessian_directional",
    "RKSXCHVPComponents": "generativeqc.rks_hessian_directional",
    "directional_rks_response": "generativeqc.rks_hessian_directional",
    "directional_rks_responses": "generativeqc.rks_hessian_directional",
    "native_rks_xc_hvp_components": "generativeqc.rks_hessian_directional",
    "RKSHessianResult": "generativeqc.rks_hessian",
    "RKSHVPBatchResult": "generativeqc.rks_hessian",
    "RKSHVPResult": "generativeqc.rks_hessian",
    "rks_hessian": "generativeqc.rks_hessian",
    "rks_hvp": "generativeqc.rks_hessian",
    "rks_hvp_many": "generativeqc.rks_hessian",
    "RHFHVPResult": "hvp",
    "rhf_hvp": "hvp",
    "RHFHVPBlockResult": "block",
    "RHFHessianResult": "block",
    "rhf_hvp_many": "block",
    "rhf_hessian": "block",
    "generated_directional_first_order": "first_order",
    "generated_directional_first_order_cuda": "first_order_cuda",
    "analytic_hessian": "analytic",
    "build_reference": "analytic",
    "cphf_relaxation": "analytic",
    "nuclear_closed_form": "analytic",
    "provider_components": "analytic",
    "StationaryNuclearResponse": "perturbation",
    "StationaryNuclearBatchResponse": "perturbation",
    "solve_stationary_nuclear_perturbation": "perturbation",
    "solve_stationary_nuclear_perturbations": "perturbation",
    "StationaryHVPContext": "generativeqc.second_order",
    "StationaryHVPContributor": "generativeqc.second_order",
    "StationaryPerturbationProvider": "generativeqc.second_order",
    "StationaryResponseDriver": "generativeqc.second_order",
    "StationarySecondOrderExecutor": "generativeqc.second_order",
    "StationarySecondOrderResult": "generativeqc.second_order",
}

__all__ = [
    "DirectionalRHFResponse",
    "DirectionalRKSBatchResponse",
    "DirectionalRKSResponse",
    "NativeRHFState",
    "RHFHVPBlockResult",
    "RHFHVPResult",
    "RHFHessianResult",
    "RKSHVPBatchResult",
    "RKSHVPResult",
    "RKSHessianResult",
    "RKSXCHVPComponents",
    "StationaryHVPContext",
    "StationaryHVPContributor",
    "StationaryNuclearBatchResponse",
    "StationaryNuclearResponse",
    "StationaryPerturbationProvider",
    "StationaryResponseDriver",
    "StationarySecondOrderExecutor",
    "StationarySecondOrderResult",
    "analytic_hessian",
    "assemble_frozen_skeleton",
    "build_reference",
    "build_stationary_nuclear_rhs",
    "cphf_relaxation",
    "directional_rhf_response",
    "directional_rks_response",
    "directional_rks_responses",
    "forces_to_gradient",
    "generated_directional_first_order",
    "generated_directional_first_order_cuda",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_translation_error",
    "metric_density_response_mo",
    "native_rks_xc_hvp_components",
    "nuclear_closed_form",
    "numerical_hessian",
    "provider_components",
    "rhf_hessian",
    "rhf_hvp",
    "rhf_hvp_many",
    "rks_hessian",
    "rks_hvp",
    "rks_hvp_many",
    "solve_stationary_nuclear_perturbation",
    "solve_stationary_nuclear_perturbations",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]


def __getattr__(name: typing.Any) -> typing.Any:
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        # Runtime owners use absolute names; tools-only implementations stay local.
        return getattr(
            importlib.import_module(
                module if "." in module else f".{module}", __name__
            ),
            name,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
