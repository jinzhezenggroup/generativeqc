"""Bind GFN2's unchanged ragged kernel envelopes to shared weighted-Gram code."""

from generativeqc_compiler.tensor.weighted_gram import canonical_regions
from generativeqc_compiler.tensor.weighted_gram_emit import (
    CheckedPairBindings,
    emit_checked_pair,
)


def emit_gfn2_density_contract() -> str:
    """Method policy contributes only buffer names and failure destinations."""
    bindings = CheckedPairBindings(
        coefficients="input.coefficients",
        weights="workspace.weights",
        energy_weights="workspace.energy_weights",
        density_output="workspace.density_scratch",
        weighted_output="workspace.weighted_density_scratch",
        scale="generativeqc::xtb::generated::gfn2_weighted_coefficient_cuda_tensor",
        contribution="generativeqc::xtb::generated::gfn2_density_contribution_cuda_tensor",
        update="generativeqc::xtb::generated::gfn2_density_update_cuda_tensor",
        density_failure=(
            "record_system_error(system_errors, system, device_error,\n"
            "                            Gfn2DensityDeviceError::kNonfiniteDensityArithmetic);"
        ),
        weighted_failure=(
            "record_system_error(system_errors, system, device_error,\n"
            "                            Gfn2DensityDeviceError::kNonfiniteWeightedDensityArithmetic);"
        ),
    )
    return emit_checked_pair(*canonical_regions("cuda", orbital_count=3), bindings)
