"""Bind GFN2's unchanged ragged kernel envelopes to shared weighted-Gram code."""

from generativeqc_compiler.tensor.weighted_gram import canonical_regions
from generativeqc_compiler.tensor.weighted_gram_emit import (
    CheckedPairBindings,
    emit_checked_pair,
)


def _instrument_checked_pairs(source: str) -> str:
    """Count visits at the selected schedule's executed loop boundaries.

    These additions have no arithmetic effect. Fail closed if the shared
    emitter changes its schedule rather than silently emitting stale counts.
    """
    insertions = (
        (
            "    const MatrixPair indices = matrix_pair(pair);",
            "    ++density_pairs_visited;",
            False,
        ),
        (
            "      const double first = @COEFFICIENTS@[matrix_begin + indices.row * count + local];",
            "      ++density_plain_visits;",
            True,
        ),
        (
            "      double weighted_left = 0.0;",
            "      ++density_plain_completed;\n      ++density_weighted_visits;",
            True,
        ),
        (
            "      weighted_density = weighted_updated;",
            "      ++density_weighted_completed;",
            False,
        ),
        ("    if (finite) {", "    if (!finite) ++density_failed_pairs;", True),
        (
            "      @WEIGHTED_OUTPUT@[second] = weighted_density;",
            "      ++density_published_pairs;",
            False,
        ),
    )
    # Bindings have already been substituted by emit_checked_pair().
    for marker, addition, before in insertions:
        bound_marker = marker.replace("@COEFFICIENTS@", "input.coefficients").replace(
            "@WEIGHTED_OUTPUT@", "workspace.weighted_density_scratch"
        )
        if source.count(bound_marker) != 1:
            raise ValueError(f"weighted-Gram diagnostic anchor changed: {bound_marker}")
        source = source.replace(
            bound_marker,
            addition + "\n" + bound_marker
            if before
            else bound_marker + "\n" + addition,
            1,
        )
    return source


def emit_gfn2_density_contract(*, instrumented: bool = False) -> str:
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
    source = emit_checked_pair(*canonical_regions("cuda", orbital_count=3), bindings)
    return _instrument_checked_pairs(source) if instrumented else source
