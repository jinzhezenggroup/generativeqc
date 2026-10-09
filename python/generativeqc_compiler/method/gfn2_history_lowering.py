"""GFN2 CUDA physical bindings for shared ordered-history emission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from generativeqc_compiler.tensor.ordered_history_artifacts import (
    emit_ordered_history_artifacts,
)
from generativeqc_compiler.tensor.ordered_history_emit import (
    CorrectionBindings,
    CudaFlagFailure,
    CudaVisitCounter,
)
from generativeqc_compiler.tensor.ordered_history_gram import GramBindings

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate


def gfn2_history_correction_bindings(backend: str) -> CorrectionBindings:
    if backend != "cuda":
        raise ValueError("unqualified GFN2 history backend")
    return CorrectionBindings(
        current="state.current_inputs[index]",
        damping="policy.damping",
        residual="workspace.residual[index]",
        coefficient="workspace.coefficients[coefficient_begin + history]",
        mixed="workspace.mixed[index]",
        capacity="policy.history_size",
        tentative_weight="new_omega",
        weights="state.omega",
        tentative_vectors="workspace.new_u",
        vectors="state.u_history",
        failure=CudaFlagFailure(
            function="record_error",
            error_buffer="device_error",
            code="Gfn2SccMixerDeviceError::kNonfiniteMixedMultipole",
            valid="valid",
        ),
        visit_counter=CudaVisitCounter("Record", "combination_visits"),
    )


def gfn2_history_gram_bindings(backend: str) -> GramBindings:
    if backend != "cuda":
        raise ValueError("unqualified GFN2 history backend")

    def device(code: str) -> CudaFlagFailure:
        return CudaFlagFailure(
            "record_error", "device_error", "Gfn2SccMixerDeviceError::" + code, "valid"
        )

    return GramBindings(
        delta_f="workspace.delta_f",
        new_u="workspace.new_u",
        residual="workspace.residual",
        df_history="state.df_history",
        u_history="state.u_history",
        weights="state.omega",
        new_weight="new_omega",
        capacity="policy.history_size",
        coefficients="workspace.coefficients",
        beta="workspace.beta",
        omega_zero="kOmegaZero",
        history_slots="",
        coefficient_dot_failure=device("kNonfiniteCoefficient"),
        coefficient_product_failure=device("kNonfiniteCoefficient"),
        overlap_failure=device("kNonfiniteHistory"),
        matrix_failure=device("kNonfiniteHistory"),
        weight_failure=device("kNonfiniteWeight"),
        coefficient_visit_counter=CudaVisitCounter("Record", "coefficient_visits"),
        overlap_visit_counter=CudaVisitCounter("Record", "gram_visits"),
    )


def emit_gfn2_history_artifacts(
    backend: str, *, candidate: LoweringCandidate | None = None
) -> dict[str, str]:
    """Bind the unchanged GFN2 CUDA consumer to shared candidate admission."""
    return emit_ordered_history_artifacts(
        backend,
        source_prefix=f"generated_gfn2_history_{backend}",
        gram=gfn2_history_gram_bindings(backend),
        correction=gfn2_history_correction_bindings(backend),
        candidate=candidate,
    )


def emit_gfn2_history_correction(backend: str) -> str:
    return emit_gfn2_history_artifacts(backend)[
        f"generated_gfn2_history_{backend}_correction.inc"
    ]
