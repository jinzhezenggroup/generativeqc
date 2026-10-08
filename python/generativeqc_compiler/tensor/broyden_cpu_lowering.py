"""Physical bindings for the solver-owned CPU Johnson-Broyden provider.

The retained diagnostic strings preserve source compatibility, not method policy.
Allocation, admission limits and state publication belong to the native solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ordered_history_artifacts import emit_ordered_history_artifacts
from .ordered_history_emit import CorrectionBindings, ReturnStatusFailure
from .ordered_history_gram import GramBindings

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate


def broyden_cpu_correction_bindings() -> CorrectionBindings:
    return CorrectionBindings(
        current="current[component]",
        damping="data.damping",
        residual="workspace.residual[component]",
        coefficient="workspace.coefficients[history]",
        mixed="workspace.mixed[component]",
        history_slots="workspace.history_slots",
        weight_accessor="slot_omega",
        vector_accessor="u_vector",
        failure=ReturnStatusFailure(
            function="record_numeric_failure",
            state="state",
            system="system",
            message="SCC mixer Broyden result is not finite",
            error="error",
        ),
    )


def broyden_cpu_gram_bindings() -> GramBindings:
    def status(message: str, message_line: str = "first") -> ReturnStatusFailure:
        return ReturnStatusFailure(
            "record_numeric_failure", "state", "system", message, "error", message_line
        )

    return GramBindings(
        delta_f="workspace.delta_f",
        new_u="workspace.new_u",
        residual="workspace.residual",
        df_history="state.df_history",
        u_history="state.u_history",
        weights="state.omega",
        new_weight="omega",
        capacity="memory",
        coefficients="workspace.coefficients",
        beta="workspace.beta",
        omega_zero="kOmegaZero",
        history_slots="workspace.history_slots",
        coefficient_dot_failure=status("SCC mixer Broyden coefficient is not finite"),
        coefficient_product_failure=status("SCC mixer Broyden coefficient overflowed"),
        overlap_failure=status(
            "SCC mixer Broyden history overlap is not finite", "second"
        ),
        matrix_failure=status("SCC mixer Broyden matrix overflowed"),
        weight_failure=None,
    )


def emit_broyden_cpu_artifacts(
    *, candidate: LoweringCandidate | None = None
) -> dict[str, str]:
    """Emit the compact CPU fragments without importing method bindings."""
    return emit_ordered_history_artifacts(
        "cpu",
        source_prefix="generated_broyden_cpu",
        gram=broyden_cpu_gram_bindings(),
        correction=broyden_cpu_correction_bindings(),
        candidate=candidate,
    )
