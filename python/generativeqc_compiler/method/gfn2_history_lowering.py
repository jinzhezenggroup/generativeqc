"""GFN2 physical bindings for candidate-selected shared ordered-history emission."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.tensor.ordered_history import (
    ordered_history_candidate,
    ordered_history_program,
    require_history_candidate,
)
from generativeqc_compiler.tensor.ordered_history_emit import (
    CorrectionBindings,
    CudaFlagFailure,
    ReturnStatusFailure,
    emit_cholesky,
    emit_correction,
    emit_cpu_dot,
)
from generativeqc_compiler.tensor.ordered_history_gram import (
    GramBindings,
    emit_history_gram,
    emit_history_window,
)

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate


def gfn2_history_correction_bindings(backend: str) -> CorrectionBindings:
    if backend == "cpu":
        bindings = CorrectionBindings(
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
    elif backend == "cuda":
        bindings = CorrectionBindings(
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
        )
    else:
        raise ValueError("unqualified history backend")
    return bindings


def gfn2_history_gram_bindings(backend: str) -> GramBindings:
    def status(message: str, message_line: str = "first") -> ReturnStatusFailure:
        return ReturnStatusFailure(
            "record_numeric_failure", "state", "system", message, "error", message_line
        )

    def device(code: str) -> CudaFlagFailure:
        return CudaFlagFailure(
            "record_error", "device_error", "Gfn2SccMixerDeviceError::" + code, "valid"
        )

    cpu = backend == "cpu"
    if backend not in ("cpu", "cuda"):
        raise ValueError("unqualified history backend")
    return GramBindings(
        delta_f="workspace.delta_f",
        new_u="workspace.new_u",
        residual="workspace.residual",
        df_history="state.df_history",
        u_history="state.u_history",
        weights="state.omega",
        new_weight="omega" if cpu else "new_omega",
        capacity="memory" if cpu else "policy.history_size",
        coefficients="workspace.coefficients",
        beta="workspace.beta",
        omega_zero="kOmegaZero",
        history_slots="workspace.history_slots" if cpu else "",
        coefficient_dot_failure=status("SCC mixer Broyden coefficient is not finite")
        if cpu
        else device("kNonfiniteCoefficient"),
        coefficient_product_failure=status("SCC mixer Broyden coefficient overflowed")
        if cpu
        else device("kNonfiniteCoefficient"),
        overlap_failure=status(
            "SCC mixer Broyden history overlap is not finite", "second"
        )
        if cpu
        else device("kNonfiniteHistory"),
        matrix_failure=status("SCC mixer Broyden matrix overflowed")
        if cpu
        else device("kNonfiniteHistory"),
        weight_failure=None if cpu else device("kNonfiniteWeight"),
    )


def emit_gfn2_history_artifacts(
    backend: str, *, candidate: LoweringCandidate | None = None
) -> dict[str, str]:
    """The admitted candidate's selected algorithm drives every native fragment."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("unqualified history backend")
    program = ordered_history_program()
    selected = "compact-cpu" if backend == "cpu" else "capacity-cuda"
    candidate = (
        ordered_history_candidate(program, selected) if candidate is None else candidate
    )
    schedule = require_history_candidate(program, candidate)
    if schedule.name != selected:
        raise ValueError("history candidate does not match its consumer backend")
    gram = gfn2_history_gram_bindings(backend)
    helpers = emit_cholesky(program, schedule)
    if schedule.name == "compact-cpu":
        helpers = emit_cpu_dot(program, schedule) + "\n\n" + helpers
    bodies = {
        f"generated_gfn2_history_{backend}_helpers.inc": helpers,
        f"generated_gfn2_history_{backend}_window.inc": emit_history_window(
            program, schedule, gram
        ),
        f"generated_gfn2_history_{backend}_gram.inc": emit_history_gram(
            program, schedule, gram
        ),
        f"generated_gfn2_history_{backend}_correction.inc": emit_correction(
            program, schedule, gfn2_history_correction_bindings(backend)
        ),
    }
    identity = {
        "schema": "generativeqc.ordered-history-source.v1",
        "scientific_identity": program.identity,
        "candidate": candidate.to_payload(),
        "sources": {
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in bodies.items()
        },
    }
    bodies[f"generated_gfn2_history_{backend}_identity.json"] = (
        json.dumps(
            {**identity, "source_identity": canonical_hash(identity)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return bodies


def emit_gfn2_history_correction(backend: str) -> str:
    return emit_gfn2_history_artifacts(backend)[
        f"generated_gfn2_history_{backend}_correction.inc"
    ]
