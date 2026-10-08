"""Candidate admission and source assembly for shared ordered-history algebra."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

from .ordered_history import (
    ordered_history_candidate,
    ordered_history_program,
    require_history_candidate,
)
from .ordered_history_emit import (
    CorrectionBindings,
    emit_cholesky,
    emit_correction,
    emit_cpu_dot,
)
from .ordered_history_gram import (
    GramBindings,
    emit_history_gram,
    emit_history_window,
)

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_provider import LoweringCandidate


def emit_ordered_history_artifacts(
    backend: str,
    *,
    source_prefix: str,
    gram: GramBindings,
    correction: CorrectionBindings,
    candidate: LoweringCandidate | None = None,
) -> dict[str, str]:
    """Bind a consumer only after admitting its complete lowering candidate."""
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
    helpers = emit_cholesky(program, schedule)
    if schedule.name == "compact-cpu":
        helpers = emit_cpu_dot(program, schedule) + "\n\n" + helpers
    bodies = {
        f"{source_prefix}_helpers.inc": helpers,
        f"{source_prefix}_window.inc": emit_history_window(program, schedule, gram),
        f"{source_prefix}_gram.inc": emit_history_gram(program, schedule, gram),
        f"{source_prefix}_correction.inc": emit_correction(
            program, schedule, correction
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
    bodies[f"{source_prefix}_identity.json"] = (
        json.dumps(
            {**identity, "source_identity": canonical_hash(identity)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return bodies
