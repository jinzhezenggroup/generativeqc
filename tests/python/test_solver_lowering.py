"""Canonical solver semantics stay independent of backend/provider preparation."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.library import LibraryRequest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.solver_lowering import (
    ordinary_eigh_header,
    ordinary_eigh_portfolio,
    symmetric_eigh_request,
)
from generativeqc_compiler.integral.runtime_backend import (
    LibraryRequest as LegacyRequest,
)


def operation(workspace_limit_bytes: int = 0) -> LibraryRequest:
    return LibraryRequest(
        "symmetric_eigh",
        (24,),
        layout="column_major",
        residual_policy="qualification",
        workspace_limit_bytes=workspace_limit_bytes,
    )


def test_solver_reuses_canonical_library_vocabulary() -> None:
    assert LegacyRequest is LibraryRequest
    request = symmetric_eigh_request(operation(), backend="cuda")
    cpu = symmetric_eigh_request(operation(), backend="cpu")
    assert request.scientific_identity == cpu.scientific_identity
    assert request.semantic_identity == cpu.semantic_identity
    assert request.identity != cpu.identity
    bounded = symmetric_eigh_request(
        operation(workspace_limit_bytes=8192), backend="cuda"
    )
    assert bounded.semantic_identity == request.semantic_identity
    assert bounded.identity != request.identity
    assert request.operands[0].access == "read-write"
    assert request.operands[0].strides == (1, 24)
    assert dict(request.effects)["residual"] == "independent-qualification-only"


def test_ordinary_binding_cannot_claim_runtime_residual_or_other_layout() -> None:
    for invalid in (
        replace(operation(), residual_policy="runtime"),
        replace(operation(), layout="row_major"),
        replace(operation(), operation="cholesky"),
    ):
        with pytest.raises(ValueError, match="qualified column-major symmetric_eigh"):
            symmetric_eigh_request(invalid, backend="cuda")
    with pytest.raises(ValueError, match="residual policy"):
        replace(operation(), residual_policy="status-is-success")


def test_two_provider_templates_share_precision_and_unknown_costs() -> None:
    source = canonical_hash({"source": "fixture"})
    request, candidates, _, _ = ordinary_eigh_portfolio(source)
    assert len(candidates) == 2
    assert all(candidate.request is request for candidate in candidates)
    assert all(candidate.cost is None for candidate in candidates)
    assert (
        len({candidate.execution.precision.identity for candidate in candidates}) == 1
    )
    assert [candidate.execution.capture_safe for candidate in candidates] == [
        True,
        False,
    ]
    changed = ordinary_eigh_portfolio(canonical_hash({"source": "changed"}))
    assert changed[0].identity == request.identity
    assert changed[1][0].identity != candidates[0].identity
    assert ordinary_eigh_header(source) == ordinary_eigh_header(source)
