"""Independent algebra and negative-evidence tests for shared SCF regions.

GPU arithmetic, replay and alias admission are qualified by the native driver,
not by these TensorIR tests or by source-pattern assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor.basis_transform import basis_transform_program
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.structured_scf import (
    emit_structured_scf_cuda,
    structured_program,
    structured_scf_portfolios,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dimension", [1, 3, 17, 33])
@pytest.mark.parametrize("inverse_root", [False, True])
def test_weighted_reconstruction_oracle(dimension: int, inverse_root: bool) -> None:
    rng = np.random.default_rng(1883 + dimension)
    vectors = rng.normal(size=(dimension, dimension))
    weights = (
        rng.uniform(0.05, 3, dimension) if inverse_root else rng.normal(size=dimension)
    )
    result = execute(
        structured_program(dimension, inverse_root=inverse_root),
        {"vectors": vectors, "weights": weights},
    ).outputs["product"]
    effective = 1 / np.sqrt(weights) if inverse_root else weights
    oracle = (vectors * effective) @ vectors.T
    np.testing.assert_allclose(result, oracle, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_invalid_domain_rejected(dimension: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        structured_program(dimension)


def test_complete_portfolios_retain_negative_evidence() -> None:
    portfolios = structured_scf_portfolios()
    assert len(portfolios) == 7
    identities = []
    for slot, portfolio in enumerate(portfolios):
        request, candidates = portfolio.request, portfolio.candidates
        identities.append(request.identity)
        assert dict(request.effects)["inactive_publication"] == "preserve-bytes"
        assert len(candidates) == 7
        assert candidates[0].status == candidates[1].status == "ready"
        assert candidates[2].status == ("ready" if slot == 0 else "unsupported")
        assert all(candidate.status == "unsupported" for candidate in candidates[3:])
        assert all(candidate.reason for candidate in candidates[3:])
        assert all(candidate.cost is None for candidate in candidates)
        assert all(candidate.execution.capture_safe for candidate in candidates)
        if slot < 2:
            assert dict(request.semantics)["symmetric_rank_k"] is True
            assert dict(request.semantics)["update_mode"] == "overwrite"
            assert request.operands[-1].triangle == "lower"
            assert request.operands[0].strides == (1, 5)
    assert len(set(identities)) == 7
    assert (
        dict(portfolios[0].request.semantics)["weight_domain"]
        == "nonnegative-occupied-counts"
    )
    assert dict(portfolios[1].request.semantics)["weight_domain"] == "signed"
    assert (
        dict(portfolios[2].request.semantics)["weight_domain"]
        == "positive-finite-eigenvalues"
    )


def test_final_basis_identity_includes_hcore_epilogue() -> None:
    program = basis_transform_program(5, 7)
    adapter = TensorLoweringAdapter(program)
    final = structured_scf_portfolios()[-1].request
    assert (
        final.scientific_identity
        == adapter.request(
            program.outputs["fock_public"], backend="cuda"
        ).scientific_identity
    )
    assert final.operation == "add"
    assert (
        adapter.request(
            program.outputs["fock_public"], backend="cuda"
        ).semantic_identity
        != adapter.request(
            program.outputs["fock_right"], backend="cuda"
        ).semantic_identity
    )


def test_emission_is_deterministic_and_shared_owners_do_not_submit_blas() -> None:
    assert emit_structured_scf_cuda() == emit_structured_scf_cuda()
    hf = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    assert "launch_structured_density(" in hf and "launch_structured_density(" in ks
    assert "launch_structured_orthogonalizer(" in hf
    assert "launch_structured_basis(" in hf
    assert "cublasDsyrk(" not in hf + ks
    assert "cublasDgemm(" not in hf + ks
