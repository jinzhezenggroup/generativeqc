"""Density work projection preserves raw candidates and complete spin outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.xc import density_work
from generativeqc_compiler.xc.bulk_point_program import (
    bind_runtime_semilocal_point_program,
    inspect_runtime_program,
)
from generativeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
    production_candidate_domain,
)
from generativeqc_compiler.xc.compiled_cpu_evidence import (
    build_result as build_compiled_result,
)
from generativeqc_compiler.xc.compiled_cpu_evidence import (
    validate_result as validate_compiled_result,
)
from generativeqc_compiler.xc.density_work import (
    DENSITY_WORK_DOMAIN,
    DENSITY_WORK_DOMAIN_VERSION,
)
from generativeqc_compiler.xc.production_domain_evidence import build_execution_binding
from generativeqc_compiler.xc.spec import UnsupportedXC

if TYPE_CHECKING:
    from generativeqc_compiler.xc.bulk_runtime import BulkRuntimeProgram

NAME = "LDA_C_CHACHIYO"


@pytest.mark.parametrize("name", [NAME, "LDA_C_VWN_4", "LDA_C_PZ_MOD"])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_public_automatic_density_domain_matches_the_native_worker(
    name: str, spin: str
) -> None:
    from generativeqc.ks import scf_domain_for_method
    from generativeqc_compiler.method import resolve_bulk_ks
    from generativeqc_compiler.xc.automatic_semilocal import automatic_scf_domain

    resolution = resolve_bulk_ks(name, spin=spin, backend="cpu")
    selector = f"libxc-uks:{name}" if spin == "polarized" else f"libxc-rks:{name}"
    assert automatic_scf_domain(name) == DENSITY_WORK_DOMAIN
    assert scf_domain_for_method(selector) == DENSITY_WORK_DOMAIN
    assert scf_domain_for_method(resolution.method) == DENSITY_WORK_DOMAIN


def _program(
    *, spin: str = "polarized", domain: str = DENSITY_WORK_DOMAIN
) -> BulkRuntimeProgram:
    return build_bulk_runtime_program(NAME, spin=spin, order=1, domain=domain)


def test_complete_empty_and_full_spin_work_outputs_match_pinned_reference() -> None:
    inputs = np.array([[0.0, 0.62, 0.8, 0.0], [0.62, 0.0, 0.0, 0.8]])
    reference = np.array(
        [
            [
                -0.02171144895067935,
                -0.02171144895067935,
                -0.02884567880213114,
                -0.02884567880213114,
            ],
            [
                -0.23632884700378914,
                -0.03906393853241106,
                -0.040160984808165864,
                -0.24409169721836327,
            ],
            [
                -0.03906393853241106,
                -0.23632884700378914,
                -0.24409169721836327,
                -0.040160984808165864,
            ],
        ]
    )
    observed = _program().evaluate(inputs)
    np.testing.assert_allclose(observed, reference, atol=2e-14, rtol=0)
    assert _program().outputs == ((), (0,), (1,))


def test_unclipped_v2_keeps_its_original_failure_and_identity() -> None:
    inputs = np.array([[0.0], [0.62]])
    original = _program(domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN)
    projected = _program()
    np.testing.assert_allclose(
        original.evaluate(inputs)[:, 0],
        [-0.02171144895067915, -0.23633115896744458, -0.03906393853241101],
        atol=2e-14,
        rtol=0,
    )
    assert original.expression_hash != projected.expression_hash
    assert "density_work" not in original.spec.to_payload()
    assert (
        projected.spec.to_payload()["density_work"]["worker"]["density_threshold"]
        == 1e-15
    )


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_vacuum_screening_precedes_work_evaluation_and_does_not_mutate(
    spin: str,
) -> None:
    program = _program(spin=spin)
    features = np.zeros((len(program.spec.features), 3))
    features[:, 1] = 0.62
    features[:, 2] = 1e-20
    original = features.copy()
    observed = program.evaluate(features)
    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(observed[:, [0, 2]], 0.0)
    assert np.all(np.isfinite(observed))


def test_total_density_energy_weight_is_not_replaced_by_floored_spin_sum() -> None:
    program = _program()
    raw = np.array([[0.0], [1.1e-15]])
    work = np.maximum(raw, 1e-15)
    interior = build_bulk_runtime_program(NAME, spin="polarized", order=1)
    expected = interior.evaluate(work)
    expected[0] *= raw.sum() / work.sum()
    np.testing.assert_allclose(program.evaluate(raw), expected, atol=2e-14, rtol=1e-14)


@pytest.mark.parametrize("name", [NAME, "LDA_C_VWN_4", "LDA_C_PZ_MOD"])
@pytest.mark.parametrize("optimization", ["none", "before", "after"])
def test_energy_reweighting_does_not_square_large_finite_density(
    name: str, optimization: str
) -> None:
    features = np.array([[1e160]])
    interior = build_bulk_runtime_program(
        name, spin="unpolarized", optimization=optimization
    )
    projected = build_bulk_runtime_program(
        name,
        spin="unpolarized",
        optimization=optimization,
        domain=DENSITY_WORK_DOMAIN,
    )
    np.testing.assert_allclose(
        projected.evaluate(features), interior.evaluate(features), rtol=2e-14, atol=0
    )


def test_projection_does_not_differentiate_clipping_into_vxc() -> None:
    program = _program()
    raw = np.array([[0.0], [0.62]])
    interior = build_bulk_runtime_program(NAME, spin="polarized", order=1)
    expected = interior.evaluate(np.maximum(raw, 1e-15))
    np.testing.assert_allclose(
        program.evaluate(raw)[1:], expected[1:], atol=2e-14, rtol=0
    )
    assert abs(program.evaluate(raw)[1, 0]) > 0.2


def test_cpu_and_cuda_share_the_projected_mathematics_and_native_domain() -> None:
    program = _program()
    cpu = inspect_runtime_program(program, "cpu")
    cuda = inspect_runtime_program(program, "cuda")
    assert cpu.ssa == cuda.ssa
    assert cpu.import_identity == cuda.import_identity
    assert cpu.features == cuda.features
    assert cpu.domain == cuda.domain == DENSITY_WORK_DOMAIN
    binding = bind_runtime_semilocal_point_program(program)
    assert binding.domain_version == DENSITY_WORK_DOMAIN_VERSION
    assert binding.to_payload()["domain"] == DENSITY_WORK_DOMAIN


@pytest.mark.parametrize("name", ["GGA_X_PBE_SOL", "MGGA_X_LTA"])
def test_unimplemented_work_ingredients_fail_before_execution(name: str) -> None:
    with pytest.raises(UnsupportedXC, match="density-only LDA"):
        build_bulk_runtime_program(name, domain=DENSITY_WORK_DOMAIN)


def test_negative_density_remains_invalid() -> None:
    with pytest.raises(UnsupportedXC, match="nonnegative density"):
        _program().evaluate(np.array([[-1e-20], [0.62]]))


@pytest.mark.parametrize("name", [NAME, "LDA_C_VWN_4", "LDA_C_PZ_MOD"])
def test_density_only_qualification_recipe_is_shared(name: str) -> None:
    assert production_candidate_domain(name) == DENSITY_WORK_DOMAIN


@pytest.mark.parametrize("name", ["GGA_X_PBE_SOL", "MGGA_X_LTA"])
def test_other_ingredients_keep_the_unclipped_qualification_recipe(name: str) -> None:
    assert production_candidate_domain(name) == PRODUCTION_DENSITY_CANDIDATE_DOMAIN


def test_work_owner_change_invalidates_receipt_but_not_prepared_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programs = {spin: _program(spin=spin) for spin in ("polarized", "unpolarized")}
    program = programs["polarized"]
    prepared_payload = program.spec.to_payload()
    binding = bind_runtime_semilocal_point_program(program)
    result = build_compiled_result(
        NAME,
        binding,
        {"status": "not-run", "reason": "staleness fixture; no compilation claimed"},
        evidence="test://density-work/staleness",
    )
    assert validate_compiled_result(NAME, result) == result
    monkeypatch.setattr(density_work, "file_hash", lambda _: "f" * 64)
    assert program.spec.to_payload() == prepared_payload
    with pytest.raises(ValueError, match="density work binding is stale"):
        validate_compiled_result(NAME, result)
    with pytest.raises(ValueError, match="density work execution is stale"):
        build_execution_binding(NAME, programs)


@pytest.mark.parametrize("name", [NAME, "LDA_C_VWN_4", "LDA_C_PZ_MOD"])
def test_restricted_binding_preserves_distinct_total_density_contract(
    name: str,
) -> None:
    polarized = build_bulk_runtime_program(name, domain=DENSITY_WORK_DOMAIN)
    total = build_bulk_runtime_program(
        name, spin="unpolarized", domain=DENSITY_WORK_DOMAIN
    )
    binding = bind_runtime_semilocal_point_program(polarized)
    assert binding.restricted is not None
    payload = binding.restricted.to_payload()
    assert payload["normalization"] == "original-total-density-no-spin-fold"
    assert payload["point_expression_identity"] == total.expression_hash
    assert payload["outputs"] == [[], [0]]
    assert binding.to_payload()["schema"].endswith("/v5")
    rho = np.array([[1.1 * binding.density_threshold]])
    folded = polarized.evaluate(np.vstack((0.5 * rho, 0.5 * rho)))
    assert abs(total.evaluate(rho)[1, 0] - folded[1:, 0].mean()) > 1e-6


@pytest.mark.parametrize("name", [NAME, "LDA_C_VWN_4", "LDA_C_PZ_MOD"])
def test_restricted_work_requires_native_total_density_smoke(name: str) -> None:
    from test_compiled_cpu_evidence import _outcome

    binding = bind_runtime_semilocal_point_program(
        build_bulk_runtime_program(name, domain=DENSITY_WORK_DOMAIN)
    )
    with pytest.raises(ValueError, match="exact interior and vacuum cases"):
        build_compiled_result(
            name, binding, _outcome(), evidence="test://density-work/missing-tail"
        )


def _restricted_compiled_result() -> dict:
    from test_compiled_cpu_evidence import _outcome

    outcome = _outcome()
    smoke = outcome["smoke"]
    smoke["case_labels"].extend(
        ["restricted-interior", "restricted-vacuum", "restricted-active-tail"]
    )
    for key in ("expected", "observed"):
        smoke[key].extend([float(index) for index in range(18)])
    return build_compiled_result(
        NAME,
        bind_runtime_semilocal_point_program(_program()),
        outcome,
        evidence="test://density-work/complete-native-contract",
    )


def test_restricted_binding_receipt_rejects_missing_or_changed_normalization() -> None:
    result = _restricted_compiled_result()
    assert validate_compiled_result(NAME, result) == result
    for key, replacement in (
        ("normalization", "polarized-spin-fold"),
        ("point_expression_identity", "0" * 64),
        ("artifact_source_sha256", "0" * 64),
        ("domain_version", True),
        ("features", ["rho_a", "rho_b"]),
    ):
        changed = deepcopy(result)
        changed["binding"]["restricted_binding"][key] = replacement
        with pytest.raises(ValueError, match="density work binding is stale"):
            validate_compiled_result(NAME, changed)
    changed = deepcopy(result)
    del changed["binding"]["restricted_binding"]
    with pytest.raises(ValueError, match="density work binding is stale"):
        validate_compiled_result(NAME, changed)


def test_restricted_binding_receipt_is_detached_from_caller_payload() -> None:
    result = _restricted_compiled_result()
    normalized = validate_compiled_result(NAME, result)
    result["binding"]["restricted_binding"]["features"].append("not-an-input")
    assert normalized["binding"]["restricted_binding"]["features"] == ["rho"]


def test_old_unclipped_binding_cannot_supply_work_compilation_evidence() -> None:
    binding = bind_runtime_semilocal_point_program(
        _program(domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN)
    )
    with pytest.raises(ValueError, match="production candidate domain"):
        build_compiled_result(
            NAME,
            binding,
            {"status": "not-run", "reason": "old recipe; not executed"},
            evidence="test://density-work/old-recipe",
        )
