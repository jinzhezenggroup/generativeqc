from __future__ import annotations

from types import SimpleNamespace

import pytest
from vibeqc_compiler.method import bulk_ks
from vibeqc_compiler.method.spec import UnsupportedMethod
from vibeqc_compiler.xc.capability_resolution import (
    CapabilityNotQualified,
    CapabilityResolution,
)
from vibeqc_compiler.xc.libxc_blacklist import (
    LIBXC_SEMILOCAL_BLACKLIST,
    blacklist_reason,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability


def _compiled_cpu() -> dict[str, str]:
    return {
        "binding_identity": "a" * 64,
        "result_identity": "b" * 64,
    }


def test_bulk_ks_default_allows_supported_import_without_evidence() -> None:
    result = bulk_ks.resolve_bulk_ks("GGA_X_APBE", spin="polarized")

    assert result.method.identifier == "LIBXC:GGA_X_APBE"
    assert result.method.reference == "unrestricted"
    assert result.required_ingredients == ("rho", "sigma")
    assert result.plan.required_lowerers == ("semilocal-xc",)
    assert result.plan.exchange == ()
    assert result.public_dft
    assert result.compiled_cpu_binding_identity is None
    assert result.compiled_cpu_result_identity is None
    assert result.to_payload()["admission"] == "default-allow/explicit-blacklist"


def test_public_bulk_ks_is_default_allow_compatibility_alias() -> None:
    first = bulk_ks.resolve_bulk_ks("MGGA_X_LTA")
    second = bulk_ks.resolve_public_bulk_ks("MGGA_X_LTA")

    assert first.method.identity == second.method.identity
    assert first.plan.identity == second.plan.identity
    assert first.public_dft and second.public_dft


def test_positive_evidence_is_not_a_public_admission_gate() -> None:
    result = bulk_ks.resolve_bulk_ks(
        "LDA_C_BR78",
        evidence={"public-method": {"status": "fail", "sentinel": True}},
    )
    assert result.public_dft
    assert result.admission == "default-allow/explicit-blacklist"


def test_bulk_ks_rejects_explicit_blacklist() -> None:
    assert blacklist_reason("GGA_C_AM05") is not None
    with pytest.raises(UnsupportedMethod, match="blacklisted"):
        bulk_ks.resolve_bulk_ks("GGA_C_AM05")


def test_blacklist_is_negative_only_and_immutable() -> None:
    assert "GGA_X_APBE" not in LIBXC_SEMILOCAL_BLACKLIST
    with pytest.raises(TypeError):
        LIBXC_SEMILOCAL_BLACKLIST["GGA_X_APBE"] = "do not allow"  # type: ignore[index]


def test_bulk_ks_rejects_non_cpu_backend_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bulk_ks,
        "functional_capability",
        lambda *args, **kwargs: pytest.fail("capability lookup must stay inactive"),
    )

    with pytest.raises(UnsupportedMethod, match="supports CPU only"):
        bulk_ks.resolve_bulk_ks("GGA_X_APBE", backend="cuda")


def test_bulk_ks_rejects_unsupported_ingredient_structurally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = functional_capability("GGA_X_APBE")
    capability = SimpleNamespace(
        name=base.name,
        identity=base.identity,
        required_ingredients=("rho", "sigma", "laplacian"),
    )
    monkeypatch.setattr(
        bulk_ks, "functional_capability", lambda *args, **kwargs: capability
    )

    with pytest.raises(UnsupportedMethod, match="laplacian"):
        bulk_ks.resolve_bulk_ks(base.name)


def test_bulk_ks_descriptive_identifier_does_not_change_semantics() -> None:
    first = bulk_ks.resolve_bulk_ks("GGA_X_APBE", identifier="candidate-a")
    second = bulk_ks.resolve_bulk_ks("GGA_X_APBE", identifier="candidate-b")

    assert first.method.identifier != second.method.identifier
    assert first.method.identity == second.method.identity
    assert first.plan.identity == second.plan.identity
    assert first.capability.identity == second.capability.identity


def test_qualification_candidate_remains_evidence_gated() -> None:
    with pytest.raises(CapabilityNotQualified) as exc:
        bulk_ks.resolve_bulk_ks_candidate("GGA_X_APBE")

    assert "compiled-cpu" in exc.value.missing_stages
    assert "production-domain" in exc.value.missing_stages


def test_qualification_candidate_can_retest_blacklisted_functional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = functional_capability("GGA_C_AM05")
    qualified = CapabilityResolution(
        name=capability.name,
        identity=capability.identity,
        required_stages=("compiled-cpu", "production-domain"),
        qualified_stages=(
            "graph-imported",
            "pointwise-validated",
            "compiled-cpu",
            "production-domain",
        ),
        public_dft=False,
    )
    monkeypatch.setattr(bulk_ks, "resolve_capability", lambda *args, **kwargs: qualified)
    monkeypatch.setattr(
        bulk_ks, "_require_exact_compiled_cpu", lambda capability: _compiled_cpu()
    )

    result = bulk_ks.resolve_bulk_ks_candidate(capability.name)

    assert not result.public_dft
    assert result.admission == "qualification-evidence"
    assert result.compiled_cpu_binding_identity == "a" * 64
    assert result.compiled_cpu_result_identity == "b" * 64


def test_candidate_detects_capability_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = functional_capability("GGA_X_APBE")
    drifted = CapabilityResolution(
        name=capability.name,
        identity="0" * 64,
        required_stages=("compiled-cpu", "production-domain"),
        qualified_stages=("compiled-cpu", "production-domain"),
        public_dft=False,
    )
    monkeypatch.setattr(bulk_ks, "resolve_capability", lambda *args, **kwargs: drifted)

    with pytest.raises(RuntimeError, match="identity changed"):
        bulk_ks.resolve_bulk_ks_candidate(capability.name)
