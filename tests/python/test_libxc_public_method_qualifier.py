"""Tests for the exact automatic Libxc public-method qualifier."""

from __future__ import annotations

import pytest
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.molecular_scf_evidence import (
    QUALIFICATION_SCHEMA as MOLECULAR_SCF_QUALIFICATION_SCHEMA,
)
from vibeqc_compiler.xc.molecular_scf_evidence import (
    RESULT_SCHEMA as MOLECULAR_SCF_RESULT_SCHEMA,
)

from tools.qualify_libxc_public_method import qualify_public_method

NAME = "GGA_X_PBE_SOL"


def _stage(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    qualification: dict | None = None,
    evidence: str | None = None,
) -> dict:
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": "pass",
        "reason": None,
        "evidence": evidence or f"test://{capability.name}/{stage}",
    }
    if qualification is not None:
        payload["qualification"] = qualification
    return payload


def _prerequisites() -> tuple[dict, dict, dict]:
    capability = libxc_bulk_capabilities.functional_capability(NAME)
    result_identity = "c" * 64
    compiled = _stage(capability, "compiled-cpu")
    production = _stage(
        capability,
        "production-domain",
        qualification=capability.production_domain_profile.to_payload(),
    )
    molecular = _stage(
        capability,
        "molecular-scf",
        qualification={
            "schema": ENDPOINT_COVERAGE_SCHEMA,
            "coverage": [
                {"backend": "cpu", "spin": spin, "products": ["energy"]}
                for spin in libxc_bulk_capabilities.SPIN_LAYOUTS
            ],
            "result_schema": MOLECULAR_SCF_RESULT_SCHEMA,
            "result_identity": result_identity,
            "qualification_schema": MOLECULAR_SCF_QUALIFICATION_SCHEMA,
        },
        evidence=f"test://{capability.name}/molecular-scf#sha256={result_identity}",
    )
    return compiled, production, molecular


def test_exact_prerequisites_produce_public_method_evidence() -> None:
    compiled, production, molecular = _prerequisites()
    campaign = qualify_public_method(
        NAME,
        compiled_cpu_evidence=compiled,
        production_domain_evidence=production,
        molecular_scf_evidence=molecular,
        evidence="test://public-method/result",
    )

    public = campaign["stage_evidence"]
    assert campaign["schema"] == "vibeqc.libxc-public-method-campaign/v1"
    assert public["stage"] == "public-method"
    assert public["status"] == "pass"

    evidence = {
        "compiled-cpu": compiled,
        "production-domain": production,
        "molecular-scf": molecular,
        "public-method": public,
    }
    for spin in libxc_bulk_capabilities.SPIN_LAYOUTS:
        endpoint = resolve_endpoint_capability(
            NAME,
            backend="cpu",
            product="energy",
            spin=spin,
            require_public=True,
            evidence=evidence,
        )
        assert endpoint.public_dft is True


def test_generic_molecular_coverage_cannot_produce_public_method() -> None:
    compiled, production, molecular = _prerequisites()
    molecular["qualification"] = {
        "schema": ENDPOINT_COVERAGE_SCHEMA,
        "coverage": [
            {"backend": "cpu", "spin": spin, "products": ["energy"]}
            for spin in libxc_bulk_capabilities.SPIN_LAYOUTS
        ],
    }
    molecular["evidence"] = "test://forged/molecular-scf"

    with pytest.raises(ValueError, match="result schema mismatch"):
        qualify_public_method(
            NAME,
            compiled_cpu_evidence=compiled,
            production_domain_evidence=production,
            molecular_scf_evidence=molecular,
            evidence="test://public-method/forged",
        )


def test_wrapped_campaign_stage_payloads_are_accepted() -> None:
    compiled, production, molecular = _prerequisites()
    campaign = qualify_public_method(
        NAME,
        compiled_cpu_evidence={"stage_evidence": compiled},
        production_domain_evidence={"stage_evidence": production},
        molecular_scf_evidence={"stage_evidence": molecular},
        evidence="test://public-method/wrapped",
    )
    assert campaign["stage_evidence"]["status"] == "pass"
