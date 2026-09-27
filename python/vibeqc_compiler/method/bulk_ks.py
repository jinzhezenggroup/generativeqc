"""Evidence-gated MethodIR/KS resolution for automatic bulk Libxc registrations.

This module is deliberately a composition boundary, not an evidence producer.
Qualification producers may resolve an execution candidate after compiled-CPU
and production-domain evidence exists; ordinary consumers require the additional
molecular-SCF evidence produced by executing that candidate.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.xc.capability_resolution import (
    CapabilityResolution,
    resolve_capability,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import validate_qualification
from vibeqc_compiler.xc.endpoint_capability import (
    ENDPOINT_COVERAGE_SCHEMA,
    resolve_endpoint_capability,
)
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    SPIN_LAYOUTS,
    BulkFunctionalCapability,
    functional_capability,
)
from vibeqc_compiler.xc.molecular_scf_evidence import (
    QUALIFICATION_SCHEMA as MOLECULAR_SCF_QUALIFICATION_SCHEMA,
    RESULT_SCHEMA as MOLECULAR_SCF_RESULT_SCHEMA,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, functional

from .ks_execution import KsExecutionPlan, compile_ks_execution_plan
from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod

BULK_KS_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-ks-resolution.v2"
_CPU_EXECUTION_STAGES = ("compiled-cpu", "production-domain")
_CPU_PROMOTION_STAGES = (*_CPU_EXECUTION_STAGES, "molecular-scf")
_CPU_PUBLIC_STAGES = (*_CPU_PROMOTION_STAGES, "public-method")
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


@dataclass(frozen=True)
class BulkKsResolution:
    """One evidence-qualified pure-semilocal bulk Libxc KS composition."""

    capability: CapabilityResolution
    method: MethodIR
    plan: KsExecutionPlan
    required_ingredients: tuple[str, ...]
    compiled_cpu_binding_identity: str
    compiled_cpu_result_identity: str
    backend: str = "cpu"

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a detached provenance record for the resolved composition."""
        return {
            "schema": BULK_KS_RESOLUTION_SCHEMA,
            "backend": self.backend,
            "capability": self.capability.to_payload(),
            "required_ingredients": list(self.required_ingredients),
            "compiled_cpu_binding_identity": self.compiled_cpu_binding_identity,
            "compiled_cpu_result_identity": self.compiled_cpu_result_identity,
            "method_identity": self.method.identity,
            "method_identifier": self.method.identifier,
            "plan_identity": self.plan.identity,
            "spin": self.method.spin,
            "reference": self.method.reference,
            "required_lowerers": list(self.plan.required_lowerers),
            "public_dft": self.capability.public_dft,
        }


def _require_exact_compiled_cpu(
    capability: BulkFunctionalCapability,
) -> dict[str, typing.Any]:
    stage = next(
        (
            item
            for item in capability.stage_evidence
            if item.stage == "compiled-cpu" and item.status == "pass"
        ),
        None,
    )
    if stage is None:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires passing compiled-CPU evidence"
        )
    try:
        qualification = validate_qualification(capability.name, stage.qualification)
        evidence = stage.evidence
        expected_anchor = f"#sha256={qualification['result_identity']}"
        if not isinstance(evidence, str) or not evidence.strip().endswith(
            expected_anchor
        ):
            raise ValueError("compiled-CPU stage evidence result identity mismatch")
        return qualification
    except (TypeError, ValueError) as exc:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires exact compiled-CPU qualification"
        ) from exc


def _require_exact_molecular_scf(capability: BulkFunctionalCapability) -> str:
    stage = next(
        (
            item
            for item in capability.stage_evidence
            if item.stage == "molecular-scf" and item.status == "pass"
        ),
        None,
    )
    if stage is None:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires passing molecular-SCF evidence"
        )
    try:
        qualification = stage.qualification
        if not isinstance(qualification, typing.Mapping):
            raise ValueError("molecular-SCF qualification must be a mapping")
        if qualification.get("schema") != ENDPOINT_COVERAGE_SCHEMA:
            raise ValueError("molecular-SCF endpoint coverage schema mismatch")
        if qualification.get("result_schema") != MOLECULAR_SCF_RESULT_SCHEMA:
            raise ValueError("molecular-SCF result schema mismatch")
        if (
            qualification.get("qualification_schema")
            != MOLECULAR_SCF_QUALIFICATION_SCHEMA
        ):
            raise ValueError("molecular-SCF qualification schema mismatch")
        expected_coverage = [
            {"backend": "cpu", "spin": spin, "products": ["energy"]}
            for spin in SPIN_LAYOUTS
        ]
        if qualification.get("coverage") != expected_coverage:
            raise ValueError("molecular-SCF coverage is not exact dual-spin CPU energy")
        result_identity = qualification.get("result_identity")
        if (
            not isinstance(result_identity, str)
            or len(result_identity) != 64
            or any(ch not in "0123456789abcdef" for ch in result_identity)
        ):
            raise ValueError("molecular-SCF result identity is invalid")
        evidence = stage.evidence
        if (
            not isinstance(evidence, str)
            or not evidence.strip().endswith(f"#sha256={result_identity}")
        ):
            raise ValueError("molecular-SCF evidence result identity mismatch")
        return result_identity
    except (TypeError, ValueError) as exc:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires exact molecular-SCF qualification"
        ) from exc


def _resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
    required_stages: tuple[str, ...],
) -> BulkKsResolution:
    if backend != "cpu":
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution is currently qualified only for CPU"
        )

    capability = functional_capability(name, evidence=evidence)
    if capability.name not in AUTO_BULK_COMPONENTS:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution requires a non-curated "
            "AUTO_BULK_COMPONENTS registration"
        )

    unsupported = tuple(
        ingredient
        for ingredient in capability.required_ingredients
        if ingredient not in _SUPPORTED_INGREDIENTS
    )
    if unsupported:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution does not support ingredients "
            f"{unsupported!r}"
        )

    qualified = resolve_capability(
        capability.name,
        required_stages=required_stages,
        evidence=evidence,
    )
    if qualified.identity != capability.identity:
        raise RuntimeError(
            "bulk Libxc capability identity changed during KS resolution"
        )
    compiled_cpu = _require_exact_compiled_cpu(capability)
    if "molecular-scf" in required_stages:
        _require_exact_molecular_scf(capability)

    functional_spec = functional(capability.name, spin=spin)
    method = MethodIR(
        identifier=(
            identifier if identifier is not None else f"LIBXC:{capability.name}"
        ),
        spin=spin,
        primitives=(SemilocalXCPrimitive(functional_spec),),
    )
    plan = compile_ks_execution_plan(method)
    if plan.exchange or plan.nonlocal_correlation is not None or plan.post_scf:
        raise RuntimeError(
            "pure semilocal bulk KS resolution produced extra primitives"
        )

    return BulkKsResolution(
        capability=qualified,
        method=method,
        plan=plan,
        required_ingredients=capability.required_ingredients,
        compiled_cpu_binding_identity=compiled_cpu["binding_identity"],
        compiled_cpu_result_identity=compiled_cpu["result_identity"],
    )


def resolve_bulk_ks_candidate(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve the CPU candidate used to produce molecular-SCF evidence.

    Candidate execution remains fail-closed on compiled-CPU and complete
    production-domain evidence. Requiring molecular-SCF here would be circular:
    this is the exact plan that the qualification runner must execute to create
    that evidence.
    """
    return _resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
        required_stages=_CPU_EXECUTION_STAGES,
    )


def resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve one promoted automatic Libxc registration into a pure KS plan.

    Ordinary consumers require compiled-CPU, production-domain and molecular-SCF
    evidence. Qualification producers must use resolve_bulk_ks_candidate instead
    of manufacturing the final endpoint stage.
    """
    return _resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
        required_stages=_CPU_PROMOTION_STAGES,
    )


def resolve_public_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve one exact public CPU-energy bulk Libxc endpoint into a KS plan.

    Public routing is stricter than ordinary promoted execution: the retained
    public-method receipt must validate for this exact backend/product/spin
    endpoint. The returned KS resolution still revalidates exact compiled-CPU
    qualification, so public evidence cannot bypass the executable artifact
    identity owned by this module.
    """
    endpoint = resolve_endpoint_capability(
        name,
        backend=backend,
        product="energy",
        spin=spin,
        require_public=True,
        evidence=evidence,
    )
    if not endpoint.public_dft:
        raise RuntimeError("public bulk Libxc endpoint resolved without public admission")

    resolved = _resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
        required_stages=_CPU_PUBLIC_STAGES,
    )
    if resolved.capability.identity != endpoint.identity:
        raise RuntimeError(
            "bulk Libxc endpoint identity changed during public KS resolution"
        )
    if resolved.method.spin != endpoint.spin:
        raise RuntimeError("bulk Libxc endpoint spin changed during public KS resolution")
    return resolved
