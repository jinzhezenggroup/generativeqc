"""Default-allow MethodIR/KS resolution for automatic bulk Libxc registrations.

Imported semilocal functionals are admitted by structural capability, not by a
positive evidence whitelist. A representable non-curated LDA/GGA/rho-sigma-tau
meta-GGA is public on the supported CPU path unless an explicit functional-
specific defect is present in the Libxc blacklist.

Qualification producers retain a stricter candidate entry point so regression
campaigns can attach compiled/runtime evidence without making that evidence a
user-facing admission gate.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.xc.capability_resolution import (
    CapabilityResolution,
    resolve_capability,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import validate_qualification
from vibeqc_compiler.xc.libxc_blacklist import blacklist_reason
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    BulkFunctionalCapability,
    functional_capability,
)
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS, functional

from .ks_execution import KsExecutionPlan, compile_ks_execution_plan
from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod

BULK_KS_RESOLUTION_SCHEMA = "vibeqc.bulk-libxc-ks-resolution.v2"
_CPU_EXECUTION_STAGES = ("compiled-cpu", "production-domain")
_SUPPORTED_INGREDIENTS = frozenset(("rho", "sigma", "tau"))


@dataclass(frozen=True)
class BulkKsResolution:
    """One structurally admitted pure-semilocal bulk Libxc KS composition."""

    capability: BulkFunctionalCapability | CapabilityResolution
    method: MethodIR
    plan: KsExecutionPlan
    required_ingredients: tuple[str, ...]
    compiled_cpu_binding_identity: str | None = None
    compiled_cpu_result_identity: str | None = None
    public_dft: bool = False
    admission: str = "qualification"
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
            "public_dft": self.public_dft,
            "admission": self.admission,
        }


def _require_exact_compiled_cpu(
    capability: BulkFunctionalCapability,
) -> dict[str, typing.Any]:
    """Validate the exact compiled-CPU receipt for qualification tooling."""
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
            "automatic bulk Libxc qualification requires passing compiled-CPU evidence"
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
            "automatic bulk Libxc qualification requires exact compiled-CPU evidence"
        ) from exc


def _structural_capability(
    name: str,
    *,
    evidence: typing.Mapping[str, typing.Any] | None = None,
    enforce_blacklist: bool,
) -> BulkFunctionalCapability:
    capability = functional_capability(name, evidence=evidence)
    if capability.name not in AUTO_BULK_COMPONENTS:
        raise UnsupportedMethod(
            "automatic bulk Libxc KS requires a non-curated imported semilocal "
            "registration"
        )

    unsupported = tuple(
        ingredient
        for ingredient in capability.required_ingredients
        if ingredient not in _SUPPORTED_INGREDIENTS
    )
    if unsupported:
        raise UnsupportedMethod(
            f"automatic bulk Libxc KS does not support ingredients {unsupported!r}"
        )

    if enforce_blacklist:
        reason = blacklist_reason(capability.name)
        if reason is not None:
            raise UnsupportedMethod(
                f"automatic bulk Libxc functional {capability.name} is blacklisted: "
                f"{reason}"
            )
    return capability


def _build_resolution(
    capability: BulkFunctionalCapability | CapabilityResolution,
    *,
    functional_name: str,
    required_ingredients: tuple[str, ...],
    spin: str,
    identifier: str | None,
    compiled_cpu: dict[str, typing.Any] | None,
    public_dft: bool,
    admission: str,
) -> BulkKsResolution:
    functional_spec = functional(functional_name, spin=spin)
    method = MethodIR(
        identifier=identifier if identifier is not None else f"LIBXC:{functional_name}",
        spin=spin,
        primitives=(SemilocalXCPrimitive(functional_spec),),
    )
    plan = compile_ks_execution_plan(method)
    if plan.exchange or plan.nonlocal_correlation is not None or plan.post_scf:
        raise RuntimeError(
            "pure semilocal bulk KS resolution produced extra primitives"
        )

    return BulkKsResolution(
        capability=capability,
        method=method,
        plan=plan,
        required_ingredients=required_ingredients,
        compiled_cpu_binding_identity=(
            None if compiled_cpu is None else compiled_cpu["binding_identity"]
        ),
        compiled_cpu_result_identity=(
            None if compiled_cpu is None else compiled_cpu["result_identity"]
        ),
        public_dft=public_dft,
        admission=admission,
    )


def resolve_bulk_ks_candidate(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve the stricter CPU candidate used by qualification campaigns.

    The blacklist is deliberately ignored here so a previously blocked
    functional can be requalified after its implementation is fixed. Evidence
    remains mandatory for this testing-only entry point.
    """
    if backend != "cpu":
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution currently supports CPU only"
        )
    capability = _structural_capability(
        name, evidence=evidence, enforce_blacklist=False
    )
    qualified = resolve_capability(
        capability.name,
        required_stages=_CPU_EXECUTION_STAGES,
        evidence=evidence,
    )
    if qualified.identity != capability.identity:
        raise RuntimeError(
            "bulk Libxc capability identity changed during KS resolution"
        )
    compiled_cpu = _require_exact_compiled_cpu(capability)
    return _build_resolution(
        qualified,
        functional_name=capability.name,
        required_ingredients=capability.required_ingredients,
        spin=spin,
        identifier=identifier,
        compiled_cpu=compiled_cpu,
        public_dft=False,
        admission="qualification-evidence",
    )


def resolve_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Resolve one imported semilocal Libxc registration by default.

    Public admission is negative-list based. Representation and ingredient
    capability are checked generically; no compiled-CPU, molecular-SCF or
    public-method receipt is required. The evidence argument remains accepted
    only for API compatibility and does not grant or revoke user capability.
    """
    if backend != "cpu":
        raise UnsupportedMethod(
            "automatic bulk Libxc KS resolution currently supports CPU only"
        )
    _ = evidence
    capability = _structural_capability(name, evidence=None, enforce_blacklist=True)
    return _build_resolution(
        capability,
        functional_name=capability.name,
        required_ingredients=capability.required_ingredients,
        spin=spin,
        identifier=identifier,
        compiled_cpu=None,
        public_dft=True,
        admission="default-allow/explicit-blacklist",
    )


def resolve_public_bulk_ks(
    name: str,
    *,
    spin: str = "unpolarized",
    backend: str = "cpu",
    evidence: typing.Mapping[str, typing.Any] | None = None,
    identifier: str | None = None,
) -> BulkKsResolution:
    """Compatibility alias for default-allow automatic Libxc public routing."""
    return resolve_bulk_ks(
        name,
        spin=spin,
        backend=backend,
        evidence=evidence,
        identifier=identifier,
    )
