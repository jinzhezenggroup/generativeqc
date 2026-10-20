"""Shared, advisory materialization policy over analyzer-owned evidence.

This module does not parse source, prove support, or change equations/layouts.
Source auditors and TensorIR retain their own evidence and certificate scope;
only the presentation and conservative recommendation policy are shared here.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SupportKind = Literal[
    "exact-address-domain",
    "full-domain",
    "union-upper-bound",
    "unknown",
    "missing-structured-ir",
]


# Bind an imported policy to its source revision. A long-lived audit process
# must never label old executable policy with a newer on-disk file's digest.
_IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_identity() -> str:
    """Return the imported policy digest, failing closed after an on-disk edit."""
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != _IMPORTED_SOURCE_SHA256
    ):
        raise ValueError(
            "materialization policy source changed after import; restart the audit"
        )
    return _IMPORTED_SOURCE_SHA256


@dataclass(frozen=True, init=False)
class MaterializationRuleRequest:
    """Immutable analyzer-owned request for a compiler or external source rule.

    Descriptors are data, not a proof of separability, ABI compatibility or
    numerical admission. Rules must prove their own applicability before use.
    """

    _payload_json: str

    def __init__(
        self,
        *,
        origin: str,
        source_identity: str,
        subject: Mapping[str, Any],
        producer: Mapping[str, Any],
        consumers: Sequence[Mapping[str, Any]],
        retained_output: bool,
        allow_reassociation: bool,
        precision_contract: Any = None,
        max_materialized_bytes: int | None = None,
        constraints: Mapping[str, Any] | None = None,
    ) -> None:
        """Snapshot source/consumer, arithmetic and resource requirements."""
        if (
            not isinstance(origin, str)
            or not origin
            or not isinstance(source_identity, str)
            or not source_identity
        ):
            raise ValueError(
                "rule requests require nonempty origin and source identity"
            )
        if type(retained_output) is not bool or type(allow_reassociation) is not bool:
            raise TypeError(
                "rule request output and arithmetic permissions must be Boolean"
            )
        if max_materialized_bytes is not None:
            # Keep the existing stdlib-only source-audit policy import independent.
            from .resources import checked_bytes

            checked_bytes(max_materialized_bytes, "materialization byte budget")
        payload = {
            "schema": "generativeqc.materialization-rule-request.v1",
            "origin": origin,
            "source_identity": source_identity,
            "subject": dict(subject),
            "producer": dict(producer),
            "consumers": [dict(consumer) for consumer in consumers],
            "retained_output": retained_output,
            "allow_reassociation": allow_reassociation,
            "precision_contract": precision_contract,
            "max_materialized_bytes": max_materialized_bytes,
            "constraints": dict(constraints or {}),
            "required_contracts": [
                "exact-source-equivalence",
                "all-consumer-layouts",
                "input-alias-and-lifetime",
                "derivative-semantics",
                "complete-resource-and-work-accounting",
            ],
            "external_rule_families": []
            if retained_output
            else [
                "factorized-source",
                "block-source",
            ],
            "automatic_rewrite": False,
        }
        object.__setattr__(
            self,
            "_payload_json",
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    @property
    def request_hash(self) -> str:
        """Bind identity to the complete immutable request, including budget."""
        return hashlib.sha256(self._payload_json.encode()).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        """Return a detached, data-only request without granting rule admission."""
        return {**json.loads(self._payload_json), "request_hash": self.request_hash}


def materialization_diagnostic(
    *,
    origin: str,
    subject: Mapping[str, Any],
    dense_elements: str | int,
    dense_growth_degree: int | None = None,
    support_kind: SupportKind = "missing-structured-ir",
    written_elements: str | None = None,
    written_growth_degree: int | None = None,
    domains: Sequence[Mapping[str, Any]] = (),
    expansion_ratio: str | None = None,
    conditions: Sequence[str] = (),
    certificate_scope: str,
    unknown_reason: str | None = None,
    layout: str = "logical-dense",
    retained_output: bool = False,
    observed_source_roles: Sequence[Mapping[str, Any]] = (),
    rule_request: MaterializationRuleRequest | None = None,
) -> dict[str, Any]:
    """Normalize evidence without upgrading upper bounds, roles, or unknowns.

    An exact address-domain certificate permits reviewing structured storage;
    it never proves strict savings, consumer compatibility, or runtime routing.
    A full-domain certificate precludes exact-zero storage advice. Tensor shape
    alone is not a zero-support certificate, even for a factorized expression.
    """
    if support_kind not in {
        "exact-address-domain",
        "full-domain",
        "union-upper-bound",
        "unknown",
        "missing-structured-ir",
    }:
        raise ValueError("unrecognized materialization support evidence")
    if support_kind in {
        "exact-address-domain",
        "full-domain",
        "union-upper-bound",
    } and (written_elements is None or not domains):
        raise ValueError("support evidence requires analyzer-owned domains and count")

    blockers: list[dict[str, str]] = []

    def blocked(category: str, code: str, reason: str) -> None:
        blockers.append({"category": category, "code": code, "reason": reason})

    recommendation = None
    representations: list[str] = []
    producer_status = "support-certified"
    if support_kind == "exact-address-domain":
        triangle = any("triangle" in str(domain.get("kind")) for domain in domains)
        representations = ["packed"] if triangle else ["block", "diagonal"]
        recommendation = (
            "Review "
            + "/".join(representations)
            + " storage and all consumers before changing the dense layout."
        )
    elif support_kind == "full-domain":
        blocked(
            "producer",
            "full-write-domain",
            "The producer writes the full dense domain; no exact-zero storage reduction is certified.",
        )
    elif support_kind == "union-upper-bound":
        blocked(
            "producer",
            "union-cardinality-unresolved",
            "The certificate bounds a union of written coordinates; overlap and exact cardinality remain unresolved.",
        )
    elif support_kind == "unknown":
        producer_status = "unsupported"
        blocked(
            "producer",
            "unsupported-write-support",
            unknown_reason
            or "The producer is outside the analyzer's support-proof subset.",
        )
    else:
        producer_status = "not-certified"
        blocked(
            "structured_ir",
            "missing-exact-support",
            "Logical tensor shape does not establish exact written or zero support.",
        )

    blocked(
        "consumer",
        "downstream-layout-unresolved",
        "Downstream dense-layout requirements have not been proved; an unknown consumer is not a dense-layout requirement.",
    )
    if retained_output:
        blocked(
            "abi_layout",
            "retained-output-layout",
            "The declared TensorIR output shape/layout is retained; a representation change requires an explicit output contract change.",
        )
    elif layout == "aggregate-member-vector":
        blocked(
            "abi_layout",
            "aggregate-member-layout",
            "The observed dense member-vector boundary does not establish aggregate aliasing or downstream ABI compatibility.",
        )
    elif layout == "dense-vector":
        blocked(
            "abi_layout",
            "dense-vector-layout",
            "The declared dense vector is a layout boundary; downstream ABI compatibility is unverified.",
        )

    result = {
        "schema": "generativeqc.materialization-diagnostic.v1",
        "origin": origin,
        "subject": deepcopy(dict(subject)),
        "storage": {
            "layout": layout,
            "dense_elements": dense_elements,
            "dense_growth_degree": dense_growth_degree,
            "retained_output": retained_output,
        },
        "support": {
            "status": support_kind,
            "producer_status": producer_status,
            "written_elements": written_elements,
            "written_growth_degree": written_growth_degree,
            "domains": deepcopy(list(domains)),
            "certificate_scope": certificate_scope,
            "conditions": list(conditions),
            "strict_reduction_proven": False,
        },
        "expansion_ratio": expansion_ratio
        if support_kind == "exact-address-domain"
        else None,
        "recommendation": recommendation,
        "representation_candidates": representations,
        "blockers": blockers,
        "observed_source_roles": deepcopy(list(observed_source_roles)),
        "consumer_abi_verified": False,
        "runtime_endpoint_selection_proven": False,
        "automatic_rewrite": False,
    }
    if rule_request is not None:
        if not isinstance(rule_request, MaterializationRuleRequest):
            raise TypeError("materialization rule request must be an immutable request")
        request = rule_request.to_payload()
        if (
            request["origin"] != origin
            or request["subject"] != dict(subject)
            or request["retained_output"] != retained_output
        ):
            raise ValueError(
                "materialization rule request does not match its diagnostic"
            )
        result["rule_request"] = request
    return result
