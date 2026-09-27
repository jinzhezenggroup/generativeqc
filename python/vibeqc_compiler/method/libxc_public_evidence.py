"""Installed evidence for public automatic Libxc endpoints.

The broad qualification campaign runs outside routine PR CI. This module keeps
consumption fail-closed: generated evidence is usable only when every retained
stage is bound to the capability identity computed from the currently installed
compiler sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

from ._generated_libxc_public_evidence import (
    PUBLIC_EVIDENCE,
    PUBLIC_EVIDENCE_PROVENANCE,
)


def installed_public_evidence(name: str) -> dict[str, Any] | None:
    """Return exact installed public evidence, or ``None`` when absent/stale."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("installed Libxc public evidence requires a non-empty name")
    key = name.upper()
    payload = PUBLIC_EVIDENCE.get(key)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"installed Libxc evidence for {key} is not a mapping")

    current_identity = functional_capability(key).identity
    stage_subjects = {
        envelope.get("subject_identity")
        for envelope in payload.values()
        if isinstance(envelope, Mapping)
    }
    if stage_subjects != {current_identity}:
        return None

    detached = deepcopy(dict(payload))
    capability = functional_capability(key, evidence=detached)
    if "public-method" not in capability.qualified_stages:
        return None
    return detached


def installed_public_functionals() -> tuple[str, ...]:
    """Return exact-current registrations with usable installed public evidence."""

    admitted: list[str] = []
    for name in sorted(PUBLIC_EVIDENCE):
        if installed_public_evidence(name) is not None:
            admitted.append(name)
    return tuple(admitted)


def installed_public_evidence_provenance() -> dict[str, Any]:
    """Return detached provenance for the generated installed inventory."""
    return deepcopy(dict(PUBLIC_EVIDENCE_PROVENANCE))


__all__ = [\n    "installed_public_evidence",\n    "installed_public_evidence_provenance",\n    "installed_public_functionals",\n]
