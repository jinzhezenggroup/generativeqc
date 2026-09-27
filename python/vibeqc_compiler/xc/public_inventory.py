"""Retained public CPU evidence for automatically imported Libxc functionals.

A campaign artifact is not a public capability by itself.  This module accepts
only source-controlled inventories whose four prerequisite stage envelopes
revalidate against the exact current compiler/source identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "vibeqc.libxc-public-cpu-inventory/v1"
REQUIRED_STAGES = (
    "compiled-cpu",
    "production-domain",
    "molecular-scf",
    "public-method",
)
DEFAULT_INVENTORY = Path(__file__).with_name("libxc_public_cpu.json")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unsupported retained Libxc public inventory schema")
    revision = payload.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("retained Libxc public inventory requires a source revision")
    functionals = payload.get("functionals")
    if not isinstance(functionals, dict):
        raise TypeError("retained Libxc public inventory functionals must be a mapping")
    return payload


def _evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    stages = record.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(REQUIRED_STAGES):
        raise ValueError("retained Libxc public record has incomplete stage evidence")
    return {stage: dict(stages[stage]) for stage in REQUIRED_STAGES}


def load_public_inventory(path: Path = DEFAULT_INVENTORY) -> dict[str, dict[str, Any]]:
    """Return exact current public evidence, rejecting stale/tampered records."""

    from vibeqc_compiler.method.bulk_ks import resolve_public_bulk_ks

    from .libxc_bulk_capabilities import functional_capability

    payload = _read(Path(path))
    result: dict[str, dict[str, Any]] = {}
    for name, raw in sorted(payload["functionals"].items()):
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise TypeError("retained Libxc public inventory entry is malformed")
        evidence = _evidence(raw)
        capability = functional_capability(name, evidence=evidence)
        expected = raw.get("capability_identity")
        if capability.identity != expected:
            raise ValueError(f"retained Libxc public identity is stale for {name}")
        for spin in ("unpolarized", "polarized"):
            resolved = resolve_public_bulk_ks(name, spin=spin, evidence=evidence)
            if not resolved.capability.public_dft:
                raise ValueError(
                    f"retained Libxc public admission is incomplete for {name}"
                )
        result[capability.name] = evidence
    return result


def available_public_functionals(path: Path = DEFAULT_INVENTORY) -> tuple[str, ...]:
    """Return the installed evidence-backed public CPU Libxc registrations."""

    return tuple(load_public_inventory(path))


def public_evidence(name: str, path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    """Return detached evidence for one retained public registration."""

    inventory = load_public_inventory(path)
    key = name.upper()
    try:
        return dict(inventory[key])
    except KeyError as exc:
        raise ValueError(
            f"Libxc functional is not retained as public: {name!r}"
        ) from exc
