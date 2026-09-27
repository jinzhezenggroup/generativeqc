"""Load retained automatic Libxc public-method evidence fail-closed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .libxc_bulk_capabilities import functional_capability

MATRIX_SCHEMA = "vibeqc.libxc-broad-promotion-matrix/v2"
PUBLIC_STAGES = (
    "production-domain",
    "compiled-cpu",
    "molecular-scf",
    "public-method",
)


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"retained Libxc evidence must be an object: {path}")
    return payload


def _stage_envelope(path: Path, expected: str) -> dict[str, Any]:
    payload = _object(path)
    envelope: Any = payload.get("stage_evidence", payload)
    if not isinstance(envelope, dict):
        raise TypeError(f"{expected} evidence must contain an object envelope")
    if envelope.get("stage") != expected:
        raise ValueError(f"expected {expected} evidence at {path}")
    return dict(envelope)


def load_retained_public_evidence(
    root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Return exact per-functional stage evidence from one retained broad campaign.

    The directory is data, not authority. Every passing row is rebound to the
    current imported functional identity and replayed through the capability
    validator before it can be returned to a public resolver.
    """
    directory = Path(root)
    summary = _object(directory / "summary.json")
    if summary.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unsupported retained Libxc broad-matrix schema")
    if summary.get("complete") is not True:
        raise ValueError("retained Libxc broad matrix is incomplete")

    rows = summary.get("functionals")
    if not isinstance(rows, list):
        raise TypeError("retained Libxc broad matrix requires functional rows")

    retained: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "pass":
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("invalid retained Libxc functional name")
        if name in retained:
            raise ValueError(f"duplicate retained Libxc functional {name!r}")

        intrinsic = functional_capability(name)
        if row.get("capability_identity") != intrinsic.identity:
            raise ValueError(
                f"retained Libxc evidence is stale for current functional {name}"
            )

        stages = {
            stage: _stage_envelope(directory / name / f"{stage}.json", stage)
            for stage in PUBLIC_STAGES
        }
        qualified = functional_capability(name, evidence=stages)
        if not qualified.public_dft:
            raise ValueError(
                f"retained Libxc evidence does not qualify public method {name}"
            )
        retained[name] = stages

    if not retained:
        raise ValueError("retained Libxc broad matrix has no public methods")
    return retained


__all__ = [
    "MATRIX_SCHEMA",
    "PUBLIC_STAGES",
    "load_retained_public_evidence",
]
