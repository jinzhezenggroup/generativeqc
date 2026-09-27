"""Render exact broad Libxc promotion evidence into an installable Python module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

STAGES = ("compiled-cpu", "production-domain", "molecular-scf", "public-method")
SCHEMA = "vibeqc.installed-libxc-public-evidence/v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def collect_public_evidence(root: Path) -> dict[str, dict[str, Any]]:
    """Collect only exact-current public passes from one broad campaign artifact."""
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("complete") is not True:
        raise ValueError("broad Libxc promotion summary is not complete")

    result: dict[str, dict[str, Any]] = {}
    for row in summary.get("functionals", []):
        if not isinstance(row, dict) or row.get("status") != "pass":
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("passing broad-matrix row is missing a functional name")
        current_identity = functional_capability(name).identity
        if row.get("capability_identity") != current_identity:
            raise ValueError(
                f"{name} broad evidence is stale for the current compiler identity"
            )
        stages: dict[str, Any] = {}
        for stage in STAGES:
            artifact = _read_json(root / name / f"{stage}.json")
            envelope = artifact.get("stage_evidence")
            if not isinstance(envelope, dict):
                raise TypeError(f"{name}/{stage} is missing stage_evidence")
            if envelope.get("subject_identity") != current_identity:
                raise ValueError(f"{name}/{stage} subject identity mismatch")
            stages[stage] = envelope
        qualified = functional_capability(name, evidence=stages)
        if "public-method" not in qualified.qualified_stages:
            raise ValueError(f"{name} evidence does not qualify public-method")
        result[name] = stages
    if not result:
        raise ValueError("broad Libxc artifact contains no exact-current public passes")
    return result


def render_module(root: Path, *, source: str) -> str:
    evidence = collect_public_evidence(root)
    summary_bytes = (root / "summary.json").read_bytes()
    provenance = {
        "schema": SCHEMA,
        "source": source,
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }
    return (
        '"""Generated installed automatic-Libxc public evidence.\n\n'
        "Do not edit by hand. Regenerate with tools/render_libxc_public_evidence.py.\n"
        '"""\n\n'
        "from types import MappingProxyType\n\n"
        f"PUBLIC_EVIDENCE_PROVENANCE = MappingProxyType({provenance!r})\n"
        f"PUBLIC_EVIDENCE = MappingProxyType({evidence!r})\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render_module(args.artifact_root, source=args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
