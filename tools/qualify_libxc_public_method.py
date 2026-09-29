"""Promote exact automatic Libxc CPU evidence into public-method evidence.

This is an evidence assembly tool, not a scientific executor. It consumes the
three already-qualified CPU stages and emits the exact public-method receipt
owned by `generativeqc_compiler.xc.public_method_evidence`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from generativeqc_compiler.common.provenance import atomic_json
from generativeqc_compiler.xc.public_method_evidence import build_result, stage_evidence


def _stage_payload(value: Mapping[str, Any], expected: str) -> dict[str, Any]:
    candidate: Any = value.get("stage_evidence", value)
    if not isinstance(candidate, Mapping):
        raise TypeError(f"{expected} evidence must be a mapping")
    if candidate.get("stage") != expected:
        raise ValueError(f"expected {expected} stage evidence")
    return dict(candidate)


def _load_stage(path: Path, expected: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{expected} evidence file must contain an object")
    return _stage_payload(raw, expected)


def qualify_public_method(
    name: str,
    *,
    compiled_cpu_evidence: Mapping[str, Any],
    production_domain_evidence: Mapping[str, Any],
    molecular_scf_evidence: Mapping[str, Any],
    evidence: str,
) -> dict[str, Any]:
    """Build exact public CPU-energy evidence from retained prerequisite stages."""
    prerequisites = {
        "compiled-cpu": _stage_payload(compiled_cpu_evidence, "compiled-cpu"),
        "production-domain": _stage_payload(
            production_domain_evidence, "production-domain"
        ),
        "molecular-scf": _stage_payload(molecular_scf_evidence, "molecular-scf"),
    }
    result = build_result(
        name,
        prerequisite_evidence=prerequisites,
        evidence=evidence,
    )
    envelope = stage_evidence(
        name,
        result,
        prerequisite_evidence=prerequisites,
    )
    return {
        "schema": "generativeqc.libxc-public-method-campaign/v1",
        "functional": name,
        "prerequisites": prerequisites,
        "result": result,
        "stage_evidence": envelope,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--compiled-cpu", type=Path, required=True)
    parser.add_argument("--production-domain", type=Path, required=True)
    parser.add_argument("--molecular-scf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = qualify_public_method(
        args.name,
        compiled_cpu_evidence=_load_stage(args.compiled_cpu, "compiled-cpu"),
        production_domain_evidence=_load_stage(
            args.production_domain, "production-domain"
        ),
        molecular_scf_evidence=_load_stage(args.molecular_scf, "molecular-scf"),
        evidence=args.evidence,
    )
    atomic_json(args.output, payload)
    status = payload["stage_evidence"]["status"]
    print(f"{args.name}: public-method={status} -> {args.output}")
    return 1 if args.require_pass and status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
