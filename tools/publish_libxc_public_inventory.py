"""Publish passing broad Libxc CPU evidence into the retained public inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.xc.public_inventory import INVENTORY_SCHEMA, REQUIRED_STAGES
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability


def publish(campaign: Path, revision: str, output: Path) -> dict:
    if not revision.strip():
        raise ValueError("retained Libxc public inventory requires a source revision")
    summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "vibeqc.libxc-broad-promotion-matrix/v2":
        raise ValueError("unsupported broad Libxc campaign schema")
    records = {}
    for row in summary.get("functionals", ()):
        if row.get("status") != "pass" or row.get("stages", {}).get("public-method") != "pass":
            continue
        name = row["name"]
        stages = {}
        for stage in REQUIRED_STAGES:
            stage_path = campaign / name / f"{stage}.json"
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
            envelope = payload.get("stage_evidence")
            if not isinstance(envelope, dict):
                raise ValueError(f"{name} {stage} omitted stage_evidence")
            stages[stage] = envelope
        capability = functional_capability(name, evidence=stages)
        if capability.identity != row.get("capability_identity") or not capability.public_dft:
            raise ValueError(f"{name} campaign evidence does not revalidate on this tree")
        records[name] = {
            "capability_identity": capability.identity,
            "family": capability.family,
            "required_ingredients": list(capability.required_ingredients),
            "stages": stages,
        }
    payload = {
        "schema": INVENTORY_SCHEMA,
        "revision": revision,
        "functionals": dict(sorted(records.items())),
    }
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = publish(args.campaign, args.revision, args.output)
    print(f"retained {len(payload['functionals'])} public Libxc CPU functionals -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
