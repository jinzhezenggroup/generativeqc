"""Audit CUDA AOT capabilities and enforce source-matched default decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from tools.generate_build_identity import _inventory, _source_identity
from tools.generativeqc_validation.cuda_aot_evidence import (
    apply_card_comparison_evidence,
    apply_compile_evidence,
    apply_endpoint_evidence,
)
from tools.generativeqc_validation.cuda_aot_policy import (
    MANIFEST,
    capability_audit,
    check_defaults,
)

DECISIONS = ROOT / "manifests/maintenance/cuda_aot_default_decisions.json"
AUDIT = ROOT / "manifests/maintenance/cuda_aot_capability_audit.json"
EVIDENCE = ROOT / "manifests/maintenance/cuda_aot_capability_evidence.json"


def main() -> int:
    """Refresh only unmeasured audit data, never a promotion or its baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument(
        "--verify-compile-inputs",
        action="store_true",
        help="Opt-in reproduction of historical compile inputs; never a required refactor fingerprint",
    )
    args = parser.parse_args()
    try:
        source_identity = _source_identity(
            ROOT, _inventory(ROOT, ROOT / "cmake/GenerativeQCSourceIdentity.json")
        )
        report = capability_audit(ROOT / MANIFEST)
        report["native_source_identity"] = source_identity
        if EVIDENCE.exists():
            evidence = json.loads(EVIDENCE.read_text())
            apply_compile_evidence(
                report,
                ROOT,
                evidence,
                source_identity,
                verify_inputs=args.verify_compile_inputs,
            )
            if "endpoint_smoke" in evidence:
                apply_endpoint_evidence(
                    report, ROOT, evidence["endpoint_smoke"], source_identity
                )
            if "same_binary_card_comparison" in evidence:
                apply_card_comparison_evidence(
                    report,
                    ROOT,
                    evidence["same_binary_card_comparison"],
                    source_identity,
                )
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.write_audit:
            AUDIT.write_text(rendered)
        elif AUDIT.read_text() != rendered:
            raise ValueError(
                "CUDA AOT audit is stale; run --write-audit (not qualification)"
            )
        result = check_defaults(
            ROOT, json.loads(DECISIONS.read_text()), source_identity=source_identity
        )
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        print(f"CUDA AOT policy rejected: {error}", file=sys.stderr)
        return 1
    print(
        f"CUDA AOT audit: {len(report['rows'])} consumer features, {len(report['routes'])} representative targets"
    )
    print("CUDA AOT default decisions: " + json.dumps(result, sort_keys=True))
    if smoke := report.get("endpoint_smoke_evidence"):
        failed = sum(
            phase["status"] == "failed-scientific-gate"
            for arms in smoke["results"].values()
            for arm in arms.values()
            for phase in arm["phases"].values()
        )
        print(
            f"CUDA AOT diagnostic endpoints: {smoke['source_status']}, {failed} failed arm/phase groups retained; not global-default qualification"
        )
    if comparison := report.get("same_binary_card_comparison"):
        secondary = comparison["secondary"]
        print(
            f"CUDA AOT same-binary card diagnostic: {secondary['device_name']}, {secondary['source_status']}; not cross-SM or performance qualification"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
