"""Regenerate M06 endpoint ledgers from checksummed, offline Git evidence.

The original capsules remain in an existing ancestor, not an external archive.
Only auxiliary profiling and vector payloads move out of the current checkout;
all endpoint observations, work counts, source identities and negative verdicts
remain visible. This does not qualify new scientific code or rerun any GPU work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.restore_retained_evidence import restore_snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "benchmarks/results/m06-rks-force-powers-20261014/full-evidence.json"
SHARED_LEDGER = "benchmarks/results/m06-rks-force-powers-20261014/summary.json.xz"
FULL_ONLY = (
    "build_files",
    "logs",
    "reproduction",
    "receipts",
    "point_arrays",
    "independent_references",
)


def _without_component_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_component_timings(child)
            for key, child in value.items()
            if not key.endswith(("_seconds", "_ms"))
        }
    if isinstance(value, list):
        return [_without_component_timings(child) for child in value]
    return value


def _project(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _project(child) for key, child in value.items()}
        if "complete_seconds" in result and "phase" in result:
            result.pop("forces", None)
        if "native_ks_diagnostic" in result:
            result["native_ks_diagnostic"].pop("history", None)
            result["native_ks_diagnostic"].pop("components", None)
        if "native_force_components" in result:
            result["native_force_components"] = _without_component_timings(
                result["native_force_components"]
            )
        return result
    if isinstance(value, list):
        return [_project(child) for child in value]
    return value


def compact(payload: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """Project every observation without rounding, filtering or recertification.

    Missing J/K execution counts remain null. Per-call force/work policy is kept,
    but intermediate SCF histories and component profiling are recoverable only
    in the original capsule. Endpoint/preparation times are never projected out.
    """
    result = copy.deepcopy(payload)
    omitted = [key for key in FULL_ONLY if key in result]
    for key in omitted:
        del result[key]
    result = _project(result)
    result["original_schema"] = result["schema"]
    result["schema"] = "generativeqc.m06-force-endpoint-ledger.v1"
    result["retention"] = {
        "snapshot_manifest": SNAPSHOT,
        "original_capsule": original,
        "full_only_top_level": omitted,
        "full_only_nested": [
            "endpoint forces",
            "native_ks_diagnostic.history/components",
            "native_force_components fields ending _seconds/_ms",
        ],
    }
    return result


def main() -> None:
    """Verify the entire original snapshot before regenerating both ledgers."""
    snapshot_path = ROOT / SNAPSHOT
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    with tempfile.TemporaryDirectory(prefix="m06-force-evidence-") as temporary:
        restored = restore_snapshot(
            Path(temporary) / "original", manifest=snapshot_path
        )
        ledgers = {}
        manifests = {}
        for entry in snapshot["files"]:
            if not entry["path"].endswith("/summary.json.xz"):
                continue
            path = Path(entry["path"])
            payload = json.loads(lzma.decompress((restored / path).read_bytes()))
            campaign = path.parent.name
            ledgers[campaign] = compact(payload, entry)
            manifest_path = path.parent / "manifest.json"
            manifests[manifest_path] = json.loads(
                (restored / manifest_path).read_bytes()
            )
        packed = lzma.compress(
            json.dumps(
                ledgers, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
            preset=9 | lzma.PRESET_EXTREME,
        )
        for manifest_path, manifest in manifests.items():
            manifest["summary"] = {
                "path": SHARED_LEDGER,
                "member": manifest_path.parent.name,
                "bytes": len(packed),
                "sha256": hashlib.sha256(packed).hexdigest(),
            }
            manifest["full_evidence"] = {
                "manifest": SNAPSHOT,
                "bytes": len(snapshot_bytes),
                "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "source_revision": snapshot["source_revision"],
            }
            (ROOT / manifest_path).write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
        (ROOT / SHARED_LEDGER).write_bytes(packed)
        for manifest_path in manifests:
            previous = manifest_path.parent / "summary.json.xz"
            if previous.as_posix() != SHARED_LEDGER:
                (ROOT / previous).unlink(missing_ok=True)
        print(f"{SHARED_LEDGER}: {len(packed)} bytes, {len(ledgers)} complete ledgers")


if __name__ == "__main__":
    main()
