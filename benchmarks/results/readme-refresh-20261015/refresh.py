"""Audit native-only remeasurements and replay unchanged historical references.

The previous publication owns every independent reference array and timing.
Only native observations are overlaid; no reference execution is needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import math
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "readme-wb97mv-20261014"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def read_summary(path: Path) -> dict[str, Any]:
    """Read a losslessly compressed scalar reduction, not a new measurement."""
    return json.loads(lzma.decompress(path.read_bytes()))


def baseline_summary(name: str) -> dict[str, Any]:
    """Resolve the preceding publication's already qualified, mixed-source panel."""
    previous = runpy.run_path(str(BASELINE / "refresh.py"))
    return previous["combined_summary"](name)


def combined_summary(name: str) -> dict[str, Any]:
    """Overlay native engines without changing a single historical reference."""
    summary = copy.deepcopy(baseline_summary(name))
    campaign = read_summary(ROOT / "native-summary.json.xz")
    mode = "cuda" if summary["density_fitting"] else "none"
    for method, updates in campaign["populations"][mode].items():
        points = {point["atoms"]: point for point in summary["methods"][method]}
        for update in updates:
            point = points[update["atoms"]]
            if point["protocol"] != update["protocol"]:
                raise ValueError(
                    "native refresh changed the historical scientific protocol"
                )
            point["engines"]["native"] = update["native"]
        summary["methods"][method] = [points[size] for size in sorted(points)]
    summary["panel_annotations"] = {
        "wb97m-v": "Native DF/96: no complete E+F" if mode == "cuda" else ""
    }
    return summary


def verify() -> dict[str, int]:
    """Verify provenance, complete populations, scalar gates and reference identity.

    Native force vectors stay in ignored raw storage. The reducer checks them
    against retained independent arrays before publication; this replay checks
    the retained errors and raw identities, not omitted native vectors.
    """
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, receipt in manifest["files"].items():
        payload = (ROOT / relative).read_bytes()
        if (
            len(payload) != receipt["bytes"]
            or hashlib.sha256(payload).hexdigest() != receipt["sha256"]
        ):
            raise ValueError(f"publication checksum mismatch: {relative}")
    for relative, receipt in manifest["baseline_files"].items():
        payload = (ROOT.parent / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            raise ValueError(f"historical evidence changed: {relative}")
    settings = json.loads((ROOT / "build-settings.json").read_bytes())
    build = settings["native_build"]
    if not (
        settings["source_commit"] == manifest["source_commit"]
        and build["library_sha256"] == manifest["native_library_sha256"]
        and build["probe"]["source_identity"] == manifest["native_source_identity"]
    ):
        raise ValueError("publication does not belong to the frozen build")
    campaign = read_summary(ROOT / "native-summary.json.xz")
    if campaign["source_commit"] != manifest["source_commit"]:
        raise ValueError("measured source identity differs from the manifest")
    if campaign.get("prerequisites") != manifest.get("prerequisites"):
        raise ValueError("combined PR source prerequisites differ from the manifest")
    if set(campaign["populations"]) != {"none", "cuda"}:
        raise ValueError("native campaign requires both direct and DF populations")
    counts: Counter[str] = Counter()
    for mode, methods in campaign["populations"].items():
        expected_methods = set(
            baseline_summary(
                "df-summary.json.xz" if mode == "cuda" else "summary.json.xz"
            )["methods"]
        )
        if set(methods) != expected_methods:
            raise ValueError("native campaign lost a requested method")
        for method, points in methods.items():
            expected_sizes = (
                [3, 12, 48]
                if mode == "cuda" and method == "wb97m-v"
                else [3, 12, 48, 96]
            )
            if [point["atoms"] for point in points] != expected_sizes:
                raise ValueError("native campaign lost a requested population")
            for point in points:
                native = point["native"]
                build = native.get("native_build") or {}
                if build.get("library_sha256") != manifest["native_library_sha256"]:
                    raise ValueError("native observations lack the qualified binary")
                if (
                    build.get("probe", {}).get("source_identity")
                    != manifest["native_source_identity"]
                ):
                    raise ValueError(
                        "native observations lack the qualified source inventory"
                    )
                experiment = native.get("native_experiment", {})
                if not (
                    experiment.get("precision_policy") == "fp64"
                    and experiment.get("preliminary_provider") == "none"
                    and experiment.get("public_initialization_policy") is True
                    and experiment.get("force_active_ao_cutoff") is None
                    and experiment.get("force_active_ao_cache_bytes") is None
                    and experiment.get("force_max_device_bytes") is None
                    and experiment.get("force_max_host_bytes") is None
                ):
                    raise ValueError(
                        "native observation enabled a benchmark-only policy"
                    )
                if point["atoms"] in (3, 12):
                    primer = point["cache_priming"]
                    if not (
                        primer["outcome"]["exit_code"] == 0
                        and primer["phase_counts"] == PHASE_COUNTS
                        and primer["checked_calls"] == 12
                        and primer["fresh_measurement_process"] is True
                    ):
                        raise ValueError(
                            "small native cold requires a separate successful primer"
                        )
                    if primer.get("native_build") != build:
                        raise ValueError("cache primer used a different native image")
                if native["status"] not in ("measured", "completed_unchecked"):
                    counts["incomplete_populations"] += 1
                    continue
                if Counter(row["phase"] for row in native["records"]) != PHASE_COUNTS:
                    raise ValueError("completed population lost an endpoint sample")
                counts["complete_populations"] += 1
                for row in native["records"]:
                    if not (
                        math.isfinite(row["energy"])
                        and math.isfinite(row["complete_seconds"])
                        and row["complete_seconds"] > 0
                        and row["converged"] is True
                        and row["status"] == 0
                    ):
                        raise ValueError(
                            "published endpoint did not complete successfully"
                        )
                    if row["gate"] is None:
                        if (
                            native["phase_status"][row["phase"]]
                            != "completed_unchecked"
                        ):
                            raise ValueError(
                                "missing independent accuracy cannot be a pass"
                            )
                        if (
                            row["energy_error"] is not None
                            or row["force_error"] is not None
                        ):
                            raise ValueError(
                                "unchecked accuracy must retain null errors"
                            )
                        counts["unchecked_calls"] += 1
                    elif not (
                        row["gate"] is True
                        and 0 <= row["energy_error"] <= 1e-8
                        and 0 <= row["force_error"] <= 1e-7
                    ):
                        raise ValueError(
                            f"independent energy/force gate failed: {mode}/{method}/{point['atoms']}"
                        )
                    else:
                        counts["checked_calls"] += 1
    for name in ("summary.json.xz", "df-summary.json.xz"):
        before = baseline_summary(name)
        after = combined_summary(name)
        for method, points in before["methods"].items():
            for original, current in zip(points, after["methods"][method], strict=True):
                if original["engines"]["reference"] != current["engines"]["reference"]:
                    raise ValueError("historical GPU4PySCF observations were modified")
                if original.get("independent_references") != current.get(
                    "independent_references"
                ):
                    raise ValueError(
                        "historical independent geometry arrays were modified"
                    )
    return dict(counts)


def main() -> None:
    """Audit stored evidence and optionally regenerate the README tables/figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = verify()
    if args.output:
        previous = runpy.run_path(str(BASELINE / "refresh.py"))
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json.xz", "df-summary.json.xz"):
            previous["render_publication"](combined_summary(name), args.output)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
