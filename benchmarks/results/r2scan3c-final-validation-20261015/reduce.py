"""Recheck source-qualified, complete r²SCAN-3c final-proof comparisons."""

from __future__ import annotations

import argparse
import json
import lzma
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from tools.render_readme_dft import collect_point

PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}
SELECTOR = "GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION"
FORCE_SELECTOR = "GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS"


def phase_statistics(records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    """Retain every complete sample and semantic SCF work, without normalization."""
    rows = [row for row in records if row["phase"] == phase]
    samples = [row["complete_seconds"] for row in rows]
    center = median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": center,
        "relative_mad": median(abs(sample - center) for sample in samples) / center,
        "iterations_fock_builds": [
            [row["iterations"], row["fock_builds"]] for row in rows
        ],
    }


def compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply the declared descriptive gate, not a significance or universal claim."""
    comparisons = {}
    for phase in PHASE_COUNTS:
        before = phase_statistics(control["records"], phase)
        after = phase_statistics(candidate["records"], phase)
        reduction = 1 - after["median_seconds"] / before["median_seconds"]
        gate = max(0.02, 2 * (before["relative_mad"] + after["relative_mad"]))
        same_work = before["iterations_fock_builds"] == after["iterations_fock_builds"]
        comparisons[phase] = {
            "control": before,
            "candidate": after,
            "relative_reduction": reduction,
            "robust_descriptive_gate": gate,
            "passes_benefit_gate": reduction >= gate,
            "passes_nonregression_gate": reduction >= -max(0.05, gate),
            "same_scf_work": same_work,
            "work_confounded": not same_work,
        }
    return comparisons


def reduce_campaign(campaign: Path) -> dict[str, Any]:
    """Reject stale, incomplete, unchecked or non-default candidate populations."""
    frozen = json.loads((campaign / "source-freeze.json").read_text())
    points: dict[tuple[str, int], dict[str, Any]] = {}
    images = set()
    for path in sorted((campaign / "remote/measurements").rglob("native.json")):
        relative = path.relative_to(campaign / "remote/measurements")
        arm, mode, method, size, filename = relative.parts
        if arm not in ("control", "candidate") or mode not in ("none", "cuda"):
            raise ValueError(f"unexpected population: {relative}")
        if method != "r2scan-3c" or filename != "native.json":
            raise ValueError(f"unexpected method: {relative}")
        document = json.loads(path.read_text())
        native_build = document["native_build"]
        if native_build["probe"]["source_identity"] != frozen["source_identity"]:
            raise ValueError(f"stale native image: {relative}")
        if document["source_file_sha256"] != frozen["consumer_source_file_sha256"]:
            raise ValueError(f"mismatched consumers: {relative}")
        settings = document["native_schedule_settings"]
        expected_selector = "0" if arm == "control" else None
        if settings.get(SELECTOR) != expected_selector or FORCE_SELECTOR in settings:
            raise ValueError(
                f"comparison does not use the declared defaults: {relative}"
            )
        point = collect_point(path.parent.parent, int(size))
        native = point["engines"]["native"]
        if native["status"] != "measured":
            raise ValueError(f"incomplete or unchecked population: {relative}")
        if Counter(row["phase"] for row in native["records"]) != PHASE_COUNTS:
            raise ValueError(f"incomplete phase repeats: {relative}")
        images.add(native_build["library_sha256"])
        point["population"] = str(relative)
        points.setdefault((mode, int(size)), {})[arm] = point
    if not points or len(images) != 1:
        raise ValueError("all comparison populations must share one frozen image")
    populations = []
    for (mode, size), arms in sorted(points.items()):
        if set(arms) != {"control", "candidate"}:
            raise ValueError(f"missing comparison arm: {mode}/{size}")
        if arms["control"]["protocol"] != arms["candidate"]["protocol"]:
            raise ValueError(f"comparison protocols differ: {mode}/{size}")
        phases = compare(
            arms["control"]["engines"]["native"],
            arms["candidate"]["engines"]["native"],
        )
        populations.append(
            {"density_fitting": mode, "atoms": size, "arms": arms, "phases": phases}
        )
        print(
            mode,
            size,
            " ".join(
                f"{phase}: {100 * result['relative_reduction']:+.2f}%"
                for phase, result in phases.items()
            ),
        )
    records = [
        row
        for arms in points.values()
        for point in arms.values()
        for row in point["engines"]["native"]["records"]
    ]
    return {
        "schema": "generativeqc.r2scan3c-final-validation.v1",
        "frozen_source": frozen,
        "native_library_sha256": images.pop(),
        "acceptance": {
            "energy_error_eh": 1e-8,
            "maximum_force_error_eh_per_bohr": 1e-7,
            "benefit": "max(2%, 2 * (relative_MAD_control + relative_MAD_candidate))",
            "nonregression": "max(5%, benefit gate)",
            "qualification": "complete accurate endpoints and matched-work nonregression; cold/moved work-confounded samples remain visible but do not establish a validation-route speedup/regression; the 2% benefit label is not an enablement threshold",
            "scope": "descriptive same-image cohorts; not statistical significance",
        },
        "complete_native_calls": len(records),
        "maximum_energy_error_eh": max(row["energy_error"] for row in records),
        "maximum_force_error_eh_per_bohr": max(row["force_error"] for row in records),
        "comparisons": populations,
    }


def main() -> None:
    """Write a compact, independently rechecked record of the raw campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = reduce_campaign(args.campaign)
    args.output.write_bytes(
        lzma.compress((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
    )


if __name__ == "__main__":
    main()
