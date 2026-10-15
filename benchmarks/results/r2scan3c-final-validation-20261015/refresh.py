"""Replay both README panels, replacing only source-qualified r²SCAN-3c points."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "readme-wb97mv-20261014"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def read_summary(name: str) -> dict[str, Any]:
    """Read only the newly measured method, leaving historical methods linked."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def verify() -> None:
    """Check retained identities, numerical gates and declared cohort acceptance."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, receipt in manifest["files"].items():
        payload = (ROOT / name).read_bytes()
        if (
            len(payload) != receipt["bytes"]
            or hashlib.sha256(payload).hexdigest() != receipt["sha256"]
        ):
            raise ValueError(f"publication checksum mismatch: {name}")
    for name, receipt in manifest["historical_files"].items():
        payload = (ROOT.parents[2] / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            raise ValueError(f"historical input checksum mismatch: {name}")
    observations = read_summary("observations.json.xz")
    if (
        observations["frozen_source"]["source_identity"]
        != manifest["native_source_identity"]
    ):
        raise ValueError("publication and observation source identities differ")
    for comparison in observations["comparisons"]:
        for phase, result in comparison["phases"].items():
            if result["work_confounded"] and phase in ("cold", "moved"):
                continue
            if not result["passes_nonregression_gate"] or not result["same_scf_work"]:
                raise ValueError("nonregression or semantic SCF work gate failed")
    for name in ("summary.json.xz", "df-summary.json.xz"):
        update = read_summary(name)
        if set(update["methods"]) != {"r2scan-3c"}:
            raise ValueError("refresh may replace only r²SCAN-3c")
        points = update["methods"]["r2scan-3c"]
        if [point["atoms"] for point in points] != [3, 12, 48, 96]:
            raise ValueError("publication requires every requested size")
        mode = "cuda" if update["density_fitting"] else "none"
        expected_points = [
            comparison["arms"]["candidate"]
            for comparison in observations["comparisons"]
            if comparison["density_fitting"] == mode
        ]
        if points != expected_points:
            raise ValueError(
                "publication does not match qualified candidate populations"
            )
        for point in points:
            native = point["engines"]["native"]
            if native["status"] != "measured":
                raise ValueError("unchecked or incomplete native population")
            if Counter(row["phase"] for row in native["records"]) != PHASE_COUNTS:
                raise ValueError("native population lacks complete phase repeats")
            if (
                native["native_build"]["probe"]["source_identity"]
                != manifest["native_source_identity"]
            ):
                raise ValueError("native population uses a different image")
            for row in native["records"]:
                if not (
                    row["gate"] is True
                    and row["converged"] is True
                    and row["status"] == 0
                    and 0 <= row["energy_error"] <= 1e-8
                    and 0 <= row["force_error"] <= 1e-7
                    and math.isfinite(row["complete_seconds"])
                    and row["complete_seconds"] > 0
                ):
                    raise ValueError(
                        "complete endpoint or independent accuracy gate failed"
                    )


def combined_summary(name: str) -> dict[str, Any]:
    """Compose prior panels without relabeling their source/build identities."""
    previous = runpy.run_path(str(PREVIOUS / "refresh.py"))
    summary = previous["combined_summary"](name)
    update = read_summary(name)
    if (
        update["sizes"] != summary["sizes"]
        or update["density_fitting"] != summary["density_fitting"]
    ):
        raise ValueError("refresh and historical panel protocols differ")
    summary["methods"]["r2scan-3c"] = update["methods"]["r2scan-3c"]
    return summary


def main() -> None:
    """Verify evidence, optionally regenerating compact SVGs and timing tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify()
    previous = runpy.run_path(str(PREVIOUS / "refresh.py"))
    previous["verify"]()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json.xz", "df-summary.json.xz"):
            previous["render_publication"](combined_summary(name), args.output)
    print("r²SCAN-3c identities, complete endpoints and acceptance gates verified.")


if __name__ == "__main__":
    main()
