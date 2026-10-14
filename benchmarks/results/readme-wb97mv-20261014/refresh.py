"""Replay the README panels without duplicating unchanged historical evidence.

Only the WB97M-V population is replaced. Other functionals keep their original
source/build identities in the linked baseline campaign, not the new revision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

from tools.render_readme_dft import phase_rows, render

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "readme-dft-20261013"
SOURCE_COMMIT = "810f081ef293256ebe20beb7298f9d21996fec81"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def read_summary(path: Path) -> dict[str, Any]:
    """Read the losslessly compressed scientific reduction, not raw run debris."""
    return json.loads(lzma.decompress(path.read_bytes()))


def combined_summary(name: str) -> dict[str, Any]:
    """Join fresh native points with the unchanged, explicitly historical panels."""
    summary = copy.deepcopy(read_summary(BASELINE / name))
    update = read_summary(ROOT / name)
    if update["source_commit"] != SOURCE_COMMIT:
        raise ValueError("WB97M-V source identity differs from this publication")
    if update["sizes"] != summary["sizes"]:
        raise ValueError("WB97M-V sizes differ from the README protocol")
    if set(update["methods"]) != {"wb97m-v"}:
        raise ValueError("refresh may replace only the WB97M-V population")
    if update["density_fitting"] != summary["density_fitting"]:
        raise ValueError("direct and density-fitted campaigns cannot be combined")
    summary["methods"]["wb97m-v"] = update["methods"]["wb97m-v"]
    summary["panel_annotations"] = update.get("panel_annotations", {})
    return summary


def render_publication(summary: dict[str, Any], destination: Path) -> None:
    """Compact generated SVG markup without changing the scientific reduction.

    Scour 0.38.2 removes unused IDs and redundant markup; eight significant
    digits retain sub-pixel positioning. Historical SVGs
    are archived losslessly instead, preserving their exact original bytes.
    """
    from scour.scour import parse_args, scourString

    render(summary, destination)
    options = parse_args(
        [
            "--enable-id-stripping",
            "--shorten-ids",
            "--strip-xml-prolog",
            "--remove-metadata",
            "--indent=none",
            "--set-precision=8",
            "--set-c-precision=8",
        ]
    )
    name = "df-cold-warm.svg" if summary["density_fitting"] else "dft-cold-warm.svg"
    path = destination / name
    path.write_text(scourString(path.read_text(), options=options))


def verify() -> None:
    """Audit stored identities, full phase populations and honest accuracy labels.

    This scalar/integrity audit cannot recompute omitted native force vectors or
    qualify a different binary. Missing independent geometry oracles remain null.
    Checked raw rows may encode execution in their combined scientific gate;
    unchecked rows must retain a separate successful execution gate.
    """
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["native_source_commit"] != SOURCE_COMMIT:
        raise ValueError("manifest source identity differs from this publication")
    for directory, key in ((ROOT, "files"), (BASELINE, "baseline_files")):
        for name, receipt in manifest[key].items():
            payload = (directory / name).read_bytes()
            if (
                len(payload) != receipt["bytes"]
                or hashlib.sha256(payload).hexdigest() != receipt["sha256"]
            ):
                raise ValueError(f"publication checksum mismatch: {name}")
    for name in ("summary.json.xz", "df-summary.json.xz"):
        summary = read_summary(ROOT / name)
        points = summary["methods"]["wb97m-v"]
        expected_sizes = [3, 12, 48] if summary["density_fitting"] else [3, 12, 48, 96]
        if [point["atoms"] for point in points] != expected_sizes:
            raise ValueError("publication must retain every completed native size")
        for point in points:
            native = point["engines"]["native"]
            if (
                native["native_build"]["library_sha256"]
                != manifest["native_library_sha256"]
            ):
                raise ValueError("native point uses a different binary")
            if native["source_file_sha256"] != manifest["native_source_file_sha256"]:
                raise ValueError(
                    "native point uses a different source/consumer inventory"
                )
            if native["status"] not in ("measured", "completed_unchecked"):
                raise ValueError(
                    f"incomplete native WB97M-V population: {point['atoms']}"
                )
            if len(native["records"]) != sum(PHASE_COUNTS.values()):
                raise ValueError("native population must contain all twelve endpoints")
            for phase, expected in PHASE_COUNTS.items():
                rows = phase_rows(native, phase)
                if len(rows) != expected:
                    raise ValueError(f"incomplete phase: {phase}")
                for row in rows:
                    if not (
                        row.get("execution_gate", row["gate"]) is True
                        and row["converged"] is True
                        and row["status"] == 0
                        and math.isfinite(row["energy"])
                    ):
                        raise ValueError("native E+F execution did not complete")
                    if (
                        not math.isfinite(row["complete_seconds"])
                        or row["complete_seconds"] <= 0
                    ):
                        raise ValueError("invalid complete endpoint timing")
                    if row["gate"] is None:
                        if native["phase_status"][phase] != "completed_unchecked":
                            raise ValueError(
                                "missing accuracy cannot have a measured label"
                            )
                        if (
                            row["energy_error"] is not None
                            or row["force_error"] is not None
                        ):
                            raise ValueError(
                                "unchecked accuracy must retain null errors"
                            )
                    elif not (
                        row["gate"] is True
                        and 0 <= row["energy_error"] <= 1e-8
                        and 0 <= row["force_error"] <= 1e-7
                    ):
                        raise ValueError("independent energy/force acceptance failed")
        combined_summary(name)


def main() -> None:
    """Verify the retained campaign and optionally replay its tables and SVGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json.xz", "df-summary.json.xz"):
            render_publication(combined_summary(name), args.output)
    print("WB97M-V identities, complete phases and scalar gates verified.")


if __name__ == "__main__":
    main()
