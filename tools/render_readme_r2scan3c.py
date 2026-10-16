"""Replay the native-only r²SCAN-3c README update from existing qualified evidence.

This is a mixed-source presentation, not a new benchmark campaign. Historical
panels and independent references remain owned by their frozen publications.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

from tools.generativeqc_validation.xc_diagonal_evidence import summarize
from tools.render_readme_dft import timing_cell, work_cell

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
PHASE_COUNTS = {"cold": 1, "warm": 5, "moved": 1, "moved-warm": 5}


def overlay_native(
    summary: dict[str, Any], updates: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Replace only independently qualified native points with identical protocols.

    Do not reuse historical native metadata with new observations, trim samples,
    or normalize complete wall times by their actual SCF work.
    """
    if summary["density_fitting"]:
        raise ValueError("this qualification contains Direct endpoints only")
    result = copy.deepcopy(summary)
    points = {point["atoms"]: point for point in result["methods"]["r2scan-3c"]}
    for atoms, update in updates.items():
        point = points[atoms]
        if point["protocol"] != update["protocol"]:
            raise ValueError("r²SCAN-3c update changed the scientific protocol")
        native = update["native"]
        rows = native["records"]
        if (
            native["status"] != "measured"
            or Counter(row["phase"] for row in rows) != PHASE_COUNTS
        ):
            raise ValueError("r²SCAN-3c update requires all twelve complete calls")
        for row in rows:
            if not (
                row["gate"] is True
                and row["converged"] is True
                and 0 <= row["energy_error"] <= 1e-8
                and 0 <= row["force_error"] <= 1e-7
                and math.isfinite(row["complete_seconds"])
                and row["complete_seconds"] > 0
            ):
                raise ValueError("r²SCAN-3c independent endpoint gate failed")
        point["engines"]["native"] = copy.deepcopy(native)
    result.setdefault("panel_annotations", {})["r2scan-3c"] = (
        "GenerativeQC 48/96: 2026-10-16 composition"
    )
    result["evidence_note"] = (
        "The [measurement report](../xc-diagonal-potential-20261016/README.md#homepage-update) "
        "identifies the mixed sources. "
        "Only native Direct r²SCAN-3c/48/96 is updated from "
        "[qualified XC/force-stack evidence](../xc-diagonal-potential-20261016/README.md); "
        "all other observations and independent arrays remain in the "
        "[frozen publication](../readme-refresh-20261015/README.md)."
    )
    return result


def combined_summary() -> dict[str, Any]:
    """Audit both retained publications before constructing the presentation.

    The evidence reader replays 32 full force gates; the 48-atom candidate's
    force gates are collection-time independent-validation receipts, not newly
    replayable arrays. Both sources retain every scalar sample and work count.
    """
    previous = runpy.run_path(str(RESULTS / "readme-refresh-20261015/refresh.py"))
    previous["verify"]()
    summary = previous["combined_summary"]("summary.json.xz")
    evidence = summarize()
    reference = runpy.run_path(
        str(RESULTS / "md-j-sparse-support-20261016/summarize.py")
    )
    updates = {}
    for atoms in (48, 96):
        case = f"r2scan-3c-{atoms}"
        key = f"qualified-stack/candidate/{case}"
        if key in evidence["populations"]:
            observation = evidence["populations"][key]
        else:
            observation = evidence["observational_controls"][key][
                "independent_validation_receipt"
            ]
        updates[atoms] = {
            "protocol": reference["load"](f"raw/reference-{case}.json.xz")["protocol"],
            "native": {
                "status": "measured",
                "records": observation["calls"],
                "library_sha256": observation["library_sha256"],
                "scheduler": observation["scheduler"],
                "evidence_key": key,
            },
        }
    return overlay_native(summary, updates)


def render_publication(summary: dict[str, Any], destination: Path) -> None:
    """Use the existing renderer and retain changed-geometry timing/work too."""
    previous = runpy.run_path(str(RESULTS / "readme-wb97mv-20261014/refresh.py"))
    destination.mkdir(parents=True, exist_ok=True)
    previous["render_publication"](summary, destination)
    lines = [
        "\n## Updated Native Changed-Geometry Endpoints\n",
        "All five moved-warm calls are retained. Counts are iterations/Fock builds.\n",
        "| Atoms | Moved (s) | Moved-warm (s) | Moved work | Moved-warm work |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in summary["methods"]["r2scan-3c"]:
        if point["atoms"] not in (48, 96):
            continue
        native = point["engines"]["native"]
        cells = [timing_cell(native, phase) for phase in ("moved", "moved-warm")]
        cells += [work_cell(native, phase) for phase in ("moved", "moved-warm")]
        lines.append(f"| {point['atoms']} | " + " | ".join(cells) + " |")
    table = destination / "timings.md"
    table.write_text(table.read_text() + "\n".join(lines) + "\n")


def main() -> None:
    """Validate without a GPU; optionally regenerate the figure and timing table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = combined_summary()
    if args.output:
        render_publication(summary, args.output)
    print(json.dumps({"updated_native_direct_atoms": [48, 96], "gpu_rerun": False}))


if __name__ == "__main__":
    main()
