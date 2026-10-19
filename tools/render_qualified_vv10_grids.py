"""Publish grid timings only after replaying joint native/reference precision gates.

The manifest names a complete requested grid/size/Direct-or-DF matrix, with a
separate endpoint qualification report for each engine. Reports are recomputed
from their hash-bound raw measurements and common reference controls. Retain
every timing/force sample and the independent accuracy oracles in the reduction;
archived reductions can replay their numerical gates without the raw filesystem.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
from dataclasses import asdict
from pathlib import Path
from typing import Any

from generativeqc import GridSpec

from benchmarks._readme_grid import parse_grid
from tools.analyze_vv10_grid_study import PROFILES, load_population
from tools.qualify_readme_rerun import physical_protocol
from tools.qualify_vv10_endpoint_precision import (
    CONTROLS,
    qualify_endpoint,
    replay_qualification,
)

QUALIFICATION = "joint-total-E+F-empirical-reference"
PHASES = ("cold", "warm", "moved", "moved-warm")


def _validate_point(
    point: dict, profile: str, mode: str, grid: tuple[int, int, int]
) -> None:
    """Do not let plotting a retained archive silently weaken its scientific gate."""
    if set(point["engines"]) != {"native", "reference"}:
        raise ValueError("both native and GPU4PySCF timing populations are required")
    protocol = point["protocol"]
    if (
        protocol["atoms"] != point["atoms"]
        or json.dumps(protocol["grid"], sort_keys=True)
        != json.dumps(asdict(GridSpec(*grid)), sort_keys=True)
        or protocol["density_fitting"] != (mode == "cuda")
    ):
        raise ValueError(
            "qualified timing population is filed under the wrong case/grid"
        )
    oracles = point["independent_accuracy_oracles"]
    if set(oracles) != {*CONTROLS, "anchor_replay"}:
        raise ValueError("retained joint reference oracles are missing")
    for oracle in oracles.values():
        if oracle["stage"] != "complete" or any(
            oracle[field] != oracle["receipt"][field]
            for field in ("scheduler", "accelerator", "source_sha256")
        ):
            raise ValueError("retained reference metadata differs from its receipt")
    reference_controls = None
    for engine, entry in point["engines"].items():
        saved = entry["endpoint_precision_qualification"]
        if saved["profile"] != profile or saved["engine"] != engine:
            raise ValueError("precision report profile or engine differs")
        if saved["total_endpoint_gate"] is not True:
            raise ValueError(
                "unqualified endpoints cannot become precision-qualified timings"
            )
        receipt = saved["candidate_receipt"]
        fields = ("scheduler", "source_file_sha256") + (
            ("native_build",) if engine == "native" else ()
        )
        if entry["raw_sha256"] != receipt["sha256"] or any(
            entry[field] != receipt[field] for field in fields
        ):
            raise ValueError("retained source/build metadata differs from its receipt")
        if reference_controls is None:
            reference_controls = saved["reference_controls"]
        elif saved["reference_controls"] != reference_controls:
            raise ValueError(
                "timing engines must share the same joint reference controls"
            )
        candidate = copy.deepcopy(entry)
        if physical_protocol(candidate["protocol"]) != physical_protocol(protocol):
            raise ValueError("retained engine protocols differ")
        candidate["status"] = entry["raw_status"]
        candidate["receipt"] = saved["candidate_receipt"]
        for row in candidate["records"]:
            row["gate"] = row.pop("measurement_gate")
        observed = qualify_endpoint(
            candidate,
            oracles["anchor_replay"],
            {name: oracles[name] for name in CONTROLS},
            profile=profile,
            anchor_fraction=saved["reference_allowance_fraction"],
        )
        if observed != saved or entry["status"] != "measured":
            raise ValueError(
                "retained precision report differs from its numerical evidence"
            )
        if entry["phase_status"] != dict.fromkeys(PHASES, "measured"):
            raise ValueError("retained timing phases are not all qualified")


def validate_summary(summary: dict[str, Any]) -> None:
    """Recheck the full declared cohort, not just whichever points can be plotted."""
    if (
        summary.get("schema") != "generativeqc.readme-qualified-grid-timings.v1"
        or summary.get("precision_qualification") != QUALIFICATION
        or summary.get("profile") not in PROFILES
        or not summary["sizes"]
        or len(set(summary["sizes"])) != len(summary["sizes"])
        or not summary["modes"]
        or not set(summary["modes"]) <= {"none", "cuda"}
        or set(summary["modes"]) != set(summary["requested_modes"])
    ):
        raise ValueError("invalid qualified grid summary")
    grids = [parse_grid(",".join(map(str, values))) for values in summary["grids"]]
    if not grids or len(set(grids)) != len(grids):
        raise ValueError("qualified grid dimensions must be nonempty and unique")
    protocols = {}
    sources = {}
    for mode, populations in summary["modes"].items():
        if [tuple(population["dimensions"]) for population in populations] != grids:
            raise ValueError("qualified grid population matrix is incomplete")
        for population in populations:
            if [point["atoms"] for point in population["points"]] != summary["sizes"]:
                raise ValueError("qualified size population matrix is incomplete")
            for point in population["points"]:
                _validate_point(
                    point, summary["profile"], mode, tuple(population["dimensions"])
                )
                problem = physical_protocol(point["protocol"])
                problem.pop("grid")
                case = (mode, point["atoms"])
                if case in protocols and protocols[case] != problem:
                    raise ValueError(
                        "grid comparison changes the physical case or controls"
                    )
                protocols[case] = problem
                for engine, entry in point["engines"].items():
                    identity = {"source_file_sha256": entry["source_file_sha256"]}
                    if engine == "native":
                        identity["library_sha256"] = entry["native_build"][
                            "library_sha256"
                        ]
                    if engine in sources and sources[engine] != identity:
                        raise ValueError(
                            "qualified grid timings mix source/build cohorts"
                        )
                    sources[engine] = identity


def collect_qualified_sweep(
    path: Path,
    grids: list[tuple[int, int, int]],
    sizes: list[int],
    modes: list[str],
) -> dict[str, Any]:
    """Replay both engines; preserve failures by refusing an incomplete publication."""
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (
        manifest.get("schema") != "generativeqc.readme-qualified-grid-inputs.v1"
        or manifest.get("profile") not in PROFILES
    ):
        raise ValueError("invalid qualified grid input manifest")
    expected = {
        (mode, grid, atoms) for mode in modes for grid in grids for atoms in sizes
    }
    points = {}
    for item in manifest["points"]:
        grid = parse_grid(",".join(map(str, item["grid"])))
        key = (item["mode"], grid, item["atoms"])
        if key not in expected or key in points:
            raise ValueError("unexpected or duplicate qualified grid population")
        if set(item["reports"]) != {"native", "reference"}:
            raise ValueError("both native and GPU4PySCF reports are required")
        point: dict[str, Any] = {"atoms": item["atoms"], "engines": {}}
        controls = None
        for engine in ("native", "reference"):
            report_path = Path(item["reports"][engine])
            if not report_path.is_absolute():
                report_path = path.parent / report_path
            candidate, report = replay_qualification(report_path)
            if (
                report["profile"] != manifest["profile"]
                or candidate["engine"] != engine
            ):
                raise ValueError(
                    "qualification profile or engine differs from manifest"
                )
            if report["total_endpoint_gate"] is not True:
                raise ValueError("joint total endpoint precision did not pass")
            if "protocol" in point and physical_protocol(
                point["protocol"]
            ) != physical_protocol(candidate["protocol"]):
                raise ValueError("qualified native/reference physical protocols differ")
            point["protocol"] = copy.deepcopy(candidate["protocol"])
            if controls is None:
                controls = report["reference_controls"]
                point["independent_accuracy_oracles"] = {
                    name: load_population(Path(control["receipt"]["file"]))
                    for name, control in controls.items()
                }
            elif controls != report["reference_controls"]:
                raise ValueError(
                    "qualified timing engines use different joint references"
                )
            entry = {
                name: copy.deepcopy(value)
                for name, value in candidate.items()
                if name != "receipt"
            }
            entry.update(
                raw_status=candidate["status"],
                status="measured",
                phase_status=dict.fromkeys(PHASES, "measured"),
                raw_sha256=report["candidate_receipt"]["sha256"],
                endpoint_precision_qualification=report,
                qualification_report_sha256=hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
            )
            for row in entry["records"]:
                row["measurement_gate"] = row.get("gate")
                row["gate"] = True
            point["engines"][engine] = entry
        points[key] = point
    if set(points) != expected:
        raise ValueError(
            "requested native/GPU4PySCF grid/size/mode cohort is incomplete"
        )
    summary = {
        "schema": "generativeqc.readme-qualified-grid-timings.v1",
        "precision_qualification": QUALIFICATION,
        "profile": manifest["profile"],
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "sizes": sizes,
        "requested_modes": modes,
        "grids": [list(grid) for grid in grids],
        "modes": {
            mode: [
                {
                    "dimensions": list(grid),
                    "points": [points[mode, grid, atoms] for atoms in sizes],
                }
                for grid in grids
            ]
            for mode in modes
        },
    }
    validate_summary(summary)
    return summary


def main() -> None:
    """Write no precision-qualified figure until the complete requested cohort passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grids", type=parse_grid, nargs="+", required=True)
    parser.add_argument(
        "--sizes", type=int, choices=(3, 12, 48, 96), nargs="+", default=[3, 12, 48, 96]
    )
    parser.add_argument(
        "--modes", choices=("none", "cuda"), nargs="+", default=["none", "cuda"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.render_readme_grids import compact_summary, render

    summary = compact_summary(
        collect_qualified_sweep(args.manifest, args.grids, args.sizes, args.modes)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    render(summary, args.output)
    (args.output / "grid-timings.json.xz").write_bytes(
        lzma.compress(
            json.dumps(
                summary, separators=(",", ":"), sort_keys=True, allow_nan=False
            ).encode(),
            preset=9,
        )
    )


if __name__ == "__main__":
    main()
