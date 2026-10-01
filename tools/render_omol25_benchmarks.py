"""Verify every OMol25-level endpoint and plot all five warm observations.

Incomplete points retain their stage and outcome but never become warm timing
bounds. Independent cold/moved oracles and raw hashes accompany scalar samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks.readme_omol25 import SCHEMA, SIZES, check_record


def digest(path: Path) -> str:
    """Bind retained evidence to the exact raw input, not just its source SHA."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    raw: dict[str, Any], reference: dict[str, Any], *, schema: str = SCHEMA
) -> None:
    """Reject missing repeats, duplicate keys, bad timings or any inaccurate call."""
    if raw.get("schema") != schema or reference.get("schema") != schema:
        raise ValueError("endpoint timing schema mismatch")
    if raw["protocol"].get("force_return") != "host_array_in_timed_endpoint":
        raise ValueError("force host export is not part of the recorded endpoint")
    if raw["protocol"] != reference["protocol"]:
        raise ValueError("scientific protocol mismatch")
    repeats = raw["protocol"]["repeats"]
    expected = {"cold": 1, "warm": repeats, "moved": 1, "moved-warm": repeats}
    rows = raw["records"]
    if Counter(row["phase"] for row in rows) != expected:
        raise ValueError("missing cold, warm or changed-geometry endpoint")
    keys = [(row["phase"], row["geometry"], row["repeat"]) for row in rows]
    required = [("cold", 0, 0), ("moved", 1, 0)] + [
        (phase, geometry, repeat)
        for phase, geometry in (("warm", 0), ("moved-warm", 1))
        for repeat in range(repeats)
    ]
    if sorted(keys) != sorted(required):
        raise ValueError("duplicate or mislabeled endpoint")
    oracles = {
        row["geometry"]: row
        for row in reference["records"]
        if row["phase"] in ("cold", "moved")
    }
    for row in rows:
        if not math.isfinite(row["complete_seconds"]) or row["complete_seconds"] <= 0:
            raise ValueError("invalid complete endpoint timing")
        checked = check_record(row, oracles[row["geometry"]])
        if not checked["gate"] or row.get("gate") is not True:
            raise ValueError("independent energy/force acceptance failed")


def collect(directory: Path, atoms: int, *, schema: str = SCHEMA) -> dict[str, Any]:
    """Preserve failures independently of the other engine's successful result."""
    point: dict[str, Any] = {"atoms": atoms, "engines": {}}
    reference_path = directory / str(atoms) / "reference.json"
    reference = (
        json.loads(reference_path.read_text()) if reference_path.exists() else None
    )
    for engine in ("reference", "native"):
        path = directory / str(atoms) / f"{engine}.json"
        outcome_path = path.with_suffix(".outcome")
        outcome = (
            json.loads(outcome_path.read_text()) if outcome_path.exists() else None
        )
        entry: dict[str, Any] = {"status": "not_run", "outcome": outcome}
        point["engines"][engine] = entry
        if outcome and outcome.get("reason") == "reference_unavailable":
            entry["status"] = "reference_unavailable"
            continue
        if not path.exists():
            if outcome is not None:
                entry["status"] = (
                    "timeout" if outcome["exit_code"] in (124, 137) else "failed"
                )
            continue
        raw = json.loads(path.read_text())
        if "protocol" in point and point["protocol"] != raw["protocol"]:
            raise ValueError("point protocols differ")
        point["protocol"] = raw["protocol"]
        entry.update(
            {
                key: raw[key]
                for key in (
                    "stage",
                    "error",
                    "environment",
                    "native_build",
                    "native_capabilities",
                    "reference_sha256",
                    "basis_file_sha256",
                    "native_force_components",
                    "grid_points",
                    "scheduler",
                    "reference_tensor_engine",
                    "reference_libxc",
                    "source_file_sha256",
                    "native_schedule_settings",
                    "native_schedule_policy",
                )
                if key in raw
            }
        )
        entry["raw_sha256"] = digest(path)
        entry["records"] = [
            {key: value for key, value in row.items() if key != "forces"}
            for row in raw["records"]
        ]
        completed = (
            raw["status"] == "measured"
            and outcome is not None
            and outcome["exit_code"] == 0
        )
        if completed:
            if reference is None or reference["status"] != "measured":
                raise ValueError("measured point has no complete independent reference")
            if engine == "native" and raw["reference_sha256"] != digest(reference_path):
                raise ValueError("native run used a different independent reference")
            validate(raw, reference, schema=schema)
            entry["status"] = "measured"
            entry["medians"] = {
                phase: median(
                    row["complete_seconds"]
                    for row in raw["records"]
                    if row["phase"] == phase
                )
                for phase in ("cold", "warm", "moved", "moved-warm")
            }
        elif raw["status"] == "unsupported":
            entry["status"] = "unsupported"
        else:
            entry["status"] = (
                "timeout"
                if outcome and outcome["exit_code"] in (124, 137)
                else "failed"
            )
        if engine == "reference":
            point["independent_references"] = [
                row for row in raw["records"] if row["phase"] in ("cold", "moved")
            ]
    return point


def figure(
    points: list[dict[str, Any]],
    destination: Path,
    *,
    title: str = "OMol25 level: ωB97M-V / def2-TZVPD",
    filename: str = "omol25.svg",
) -> None:
    """Use HF's all-repeat medians/ranges, exposing variable iteration branches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": filename})
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"native": "#d55e00", "reference": "#0072b2"}
    native_unsupported = all(
        point["engines"]["native"]["status"] == "unsupported" for point in points
    )
    canonical_native = any(
        point["engines"]["native"]
        .get("native_schedule_settings", {})
        .get("GENERATIVEQC_CUDA_CANONICAL_JK")
        == "1"
        for point in points
        if point["engines"]["native"]["status"] == "measured"
    )
    automatic_native = any(
        point["engines"]["native"].get("native_schedule_policy")
        == "automatic-generated-SPD/canonical-through-f"
        for point in points
        if point["engines"]["native"]["status"] == "measured"
    )
    for engine, default_label in (
        (
            "native",
            "GenerativeQC direct (automatic through-f)"
            if automatic_native and not canonical_native
            else "GenerativeQC direct (canonical J/K opt-in)"
            if canonical_native
            else "GenerativeQC direct",
        ),
        ("reference", "GPU4PySCF direct"),
    ):
        label = (
            "GenerativeQC: f-shell force API unavailable (no native timings)"
            if engine == "native" and native_unsupported
            else default_label
        )
        xs, ys, lows, highs = [], [], [], []
        for point in points:
            entry = point["engines"][engine]
            if entry["status"] != "measured":
                continue
            rows = [row for row in entry["records"] if row["phase"] == "warm"]
            times = [row["complete_seconds"] for row in rows]
            aos = point["protocol"]["aos"]
            center = median(times)
            xs.append(aos)
            ys.append(center)
            lows.append(center - min(times))
            highs.append(max(times) - center)
            if len({row["iterations"] for row in rows}) != 1:
                ax.scatter(
                    [aos] * len(times), times, marker="x", color=colors[engine], s=25
                )
        ax.errorbar(
            xs,
            ys,
            yerr=[lows, highs] if xs else None,
            fmt="x" if engine == "native" and native_unsupported else "o-",
            color=colors[engine],
            label=label,
            capsize=3,
            linewidth=1.8,
        )
    ticks = [point["protocol"]["aos"] for point in points if "protocol" in point]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_xlabel("Spherical AOs (3–96 atoms; same water clusters as HF)")
    ax.set_ylabel("Complete warm SCF energy + analytic forces / s")
    fig.suptitle(title, x=0.105, y=0.98, ha="left", weight="bold")
    missing = [
        f"{point['atoms']}: "
        + ", ".join(
            f"{'GQC' if engine == 'native' else 'GPU4PySCF'} {entry['status']}"
            for engine, entry in point["engines"].items()
            if entry["status"] != "measured"
            and not (engine == "native" and native_unsupported)
        )
        for point in points
        if any(
            entry["status"] != "measured"
            and not (engine == "native" and native_unsupported)
            for engine, entry in point["engines"].items()
        )
    ]
    if missing:
        lines = [
            "; ".join(missing[index : index + 2]) for index in range(0, len(missing), 2)
        ]
        fig.text(
            0.5,
            0.025,
            "Incomplete points (atoms):\n" + "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.80, bottom=0.23 if missing else 0.15
    )
    ax.grid(True, alpha=0.17, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.10, 0.935), frameon=False
    )
    output = destination / filename
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main() -> None:
    """Write compact reviewed evidence while leaving raw logs in ignored storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--basis-directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    points = [collect(args.raw_directory, atoms) for atoms in SIZES]
    for point in points:
        (args.destination / f"water{point['atoms']}.json").write_text(
            json.dumps(point, indent=2, allow_nan=False) + "\n"
        )
    for name in ("def2-tzvpd-ho.json", "def2-tzvpd-ho.bse.json"):
        source = args.basis_directory / name
        target = args.destination / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    figure(points, args.destination)


if __name__ == "__main__":
    main()
