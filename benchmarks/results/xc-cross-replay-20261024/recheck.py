"""Requalify frozen XC tile-replay receipts without loading a GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults_qualification import qualify


def require(condition: bool, message: object) -> None:
    """Keep acceptance active under Python optimization."""
    if not condition:
        raise RuntimeError(str(message))


def read_archive(path: Path) -> dict:
    """Read a lossless JSON archive."""
    return json.loads(lzma.decompress(path.read_bytes()))


def statistics_for(values: list[float]) -> dict:
    """Retain all observations and robust location/spread, without trimming."""
    median = statistics.median(values)
    return {
        "samples": values,
        "median": median,
        "mad": statistics.median(abs(x - median) for x in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def timing_summary(records: dict) -> dict:
    """Summarize only the three prespecified fresh-process clean rounds."""
    result = {}
    for method in ("pbe", "pbe0", "r2scan-3c"):
        arms = {}
        for variant in ("baseline", "candidate", "reference"):
            rows = [
                records[f"clean/r{r}-{variant}-{method}-96.json"]["samples"]
                for r in (1, 2, 3)
            ]
            arms[variant] = {
                "cold": statistics_for([row[0]["seconds"] for row in rows]),
                "warm": statistics_for([s["seconds"] for row in rows for s in row[1:]]),
                "warm_process_medians": [
                    statistics.median(s["seconds"] for s in row[1:]) for row in rows
                ],
            }
        baseline, candidate = arms["baseline"]["cold"], arms["candidate"]["cold"]
        arms["cold_reduction"] = 1 - candidate["median"] / baseline["median"]
        arms["robustness_floor"] = max(
            0.02,
            2
            * (
                baseline["mad"] / baseline["median"]
                + candidate["mad"] / candidate["median"]
            ),
        )
        arms["candidate_reference_cold_gap"] = (
            candidate["median"] / arms["reference"]["cold"]["median"] - 1
        )
        result[method] = arms
    return result


def validate(summary: dict, texts: dict, oracles: list, reuse: dict) -> dict:
    """Check receipt identity, complete populations, every oracle gate and timings."""
    records = {name: json.loads(value) for name, value in texts.items()}
    require(set(records) == set(summary["populations"]), "receipt inventory changed")
    counts = dict.fromkeys(
        ("clean", "holdout", "moved", "diagnostic", "profile", "stability"), 0
    )
    populations = counts.copy()
    for name, record in records.items():
        entry = summary["populations"][name]
        require(
            hashlib.sha256(texts[name].encode()).hexdigest() == entry["sha256"], name
        )
        group = name.split("/")[0]
        require(record["status"] == "complete", (name, "incomplete"))
        phases = [s["phase"] for s in record["samples"]]
        expected = (
            ["cold", "warm", "moved"]
            if group in ("moved", "stability")
            else (
                ["cold", "warm", "warm", "warm"]
                if group == "clean"
                else ["cold", "warm"]
            )
        )
        require(phases == expected, (name, phases))
        require(len(entry["samples"]) == len(record["samples"]), (name, "sample count"))
        for frozen, sample in zip(entry["samples"], record["samples"], strict=True):
            work = {key: sample.get(key) for key in frozen if key != "grid_points"}
            work["grid_points"] = sample.get("settings", {}).get("xc_grid_points")
            require(work == frozen, (name, "sample timing/work metadata"))
        require(
            all(
                np.isfinite(s["seconds"]) and s["seconds"] > 0
                for s in record["samples"]
            ),
            name,
        )
        if "native_build" in record:
            variant = "candidate" if "candidate" in name else "baseline"
            build = record["native_build"]
            require(
                all(
                    build[key] == value
                    for key, value in summary["native_builds"][variant].items()
                ),
                (name, "binary identity"),
            )
        if group in ("moved", "stability"):
            require(record["fp32_block_screening"] is False, name)
            reference_record = records[f"moved/candidate-{record['case']}.json"]
            require(
                record["geometries_bohr"] == reference_record["geometries_bohr"]
                and record["driver_sha256"] == reference_record["driver_sha256"],
                (name, "moved reference inputs"),
            )
            references = reference_record["references"]
            gates = {
                "energy_per_atom": 2.5e-6,
                "force_max": 2.5e-5,
                "force_rms": 7.5e-6,
            }
            require(record["gates"] == gates, (name, "moved gates"))
            for sample in record["samples"]:
                oracle = references[sample["geometry"]]
                atoms = len(record["geometries_bohr"][sample["geometry"]])
                actual, expected_force = (
                    np.asarray(sample["forces"]),
                    np.asarray(oracle["forces"]),
                )
                require(actual.shape == expected_force.shape == (atoms, 3), name)
                delta = actual - expected_force
                errors = {
                    "energy_per_atom": abs(sample["energy"] - oracle["energy"]) / atoms,
                    "force_max": float(np.max(np.abs(delta))),
                    "force_rms": float(np.sqrt(np.mean(delta**2))),
                }
                require(
                    all(
                        np.isfinite(errors[k]) and errors[k] <= v
                        for k, v in gates.items()
                    ),
                    (name, errors),
                )
        else:
            qualification = qualify([record], oracles, reference_reuse=reuse)[0]
            require(qualification["status"] == "qualified", (name, qualification))
        counts[group] += len(record["samples"])
        populations[group] += 1
    require(
        populations
        == {
            "clean": 27,
            "holdout": 20,
            "moved": 6,
            "diagnostic": 5,
            "profile": 2,
            "stability": 4,
        },
        populations,
    )
    require(
        counts
        == {
            "clean": 108,
            "holdout": 40,
            "moved": 18,
            "diagnostic": 10,
            "profile": 4,
            "stability": 12,
        },
        counts,
    )
    require(timing_summary(records) == summary["timings"], "timing summary changed")
    failures = 0
    for pair in summary["comparisons"]:
        rows = zip(
            records[pair["baseline"]]["samples"],
            records[pair["candidate"]]["samples"],
            strict=True,
        )
        errors = [
            (
                abs(a["energy"] - b["energy"]),
                float(np.max(np.abs(np.asarray(a["forces"]) - b["forces"]))),
            )
            for a, b in rows
        ]
        de, df = max(x[0] for x in errors), max(x[1] for x in errors)
        require(de == pair["energy_abs"] and df == pair["force_max_abs"], pair)
        passed = de <= 1e-9 and df <= 1e-8
        require(passed == pair["strict_diagnostic_pass"], pair)
        failures += not passed
    return {**counts, "strict_diagnostic_failures": failures}


def main() -> None:
    """Verify the raw bundles before rechecking their scientific contents."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory", type=Path, default=root / ".artifacts/remote"
    )
    args = parser.parse_args()
    summary = read_archive(directory / "summary.json.xz")
    for name, expected in summary["raw_bundles"].items():
        require(
            hashlib.sha256((args.raw_directory / name).read_bytes()).hexdigest()
            == expected,
            name,
        )
    for name, expected in summary["references"].items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
        )
    reference = directory.parent / "readme-final-20261022"
    counts = validate(
        summary,
        read_archive(args.raw_directory / "records.json.xz"),
        read_archive(reference / "oracles.json.xz"),
        json.loads((reference / "reference-reuse.json").read_text()),
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
