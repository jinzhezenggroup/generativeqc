"""Recompute numerical, population and endpoint timing gates from retained data."""

from __future__ import annotations

import json
import lzma
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def median_mad(values: list[float]) -> tuple[float, float]:
    """Reduce complete processes, retaining variability rather than best time."""
    center = statistics.median(values)
    return center, statistics.median(abs(value - center) for value in values)


def reproduce(payload: dict) -> dict:
    """Reject missing samples, mixed binaries and any independent observable miss."""
    records = payload["qualification"]
    expected = {
        f"fast-pbe0-{size}-{policy}-r{repeat}.json"
        for size in (12, 48, 96)
        for policy in ("off", "default")
        for repeat in range(1, 6)
    }
    expected |= {
        f"fast-{method}-{size}-{policy}-lifecycle.json"
        for method, size in (
            ("pbe0", 12),
            ("b3lyp", 12),
            ("m06-2x", 12),
            ("pbe0", 48),
            ("pbe0", 96),
        )
        for policy in ("off", "default")
    }
    expected |= {
        f"{prefix}-{policy}.json"
        for prefix in (
            "control-standard-direct",
            "control-standard-df",
            "control-fast-df",
            "capacity-fast12",
            "diagnostic-fast96",
        )
        for policy in ("off", "default")
    }
    assert set(records) == expected, ("missing", expected - set(records))
    maxima = {key: 0.0 for key in payload["protocol"]["numerical_gates"]}
    endpoint_count = 0
    work = {}
    for name, record in sorted(records.items()):
        assert (
            record["native_build"]["library_sha256"]
            == payload["binary"]["library_sha256"]
        )
        selector = record["runtime_controls"][
            "GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING"
        ]
        assert selector == ("0" if "-off" in name else None), name
        expected_phases = ["cold", "warm", "warm"]
        if "lifecycle" in name:
            expected_phases += ["moved", "moved-warm", "reset"]
        assert [sample["phase"] for sample in record["samples"]] == expected_phases
        reference = payload["references"][
            f"{record['method']}-{record['mode']}-{len(record['atoms'])}.json"
        ]
        assert reference["atoms"] == record["atoms"]
        reference_samples = {row["phase"]: row for row in reference["samples"]}
        for sample in record["samples"]:
            target = reference_samples[
                "original" if sample["phase"] in ("cold", "warm") else "moved"
            ]
            differences = [
                value - expected_value
                for force, expected_force in zip(
                    sample["forces"], target["forces"], strict=True
                )
                for value, expected_value in zip(force, expected_force, strict=True)
            ]
            errors = {
                "energy_per_atom": abs(sample["energy"] - target["energy"])
                / len(record["atoms"]),
                "force_max": max(map(abs, differences)),
                "force_rms": math.sqrt(
                    sum(value * value for value in differences) / len(differences)
                ),
            }
            for observable, error in errors.items():
                assert (
                    math.isfinite(error)
                    and error < payload["protocol"]["numerical_gates"][observable]
                ), (name, sample["phase"], observable, error)
                maxima[observable] = max(maxima[observable], error)
            assert sample["passed"] and sample["admission_passed"], name
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            endpoint_count += 1
        work[name] = {
            "phases": expected_phases,
            "fock_builds": [row["fock_builds"] for row in record["samples"]],
            "iterations": [row["iterations"] for row in record["samples"]],
        }
    timings = []
    for size in (12, 48, 96):
        for phase in ("cold", "warm"):
            statistics_by_policy = {}
            for policy in ("off", "default"):
                values = []
                for repeat in range(1, 6):
                    record = records[f"fast-pbe0-{size}-{policy}-r{repeat}.json"]
                    values.append(
                        statistics.median(
                            row["seconds"]
                            for row in record["samples"]
                            if row["phase"] == phase
                        )
                    )
                center, mad = median_mad(values)
                statistics_by_policy[policy] = {
                    "median_seconds": center,
                    "mad_seconds": mad,
                }
            off, default = (statistics_by_policy[key] for key in ("off", "default"))
            gain = 1 - default["median_seconds"] / off["median_seconds"]
            floor = max(
                0.02,
                2
                * sum(
                    row["mad_seconds"] / row["median_seconds"] for row in (off, default)
                ),
            )
            assert gain >= -floor, ("regression", size, phase, gain, floor)
            if size == 96 and phase == "cold":
                assert gain > floor, ("gain floor missed", gain, floor)
            timings.append(
                {
                    "atoms": size,
                    "phase": phase,
                    **statistics_by_policy,
                    "gain": gain,
                    "noise_floor": floor,
                    "significant_gain": gain > floor,
                }
            )
    # Pair work independently of latency. Equal Fock counts alone never qualify
    # observables, and different counts would not justify deleting a sample.
    pairs = []
    for name in sorted(records):
        if "-off" in name:
            other = name.replace("-off", "-default")
            pairs.append(
                {"off": name, "default": other, "same_work": work[name] == work[other]}
            )
    assert endpoint_count == 180
    return {
        "processes": len(records),
        "endpoints": endpoint_count,
        "maximum_errors": maxima,
        "timings": timings,
        "pairs": pairs,
        "work": work,
    }


if __name__ == "__main__":
    with lzma.open(ROOT / "qualification.json.xz", "rt") as stream:
        retained = json.load(stream)
    print(json.dumps(reproduce(retained), indent=2, allow_nan=False))
