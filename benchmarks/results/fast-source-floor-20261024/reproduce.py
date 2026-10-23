"""Replay the frozen proposal-floor experiment without a GPU or assert gates."""

from __future__ import annotations

import json
import lzma
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATES = {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6}
POLICIES = ("baseline", "candidate")


def require(condition: bool, context: object) -> None:
    """Keep evidence validation active under -O and PYTHONOPTIMIZE."""
    if not condition:
        raise ValueError(context)


def expected_records() -> set[str]:
    """Declare all repetitions and controls before reading retained outcomes."""
    names = {
        f"fast-pbe0-{size}-{policy}-r{repeat}.json"
        for size in (12, 48, 96)
        for policy in POLICIES
        for repeat in range(1, 6)
    }
    names |= {
        f"fast-{method}-{size}-{policy}-lifecycle.json"
        for method, size in (
            ("pbe0", 12),
            ("b3lyp", 12),
            ("m06-2x", 12),
            ("pbe0", 48),
            ("pbe0", 96),
        )
        for policy in POLICIES
    }
    names |= {
        f"{prefix}-{policy}.json"
        for prefix in (
            "control-standard-direct",
            "control-standard-df",
            "control-fast-df",
            "capacity-fast12",
            "diagnostic-fast96",
        )
        for policy in POLICIES
    }
    return names | {"rollback-candidate.json"}


def observable_errors(sample: dict, target: dict, atoms: int) -> dict:
    """Recompute independent errors from full forces, including shape checks."""
    require(len(sample["forces"]) == len(target["forces"]) == atoms, "force shape")
    differences = []
    for force, reference in zip(sample["forces"], target["forces"], strict=True):
        require(len(force) == len(reference) == 3, "force components")
        differences.extend(
            value - expected for value, expected in zip(force, reference, strict=True)
        )
    require(all(math.isfinite(value) for value in differences), "nonfinite forces")
    return {
        "energy_per_atom": abs(sample["energy"] - target["energy"]) / atoms,
        "force_max": max(map(abs, differences)),
        "force_rms": math.sqrt(
            sum(value * value for value in differences) / len(differences)
        ),
    }


def reproduce(payload: dict) -> dict:
    """Check exact populations, source/binary identity, observables and timing."""
    records = payload["qualification"]
    require(
        set(records) == expected_records(),
        ("population", set(records) ^ expected_records()),
    )
    require(payload["protocol"]["numerical_gates"] == GATES, "changed numerical gates")
    completed = {}
    for group in ("final-timing", "final-controls"):
        for line in payload["exit_status"][group].splitlines():
            name, status = line.split()
            require(name not in completed and status == "0", ("process exit", line))
            completed[name] = status
    require(
        {name + ".json" for name in completed} == expected_records(),
        "incomplete process exits",
    )
    tests = dict(
        line.split() for line in payload["exit_status"]["final-tests"].splitlines()
    )
    require(
        tests
        == dict.fromkeys(
            ("incremental", "df-cold-factor", "fast-anchor-delta", "public"), "0"
        ),
        ("GPU regressions", tests),
    )
    maxima = dict.fromkeys(GATES, 0.0)
    work = {}
    endpoints = 0
    for name, record in sorted(records.items()):
        policy = "baseline" if "-baseline" in name else "candidate"
        require(
            record["native_build"]["library_sha256"]
            == payload["binary"][policy]["library_sha256"],
            (name, "binary"),
        )
        require(
            record["source_revision"] == payload["base"] + "+final-" + policy,
            (name, "source"),
        )
        require(record["device"] == "NVIDIA GeForce RTX 5090", (name, "device"))
        require(record["gates"] == GATES, (name, "record gates"))
        controls = record["runtime_controls"]
        require(
            controls["GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING"]
            == ("0" if name.startswith("rollback-") else None),
            (name, "selector"),
        )
        require(
            controls["GENERATIVEQC_KS_RESIDENT_MAX_BYTES"]
            == ("1" if name.startswith("capacity-") else None),
            (name, "capacity"),
        )
        for key in (
            "GENERATIVEQC_KS_RESIDENT_VALUES",
            "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
        ):
            require(controls[key] is None, (name, key))
        phases = ["cold", "warm", "warm"]
        if "lifecycle" in name:
            phases += ["moved", "moved-warm", "reset"]
        require(
            [sample["phase"] for sample in record["samples"]] == phases,
            (name, "phases"),
        )
        atoms = len(record["atoms"])
        reference = payload["references"][
            f"{record['method']}-{record['mode']}-{atoms}.json"
        ]
        for key in ("method", "mode", "atoms"):
            require(reference[key] == record[key], (name, "reference", key))
        targets = {row["phase"]: row for row in reference["samples"]}
        work[name] = []
        for sample in record["samples"]:
            phase = sample["phase"]
            target = targets["original" if phase in ("cold", "warm") else "moved"]
            errors = observable_errors(sample, target, atoms)
            for observable, error in errors.items():
                require(
                    math.isfinite(error) and error < GATES[observable],
                    (name, phase, observable, error),
                )
                maxima[observable] = max(maxima[observable], error)
            require(
                sample["passed"] and sample["admission_passed"],
                (name, phase, "acceptance"),
            )
            require(
                math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                (name, "time"),
            )
            requested = record["arm"] == "standard" and record["mode"] == "direct"
            incremental = sample["incremental"]
            require(
                incremental["requested"] == requested, (name, "incremental requested")
            )
            require(
                incremental["active"] == (requested and phase in ("cold", "reset")),
                (name, "incremental active"),
            )
            precision = sample["precision"]
            require(
                precision["complete"] and precision["operator_work_counters_valid"],
                (name, "work completeness"),
            )
            work[name].append(
                {
                    "phase": phase,
                    "focks": sample["fock_builds"],
                    "iterations": sample["iterations"],
                    "mixed": precision["mixed_stage_fock_builds"],
                    "strict": precision["strict_stage_fock_builds"],
                    "post_scf": precision["post_scf_fock_builds"],
                    "audits": precision["final_residual_audits"],
                    "mixed_recurrence_values": sum(
                        op["count"]
                        for op in precision["operators"]
                        if op["name"] == "coulomb_recurrence"
                        and op["arithmetic_mode"] == "mixed"
                    ),
                }
            )
            endpoints += 1
    timings = []
    for size in (12, 48, 96):
        for phase in ("cold", "warm"):
            populations = {}
            for policy in POLICIES:
                values = [
                    statistics.median(
                        sample["seconds"]
                        for sample in records[
                            f"fast-pbe0-{size}-{policy}-r{repeat}.json"
                        ]["samples"]
                        if sample["phase"] == phase
                    )
                    for repeat in range(1, 6)
                ]
                median = statistics.median(values)
                populations[policy] = {
                    "seconds": values,
                    "median": median,
                    "mad": statistics.median(abs(value - median) for value in values),
                }
            baseline, candidate = (populations[policy] for policy in POLICIES)
            gain = 1 - candidate["median"] / baseline["median"]
            floor = max(
                0.02,
                2 * sum(row["mad"] / row["median"] for row in (baseline, candidate)),
            )
            require(gain >= -floor, ("regression", size, phase, gain, floor))
            if size == 96 and phase == "cold":
                require(gain > floor, ("gain floor", gain, floor))
            timings.append(
                {
                    "atoms": size,
                    "phase": phase,
                    **populations,
                    "gain": gain,
                    "noise_floor": floor,
                }
            )
    require(endpoints == 183, ("endpoint count", endpoints))
    return {
        "processes": len(records),
        "endpoints": endpoints,
        "maximum_errors": maxima,
        "timings": timings,
        "work": work,
    }


if __name__ == "__main__":
    with lzma.open(ROOT / "qualification.json.xz", "rt") as stream:
        retained = json.load(stream)
    print(json.dumps(reproduce(retained), indent=2, allow_nan=False))
