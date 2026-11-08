"""Replay the cross-functional survey, including unprofitable and rejected cases."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmarks.occupied_xc_functionals import independent_gate, replay_records
from benchmarks.occupied_xc_proposals import check_vectors, load
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT.parent / "readme-final-20261022"


def image(record: dict, capsule: dict, *, legacy: bool = False) -> None:
    """Bind every observation to the original immutable native image, not HEAD."""
    observed = (
        (
            record["source_revision"],
            record["native_build"]["library_sha256"],
            record["native_build"]["source_identity"],
        )
        if legacy
        else (
            record["source_commit"],
            record["library_sha256"],
            record["source_identity"],
        )
    )
    expected = tuple(
        capsule[key] for key in ("source_commit", "library_sha256", "source_identity")
    )
    if observed != expected:
        raise ValueError("native source or binary identity changed")


def scaling(records: dict, capsule: dict) -> dict:
    """Keep the two replication cohorts separate rather than pooling drift."""
    if set(records) != {
        f"r{repeat}-{control}" for repeat in range(3) for control in ("0", "unset")
    }:
        raise ValueError("incomplete legacy population")
    oracles = load(REFERENCES / "oracles.json.xz")
    reuse = load(REFERENCES / "reference-reuse.json")
    for name, record in records.items():
        image(record, capsule, legacy=True)
        expected_control = "0" if name.endswith("-0") else None
        if (
            record.get("occupied_xc_env", "missing") != expected_control
            or record["status"] != "complete"
        ):
            raise ValueError("invalid actual automatic/matrix control")
        if (
            record["method"] != "m06-2x"
            or record["arm"] != "standard"
            or record["mode"] != "direct"
        ):
            raise ValueError("wrong legacy execution contract")
        if [row["phase"] for row in record["samples"]] != ["cold", "warm"]:
            raise ValueError("incomplete legacy E+F phases")
        if (
            qualify([record], oracles, reference_reuse=reuse)[0]["status"]
            != "qualified"
        ):
            raise ValueError("independent dense reference gate failed")
    summary = {}
    for index, phase in enumerate(("cold", "warm")):
        values, work = {}, {}
        for control in ("0", "unset"):
            rows = [
                records[f"r{repeat}-{control}"]["samples"][index] for repeat in range(3)
            ]
            if any(
                not math.isfinite(row["seconds"])
                or row["seconds"] <= 0
                or row["iterations"] < 1
                or row["fock_builds"] < row["iterations"]
                for row in rows
            ):
                raise ValueError("invalid legacy duration or semantic work")
            values[control] = [row["seconds"] for row in rows]
            work[control] = [(row["iterations"], row["fock_builds"]) for row in rows]
        for repeat in range(3):
            baseline, default = (
                records[f"r{repeat}-{control}"]["samples"][index]
                for control in ("0", "unset")
            )
            check_vectors(default, baseline)
            if baseline["settings"] != default["settings"]:
                raise ValueError("legacy paired grid/solver settings changed")
        medians = {
            control: statistics.median(samples) for control, samples in values.items()
        }
        floor = max(
            0.02,
            2
            * sum(
                statistics.median(abs(value - medians[control]) for value in samples)
                / medians[control]
                for control, samples in values.items()
            ),
        )
        gain = 1 - medians["unset"] / medians["0"]
        summary[phase] = {
            "seconds": values,
            "medians": medians,
            "improvement_fraction": gain,
            "variability_floor": floor,
            "profitable": gain > floor,
            "regresses": -gain > floor,
            "iterations_fock_builds": work,
        }
    return summary


def survey(capsule: dict) -> dict:
    """Do not turn a failed oracle, paired gate or timing floor into a win."""
    if capsule["schema"] != "generativeqc.occupied-functional-survey.v1":
        raise ValueError("unsupported survey capsule")
    for name, expected in capsule["reference_sha256"].items():
        if hashlib.sha256((REFERENCES / name).read_bytes()).hexdigest() != expected:
            raise ValueError("historical independent reference changed")
    summary = {"functional": {}, "scaling": {}}
    for name, population in capsule["functional"].items():
        for record in population["records"]:
            image(record, capsule)
        try:
            result = replay_records(population["records"])
        except ValueError as error:
            if population["expected_rejection"] != str(error):
                raise
            result = {"qualified": False, "reason": str(error)}
        else:
            if population["expected_rejection"] is not None:
                raise ValueError("a rejected population unexpectedly passed")
        summary["functional"][name] = result
    for name, records in capsule["scaling"].items():
        summary["scaling"][name] = scaling(records, capsule)
    for label in ("discovery48", "confirmation48"):
        result = summary["scaling"][label]
        if not result["cold"]["profitable"] or result["warm"]["regresses"]:
            raise ValueError("M06-2X replication no longer clears the cost floor")
    if summary["scaling"]["size96"]["cold"]["profitable"]:
        raise ValueError("96-atom noise-level observation relabelled as a speedup")
    mn15 = summary["functional"]["mn15-48"]["phases"]
    if not all(mn15[phase]["regresses"] for phase in ("cold", "moved")):
        raise ValueError("MN15 regression was lost from the survey")
    dense6 = capsule["mn15_grid6"]
    dense5 = capsule["functional"]["mn15-48"]["records"][0]["oracle"]
    if (
        dense5["grid"]["level"] != 5
        or dense6["grid"]["level"] != 6
        or any(
            dense5[key] != dense6[key]
            for key in ("method", "basis", "spin", "atoms", "geometries")
        )
    ):
        raise ValueError("MN15 grid stability identity changed")
    stability = []
    for first, last in zip(dense5["observations"], dense6["observations"], strict=True):
        energy, force = independent_gate(first, last, 48)
        rms = math.sqrt(
            sum(
                (left - right) ** 2
                for first_row, last_row in zip(
                    first["forces"], last["forces"], strict=True
                )
                for left, right in zip(first_row, last_row, strict=True)
            )
            / (3 * 48)
        )
        if energy / 48 > 0.5e-5 or force > 0.5e-4 or rms > 1.5e-5:
            raise ValueError("MN15 independent dense grid stability failed")
        stability.append({"energy": energy, "force_max": force, "force_rms": rms})
    for record in capsule["functional"]["mn15-48"]["records"]:
        for row in record["observations"]:
            independent_gate(
                row, dense6["observations"][int(row["phase"].startswith("moved"))], 48
            )
    summary["mn15_grid_stability"] = stability
    for name, diagnostic in capsule["traces"].items():
        record = diagnostic["record"]
        image(record, capsule)
        frames = diagnostic["occupied_frame_generations"]
        if (
            not record["completed"]
            or not record["traced"]
            or record["occupied_xc_env"] is not None
            or not any(frames)
            or frames[-1]
        ):
            raise ValueError(f"missing default proposal/strict-tail diagnostic: {name}")
        for row in record["observations"]:
            independent_gate(
                row,
                record["oracle"]["observations"][int(row["phase"].startswith("moved"))],
                3,
            )
    for spot in capsule["cpu_gpu_spots"]:
        for cpu, gpu in zip(
            spot["cpu"]["observations"], spot["gpu"]["observations"], strict=True
        ):
            check_vectors(cpu, gpu)
    return summary


def main() -> None:
    """Corruption probes remain active under optimized Python as well."""
    capsule = load(ROOT / "survey.json.xz")
    result = survey(capsule)
    for corruption in ("source", "control", "energy", "force", "missing"):
        changed = copy.deepcopy(capsule)
        record = changed["scaling"]["discovery48"]["r0-unset"]
        if corruption == "source":
            record["source_revision"] = "0" * 40
        elif corruption == "control":
            record["occupied_xc_env"] = "1"
        elif corruption == "missing":
            changed["scaling"]["discovery48"].pop("r2-0")
        else:
            record["samples"][0][corruption if corruption == "energy" else "forces"] = (
                1.0 if corruption == "energy" else [[float("nan")] * 3] * 48
            )
        try:
            survey(changed)
        except ValueError:
            continue
        raise ValueError(f"corrupted {corruption} receipt was accepted")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
