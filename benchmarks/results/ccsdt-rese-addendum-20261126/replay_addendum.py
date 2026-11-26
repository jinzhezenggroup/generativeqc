"""Collect and replay the local addendum against the immutable #3178 baseline."""

import argparse
import hashlib
import itertools
import json
import lzma
import math
import sys
from importlib import import_module
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
JsonObject = dict[str, Any]


def read_archive(path: Path) -> JsonObject | list[JsonObject]:
    with lzma.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def archive(path: Path, records: JsonObject | list[JsonObject]) -> None:
    with lzma.open(path, "xt", encoding="utf8") as stream:
        json.dump(records, stream, separators=(",", ":"), allow_nan=False)


def numerical_range(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"minimum": min(values), "median": median(values), "maximum": max(values)}


def trajectory(
    records: JsonObject, native: JsonObject, replay: ModuleType
) -> JsonObject:
    lines = replay.decode(records, "stderr.txt").splitlines()
    events = []
    incomplete_last_record = False
    for index, line in enumerate(lines):
        if not line.startswith('{"event":'):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or native["status"] == "converged":
                raise
            incomplete_last_record = True
    ends = [row for row in events if row["event"] == "evaluation_end"]
    states = [row for row in events if row["event"] == "accepted"]
    if native["status"] == "converged":
        if len(ends) != native["evaluations"] or len(states) != native["steps"] + 1:
            raise ValueError("trace count disagrees with the native optimizer")
        assert states[-1]["max_force"] == native["max_force"]
    if not ends:
        return {"completed_evaluations": 0}
    assert [row["evaluation"] for row in ends] == list(range(1, len(ends) + 1))
    assert [row["step"] for row in states] == list(range(len(states)))
    assert all(
        b["energy_hartree"] <= a["energy_hartree"]
        for a, b in itertools.pairwise(states)
    )
    elapsed = []
    total = 0.0
    for row in ends:
        total += row["seconds"]
        elapsed.append(total)
    displacements = [
        replay.difference(b["coordinates_bohr"], a["coordinates_bohr"])
        for a, b in itertools.pairwise(states)
    ]
    cosines = [
        math.fsum(x * y for x, y in zip(a, b, strict=True))
        / (math.hypot(*a) * math.hypot(*b))
        for a, b in itertools.pairwise(displacements)
        if math.hypot(*a) and math.hypot(*b)
    ]
    legacy_limits = []
    for seconds in (1200, 2400):
        n = sum(value <= seconds for value in elapsed)
        accepted = [row for row in states if row["evaluations"] <= n]
        legacy_limits.append(
            {
                "seconds": seconds,
                "completed_evaluations": n,
                "accepted_steps": accepted[-1]["step"] if accepted else None,
                "last_accepted_force": accepted[-1]["max_force"] if accepted else None,
            }
        )
    return {
        "completed_evaluations": len(ends),
        "incomplete_last_record": incomplete_last_record,
        "accepted_steps": states[-1]["step"] if states else None,
        "recorded_rejected_trials": sum(
            b["evaluations"] - a["evaluations"] - 1
            for a, b in itertools.pairwise(states)
        ),
        "force_increase_steps": sum(
            b["max_force"] > a["max_force"] for a, b in itertools.pairwise(states)
        ),
        "accepted_step_direction_cosine": numerical_range(cosines),
        "sum_evaluation_seconds": total,
        "evaluation_seconds": numerical_range([row["seconds"] for row in ends]),
        "step_seconds": numerical_range([row["step_seconds"] for row in states[1:]])
        if len(states) > 1
        else None,
        # At 22f72d069 the DF-RCCSD(T) result's generic iterations are CCSD.
        # The original trace label is retained verbatim in the raw records.
        "reference_scf_iterations": None,
        "ccsd_iterations": numerical_range(
            [
                row["ccsd_iterations"]
                for row in ends
                if row["ccsd_iterations"] is not None
            ]
        ),
        "progress_at_legacy_limits_by_summed_evaluation_time": legacy_limits,
    }


def verify_minimum(
    record: JsonObject, native_sha: str, gates: JsonObject, replay: ModuleType
) -> JsonObject:
    for name in record:
        replay.decode(record, name)
    if "check.json" not in record:
        return {"status": "interrupted", "retained_files": len(record)}
    result = json.loads(replay.decode(record, "check.json"))
    if result["status"] == "failed":
        return result
    assert result["native_receipt_sha256"] == native_sha
    rank = result["rigid_rank"]
    dimension = len(result["curvatures_hartree_per_bohr2"][0])
    assert len(record) == 4 * dimension**2 + 2
    assert dimension + rank == len(result["mode_basis"])
    lowest = min(row[0] for row in result["curvatures_hartree_per_bohr2"])
    passed = (
        lowest > gates["curvature_floor"]
        and result["center_energy_difference_hartree"] <= gates["energy_hartree"]
        and result["center_projected_max_force_hartree_per_bohr"]
        <= gates["force_hartree_per_bohr"]
        and result["max_curvature_change_between_steps"] <= gates["curvature_stability"]
    )
    assert result["status"] == ("minimum_verified" if passed else "gate_failed")
    return {
        key: result[key]
        for key in (
            "status",
            "rigid_rank",
            "center_energy_difference_hartree",
            "center_projected_max_force_hartree_per_bohr",
            "max_curvature_change_between_steps",
            "curvatures_hartree_per_bohr2",
            "native_receipt_sha256",
            "seconds",
        )
    }


def report(
    baseline: Path, gpu: list[JsonObject], cpu: list[JsonObject], replay: ModuleType
) -> JsonObject:
    old_gpu = read_archive(baseline / "records.json.xz")
    old_cpu = read_archive(baseline / "minima.json.xz")["df"]
    old_cpu["provenance"] = json.loads(
        replay.decode(old_cpu["provenance"], "provenance.json")
    )
    cohort = replay.read_json(baseline / "cohort.json")
    attempts = old_gpu + gpu
    summary = replay.summarize(cohort, attempts)
    original = {attempt["id"]: attempt["records"]["cases"] for attempt in attempts}
    new_ids = {attempt["id"] for attempt in gpu}
    minima = {}
    for batch in [old_cpu, *cpu]:
        if batch["provenance"] != old_cpu["provenance"]:
            raise ValueError("PySCF version, basis inputs, or gates differ from #3178")
        gates = batch["provenance"]["gates"]
        for name, records in batch["cases"].items():
            if name in minima:
                raise ValueError(f"duplicate minimum check for {name}")
            minima[name] = (records, gates)
    angles = summary["parameters"]["df-cc-pcvtz/re_se/angles"]
    nh3_experimental_angle = 106.7
    alternative_angles = [
        {
            **row,
            "reference": nh3_experimental_angle,
            "error": row["measured"] - nh3_experimental_angle,
        }
        if row["molecule"] == "nh3" and row["label"] == "H-N-H"
        else row
        for row in angles
    ]
    result = {
        "baseline_cohort_sha256": hashlib.sha256(
            (baseline / "cohort.json").read_bytes()
        ).hexdigest(),
        "comparison": summary["comparison"],
        "nh3_experimental_angle_sensitivity": {
            "reference_degrees": nh3_experimental_angle,
            "comparison": replay.stats(alternative_angles),
        },
        "molecules": [],
        "trace": {},
        "counts": {
            "df_converged": 0,
            "df_minimum_verified": 0,
            "df_minimum_checked": 0,
            "df_minimum_attempted": 0,
        },
    }
    for molecule in summary["molecules"]:
        name = molecule["id"]
        result_row = {"id": name, "optimization": {"status": "unresolved"}}
        for attempt in molecule["attempts"]:
            stage = attempt.get("df-cc-pcvtz", {})
            arm = attempt["attempt"]
            if arm in new_ids and "stderr.txt" in original[arm][name].get(
                "df-cc-pcvtz", {}
            ):
                trace = trajectory(original[arm][name]["df-cc-pcvtz"], stage, replay)
                rhf = original[arm][name]["rhf"]
                rhf_events = (
                    json.loads(line)
                    for line in replay.decode(rhf, "stderr.txt").splitlines()
                    if line.startswith('{"event":"evaluation_end"')
                )
                trace["preoptimization_scf_iterations"] = numerical_range(
                    [event["reference_scf_iterations"] for event in rhf_events]
                )
                rhf_out = json.loads(replay.decode(rhf, "stdout.json"))
                rhf_xyz = list(
                    itertools.chain.from_iterable(rhf_out["coordinates_bohr"])
                )
                trace["preoptimization_against_previous_attempts"] = {}
                for previous in old_gpu:
                    case = original[previous["id"]].get(name, {})
                    previous_rhf = case.get("rhf", {})
                    if not previous_rhf or not case.get("df-cc-pcvtz"):
                        continue
                    if "stdout.json" not in previous_rhf:
                        continue
                    earlier = json.loads(replay.decode(previous_rhf, "stdout.json"))
                    earlier_xyz = itertools.chain.from_iterable(
                        earlier["coordinates_bohr"]
                    )
                    delta = replay.difference(rhf_xyz, earlier_xyz)
                    trace["preoptimization_against_previous_attempts"][
                        previous["id"]
                    ] = {
                        "max_absolute_component_bohr": max(map(abs, delta)),
                        "energy_difference_hartree": (
                            rhf_out["stages"][0]["energy_hartree"]
                            - earlier["stages"][0]["energy_hartree"]
                        ),
                    }
                result["trace"][name] = trace
            if stage.get("status") != "converged":
                continue
            result["counts"]["df_converged"] += 1
            result_row["optimization"] = {"attempt": arm, **stage}
            if name in minima:
                result["counts"]["df_minimum_attempted"] += 1
                records, gates = minima.pop(name)
                stdout = original[arm][name]["df-cc-pcvtz"]["stdout.json"]
                found = verify_minimum(
                    records, stdout["original_sha256"], gates, replay
                )
                result_row["minimum"] = found
                result["counts"]["df_minimum_checked"] += found["status"] in {
                    "minimum_verified",
                    "gate_failed",
                }
                result["counts"]["df_minimum_verified"] += (
                    found["status"] == "minimum_verified"
                )
        result["molecules"].append(result_row)
    if minima:
        raise ValueError(
            f"curvature inputs without a qualified GPU endpoint: {', '.join(minima)}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT.parent / "ccsdt-rese-20261125"
    )
    parser.add_argument("--gpu", type=Path, action="append", default=[])
    parser.add_argument("--cpu", type=Path, action="append", default=[])
    parser.add_argument("--superseded-cpu", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.baseline.resolve()))
    replay = import_module("replay")
    if args.gpu:
        archive(
            ROOT / "gpu-records.json.xz",
            [
                {"id": raw.parent.name, "records": replay.capture(raw)}
                for raw in args.gpu
            ],
        )
    for raw in args.cpu + args.superseded_cpu:
        cases = {
            case.name: replay.files(case)
            for case in sorted(raw.iterdir())
            if case.is_dir()
        }
        prefix = "superseded-" if raw in args.superseded_cpu else ""
        archive(
            ROOT / f"{prefix}{raw.parent.name}.json.xz",
            {"provenance": replay.read_json(raw / "provenance.json"), "cases": cases},
        )
    gpu_path = ROOT / "gpu-records.json.xz"
    gpu = read_archive(gpu_path) if gpu_path.is_file() else []
    cpu = [read_archive(path) for path in sorted(ROOT.glob("cpu-*.json.xz"))]
    summary = report(args.baseline, gpu, cpu, replay)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
