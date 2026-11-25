"""Recompute comparison and finite-difference gates using retained raw records."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

Record = dict[str, Any]
Dataset = dict[str, dict[str, Record]]
Force = list[list[float]]

HERE = Path(__file__).resolve().parent
PAIRS = (
    ("native-conv", "psi4-conv"),
    ("native-conv", "pyscf-conv"),
    ("psi4-conv", "pyscf-conv"),
    ("native-df", "psi4-df"),
)
STEPS = (0.001, 0.0005)
LIMITS = {
    "energy": 1e-7,
    "max_force": 3e-6,
    "rms_force": 1e-6,
    "fd_max_force": 3e-6,
    "fd_step_stability": 1e-6,
}
REPEATS = ("0", "1")


def read(path: Path) -> Record:
    return json.loads(path.read_text())


def check_record(
    row: Record, case: str, arm: str, trial: str, n_atoms: int | None
) -> None:
    if (row["case"], row["arm"], row["trial"]) != (case, arm, trial):
        raise ValueError(f"Invalid row identity for {case}/{arm}/{trial}")
    if not math.isfinite(row["energy_hartree"]):
        raise ValueError(f"Nonfinite energy for {case}/{arm}/{trial}")
    if not math.isfinite(row["endpoint_seconds"]) or row["endpoint_seconds"] <= 0:
        raise ValueError(f"Invalid timing for {case}/{arm}/{trial}")
    force = row["force_hartree_per_bohr"]
    if n_atoms is None:
        if force is not None:
            raise ValueError("An energy-only trial returned a force")
        return
    if (
        row["displacement"] is not None
        or len(force) != n_atoms
        or any(len(xyz) != 3 or not all(map(math.isfinite, xyz)) for xyz in force)
    ):
        raise ValueError(f"Invalid force vector for {case}/{arm}/{trial}")


def difference(a: Force, b: Force) -> tuple[float, float]:
    diffs = [
        x - y for u, v in zip(a, b, strict=True) for x, y in zip(u, v, strict=True)
    ]
    return max(map(abs, diffs)), math.sqrt(statistics.mean(d * d for d in diffs))


def summarize(data: Dataset, cases: dict[str, Record]) -> tuple[list[Record], Record]:
    def completed(arm: str, case: str) -> list[Record]:
        out = []
        for trial in REPEATS:
            row = data.get(arm, {}).get(f"{case}.{trial}", {})
            if row.get("status") == "completed":
                check_record(row, case, arm, trial, len(cases[case]["atoms_bohr"]))
                out.append(row)
        return out

    def pair_rows(case: str, left: str, right: str) -> list[Record]:
        errors = []
        for a, b in itertools.product(completed(left, case), completed(right, case)):
            energy = abs(a["energy_hartree"] - b["energy_hartree"])
            mx, rms = difference(
                a["force_hartree_per_bohr"], b["force_hartree_per_bohr"]
            )
            errors.append(
                {
                    "trial_left": a["trial"],
                    "trial_right": b["trial"],
                    "energy": energy,
                    "max_force": mx,
                    "rms_force": rms,
                }
            )
        return errors

    comparisons = []
    timings = {}
    for case in cases:
        timings[case] = {}
        for arm in dict.fromkeys(arm for pair in PAIRS for arm in pair):
            times = [row["endpoint_seconds"] for row in completed(arm, case)]
            timings[case][arm] = {
                "samples": times,
                "median": statistics.median(times)
                if len(times) == len(REPEATS)
                else None,
            }
        for left, right in PAIRS:
            errors = pair_rows(case, left, right)
            worst = (
                {
                    key: max(r[key] for r in errors)
                    for key in ("energy", "max_force", "rms_force")
                }
                if errors
                else {}
            )
            complete = len(errors) == len(REPEATS) ** 2
            outcomes = "incomplete"
            if complete:
                outcomes = (
                    "passed"
                    if all(value <= LIMITS[key] for key, value in worst.items())
                    else "failed"
                )
            comparisons.append(
                {
                    "case": case,
                    "left": left,
                    "right": right,
                    "status": outcomes,
                    "worst": worst,
                    "repeat_pairs": errors,
                }
            )
    return comparisons, timings


def finite_differences(data: Dataset, cases: dict[str, Record]) -> list[Record]:
    results = []
    n_atoms = len(cases["C1"]["atoms_bohr"])
    for reference in ("pyscf-conv", "psi4-df"):
        arm_data = data.get(reference, {})
        forces = []
        missing = []
        for step_index, step in enumerate(STEPS):
            values = []
            for atom, axis in itertools.product(range(n_atoms), range(3)):
                energies = {}
                for sign in (-1, 1):
                    key = f"C1.fd.{step_index}.{atom}.{axis}.{sign}"
                    row = arm_data.get(key, {})
                    if row.get("status") != "completed":
                        missing.append(key)
                        continue
                    check_record(row, "C1", reference, "fd", None)
                    if row["displacement"] != {
                        "atom": atom,
                        "axis": axis,
                        "sign": sign,
                        "step_bohr": step,
                    }:
                        raise ValueError(f"Invalid displacement metadata for {key}")
                    energies[sign] = row["energy_hartree"]
                if len(energies) == 2:
                    values.append(-(energies[1] - energies[-1]) / (2 * step))
            if len(values) == 3 * n_atoms:
                forces.append([values[i : i + 3] for i in range(0, len(values), 3)])
        if missing:
            results.append(
                {"reference": reference, "status": "incomplete", "missing": missing}
            )
            continue
        max_stability, rms_stability = difference(*forces)
        comparators = (
            ("native-conv", "psi4-conv", "pyscf-conv")
            if reference == "pyscf-conv"
            else ("native-df", "psi4-df")
        )
        errors = []
        for arm, trial in itertools.product(comparators, REPEATS):
            row = data.get(arm, {}).get("C1." + trial, {})
            if row.get("status") != "completed":
                continue
            check_record(row, "C1", arm, trial, n_atoms)
            mx, rms = difference(row["force_hartree_per_bohr"], forces[-1])
            errors.append({"arm": arm, "trial": trial, "max": mx, "rms": rms})
        passed = max_stability <= LIMITS["fd_step_stability"] and all(
            r["max"] <= LIMITS["fd_max_force"] for r in errors
        )
        results.append(
            {
                "reference": reference,
                "steps_bohr": STEPS,
                "status": (
                    "incomplete"
                    if len(errors) != len(REPEATS) * len(comparators)
                    else "passed"
                    if passed
                    else "failed"
                ),
                "max_step_difference": max_stability,
                "rms_step_difference": rms_stability,
                "forces_hartree_per_bohr": forces,
                "errors_at_finer_step": errors,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=HERE / "records.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()
    cases = read(HERE / "fixtures.json")["cases"]
    expected_cases = {
        *(f"C{n}" for n in range(1, 8)),
        "water",
        "ammonia",
        "formaldehyde",
    }
    if set(cases) != expected_cases:
        raise ValueError("Fixture population differs from the declared cohort")
    for case, fixture in cases.items():
        for role in ("orbital", "auxiliary"):
            identity = fixture[role + "_basis"]
            raw = (HERE / "basis" / (identity["identity"] + ".json")).read_bytes()
            if (
                len(raw) != identity["bytes"]
                or hashlib.sha256(raw).hexdigest() != identity["sha256"]
            ):
                raise ValueError(f"Basis payload mismatch for {case}/{role}")
    data = read(args.records)
    if args.records.resolve() == HERE / "records.json":
        execution = read(HERE / "execution.json")
        for arm, rows in data.items():
            locations = execution["primary_row_locations"][arm]
            if set(rows) != set(locations):
                raise ValueError(f"Primary row locations differ for {arm}")
            for label, row in rows.items():
                origin = execution["executions"][f"{arm}/{locations[label]}"]
                if arm.startswith("native") and origin["hardware"][
                    "source_revision"
                ] != ("22f72d0692a760c2892cd7791022584d9ae1c3e5"):
                    raise ValueError(
                        "A primary native row is from a different revision"
                    )
                raw = (json.dumps(row, indent=2, allow_nan=False) + "\n").encode()
                if hashlib.sha256(raw).hexdigest() != origin["row_shas"][label]:
                    raise ValueError(
                        f"Primary sample checksum differs for {arm}/{label}"
                    )
    comparisons, timings = summarize(data, cases)
    fd = finite_differences(data, cases)
    status = (
        "passed"
        if all(row["status"] == "passed" for row in comparisons + [fd[0]])
        and fd[1]["status"] != "failed"
        else "unqualified"
    )
    summary = {
        "status": status,
        "limits": LIMITS,
        "comparisons": comparisons,
        "endpoint_seconds": timings,
        "C1_finite_differences": fd,
    }
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return int(args.require_passed and status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
