"""Recompute the per-step timing table and sampled observable gates on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

HERE = Path(__file__).resolve().parent


def identity(record: list) -> str:
    data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def observable_delta(
    sample: dict, reference: dict, gates: dict, fraction: float
) -> dict:
    forces = sample["forces"]
    reference_forces = reference["forces"]
    if any(
        len(values) != 96
        or any(len(vector) != 3 for vector in values)
        or not all(math.isfinite(x) for vector in values for x in vector)
        for values in (forces, reference_forces)
    ):
        raise ValueError("invalid force vector in endpoint evidence")
    delta = [
        x - y
        for a, b in zip(forces, reference_forces, strict=True)
        for x, y in zip(a, b, strict=True)
    ]
    values = {
        "energy_abs_per_atom_hartree": abs(sample["energy"] - reference["energy"]) / 96,
        "force_max_abs_hartree_per_bohr": max(map(abs, delta)),
        "force_rms_hartree_per_bohr": math.sqrt(math.fsum(d * d for d in delta) / 288),
    }
    return {
        **values,
        "passed": all(values[key] <= fraction * limit for key, limit in gates.items()),
    }


def sample_index(record: dict, trajectory: dict) -> dict:
    if (
        record["trajectory"] != trajectory
        or record["source_revision"] != trajectory["source_revision"]
    ):
        raise ValueError("acquisition does not own exact trajectory/source")
    rows = {}
    oracle = record["arm"] == "oracle"
    for sample in record["samples"]:
        step = sample["step"]
        if not 0 <= step < len(trajectory["steps"]):
            raise ValueError("step outside the frozen trajectory")
        frame = trajectory["steps"][step]
        key = (step, sample["level"]) if oracle else step
        if oracle and (
            step not in trajectory["oracle_steps"]
            or sample["level"] not in trajectory["oracle_xc_grid_levels"]
        ):
            raise ValueError("oracle sample outside the declared steps and grids")
        if key in rows or sample["geometry_sha256"] != frame["geometry_sha256"]:
            raise ValueError("invalid or duplicate acquired sample")
        if not 0 < sample["seconds"] < math.inf:
            raise ValueError("invalid endpoint wall time")
        rows[key] = sample
    return rows


def replay(raw: dict) -> tuple[list[dict], list[dict], list[dict]]:
    trajectory = raw["trajectory"]
    steps = list(range(len(trajectory["steps"])))
    for step, frame in enumerate(trajectory["steps"]):
        if (
            frame["step"] != step
            or identity(frame["geometry_bohr"]) != frame["geometry_sha256"]
        ):
            raise ValueError("invalid frozen trajectory")
    cases = {
        (method, mode)
        for method in trajectory["methods"]
        for mode in trajectory["modes"]
    }
    attempts = {(r["method"], r["mode"], r["arm"]): r for r in raw["records"]}
    if len(attempts) != len(raw["records"]):
        raise ValueError("select one explicitly identified attempt per population")
    expected = {
        (m, mode, arm) for m, mode in cases for arm in ("standard", "gpu4pyscf")
    }
    expected |= {(m, "direct", "oracle") for m in trajectory["methods"]}
    if attempts.keys() - expected:
        raise ValueError("acquisition outside the declared populations")
    records = {key: sample_index(row, trajectory) for key, row in attempts.items()}
    gates = trajectory["gates"]
    levels = trajectory["oracle_xc_grid_levels"]
    fraction = trajectory["oracle_grid_stability_fraction"]
    sample_steps = trajectory["oracle_steps"]
    table = []
    summary = []
    qualification = []
    for method, mode in sorted(cases):
        native_key = (method, mode, "standard")
        reference_key = (method, mode, "gpu4pyscf")
        native_steps = records.get(native_key, {})
        reference_steps = records.get(reference_key, {})
        oracle_steps = records.get((method, "direct", "oracle"), {})
        for step in sample_steps:
            lower, higher = (oracle_steps.get((step, level)) for level in levels)
            grid_check = None
            if lower and higher and lower["converged"] and higher["converged"]:
                grid_check = observable_delta(lower, higher, gates, fraction)
            checks = {}
            for arm, endpoints in (
                ("standard", native_steps),
                ("gpu4pyscf", reference_steps),
            ):
                sample = endpoints.get(step)
                checks[arm] = (
                    observable_delta(sample, higher, gates, 1.0)
                    if sample and sample["converged"] and higher and higher["converged"]
                    else None
                )
            available_checks = (grid_check, *checks.values())
            qualification.append(
                {
                    "method": method,
                    "mode": mode,
                    "step": step,
                    "grid_stability": grid_check,
                    **checks,
                    "status": "pass"
                    if all(
                        check is not None and check["passed"]
                        for check in available_checks
                    )
                    else "fail"
                    if any(
                        check is not None and not check["passed"]
                        for check in available_checks
                    )
                    else "incomplete",
                }
            )
        totals = {"standard": 0.0, "gpu4pyscf": 0.0}
        for step in steps:
            native = native_steps.get(step, {})
            reference = reference_steps.get(step, {})
            row = {
                "method": method,
                "mode": mode,
                "step": step,
                "phase": "cold" if step == 0 else "moved",
            }
            for arm, sample in (("standard", native), ("gpu4pyscf", reference)):
                seconds = sample.get("seconds")
                if seconds is not None:
                    totals[arm] += seconds
                row[f"{arm}_seconds"] = seconds
                row[f"{arm}_cumulative_seconds"] = (
                    totals[arm] if seconds is not None else None
                )
                row[f"{arm}_iterations"] = sample.get("iterations")
                row[f"{arm}_converged"] = sample.get("converged")
                row[f"{arm}_used_seed"] = sample.get("warm_start_used")
                row[f"{arm}_force_evaluations"] = sample.get("force_evaluations")
            history = (native.get("ks_diagnostic") or {}).get("history")
            scopes = [entry.get("operator_scope") for entry in history or ()]
            reported_scopes = bool(scopes) and None not in scopes
            operator_work = (native.get("precision") or {}).get("operators", [])
            recurrence = next(
                (
                    o["count"]
                    for o in operator_work
                    if o["name"] == "coulomb_recurrence"
                ),
                None,
            )
            row.update(
                standard_fock_builds=native.get("fock_builds"),
                standard_full_target_cycles=sum(
                    scope == "full-target" for scope in scopes
                )
                if reported_scopes
                else None,
                standard_frozen_vv10_cycles=sum(
                    scope == "provisional-frozen-vv10" for scope in scopes
                )
                if reported_scopes
                else None,
                standard_coulomb_recurrence_count=recurrence,
                standard_seed_fallback=native.get("warm_start_fallback"),
                gpu4pyscf_effective_potential_builds=reference.get(
                    "effective_potential_builds"
                ),
            )
            table.append(row)
        arm_summary = {}
        for arm, key, records_by_step in (
            ("standard", native_key, native_steps),
            ("gpu4pyscf", reference_key, reference_steps),
        ):
            arm_complete = (
                attempts.get(key, {}).get("status") == "complete"
                and set(records_by_step) == set(steps)
                and all(row["converged"] for row in records_by_step.values())
            )
            arm_summary[arm] = {
                "complete": arm_complete,
                "completed_endpoints": sum(
                    row["converged"] for row in records_by_step.values()
                ),
                "cold_seconds": records_by_step.get(0, {}).get("seconds"),
                "moved_median_seconds": statistics.median(
                    records_by_step[k]["seconds"] for k in steps[1:]
                )
                if arm_complete
                else None,
                "moved_total_seconds": math.fsum(
                    records_by_step[k]["seconds"] for k in steps[1:]
                )
                if arm_complete
                else None,
                "cold_plus_moved_total_seconds": totals[arm] if arm_complete else None,
            }
        observed_ratio = None
        if all(arm["complete"] for arm in arm_summary.values()):
            observed_ratio = (
                arm_summary["gpu4pyscf"]["moved_total_seconds"]
                / arm_summary["standard"]["moved_total_seconds"]
            )
        summary.append(
            {
                "method": method,
                "mode": mode,
                **arm_summary,
                "moved_gpu4pyscf_over_standard": observed_ratio,
                "sampled_gates_passed": sum(
                    q["status"] == "pass"
                    for q in qualification
                    if (q["method"], q["mode"]) == (method, mode)
                ),
                "sampled_gates_total": len(sample_steps),
            }
        )
    return table, summary, qualification


def print_csv(rows: Iterable[dict]) -> None:
    iterator = iter(rows)
    first = next(iterator)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(first), lineterminator="\n")
    writer.writeheader()
    writer.writerow(first)
    writer.writerows(iterator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "records.json.xz")
    parser.add_argument("--steps-csv", action="store_true")
    args = parser.parse_args()
    with lzma.open(args.raw, "rt", encoding="utf-8") as stream:
        table, summary, qualification = replay(json.load(stream))
    if args.steps_csv:
        print_csv(table)
    else:
        print(
            json.dumps({"summary": summary, "qualification": qualification}, indent=2)
        )


if __name__ == "__main__":
    main()
