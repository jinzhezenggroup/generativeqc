"""Prepare independent tblite fixtures and validate a real native mixer journal.

The native bootstrap runs the endpoint. This module never substitutes a Python
CUDA mock or planner count for a device receipt. Cycle shares are per invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

LIMIT = 64 * 1024 * 1024
GATE = 5e-7
PARITY = 5e-14
STAGES = ("residual", "history", "solve", "combination")
ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_json(path: Path) -> dict:
    if path.stat().st_size > LIMIT:
        raise ValueError("input exceeds 64 MiB")
    return json.loads(path.read_text(), parse_constant=lambda _: None)


def prepare(directory: Path, waters: list[int]) -> None:
    """Freeze input geometry and fresh references before any CUDA run."""
    import importlib.metadata

    import numpy as np
    from tblite.interface import Calculator

    if importlib.metadata.version("tblite") != "0.6.0":
        raise ValueError("qualification requires pinned tblite 0.6.0")
    from compare_xtbloom import cases

    if directory.exists():
        raise ValueError("oracle output directory already exists")
    if any(count not in (8, 32, 128) for count in waters):
        raise ValueError("only existing 48/192/768-AO water domains are admitted")
    directory.mkdir(parents=True)
    symbols = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "Cl": 17}
    records = []
    for original in cases(waters):
        positions = np.asarray(original["positions"], dtype=float)
        moved = positions.copy()
        moved.flat[0] += 0.015
        for geometry, coordinates in enumerate((positions, moved)):
            numbers = np.asarray([symbols[s] for s in original["symbols"]])
            calculator = Calculator("GFN2-xTB", numbers, coordinates)
            calculator.set("accuracy", 0.01)
            calculator.set("max-iter", 300)
            calculator.set("temperature", 300 * 3.166808578545117e-6)
            calculator.set("verbosity", 0)
            result = calculator.singlepoint()  # no restart object
            shells = len(calculator.get("shell-map"))
            record = {
                "name": original["name"] + ("-moved" if geometry else ""),
                "numbers": numbers.tolist(),
                "positions": coordinates.tolist(),
                "energy": float(result.get("energy")),
                "forces": (-result.get("gradient")).reshape(-1).tolist(),
                "vector_elements": shells + 9 * len(numbers),
                "intrinsic_aos": len(calculator.get("orbital-map")),
            }
            # Retain the original frozen small-fixture comparison as a separate
            # gate, so a regenerated reference cannot hide a changed fixture.
            if geometry == 0 and "energy" in original:
                assert abs(record["energy"] - original["energy"]) < GATE
                assert (
                    np.max(
                        np.abs(
                            np.asarray(record["forces"])
                            - np.asarray(original["forces"]).reshape(-1)
                        )
                    )
                    < GATE
                )
            records.append(record)
            (directory / "oracle.json").write_text(
                json.dumps(
                    {
                        "oracle": "tblite 0.6.0",
                        "accuracy": 0.01,
                        "fixture_sha256": digest(
                            ROOT / "tests/data/gfn2_native_tblite.json"
                        ),
                        "cases": records,
                    },
                    indent=2,
                )
                + "\n"
            )
    lines = [str(len(records))]
    for record in records:
        lines.append(
            f"{record['name']} {len(record['numbers'])} {record['energy']:.17g}"
        )
        for index, atomic_number in enumerate(record["numbers"]):
            values = [
                atomic_number,
                *record["positions"][index],
                *record["forces"][3 * index : 3 * index + 3],
            ]
            lines.append(" ".join(str(value) for value in values))
    (directory / "oracle.txt").write_text("\n".join(lines) + "\n")


def integer(value: object, name: str, low: int = 0, high: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {name}")
    return value


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_input(row: dict, expected: dict, nonfinite_position: bool = False) -> None:
    for atomic_number in row["numbers"]:
        integer(atomic_number, "atomic number", low=1, high=86)
    positions = [coordinate for atom in expected["positions"] for coordinate in atom]
    if nonfinite_position:
        positions[0] = None
    if row["numbers"] != expected["numbers"] or row["positions"] != positions:
        raise ValueError("actual input differs from independent fixture")
    if any(
        not finite(value)
        for value in row["positions"][(1 if nonfinite_position else 0) :]
    ):
        raise ValueError("invalid actual coordinate")


def validate_metadata(row: dict) -> None:
    integer(row["schema"], "schema", low=1, high=1)
    for flag in (
        "enabled",
        "snapshot_available",
        "graph_submitted",
        "endpoint_completed",
    ):
        if type(row[flag]) is not bool:
            raise ValueError("invalid snapshot boolean")
    for field, low, high in (
        ("iterations", -42, 301),
        ("status", -1, 100),
        ("system_status", -43, 100),
        ("converged", 0, 255),
        ("flags", 0, (1 << 32) - 1),
        ("order", 0, 1),
        ("call_id", 0, (1 << 64) - 1),
        ("plan_token", 0, (1 << 64) - 1),
        ("device_id", -1, (1 << 31) - 1),
        ("graph_family", 0, 2),
        ("attempted_receipts", 0, (1 << 64) - 1),
        ("receipt_bytes", 1, LIMIT),
        ("maximum_iterations", 0, 301),
    ):
        integer(row[field], field, low=low, high=high)


@lru_cache(maxsize=1024)
def work_oracle(index: int, dimension: int) -> tuple[int, int, int, int, int]:
    """Enumerate chronological occupied slots and component visits independently.

    This deliberately does not call the production history-window generator.
    Restarted/partial trajectories stay INCOMPLETE rather than being guessed.
    """
    slots = []
    for step in range(1, index + 1):
        slot = (step - 1) % 8
        if slot in slots:
            slots.remove(slot)
        slots.append(slot)
    coefficient = gram = combination = 0
    for _slot in slots:
        for _component in range(dimension):
            coefficient += 1
            combination += 1
        for _other in slots:
            for _component in range(dimension):
                gram += 1
    return len(slots), (index - 1) % 8 if index else -1, coefficient, gram, combination


def validate_receipts(row: dict, expected: dict) -> dict:
    receipts = row["receipts"]
    attempted = integer(row["attempted_receipts"], "attempted")
    if len(receipts) > 300:
        raise ValueError("retained receipt capacity exceeded")
    if not row["snapshot_available"] or not row["endpoint_completed"]:
        return {"status": "INCOMPLETE", "reason": "snapshot or endpoint unavailable"}
    if attempted != len(receipts) or not receipts:
        return {"status": "INCOMPLETE", "reason": "empty or truncated journal"}
    integer(row["device_id"], "device identity")
    integer(row["graph_family"], "graph family", high=2)
    integer(row["plan_token"], "plan identity", low=1)
    if (
        row["device_id"] != 0
        or row["graph_family"] not in (0, 1, 2)
        or row["plan_token"] <= 0
    ):
        raise ValueError("snapshot identity mismatch")
    if row["graph_family"] in (1, 2) and not row["graph_submitted"]:
        return {"status": "INCOMPLETE", "reason": "graph was not submitted"}
    dimension = integer(expected["vector_elements"], "oracle dimension", 1, 100000)
    shares = []
    for index, receipt in enumerate(receipts):
        for key in (
            "invocation",
            "iteration_before",
            "restart_before",
            "completed_stages",
            "coefficient_dot_elements",
            "gram_dot_elements",
            "combination_elements",
        ):
            integer(receipt[key], key)
        for key in ("system", "vector_elements", "live_history", "status"):
            integer(receipt[key], key)
        integer(receipt["new_slot"], "ring slot", low=-1, high=7)
        if (
            receipt["system"] != 0
            or receipt["invocation"] != index
            or receipt["iteration_before"] != index
            or receipt["vector_elements"] != dimension
        ):
            raise ValueError("receipt chronological identity or dimension mismatch")
        if receipt["status"] != 0 or receipt["restart_before"] != 0:
            return {"status": "INCOMPLETE", "reason": "partial or restarted trajectory"}
        live, slot, coefficient, gram, combination = work_oracle(index, dimension)
        if (
            receipt["live_history"],
            receipt["coefficient_dot_elements"],
            receipt["gram_dot_elements"],
            receipt["combination_elements"],
        ) != (live, coefficient, gram, combination):
            raise ValueError(
                "actual executed visits disagree with independent work oracle"
            )
        if receipt["new_slot"] != slot:
            raise ValueError("ring slot mismatch")
        if receipt["completed_stages"] != (31 if live else 17):
            return {
                "status": "INCOMPLETE",
                "reason": "missing or invalid completion bits",
            }
        cycles = [
            integer(receipt[f"{stage}_elapsed_cycles"], "stage cycles")
            for stage in STAGES
        ]
        if not live and any(cycles[1:]):
            raise ValueError("unexecuted history stages reported clock intervals")
        total = sum(cycles)
        if total == 0:
            return {
                "status": "INCOMPLETE",
                "reason": "stage clock intervals unavailable",
            }
        # All intervals in this denominator belong to this one invocation/CTA.
        # It excludes unmeasured normalization, commit and launch overhead.
        shares.append({stage: cycles[i] / total for i, stage in enumerate(STAGES)})
    if attempted > row["iterations"]:
        raise ValueError("receipt count exceeds SCC iterations")
    return {
        "status": "PASS",
        "per_invocation_measured_stage_shares": shares,
        "mixer_scc_fraction": "UNAVAILABLE",
        "submitted_graph_count": "UNAVAILABLE",
    }


def numerical(row: dict, expected: dict) -> dict:
    energy, forces = row["energy"], row["forces"]
    if len(forces) != len(expected["forces"]):
        raise ValueError("incomplete host forces")
    valid = (
        row["status"] == row["system_status"] == 0
        and row["converged"] == 1
        and finite(energy)
        and all(finite(v) for v in forces)
    )
    if not valid:
        return {
            "status": "FAIL",
            "reason": "endpoint did not publish finite converged E+F",
        }
    de = abs(energy - expected["energy"])
    df = max(abs(a - b) for a, b in zip(forces, expected["forces"], strict=True))
    return {
        "status": "PASS" if max(de, df) < GATE else "FAIL",
        "energy_error": de,
        "force_error": df,
    }


def report(journal: Path, oracle: dict, repetitions: int) -> dict:
    if not 3 <= repetitions <= 20 or journal.stat().st_size > LIMIT:
        raise ValueError("invalid repetitions or journal capacity")
    rows = [
        json.loads(line, parse_constant=lambda _: None)
        for line in journal.read_text().splitlines()
    ]
    cases = oracle["cases"]
    if (
        not cases
        or len(cases) % 2
        or len(cases) > 32
        or oracle["oracle"] != "tblite 0.6.0"
    ):
        raise ValueError("invalid paired independent oracle")
    expected_keys = {
        (case, rep, arm)
        for case in range(len(cases))
        for rep in range(repetitions + 1)
        for arm in (False, True)
    }
    seen = set()
    findings = []
    plans = {}
    for row in rows:
        validate_metadata(row)
        integer(row["iterations"], "positive protocol iterations", low=-42, high=300)
        case = integer(row["case_index"], "case", high=len(cases) - 1)
        validate_input(row, cases[case])
        integer(row["maximum_iterations"], "maximum iterations", low=300, high=300)
        rep = integer(row["repetition"], "repetition", high=repetitions)
        key = case, rep, row["enabled"]
        if key in seen:
            raise ValueError("duplicate endpoint row")
        seen.add(key)
        phase = (
            ("moved_replay" if rep else "moved")
            if case % 2
            else ("warm" if rep else "cold")
        )
        if row["phase"] != phase or row["order"] not in (0, 1):
            raise ValueError("phase or order mismatch")
        expected_order = (int(row["enabled"]) - rep - case % 2 - case // 2) % 2
        if row["order"] != expected_order:
            raise ValueError("arms did not alternate")
        for field in ("endpoint_seconds", "readback_seconds", "construction_seconds"):
            if not finite(row[field]) or row[field] < 0:
                raise ValueError("invalid endpoint timing")
        science = numerical(row, cases[case])
        item = {
            "case": cases[case]["name"],
            "phase": phase,
            "enabled": row["enabled"],
            "repetition": rep,
            "numerical": science,
        }
        if row["enabled"]:
            cache = case // 2
            if row["snapshot_available"]:
                call = integer(row["call_id"], "call", 1)
                if call != (case % 2) * (repetitions + 1) + rep + 1:
                    raise ValueError("call identity did not reset or advance")
                if cache in plans and plans[cache] != row["plan_token"]:
                    raise ValueError("same-topology plan identity changed")
                plans[cache] = row["plan_token"]
            item["work"] = validate_receipts(row, cases[case])
        elif row["snapshot_available"] or row["receipts"] or row["attempted_receipts"]:
            raise ValueError("disabled arm published mixer receipts")
        findings.append(item)
    for index, row in enumerate(rows):
        # Physical journal order is part of the alternating experiment, not a
        # user-supplied label which could hide an all-OFF/all-ON sequence.
        case = index // (2 * (repetitions + 1))
        rep = (index // 2) % (repetitions + 1)
        if (
            row["case_index"] != case
            or row["repetition"] != rep
            or row["order"] != index % 2
        ):
            raise ValueError("journal execution order mismatch")
    complete = seen == expected_keys
    pairs = {}
    for row in rows:
        pairs.setdefault((row["case_index"], row["repetition"]), {})[row["enabled"]] = (
            row
        )
    timing = []
    parity = True
    for (case, rep), arms in pairs.items():
        if set(arms) != {False, True}:
            parity = False
            continue
        off, on = arms[False], arms[True]
        eq = all(
            off[k] == on[k]
            for k in ("iterations", "status", "system_status", "converged", "flags")
        )
        if (
            finite(off["energy"])
            and finite(on["energy"])
            and all(finite(v) for v in off["forces"] + on["forces"])
        ):
            eq &= (
                abs(off["energy"] - on["energy"]) <= PARITY
                and max(
                    abs(a - b) for a, b in zip(off["forces"], on["forces"], strict=True)
                )
                <= PARITY
            )
        else:
            eq = False
        parity &= eq
        left = (
            off["construction_seconds"]
            + off["endpoint_seconds"]
            + off["readback_seconds"]
        )
        right = (
            on["construction_seconds"] + on["endpoint_seconds"] + on["readback_seconds"]
        )
        timing.append(
            {
                "case": cases[case]["name"],
                "phase": off["phase"],
                "repetition": rep,
                "off_seconds": left,
                "on_seconds": right,
                "parity": eq,
                "overhead_fraction": right / left - 1 if left > 0 else None,
            }
        )
    qualified = (
        complete
        and parity
        and all(
            x["numerical"]["status"] == "PASS"
            and x.get("work", {"status": "PASS"})["status"] == "PASS"
            for x in findings
        )
    )
    timing_summary = []
    for case in cases:
        for phase in ("cold", "warm", "moved", "moved_replay"):
            samples = [
                x for x in timing if x["case"] == case["name"] and x["phase"] == phase
            ]
            if not samples:
                continue
            off = [x["off_seconds"] for x in samples]
            on = [x["on_seconds"] for x in samples]
            overhead = [
                x["overhead_fraction"]
                for x in samples
                if x["overhead_fraction"] is not None
            ]
            timing_summary.append(
                {
                    "case": case["name"],
                    "phase": phase,
                    "samples": len(samples),
                    "off_median_seconds": statistics.median(off),
                    "on_median_seconds": statistics.median(on),
                    "off_range_seconds": [min(off), max(off)],
                    "on_range_seconds": [min(on), max(on)],
                    "paired_overhead_median": statistics.median(overhead)
                    if overhead
                    else None,
                    "paired_overhead_range": [min(overhead), max(overhead)]
                    if overhead
                    else None,
                }
            )
    return {
        "status": "PASS" if qualified else "INCOMPLETE",
        "coverage_complete": complete,
        "instrumentation_parity": parity,
        "journal_sha256": digest(journal),
        "mixer_scc_fraction": "UNAVAILABLE",
        "rows": findings,
        "endpoint_pairs": timing,
        "timing_summary": timing_summary,
        "retained_768_ao_gate": "historical independent FAIL; unresolved",
        "cycle_contract": "shares of measured stages within one CTA invocation only; no cycles-to-ms or cross-SM sum",
    }


def report_lifecycle(journal: Path, oracle: dict) -> dict:
    if journal.stat().st_size > LIMIT:
        raise ValueError("lifecycle journal capacity exceeded")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    phases = (
        "first_failed",
        "recovered",
        "replacement_failed",
        "replay",
        "rejected",
        "after_rejection",
    )
    if [row["phase"] for row in rows] != list(phases):
        return {
            "status": "INCOMPLETE",
            "reason": "missing lifecycle calls",
            "journal_sha256": digest(journal),
        }
    findings = []
    for index, row in enumerate(rows):
        validate_metadata(row)
        case = integer(
            row["case_index"], "lifecycle case", high=len(oracle["cases"]) - 1
        )
        expected = oracle["cases"][case]
        validate_input(row, expected, index in (0, 2))
        integer(row["maximum_iterations"], "lifecycle maximum", low=0, high=301)
        if row["maximum_iterations"] != (
            301 if index == 2 else 0 if index == 4 else 300
        ):
            raise ValueError("lifecycle settings mismatch")
        if row["enabled"] is not True:
            raise ValueError("lifecycle instrumentation disabled")
        if index in (0, 2, 4):
            if row["status"] != 0:
                preserved = (
                    row["energy"] == 123.0
                    and row["iterations"] == -42
                    and row["system_status"] == -43
                    and row["converged"] == 165
                    and row["flags"] == 0
                    and len(row["forces"]) == len(expected["forces"])
                    and all(value == -321.0 for value in row["forces"])
                )
            else:
                preserved = (
                    row["system_status"] != 0
                    and row["converged"] == 0
                    and row["energy"] is None
                    and len(row["forces"]) == len(expected["forces"])
                    and all(value is None for value in row["forces"])
                )
            if not preserved:
                raise ValueError(
                    "failed call published incorrect science or changed sentinels"
                )
            findings.append({"phase": row["phase"], "failure_observed": True})
        else:
            findings.append(
                {
                    "phase": row["phase"],
                    "numerical": numerical(row, expected),
                    "work": validate_receipts(row, expected),
                }
            )
        if index == 4:
            if (
                row["snapshot_available"]
                or row["receipts"]
                or row["attempted_receipts"]
            ):
                raise ValueError("pre-reset rejection exposed stale diagnostics")
        elif not row["snapshot_available"]:
            return {
                "status": "INCOMPLETE",
                "reason": "call-owned lifecycle snapshot unavailable",
                "rows": findings,
            }
        elif row["call_id"] != index + 1:
            raise ValueError("lifecycle call identity mismatch")
    if rows[2]["plan_token"] == rows[1]["plan_token"]:
        raise ValueError("failed replacement read prior plan")
    if rows[2]["status"] != 0 and rows[3]["plan_token"] != rows[1]["plan_token"]:
        raise ValueError("failed replacement published candidate topology")
    if rows[5]["plan_token"] != rows[3]["plan_token"]:
        raise ValueError("pre-reset rejection changed prepared topology")
    passed = all(
        x.get("numerical", {"status": "PASS"})["status"] == "PASS"
        and x.get("work", {"status": "PASS"})["status"] == "PASS"
        for x in findings
    )
    return {
        "status": "PASS" if passed else "INCOMPLETE",
        "rows": findings,
        "journal_sha256": digest(journal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--waters", nargs="+", type=int, default=[8, 32])
    check = sub.add_parser("report")
    check.add_argument("journal", type=Path)
    check.add_argument("oracle", type=Path)
    check.add_argument("output", type=Path)
    check.add_argument("--repetitions", type=int, default=5)
    check.add_argument("--failures", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.waters)
    else:
        result = report(args.journal, bounded_json(args.oracle), args.repetitions)
        if args.failures:
            if args.journal.stat().st_size + args.failures.stat().st_size > LIMIT:
                raise ValueError("combined journal capacity exceeded")
            result["lifecycle"] = report_lifecycle(
                args.failures, bounded_json(args.oracle)
            )
            if result["lifecycle"]["status"] != "PASS":
                result["status"] = "INCOMPLETE"
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
