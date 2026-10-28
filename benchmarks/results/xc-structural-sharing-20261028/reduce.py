"""Reduce all retained structural-sharing samples, including failed timing gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path


def restore_events(data: dict) -> None:
    """Restore exact archived event samples and verify the pre-compaction values."""
    encoding = data.pop("event_encoding")
    assert encoding["schema"] == "sorted-duration-offsets-v1"
    values = encoding["values_milliseconds"]
    for path, expected in encoding["population_sha256"].items():
        population = data
        for component in path.split("/"):
            population = population[component]
        for row in population["records"]:
            base = row.pop("milliseconds_base")
            row["milliseconds"] = [
                values[base + offset] for offset in row.pop("milliseconds_offsets")
            ]
        canonical = json.dumps(
            population, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == expected, path


def point_summary(payload: dict) -> dict:
    """Compare paired populations without dropping slow cases or pilot samples."""
    cases = defaultdict(lambda: defaultdict(list))
    for row in payload["records"]:
        key = row["name"], row["spins"], row["points"], row["tile"]
        cases[key][row["arm"]].append(statistics.median(row["milliseconds"]))
    ratios = {}
    for case, arms in cases.items():
        assert set(arms) == {"literal", "smart"}
        assert len(arms["literal"]) == len(arms["smart"])
        ratios[str(case)] = statistics.median(arms["smart"]) / statistics.median(
            arms["literal"]
        )
    return {
        "cases": len(cases),
        "paired_measurements": len(payload["records"]) // 2,
        "median_ratio": statistics.median(ratios.values()),
        "maximum_ratio": max(ratios.values()),
        "cases_above_one_percent": {
            case: ratio for case, ratio in ratios.items() if ratio > 1.01
        },
        "ratios": ratios,
    }


def main() -> None:
    # Both stages share provenance storage; retain their separate populations.
    archive = (
        Path(__file__).parent.parent / "xc-shared-point-abi-20261028/receipts.json.xz"
    )
    with lzma.open(archive, "rt") as stream:
        data = json.load(stream)["structural_discovery"]
    restore_events(data)
    for arm in ("literal", "smart"):
        hashes = set(data["generated"][arm].values())
        for row in data["builds"][arm]["objects"]:
            assert row["source_sha256"] in hashes
    pilot = point_summary(data["point_pilot"])
    final = point_summary(data["point_final"])
    endpoints = defaultdict(lambda: defaultdict(list))
    maximum_energy_error = 0.0
    calls = 0
    work_mismatches = []
    unpaired = []
    for filename, before in data["endpoints"].items():
        if not filename.startswith("literal-"):
            continue
        paired_name = filename.replace("literal-", "smart-", 1)
        if paired_name not in data["endpoints"]:
            unpaired.append(filename)
            continue
        after = data["endpoints"][paired_name]
        for field in (
            "name",
            "spins",
            "case",
            "copies",
            "atoms",
            "moved",
            "grid",
            "tile_points",
            "energy_tolerance",
            "density_tolerance",
            "max_iterations",
        ):
            assert before[field] == after[field], (filename, field)
        for left, right in zip(before["samples"], after["samples"], strict=True):
            for field in ("phase", "iterations", "fock_builds", "precision"):
                if left[field] != right[field]:
                    work_mismatches.append(
                        {
                            "file": filename,
                            "phase": left["phase"],
                            "field": field,
                            "literal": left[field],
                            "smart": right[field],
                        }
                    )
            maximum_energy_error = max(
                maximum_energy_error, abs(left["energy"] - right["energy"])
            )
            calls += 2
        key = before["name"], before["spins"], before["case"], before["copies"]
        for arm, record in (("literal", before), ("smart", after)):
            assert (
                record["library_sha256"]
                == data["binaries"][arm + "/libgenerativeqc.so"]["sha256"]
            )
            for phase in ("cold", "warm", "moved"):
                endpoints[key][arm, phase].append(
                    statistics.median(
                        row["seconds"]
                        for row in record["samples"]
                        if row["phase"].startswith(phase)
                    )
                )
    timing = {
        str(key): {
            phase: statistics.median(rows["smart", phase])
            / statistics.median(rows["literal", phase])
            for phase in ("cold", "warm", "moved")
        }
        for key, rows in endpoints.items()
    }
    failures = [row for row in data["endpoint_processes"] if row["returncode"] != 0]
    sizes = {
        arm: sum(row["bytes"] for row in data["builds"][arm]["objects"])
        for arm in ("literal", "smart")
    }
    print(
        json.dumps(
            {
                "decision": "retain qualified default; structural discovery remains opt-in",
                "catalog_registrations": data["catalog_plan"]["registrations"],
                "candidate_groups": len(data["catalog_plan"]["candidate_groups"]),
                "candidate_registrations": sum(
                    len(group["names"])
                    for group in data["catalog_plan"]["candidate_groups"]
                ),
                "point_pilot": pilot,
                "point_confirmation": final,
                "object_bytes": sizes,
                "object_reduction": 1 - sizes["smart"] / sizes["literal"],
                "endpoint_cohorts": len(endpoints),
                "complete_calls": calls,
                "maximum_energy_error": maximum_energy_error,
                "energy_gate_passed": maximum_energy_error <= 3e-9,
                "work_mismatches": work_mismatches,
                "failed_endpoint_processes": failures,
                "unpaired_successes": unpaired,
                "endpoint_phase_ratios": timing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
