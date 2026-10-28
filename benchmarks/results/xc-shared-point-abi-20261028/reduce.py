"""Authenticate and reduce every retained XC ABI experiment, including failures."""

from __future__ import annotations

import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path

from defusedxml import ElementTree as ET

TARGETS = {"GGA_XC_TH_FCO", "GGA_XC_TH_FCFO", "GGA_X_BKL1", "GGA_X_BKL2"}


def points(payload: dict) -> dict:
    """Reduce process medians per case, keeping every arm and repeat."""
    cases = defaultdict(lambda: defaultdict(list))
    for row in payload["records"]:
        key = row["name"], row["spins"], row["points"], row["tile"]
        cases[key][row["arm"]].append(statistics.median(row["milliseconds"]))
    ratios = {}
    targets = {}
    for key, arms in cases.items():
        assert len({len(samples) for samples in arms.values()}) == 1
        baseline = statistics.median(arms["literal"])
        ratios[str(key)] = {
            arm: statistics.median(samples) / baseline for arm, samples in arms.items()
        }
        if key[0] in TARGETS and key[1:] == (1, 16384, 127):
            targets[key[0]] = ratios[str(key)]
    return {
        "cases": len(cases),
        "measurements": len(payload["records"]),
        "target_ratios": targets,
        "median_by_arm": {
            arm: statistics.median(row[arm] for row in ratios.values())
            for arm in next(iter(ratios.values()))
        },
        "above_one_percent": {
            key: row
            for key, row in ratios.items()
            if any(value > 1.01 for value in row.values())
        },
        "ratios": ratios,
    }


def main() -> None:
    """Verify source/binary bindings, scientific gates and semantic work ledgers."""
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)["paired_batch"]
    # Generated code and disassembly are transient products. Source reproduction
    # is an opt-in audit; the normal reducer needs no compiler or Git history.
    sizes = {}
    kernels = {}
    for name, package in data["packages"].items():
        assert all(
            row["source_sha256"] in package["generated"].values()
            for row in package["build"]["objects"]
        )
        sizes[name] = sum(row["bytes"] for row in package["build"]["objects"])
        if package["resources"]:
            kernels[name] = sum(
                "Function " in line and "automatic_evaluate_points" in line
                for line in package["resources"].splitlines()
            )
    for row in data["cpu_build"]["objects"]:
        assert row["source_sha256"] in data["cpu_generated"].values()
    final_points = points(data["point_final"])
    assert set(final_points["target_ratios"]) == TARGETS
    assert all(row["smart"] <= 1.01 for row in final_points["target_ratios"].values())
    assert all(
        row["smart"] / row["before"] <= 1.01 for row in final_points["ratios"].values()
    )

    endpoints = defaultdict(lambda: defaultdict(list))
    initial_endpoints = defaultdict(lambda: defaultdict(list))
    confirmation_endpoints = defaultdict(lambda: defaultdict(list))
    energy_error = 0.0
    calls = 0
    work_mismatches = []
    for filename, before in data["endpoints"].items():
        if not filename.startswith("literal-"):
            continue
        after = data["endpoints"][filename.replace("literal-", "smart-", 1)]
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
                    work_mismatches.append((filename, left["phase"], field))
            energy_error = max(energy_error, abs(left["energy"] - right["energy"]))
            calls += 2
        key = before["name"], before["spins"], before["case"], before["copies"]
        for arm, record in (("literal", before), ("smart", after)):
            binary = data["packages"]["accepted/" + arm]["binaries"][
                "libgenerativeqc.so"
            ]
            assert record["library_sha256"] == binary["sha256"]
            for phase in ("cold", "warm", "moved"):
                median = statistics.median(
                    row["seconds"]
                    for row in record["samples"]
                    if row["phase"].startswith(phase)
                )
                endpoints[key][arm, phase].append(median)
                repeat = int(Path(filename).stem.rsplit("-", 1)[1])
                population = initial_endpoints if repeat < 3 else confirmation_endpoints
                population[key][arm, phase].append(median)

    def phase_summary(population: dict) -> dict:
        return {
            str(key): {
                phase: statistics.median(rows["smart", phase])
                / statistics.median(rows["literal", phase])
                for phase in ("cold", "warm", "moved")
            }
            for key, rows in population.items()
        }

    phase_ratios = phase_summary(endpoints)
    assert all(ratio <= 1.02 for row in phase_ratios.values() for ratio in row.values())
    assert energy_error <= 3e-9 and not work_mismatches
    failures = [row for row in data["endpoint_processes"] if row["returncode"]]
    assert not failures
    suites = {}
    for name in ("oracle.xml", "ks-oracle.xml", "target-127.xml"):
        root = ET.fromstring(data["logs"]["accepted/" + name])
        suites[name] = [dict(suite.attrib) for suite in root.iter("testsuite")]
        assert all(
            int(row["failures"]) == int(row["errors"]) == 0 for row in suites[name]
        )
    print(
        json.dumps(
            {
                "decision": "repair paired-batch regressions; discovery remains opt-in",
                "point_final": final_points,
                "pilots": {name: points(row) for name, row in data["pilots"].items()},
                "rejected_broad_candidate": points(data["rejected_point"]),
                "object_bytes": sizes,
                "point_kernel_counts": kernels,
                "selected_object_reduction": 1
                - sizes["accepted/smart"] / sizes["accepted/literal"],
                "endpoint_cohorts": len(endpoints),
                "complete_calls": calls,
                "maximum_energy_error": energy_error,
                "work_mismatches": work_mismatches,
                "endpoint_failures": failures,
                "endpoint_phase_ratios": phase_ratios,
                "initial_endpoint_phase_ratios": phase_summary(initial_endpoints),
                "confirmation_endpoint_phase_ratios": phase_summary(
                    confirmation_endpoints
                ),
                "endpoint_phase_maximum": max(
                    ratio for row in phase_ratios.values() for ratio in row.values()
                ),
                "numerical_suites": suites,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
