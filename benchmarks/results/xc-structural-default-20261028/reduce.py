"""Replay default-sharing gates without CUDA, generated products or Git history."""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path

from defusedxml import ElementTree as ET


def load_data() -> dict:
    """Restore every exact event value and authenticate the original populations."""
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)
    encoding = data["event_encoding"]
    assert encoding["schema"] == "sorted-duration-offsets-v1"
    values = encoding["values_milliseconds"]
    for name, expected in encoding["population_sha256"].items():
        for row in data[name]["records"]:
            base = row.pop("milliseconds_base")
            row["milliseconds"] = [
                values[base + offset] for offset in row.pop("milliseconds_offsets")
            ]
        canonical = json.dumps(
            data[name], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == expected, name
    return data


def point_ratios(payload: dict, *, repeats: int | None = 3) -> dict:
    """Keep every process; compare medians of its 50 CUDA-event observations."""
    cases = defaultdict(lambda: defaultdict(list))
    for row in payload["records"]:
        assert len(row["milliseconds"]) == 50
        key = row["name"], row["spins"], row["points"], row["tile"]
        cases[key][row["arm"]].append(statistics.median(row["milliseconds"]))
    ratios = {}
    for key, arms in cases.items():
        assert len({len(samples) for samples in arms.values()}) == 1, key
        if repeats is not None:
            assert all(len(samples) == repeats for samples in arms.values()), key
        baseline = statistics.median(arms["literal"])
        ratios[str(key)] = {
            arm: statistics.median(samples) / baseline for arm, samples in arms.items()
        }
    return ratios


def main() -> None:
    """Authenticate compiled inputs, numerical gates and complete endpoint work."""
    data = load_data()
    sizes = {}
    for arm, package in data["packages"].items():
        for row in package["build"]["objects"]:
            assert row["source_sha256"] in package["generated"].values()
        sizes[arm] = sum(row["bytes"] for row in package["build"]["objects"])
    assert data["cpu_generated"] == data["native_cpu_generated"]
    assert data["packages"]["smart"]["generated"] == data["native_cuda_generated"]
    assert data["packages"]["smart"]["point_kernel_count"] == 4 * len(data["groups"])
    assert data["packages"]["literal"]["point_kernel_count"] == 4 * len(data["names"])
    expected = set(itertools.product(data["names"], (1, 2), (16384, 262144), (0, 127)))
    seen = defaultdict(set)
    for row in data["points"]["records"]:
        key = row["name"], row["spins"], row["points"], row["tile"]
        pair = row["arm"], row["repeat"]
        assert pair not in seen[key]
        seen[key].add(pair)
    assert set(seen) == expected
    assert all(
        v == set(itertools.product(("literal", "smart"), range(3)))
        for v in seen.values()
    )
    initial_points = point_ratios(data["points"])
    confirmation = data["confirmation"]
    confirmed = {tuple(case) for case in confirmation["cases"]}
    assert {tuple(case) for case in confirmation["initial_failures"]} == {
        key for key in expected if initial_points[str(key)]["smart"] > 1.01
    }
    confirm_seen = defaultdict(set)
    for row in confirmation["records"]:
        key = row["name"], row["spins"], row["points"], row["tile"]
        pair = row["arm"], row["repeat"]
        assert pair not in confirm_seen[key]
        confirm_seen[key].add(pair)
    assert set(confirm_seen) == confirmed
    assert all(
        v == set(itertools.product(("literal", "smart"), range(3, 8)))
        for v in confirm_seen.values()
    )
    points = point_ratios(
        {"records": data["points"]["records"] + confirmation["records"]}, repeats=None
    )
    # Direct event spans retain host submission gaps; isolate every ordinary
    # entry with the same one-node Graph protocol, not just the failed cases.
    steady = data["steady_points"]
    steady_seen = defaultdict(set)
    for row in steady["records"]:
        assert row["launch_mode"] == "graph" and row["warm_seconds"] >= 0.1
        key = row["name"], row["spins"], row["points"], row["tile"]
        pair = row["arm"], row["repeat"]
        assert pair not in steady_seen[key]
        steady_seen[key].add(pair)
    assert set(steady_seen) == {key for key in expected if key[-1] == 0}
    assert all(
        v == set(itertools.product(("literal", "smart"), range(3)))
        for v in steady_seen.values()
    )
    initial_steady_ratios = point_ratios(steady)
    steady_confirmation = data["steady_confirmation"]
    steady_failures = {
        key for key in steady_seen if initial_steady_ratios[str(key)]["smart"] > 1.01
    }
    assert {
        tuple(key) for key in steady_confirmation["initial_failures"]
    } == steady_failures
    confirmation_names = {"GGA_X_APBE", "MGGA_X_KTBM_0", "LDA_C_OW"}
    for group in data["groups"]:
        if set(group["names"]) & {key[0] for key in steady_failures}:
            confirmation_names.update(group["names"])
    required_confirmation = set(
        itertools.product(confirmation_names, (1, 2), (16384, 262144), (0,))
    )
    final_seen = defaultdict(set)
    for row in steady_confirmation["records"]:
        assert row["launch_mode"] == "graph" and row["warm_seconds"] >= 0.1
        key = row["name"], row["spins"], row["points"], row["tile"]
        pair = row["arm"], row["repeat"]
        assert pair not in final_seen[key]
        final_seen[key].add(pair)
    assert set(final_seen) == required_confirmation
    assert all(
        v == set(itertools.product(("literal", "smart"), range(3, 8)))
        for v in final_seen.values()
    )
    steady_ratios = point_ratios(
        {"records": steady["records"] + steady_confirmation["records"]}, repeats=None
    )
    qualified = {
        str(key): steady_ratios[str(key)] if key[-1] == 0 else initial_points[str(key)]
        for key in expected
    }
    assert all(row["smart"] <= 1.01 for row in qualified.values())
    pilot = point_ratios(data["pilot"])
    assert all(row["ordinary-grid"] <= 1.01 for row in pilot.values())

    failures = [row for row in data["endpoint_processes"] if row["returncode"]]
    assert not failures, failures
    cohorts = defaultdict(lambda: defaultdict(list))
    max_energy_error = 0.0
    calls = 0
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
                assert left[field] == right[field], (filename, left["phase"], field)
            max_energy_error = max(
                max_energy_error, abs(left["energy"] - right["energy"])
            )
            calls += 2
        key = before["name"], before["spins"], before["case"], before["copies"]
        for arm, record in (("literal", before), ("smart", after)):
            assert (
                record["library_sha256"]
                == data["packages"][arm]["binaries"]["libgenerativeqc.so"]["sha256"]
            )
            for phase in ("cold", "warm", "moved"):
                cohorts[key][arm, phase].append(
                    statistics.median(
                        row["seconds"]
                        for row in record["samples"]
                        if row["phase"].startswith(phase)
                    )
                )
    representative_names = {group["names"][0] for group in data["groups"]}
    representative_names.update(
        {
            "GGA_X_APBE",
            "GGA_X_PBE_SOL",
            "GGA_X_XPBE",
            "GGA_XC_TH_FCO",
            "GGA_XC_TH_FCFO",
            "GGA_X_BKL1",
            "GGA_X_BKL2",
            "MGGA_X_KTBM_0",
            "MGGA_X_KTBM_GAP",
        }
    )
    required_cohorts = {
        (name, spin, "h4", 1)
        for name in representative_names
        for spin in (1, 2)
        if (name, spin) != ("MGGA_X_MVSB", 2)
    }
    required_cohorts.add(("GGA_X_APBE", 1, "water", 4))
    assert set(cohorts) == required_cohorts
    assert max_energy_error <= 3e-9
    phases = {}
    for key, rows in cohorts.items():
        assert all(len(samples) >= 5 for samples in rows.values()), key
        phases[str(key)] = {
            phase: statistics.median(rows["smart", phase])
            / statistics.median(rows["literal", phase])
            for phase in ("cold", "warm", "moved")
        }
    assert all(value <= 1.02 for row in phases.values() for value in row.values())
    suites = [
        dict(s.attrib)
        for s in ET.fromstring(data["logs"]["oracle.xml"]).iter("testsuite")
    ]
    assert all(int(s["errors"]) == 0 for s in suites)
    failed_tests = [
        case.attrib["name"]
        for case in ET.fromstring(data["logs"]["oracle.xml"]).iter("testcase")
        if case.find("failure") is not None
    ]
    assert failed_tests == ["test_larger_water[2]"]
    # A failed solve is never a passing endpoint. Both literal and shared fail
    # the unchanged 400-iteration larger UKS fixture, so retain it separately.
    assert all(not row["returncode"] for row in data["convergence_processes"])
    for arm, record in data["convergence_probes"].items():
        assert not record["converged"] and not record["succeeded"]
        assert record["status_message"] == "SCF did not converge"
        assert record["iterations"] == 400
        assert (
            record["library_sha256"]
            == data["packages"][arm]["binaries"]["libgenerativeqc.so"]["sha256"]
        )
    for row in data["fock_oracles"].values():
        assert (
            row["binary_sha256"]
            == data["packages"]["smart"]["binaries"]["xc-driver"]["sha256"]
        )
        assert row["energy_error"] < 2e-10
        assert row["directional_error"] < 2e-8
    for record in data["ks_oracles"].values():
        assert (
            record["native_library_sha256"]
            == data["packages"]["smart"]["binaries"]["libgenerativeqc.so"]["sha256"]
        )
        for row in record["rows"]:
            assert row["energy_error"] < 3e-9 and row["fixed_energy_error"] < 3e-10
            assert row["fock_max_abs_error"] < 3e-9
            assert row["density_max_abs_error"] < 2e-7
            assert row["directional_error"] < 5e-8
    for record in data["water_oracles"].values():
        for row in record["rows"]:
            errors = row["errors"]
            assert errors["energy"] < 3e-9 and errors["fixed_energy"] < 3e-10
            assert errors["fock"] < 3e-9 and errors["density"] < 2e-7
            assert errors["directional"] < 5e-8
    print(
        json.dumps(
            {
                "decision": "default structural sharing within enabled automatic XC AOT",
                "object_bytes": sizes,
                "object_reduction": 1 - sizes["smart"] / sizes["literal"],
                "point_kernel_counts": {
                    arm: p["point_kernel_count"] for arm, p in data["packages"].items()
                },
                "point_cases": len(points),
                "point_measurements": len(data["points"]["records"]),
                "confirmation_measurements": len(confirmation["records"]),
                "initial_point_failures": {
                    k: v for k, v in initial_points.items() if v["smart"] > 1.01
                },
                "confirmation_ratios": point_ratios(confirmation, repeats=5),
                "qualified_point_maximum_ratio": max(
                    row["smart"] for row in qualified.values()
                ),
                "steady_point_ratios": steady_ratios,
                "initial_steady_failures": {
                    k: v for k, v in initial_steady_ratios.items() if v["smart"] > 1.01
                },
                "steady_confirmation_measurements": len(steady_confirmation["records"]),
                "steady_confirmation_ratios": point_ratios(
                    steady_confirmation, repeats=5
                ),
                "direct_pooled_failures": {
                    k: v for k, v in points.items() if v["smart"] > 1.01
                },
                "graph_pilot": point_ratios(data["graph_pilot"]),
                "block_pilot": point_ratios(data["block_pilot"]),
                "point_ratios": points,
                "pilot_ratios": pilot,
                "endpoint_cohorts": len(cohorts),
                "complete_calls": calls,
                "maximum_energy_error": max_energy_error,
                "endpoint_phase_ratios": phases,
                "endpoint_phase_maximum": max(
                    v for r in phases.values() for v in r.values()
                ),
                "numerical_suites": suites,
                "retained_convergence_failures": {
                    arm: {
                        key: row[key]
                        for key in (
                            "converged",
                            "status_message",
                            "iterations",
                            "fock_builds",
                            "energy",
                        )
                    }
                    for arm, row in data["convergence_probes"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
