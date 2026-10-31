"""Recompute the retained point, endpoint, numerical and package-size gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    with lzma.open(Path(__file__).with_name("receipts.json.xz"), "rt") as stream:
        data = json.load(stream)
    policy = data["promotion_policy"]
    points = defaultdict(lambda: defaultdict(list))
    for row in data["points"]["records"]:
        assert len(row["milliseconds"]) == policy["samples_per_arm_per_round"]
        key = row["name"], row["spins"], row["points"], row["tile"]
        points[key][row["arm"]].append(statistics.median(row["milliseconds"]))
    assert len(points) == policy["point_cases"]
    ratios = {}
    paths = defaultdict(list)
    for key, arms in points.items():
        assert set(arms) == {"literal", "prepared"}
        assert all(len(samples) == policy["paired_rounds"] for samples in arms.values())
        ratio = statistics.median(arms["prepared"]) / statistics.median(arms["literal"])
        ratios[key] = ratio
        paths[key[1], key[3]].append(ratio)
    assert (
        statistics.median(ratios.values()) <= policy["point_aggregate_median_maximum"]
    )
    assert max(ratios.values()) <= policy["point_single_case_maximum"]
    assert all(
        statistics.median(values) <= policy["point_spin_layout_median_maximum"]
        for values in paths.values()
    )

    endpoints = defaultdict(lambda: defaultdict(list))
    maximum_energy_error = 0.0
    calls = 0
    for filename, before in data["isolated_endpoints"].items():
        if "-baseline-" not in filename:
            continue
        after = data["isolated_endpoints"][filename.replace("-baseline-", "-prepared-")]
        for field in (
            "case",
            "name",
            "spins",
            "copies",
            "atoms",
            "moved",
            "grid",
            "tile_points",
            "energy_tolerance",
            "density_tolerance",
            "max_iterations",
        ):
            assert before[field] == after[field]
        for left, right in zip(before["samples"], after["samples"], strict=True):
            for field in ("phase", "iterations", "fock_builds", "precision"):
                assert left[field] == right[field], (filename, field)
            maximum_energy_error = max(
                maximum_energy_error, abs(left["energy"] - right["energy"])
            )
            calls += 2
        key = before["case"], before["name"], before["spins"], before["copies"]
        for arm, record in (("literal", before), ("prepared", after)):
            assert (
                record["library_sha256"] == data["binaries"][arm + "-library"]["sha256"]
            )
            for phase in ("cold", "warm", "moved"):
                endpoints[key][arm, phase].append(
                    statistics.median(
                        row["seconds"]
                        for row in record["samples"]
                        if row["phase"].startswith(phase)
                    )
                )
    assert len(endpoints) == policy["endpoint_cohorts"]
    assert maximum_energy_error <= policy["endpoint_energy_absolute_maximum"]
    timing = {}
    for key, rows in endpoints.items():
        timing[str(key)] = {}
        for phase in ("cold", "warm", "moved"):
            for arm in ("literal", "prepared"):
                assert (
                    len(rows[arm, phase])
                    == policy["endpoint_isolated_pairs_per_cohort"]
                )
            ratio = statistics.median(rows["prepared", phase]) / statistics.median(
                rows["literal", phase]
            )
            assert ratio <= policy["endpoint_phase_median_ratio_maximum"]
            timing[str(key)][phase] = ratio
    assert len(data["oracle_receipts"]) == 52
    for receipt in data["oracle_receipts"].values():
        for row in receipt["rows"]:
            assert row["energy_error"] < 3e-9
            assert row["fixed_energy_error"] < 3e-10
            assert row["density_max_abs_error"] < 2e-7
            assert row["fock_max_abs_error"] < 3e-9
            assert row["directional_error"] < 5e-8
    assert data["default_equivalence"]["default"] == "ON"
    assert not data["default_equivalence"]["changed_generated_files"]
    for name, source in data["qualified_cuda_generated_sources"].items():
        expected = data["default_equivalence"]["source_hashes"][
            "libxc_semilocal_cuda/" + name
        ]
        assert hashlib.sha256(source.encode()).hexdigest() == expected
    assert len(set(data["worker_equivalence"].values())) == 1
    compilation = data["compile_trial"]["records"]
    sizes = {
        arm: sum(row["bytes"] for row in value["objects"])
        for arm, value in compilation.items()
    }
    assert sizes["candidate"] < sizes["baseline"]
    print(
        json.dumps(
            {
                "point_cases": len(points),
                "point_median_ratio": statistics.median(ratios.values()),
                "point_maximum_ratio": max(ratios.values()),
                "endpoint_calls": calls,
                "maximum_energy_error": maximum_energy_error,
                "endpoint_phase_ratios": timing,
                "object_bytes": sizes,
                "object_byte_reduction": 1 - sizes["candidate"] / sizes["baseline"],
                "compile_seconds": {
                    arm: row["compile_seconds"] for arm, row in compilation.items()
                },
                "generation_seconds": {
                    arm: row["generation_seconds"] for arm, row in compilation.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
