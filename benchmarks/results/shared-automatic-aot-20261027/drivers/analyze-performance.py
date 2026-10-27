"""Recompute paired complete-endpoint gates without dropping slow samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze(directory: Path) -> dict:
    """Require all ABBA/BAAB arms and compare public work and full vectors."""
    order = (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    )
    runs = [
        json.loads((directory / f"{i}-{arm}.json").read_text())
        for i, arm in enumerate(order)
    ]
    assert all(run["complete"] for run in runs)
    for arm in set(order):
        assert (
            len(
                {
                    runs[i]["library_sha256"]
                    for i, name in enumerate(order)
                    if name == arm
                }
            )
            == 1
        )
    assert len({len(run["rows"]) for run in runs}) == 1
    expected = 120 if directory.name.endswith("full") else 140
    assert len(runs[0]["rows"]) == expected
    assert len({run["slurm_job_id"] for run in runs}) == 1
    gates = defaultdict(lambda: {"baseline": [], "candidate": [], "paired_ratios": []})
    max_energy = max_force = 0.0
    control_flow_differences = []
    for before, after in ((0, 1), (3, 2), (5, 4), (6, 7)):
        for left, right in zip(runs[before]["rows"], runs[after]["rows"], strict=True):
            key = (left["system"], left["method"], left["phase"])
            for field in (
                "system",
                "method",
                "phase",
                "repeat",
                "iterations",
                "fock_builds",
                "coulomb_exchange_work",
                "warm_used",
                "warm_fallback",
            ):
                assert left[field] == right[field], (
                    key,
                    field,
                    left[field],
                    right[field],
                )
            transport_a, transport_b = left["transport_delta"], right["transport_delta"]
            if transport_a != transport_b:
                # The unchanged SCF gate probes canonicality when the ordinary
                # density gate passes. At iteration 27 this LTA UKS incumbent
                # probes and rejects, while the repackaged worker reaches the
                # existing occupation-stabilized proposal. Both finish at 28
                # physical Focks. One 2-spin ValidationPartial is 224 bytes.
                # Preserve this observed work difference, rather than declaring
                # exact traffic parity or attributing its cost to force staging.
                assert key == ("h4", "libxc-uks:MGGA_X_LTA", "cold"), (
                    key,
                    transport_a,
                    transport_b,
                )
                expected_delta = dict(
                    scalar_d2h_bytes=-224,
                    synchronizations=-1,
                    occupation_stabilized_proposals=1,
                )
                delta = {k: transport_b[k] - transport_a[k] for k in transport_a}
                assert all(delta[k] == expected_delta.get(k, 0) for k in delta), delta
                control_flow_differences.append(
                    dict(
                        pair=[before, after],
                        repeat=left["repeat"],
                        system=key[0],
                        method=key[1],
                        phase=key[2],
                        baseline=transport_a,
                        candidate=transport_b,
                        delta=delta,
                        reason="one fewer rejected canonical probe; one existing UKS stabilized proposal",
                    )
                )
            assert (
                left["precision"]["native_provenance"]
                == right["precision"]["native_provenance"]
            ), (key, "native_provenance")
            expected_operators = [dict(op) for op in left["precision"]["operators"]]
            if transport_a != transport_b:
                for op in expected_operators:
                    if op["name"] == "matrix_product":
                        op["count"] += 2
                expected_operators.append(
                    dict(
                        expected_operators[0], name="occupation_stabilization", count=1
                    )
                )
                control_flow_differences[-1]["operator_count_delta"] = dict(
                    matrix_product=2, occupation_stabilization=1
                )
            assert expected_operators == right["precision"]["operators"], (
                key,
                "operators",
            )
            assert not right["warm_fallback"]
            np.testing.assert_allclose(
                left["energy"], right["energy"], atol=3e-9, rtol=0
            )
            np.testing.assert_allclose(
                left["forces"], right["forces"], atol=3e-8, rtol=0
            )
            max_energy = max(max_energy, abs(left["energy"] - right["energy"]))
            max_force = max(
                max_force,
                float(np.max(np.abs(np.asarray(left["forces"]) - right["forces"]))),
            )
            a, b = left["artifact"], right["artifact"]
            if a is not None:
                for receipt, run in ((a, runs[before]), (b, runs[after])):
                    assert (
                        receipt["force_completed"]
                        and receipt["fallback_reason"] == "none"
                    )
                    for kind in ("point_artifact", "force_artifact"):
                        assert receipt[kind]["sha256"] == run["library_sha256"]
                for field in (
                    "identifier",
                    "domain_version",
                    "spin_blocks",
                    "source_count",
                    "grid_work",
                ):
                    assert a[field] == b[field], (key, field)
                aw, bw = a["stationary_work_v1"], b["stationary_work_v1"]
                # Storage, extra point launches and process-local stream address
                # differ. Every other semantic count must remain identical.
                for i in range(24):
                    if i not in (0, 3, 7):
                        assert aw[i] == bw[i], (key, "stationary_work", i, aw[i], bw[i])
                point = b["staged_point_work"]
                assert bw[3] - aw[3] == point["launches"]
                assert bw[0] - aw[0] == point["tile_bytes"]
                for field in (
                    "device_upper_bytes",
                    "minimum_device_bytes",
                    "stationary_numeric_bytes",
                ):
                    assert (
                        b["capacity"][field] - a["capacity"][field]
                        == point["tile_bytes"]
                    ), (key, field)
                if b["force_consumer_profile"].startswith("shared_"):
                    assert point["launches"] == bw[13] and point["points"] == bw[5]
                else:
                    assert all(value == 0 for value in point.values())
            assert left["seconds"] > 0 and right["seconds"] > 0
            gates[key]["baseline"].append(left["seconds"])
            gates[key]["candidate"].append(right["seconds"])
            gates[key]["paired_ratios"].append(right["seconds"] / left["seconds"])
    rows = []
    for key, samples in sorted(gates.items()):
        baseline = statistics.median(samples["baseline"])
        candidate = statistics.median(samples["candidate"])
        rows.append(
            dict(
                system=key[0],
                method=key[1],
                phase=key[2],
                baseline_median=baseline,
                candidate_median=candidate,
                ratio=candidate / baseline,
                paired_ratio_median=statistics.median(samples["paired_ratios"]),
                passed=candidate <= 1.05 * baseline,
                samples=samples,
            )
        )
    return dict(
        schema="generativeqc.shared-automatic-endpoint-comparison.v1",
        order=order,
        endpoint_count=sum(len(run["rows"]) for run in runs),
        library_sha256={
            arm: runs[order.index(arm)]["library_sha256"] for arm in set(order)
        },
        slurm_job_id=runs[0]["slurm_job_id"],
        energy_parity_max=max_energy,
        force_parity_max=max_force,
        fock_iteration_force_work_and_capacity_parity=True,
        exact_transport_parity=not control_flow_differences,
        control_flow_differences=control_flow_differences,
        limit=1.05,
        passed=all(row["passed"] for row in rows),
        gates=rows,
        input_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("[0-7]-*.json"))
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for gate in result["gates"]:
        print(
            gate["system"],
            gate["method"],
            gate["phase"],
            f"{gate['ratio']:.4f}",
            gate["passed"],
        )
    if not result["passed"]:
        raise SystemExit(1)
